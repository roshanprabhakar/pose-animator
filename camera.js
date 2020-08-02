/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */


    // bandwidth limit starts off as unlimited, but can be changed by entering a value in
    // the input box. This updates the session storage value and reloads the page.
    // -------------------------------------------
var bandwidthLimit = sessionStorage.getItem('bandwidthLimit'); //kbits/second
if (bandwidthLimit === null) {
    bandwidthLimit = 'unlimited';
    // EDIT THIS VALUE TO TOGGLE INITIAL BANDWIDTH LIMIT
}
// -------------------------------------------


import * as posenet_module from '@tensorflow-models/posenet';
import * as facemesh_module from '@tensorflow-models/facemesh';
import * as paper from 'paper';
import dat from 'dat.gui';
import Stats from 'stats.js';
import 'babel-polyfill';

import {
    drawKeypoints,
    drawSkeleton,
    isMobile,
    setStatusText,
    toggleLoadingUI,
} from './utils/demoUtils';
import {SVGUtils} from './utils/svgUtils';
import {PoseIllustration} from './illustrationGen/illustration';
import {Skeleton} from './illustrationGen/skeleton';
import {FileUtils} from './utils/fileUtils';

import * as girlSVG from './resources/illustration/girl.svg';
import * as boySVG from './resources/illustration/boy.svg';
import * as abstractSVG from './resources/illustration/abstract.svg';
import * as blathersSVG from './resources/illustration/blathers.svg';
import * as tomNookSVG from './resources/illustration/tom-nook.svg';

// Camera stream video element
let video;
let videoWidth = 500;
let videoHeight = 500;

// Canvas
let faceDetection = null;
let illustration = null;
let canvasScope;
let canvasWidth = 500;
let canvasHeight = 500;

// ML models
let facemesh;
let posenet;
let minPoseConfidence = 0.15;
let minPartConfidence = 0.1;
let nmsRadius = 30.0;

// Misc
let mobile = false;
const stats = new Stats();
const avatarSvgs = {
    'girl': girlSVG.default,
    'boy': boySVG.default,
    'abstract': abstractSVG.default,
    'blathers': blathersSVG.default,
    'tom-nook': tomNookSVG.default,
};

const bandwidthButton = document.querySelector('input#bandwidth_button');
const bandwidthInput = document.querySelector('input#bandwidth_input');

// WebRTC connection nodes
let pc1;
let pc2;

// WebRTC streaming channel
let channel;

// Analysis monitors
// const monitors = ['bytesReceived', 'packetsReceived', 'headerBytesReceived', 'packetsLost', 'totalDecodeTime', 'totalInterFrameDelay', 'codecId'];
const monitors = ['bytesReceived'];

// order list for poses deconstruction and reconstruction
const parts = ['nose', 'leftEye', 'rightEye', 'leftEar', 'rightEar', 'leftShoulder', 'rightShoulder', 'leftElbow', 'rightElbow', 'leftWrist', 'rightWrist', 'leftHip', 'rightHip', 'leftKnee', 'rightKnee', 'leftAnkle', 'rightAnkle'];

let previousTime;
let previousBytesIntegral = 0;

function getOtherPeerConnection(pc) {
    if (pc === pc1) {
        return pc2;
    } else {
        return pc1;
    }
}

function onIceCandidate(pc, event) {
    (getOtherPeerConnection(pc)).addIceCandidate(event.candidate);
}

async function initiateRtcStreamingChannel() {

    // setting up pc1 (receiving end)
    pc1 = new RTCPeerConnection({});
    pc1.addEventListener('icecandidate', e => onIceCandidate(pc1, e));

    const dataChannel = pc1.createDataChannel('pose-animator data channel');

    let message = [];
    dataChannel.onmessage = function(event) {

        message.push(event.data);

        if (message.length === 2) {

            // console.log("received: ");
            // console.log(new Float32Array(message[0]));
            // console.log(new Float32Array(message[1]));

            let poses = [reconstructPose(new Float32Array(message[0]), new Float32Array(message[1]))];
            // console.log(JSON.stringify(poses));

            // clears the output canvas
            canvasScope.project.clear();

            // projects the poses skeleton on the existing svg skeleton
            Skeleton.flipPose(poses[0]);
            illustration.updateSkeleton(poses[0], null);
            illustration.draw(canvasScope, videoWidth, videoHeight);
            if (guiState.debug.showIllustrationDebug) {
                illustration.debugDraw(canvasScope);
            }

            canvasScope.project.activeLayer.scale(
                canvasWidth / videoWidth,
                canvasHeight / videoHeight,
                new canvasScope.Point(0, 0));

            message = [];
        }
    };

    // setting up pc2 (transmitting end)
    pc2 = new RTCPeerConnection({});
    pc2.addEventListener('icecandidate', e => onIceCandidate(pc2, e));

    pc2.ondatachannel = function(event) {
        channel = event.channel;
    };

    let statsInterval = window.setInterval(getConnectionStats, 100);

    // connects pc1 and pc2
    let offer = await pc1.createOffer({
        offerToReceiveAudio: 0,
        offerToReceiveVideo: 0,
    });
    offer.sdp = setMediaBitrate(offer.sdp, 'application', bandwidthLimit);

    await pc2.setRemoteDescription(offer);
    await pc1.setLocalDescription(offer);

    let answer = await pc2.createAnswer();
    answer.sdp = setMediaBitrate(answer.sdp, 'application', bandwidthLimit);

    await pc1.setRemoteDescription(answer);
    await pc2.setLocalDescription(answer);

    // get elements for pose animator to access
    const canvas = document.getElementById('output');
    const keypointCanvas = document.getElementById('keypoints');
    const videoCtx = canvas.getContext('2d');
    const keypointCtx = keypointCanvas.getContext('2d');

    // setup html dimensions
    canvas.width = videoWidth;
    canvas.height = videoHeight;
    keypointCanvas.width = videoWidth;
    keypointCanvas.height = videoHeight;
}

async function transmit() {
    // Begin monitoring code for frames per second
    stats.begin();

    // initializes poses
    let poses = [];

    // populates poses
    let all_poses = await posenet.estimatePoses(video, {
        flipHorizontal: true,
        decodingMethod: 'multi-person',
        maxDetections: 1,
        scoreThreshold: minPartConfidence,
        nmsRadius: nmsRadius,
    });

    // merges all poses
    poses = poses.concat(all_poses);

    const keypointCanvas = document.getElementById('keypoints');
    const canvas = document.getElementById('output');
    const keypointCtx = keypointCanvas.getContext('2d');
    const videoCtx = canvas.getContext('2d');

    videoCtx.clearRect(0, 0, videoWidth, videoHeight);
    // Draw video
    videoCtx.save();
    videoCtx.scale(-1, 1);
    videoCtx.translate(-videoWidth, 0);
    videoCtx.drawImage(video, 0, 0, videoWidth, videoHeight);
    videoCtx.restore();

    keypointCtx.clearRect(0, 0, videoWidth, videoHeight);
    if (guiState.debug.showDetectionDebug) {
        poses.forEach(({score, keypoints}) => {
            if (score >= minPoseConfidence) {
                drawKeypoints(keypoints, minPartConfidence, keypointCtx);
                drawSkeleton(keypoints, minPartConfidence, keypointCtx);
            }
        });
    }


    let deconstructedPose = deconstructPose(poses[0]);

    // console.log("to be transmitted: ")
    // console.log(deconstructedPose[0]);
    // console.log(deconstructedPose[1]);

    channel.send(deconstructedPose[0].buffer);
    channel.send(deconstructedPose[1].buffer);

    // // transmit poses representation
    // channel.send(JSON.stringify(poses));

//    channel.send(new Date().getTime());

    // End monitoring code for frames per second
    stats.end();

    // loop back
    setTimeout(transmit, 10);
}

// pass in pose[0]
function deconstructPose(pose) {

    // let confidences = [];
    // let positions = [];

    let confidences = new Float32Array(18);
    let positions = new Float32Array(34);

    confidences[0] = pose.score;
    for (let i = 0; i < pose.keypoints.length; i++) {
        confidences[i + 1] = pose.keypoints[i].score;
        positions[i * 2] = pose.keypoints[i].position.x;
        positions[i * 2 + 1] = pose.keypoints[i].position.y;
    }

    // let transmittableConfidences = new Uint8Array(confidences);
    // let transmittablePositions = new Uint16Array(positions);

    // let transmittableConfidences = new Float32Array(confidences);
    // let transmittablePositions = new Float32Array(positions);

    // return [transmittableConfidences, transmittablePositions];

    return [confidences, positions];
}

// reconstructs poses[0]
function reconstructPose(confidences, positions) {

    // confidences = Uint8Array [18], 1st belongs to general, following 17 belong to each keypoint
    // positions = Uint16Array [34], each pair belongs to one of the 14 keypoints

    let pose = {
        'score': confidences[0],
        'keypoints': [],
    };
    for (let i = 0; i < 17; i += 1) {
        pose.keypoints.push({
            'score': confidences[i + 1],
            'part': parts[i],
            'position': {
                'x': positions[i * 2],
                'y': positions[i * 2 + 1],
            },
        });
    }
    return pose;
}


/**
 * Loads a the camera to be used in the demo
 *
 */
async function setupCamera() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error(
            'Browser API navigator.mediaDevices.getUserMedia not available');
    }

    const video = document.getElementById('video');
    video.width = videoWidth;
    video.height = videoHeight;

    const stream = await navigator.mediaDevices.getUserMedia({
        'audio': false,
        'video': {
            facingMode: 'user',
            width: videoWidth,
            height: videoHeight,
        },
    });
    video.srcObject = stream;

    return new Promise((resolve) => {
        video.onloadedmetadata = () => {
            resolve(video);
        };
    });
}

async function loadVideo() {
    const video = await setupCamera();
    video.play();

    return video;
}

const defaultPoseNetArchitecture = 'MobileNetV1';
const defaultQuantBytes = 2;
const defaultMultiplier = 1.0;
const defaultStride = 16;
const defaultInputResolution = 200;

const guiState = {
    avatarSVG: Object.keys(avatarSvgs)[0],
    debug: {
        showDetectionDebug: true,
        showIllustrationDebug: false,
    },
};

/**
 * Sets up dat.gui controller on the top-right of the window
 */
function setupGui(cameras) {

    if (cameras.length > 0) {
        guiState.camera = cameras[0].deviceId;
    }

    const gui = new dat.GUI({width: 300});

    let multi = gui.addFolder('Image');
    gui.add(guiState, 'avatarSVG', Object.keys(avatarSvgs)).onChange(() => parseSVG(avatarSvgs[guiState.avatarSVG]));
    multi.open();

    let output = gui.addFolder('Debug control');
    output.add(guiState.debug, 'showDetectionDebug');
    output.add(guiState.debug, 'showIllustrationDebug');
    output.open();
}

/**
 * Sets up a frames per second panel on the top-left of the window
 */
function setupFPS() {
    stats.showPanel(0);
    document.getElementById('main').appendChild(stats.dom);
}

function setupCanvas() {
    mobile = isMobile();
    if (mobile) {
        canvasWidth = Math.min(window.innerWidth, window.innerHeight);
        canvasHeight = canvasWidth;
        videoWidth *= 0.7;
        videoHeight *= 0.7;
    }

    canvasScope = paper.default;
    let canvas = document.querySelector('.illustration-canvas');
    canvas.width = canvasWidth;
    canvas.height = canvasHeight;
    canvasScope.setup(canvas);
}

/**
 * Kicks off the demo by loading the posenet model, finding and loading
 * available camera devices, and setting off pose transmission device.
 */
export async function bindPage() {
    setupCanvas();

    toggleLoadingUI(true);
    setStatusText('Loading PoseNet model...');
    posenet = await posenet_module.load({
        architecture: defaultPoseNetArchitecture,
        outputStride: defaultStride,
        inputResolution: defaultInputResolution,
        multiplier: defaultMultiplier,
        quantBytes: defaultQuantBytes,
    });
    setStatusText('Loading FaceMesh model...');
    facemesh = await facemesh_module.load();

    setStatusText('Loading Avatar file...');
    let t0 = new Date();
    await parseSVG(Object.values(avatarSvgs)[0]);

    setStatusText('Setting up camera...');
    try {
        video = await loadVideo();
    } catch (e) {
        let info = document.getElementById('info');
        info.textContent = 'this device type is not supported yet, ' +
            'or this browser does not support video capture: ' + e.toString();
        info.style.display = 'block';
        throw e;
    }

    setupGui([], posenet);
    setupFPS();

    toggleLoadingUI(false);
}

navigator.getUserMedia = navigator.getUserMedia ||
    navigator.webkitGetUserMedia || navigator.mozGetUserMedia;
FileUtils.setDragDropHandler((result) => {
    parseSVG(result);
});

async function parseSVG(target) {
    let svgScope = await SVGUtils.importSVG(target /* SVG string or file path */);
    let skeleton = new Skeleton(svgScope);
    illustration = new PoseIllustration(canvasScope);
    illustration.bindSkeleton(skeleton, svgScope);
}

// monitors inbound bytestream according to provided monitors
function getConnectionStats() {

    let taken = [];
    pc1.getStats(null).then(stats => {
        let statsOutput = '';

        stats.forEach(report => {
            if (!report.id.startsWith('RTCDataChannel_')) return;
            Object.keys(report).forEach(statName => {
                if (monitors.includes(statName)) {

                    let bytesIntegral = parseInt(report[statName]);


                    if (bytesIntegral !== 0 && !taken.includes(statName)) {
                        let currentTime = new Date().getTime();
                        let timeIntegral = (currentTime - previousTime) / 1000;

                        let kbytesPerSecond = (bytesIntegral - previousBytesIntegral) / timeIntegral / 1000;
                        previousBytesIntegral = bytesIntegral;
                        previousTime = currentTime;
                        if (statName === 'bytesReceived') {
                            statsOutput += `<strong>kilobit rate: </strong> ${(kbytesPerSecond * 8).toFixed(2)} kb/s <br>`;
                            taken.push(statName);
                        } else {
                            statsOutput += `<strong>${statName}:</strong> ${kbytesPerSecond * 8} kb/s <br>`;
                            taken.push(statName);
                        }
                    }
                }
            });
        });
        document.querySelector('#bitstream-box').innerHTML = statsOutput;
    });
    return 0;
}

function startTimer() {
    previousTime = new Date().getTime();
}

function setMediaBitrate(sdp, media, bitrate) {
    if (bandwidthLimit === 'unlimited') {
        return sdp;
    }
    bandwidthLimit = parseInt(bandwidthLimit);

    var lines = sdp.split('\n');
    var line = -1;
    for (var i = 0; i < lines.length; i++) {
        if (lines[i].indexOf('m=' + media) === 0) {
            line = i;
            break;
        }
    }
    if (line === -1) {
        console.debug('Could not find the m line for', media);
        return sdp;
    }
    console.debug('Found the m line for', media, 'at line', line);

    // Pass the m line
    line++;

    // Skip i and c lines
    while (lines[line].indexOf('i=') === 0 || lines[line].indexOf('c=') === 0) {
        line++;
    }

    // If we're on a b line, replace it
    if (lines[line].indexOf('b') === 0) {
        console.debug('Replaced b line at line', line);
        lines[line] = 'b=AS:' + bitrate;
        return lines.join('\n');
    }

    // Add a new b line
    console.debug('Adding new b line before line', line);
    var newLines = lines.slice(0, line);
    newLines.push('b=AS:' + bitrate);
    newLines = newLines.concat(lines.slice(line, lines.length));

    return newLines.join('\n');
}

// when button clicked set new bandwidthLimit to session storage and reload
bandwidthButton.onclick = () => {
    bandwidthLimit = document.getElementById('bandwidth_input').value;
    sessionStorage.setItem('bandwidthLimit', bandwidthLimit);
    location.reload();
};

// Execute a function when the user releases a key on the keyboard
bandwidthInput.addEventListener('keyup', function(event) {
    // Number 13 is the "Enter" key on the keyboard
    if (event.keyCode === 13) {
        // Cancel the default action, if needed
        event.preventDefault();
        // Trigger the button element with a click
        document.getElementById('bandwidth_button').click();
    }
});


bindPage().then(initiateRtcStreamingChannel).then(startTimer).then(transmit);

document.querySelector('#bitratelimit-box').innerHTML = `<strong>bitrate limit:</strong> ${bandwidthLimit} kb/s`;
