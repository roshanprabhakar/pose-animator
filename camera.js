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

// TODO implement finding multiple poses
// TODO modularize receive and transmit

import * as posenet_module from '@tensorflow-models/posenet';
import * as facemesh_module from '@tensorflow-models/facemesh';
import * as tf from '@tensorflow/tfjs';
import * as paper from 'paper';
import dat from 'dat.gui';
import Stats from 'stats.js';
import 'babel-polyfill';

import {drawKeypoints, drawPoint, drawSkeleton, isMobile, toggleLoadingUI, setStatusText} from './utils/demoUtils';

import {SVGUtils} from './utils/svgUtils';
import {PoseIllustration} from './illustrationGen/illustration';
import {Skeleton, facePartName2Index} from './illustrationGen/skeleton';
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

// references for render setup
const keypointCanvas = document.getElementById('keypoints');
const canvas = document.getElementById('output');
const keypointCtx = keypointCanvas.getContext('2d');
const videoCtx = canvas.getContext('2d');

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

// summations for finding necessary statistics
let previousTime;
let previousBytesIntegral = 0;

function getOtherPeerConnection(pc) {
    if (pc === pc1) {
        return pc2;
    } else {
        return pc1;
    }
}

/**
 * Adds the passed candidate to the opposite node of the connection provided
 *
 */
function onIceCandidate(pc, event) {
    (getOtherPeerConnection(pc)).addIceCandidate(event.candidate);
}

/**
 * Connects the two peer connections, adds handlers for messages received
 * and data channel detected.
 *
 */
async function initiateRtcStreamingChannel() {

    // setting up pc1 (receiving end)
    pc1 = new RTCPeerConnection({});
    pc1.addEventListener('icecandidate', e => onIceCandidate(pc1, e));

    // creates the data channel at the receiving end, transmission starts when the transmitting end detects this channel
    const dataChannel = pc1.createDataChannel('pose-animator data channel');

    // for messages received, parse the transmitted arrays as poses and project them
    let message = [];
    let faceDetection;
    dataChannel.onmessage = function(event) {

        message.push(event.data);

        if (message.length === 4) {

            // builds pose object
            let pose = reconstructPose(new Int16Array(message[0]), new Int16Array(message[1]));

            // clears the output canvas
            canvasScope.project.clear();

            // projects the poses skeleton on the existing svg skeleton
            Skeleton.flipPose(pose);
            illustration.updateSkeleton(pose, null);
            // illustration.draw(canvasScope, videoWidth, videoHeight);
            if (guiState.debug.showIllustrationDebug) {
                illustration.debugDraw(canvasScope);
            }

            canvasScope.project.activeLayer.scale(
                canvasWidth / videoWidth,
                canvasHeight / videoHeight,
                new canvasScope.Point(0, 0));

            // faceDetection = JSON.parse(message[2]);
            let faceData = message[2];
            if (faceData !== 0) {


                let face = {
                    positions: reconstructFaceData(message[2]),
                    faceInViewConfidence: message[3]
                };

                illustration.updateSkeleton(pose, face);

                // if (faceDetection && faceDetection.length > 0) {
                //     let face = Skeleton.toFaceFrame(faceDetection[0]);
                //     illustration.updateSkeleton(pose, face);
                // }
                illustration.draw(canvasScope, videoWidth, videoHeight);
            }
            message = [];
        }
    };

    // setting up pc2 (transmitting end)
    pc2 = new RTCPeerConnection({});
    pc2.addEventListener('icecandidate', e => onIceCandidate(pc2, e));

    // sets the pc2 data channel to the global context
    pc2.ondatachannel = function(event) {
        channel = event.channel;
    };

    let statsInterval = window.setInterval(getConnectionStats, 1000);

    // connects pc1 and pc2
    let offer = await pc1.createOffer({
        offerToReceiveAudio: 0,
        offerToReceiveVideo: 0,
    });

    await pc2.setRemoteDescription(offer);
    await pc1.setLocalDescription(offer);

    let answer = await pc2.createAnswer();

    await pc1.setRemoteDescription(answer);
    await pc2.setLocalDescription(answer);
}

// in: buffer for a Uint32Array
function reconstructFaceData(positionsBuffer) {
    let view = new Float32Array(positionsBuffer);
    let out = [];

    view.forEach(coordinate => {
        out.push(coordinate);
    });

    return out;
}

/**
 * Loops the transmission of deconstructed poses
 *
 */
async function transmit() {

    // Begin monitoring code for frames per second
    stats.begin();

    // get face information
    const input = tf.browser.fromPixels(canvas);
    faceDetection = await facemesh.estimateFaces(input, false, false);
    input.dispose();

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

    // clears previous render
    videoCtx.clearRect(0, 0, videoWidth, videoHeight);

    // draw video
    videoCtx.save();
    videoCtx.scale(-1, 1);
    videoCtx.translate(-videoWidth, 0);
    videoCtx.drawImage(video, 0, 0, videoWidth, videoHeight);
    videoCtx.restore();

    // projects pose and face onto svg
    keypointCtx.clearRect(0, 0, videoWidth, videoHeight);
    if (guiState.debug.showDetectionDebug) {
        poses.forEach(({score, keypoints}) => {
            if (score >= minPoseConfidence) {
                drawKeypoints(keypoints, minPartConfidence, keypointCtx);
                drawSkeleton(keypoints, minPartConfidence, keypointCtx);
            }
        });
        faceDetection.forEach(face => {
            for (let i = 0; i < face.scaledMesh.length; i++) {
                let p = face.scaledMesh[i];
                drawPoint(keypointCtx, p[1], p[0], 2, 'red');
            }
            //
            // Object.values(facePartName2Index).forEach(index => {
            //     let p = face.scaledMesh[index];
            //     drawPoint(keypointCtx, p[1], p[0], 2, 'red');
            // });
        });
    }

    // converts pose to streamable buffers
    let deconstructedPose = deconstructPose(poses[0]);

    // deconstructedPose === null if difference between consecutive frames is 0
    if (deconstructedPose !== null) {
        channel.send(deconstructedPose[0].buffer);
        channel.send(deconstructedPose[1].buffer);
    }

    // channel.send(JSON.stringify(faceDetection));

    if (faceDetection && faceDetection.length > 0) {
        // let face = Skeleton.toFaceFrame(faceDetection[0]);
        let face = Skeleton.toBufferedFaceFrame(faceDetection[0]);
        channel.send(face.positions.buffer);
        channel.send(face.faceInViewConfidence);
    } else {
        channel.send(0);
        channel.send(0);
    }

    // channel.send(JSON.stringify(Skeleton.toFaceFrame(faceDetection[0])));

    // End monitoring code for frames per second
    stats.end();

    // loop back
    setTimeout(transmit, 10);
}

/**
 * Converts a pose object to streamable array views, the corresponding
 * buffers are streamed
 *
 */
function deconstructPose(pose) {
    if (pose == null) return null;

    let confidences = new Int16Array(18);
    let positions = new Int16Array(34);

    confidences[0] = 10000 * pose.score; // to reduce transmission size
    for (let i = 0; i < pose.keypoints.length; i++) {
        confidences[i + 1] = 10000 * pose.keypoints[i].score;
        positions[i * 2] = pose.keypoints[i].position.x;
        positions[i * 2 + 1] = pose.keypoints[i].position.y;
    }

    return [confidences, positions];
}

/**
 * Converts streamed arrays (after view initialized) into a pose object for
 * animation rendering.
 *
 */
function reconstructPose(confidences, positions) {

    let pose = {
        'score': confidences[0] / 10000,
        'keypoints': [],
    };
    for (let i = 0; i < 17; i += 1) {
        pose.keypoints.push({
            'score': confidences[i + 1] / 10000,
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
 *
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
 *
 */
function setupFPS() {
    stats.showPanel(0);
    document.getElementById('main').appendChild(stats.dom);
}

// more render configuration
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

// initiates svg skeleton to be used
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

/**
 * Monitors inbound byte stream for the calculation of network transmission rate
 *
 */
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

/**
 * Sets up local and receiving renderers
 */
function configureRender() {
    canvas.width = videoWidth;
    canvas.height = videoHeight;
    keypointCanvas.width = videoWidth;
    keypointCanvas.height = videoHeight;
}


bindPage().then(initiateRtcStreamingChannel).then(configureRender).then(startTimer).then(transmit);
