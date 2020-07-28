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

import * as posenet_module from '@tensorflow-models/posenet';
import * as facemesh_module from '@tensorflow-models/facemesh';
import * as tf from '@tensorflow/tfjs';
import * as paper from 'paper';
import dat from 'dat.gui';
import Stats from 'stats.js';
import 'babel-polyfill';

import {
    drawKeypoints,
    drawPoint,
    drawSkeleton,
    isMobile,
    toggleLoadingUI,
    setStatusText,
} from './utils/demoUtils';
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
let videoWidth = 300;
let videoHeight = 300;

// Canvas
let faceDetection = null;
let illustration = null;
let canvasScope;
let canvasWidth = 800;
let canvasHeight = 800;

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

// WebRTC connection nodes
let pc1;
let pc2;

// WebRTC streaming channel
let channel;

function getOtherPeerConnection(pc) {
    if (pc === pc1) {
        return pc2;
    } else return pc1;
}

function onIceCandidate(pc, event) {
    (getOtherPeerConnection(pc)).addIceCandidate(event.candidate);
}

async function initiateRtcStreamingChannel() {

    // setting up pc1 (receiving end)
    pc1 = new RTCPeerConnection({});
    pc1.addEventListener('icecandidate', e => onIceCandidate(pc1, e));

    const dataChannel = pc1.createDataChannel("pose-animator data channel");

    dataChannel.onmessage = function (event) {
        let poses = JSON.parse(event.data);

        // clears the output canvas
        keypointCtx.clearRect(0, 0, videoWidth, videoHeight);
        canvasScope.project.clear();

        // projects the poses skeleton on the existing svg skeleton
        Skeleton.flipPose(poses[0]);
        illustration.updateSkeleton(poses[0], null);
        illustration.draw(canvasScope, videoWidth, videoHeight);

        canvasScope.project.activeLayer.scale(
            canvasWidth / videoWidth,
            canvasHeight / videoHeight,
            new canvasScope.Point(0, 0));
    }

    // setting up pc2 (transmitting end)
    pc2 = new RTCPeerConnection({});
    pc2.addEventListener('icecandidate', e => onIceCandidate(pc2, e));

    pc2.ondatachannel = function (event) {
        channel = event.channel;
    }

    // connects pc1 and pc2
    let offer = await pc1.createOffer({offerToReceiveAudio: 0, offerToReceiveVideo: 0});

    await pc2.setRemoteDescription(offer);
    await pc1.setLocalDescription(offer);

    let answer = await pc2.createAnswer();

    await pc1.setRemoteDescription(answer);
    await pc2.setLocalDescription(answer);

    // get elements for pose animator to access
    const canvas = document.getElementById("output");
    const keypointCanvas = document.getElementById("keypoints");
    const videoCtx = canvas.getContext('2d');
    const keypointCtx = keypointCanvas.getContext('2d');

    // setup html dimensions
    canvas.width = videoWidth;
    canvas.height = videoHeight;
    keypointCanvas.width = videoWidth;
    keypointCanvas.height = videoHeight;
}

async function transmit() {

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

    // transmit poses object
    channel.send(JSON.stringify(poses));

    // loop back
    setTimeout(transmit, 10);
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

bindPage().then(initiateRtcStreamingChannel).then(transmit);
