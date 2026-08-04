const PICK_SIGMA = 2.5;
// Sigmoid opacities never reach zero, so ignore imperceptible training outliers.
const PICK_OPACITY_THRESHOLD = 0.01;
// GaussianSplats3D 0.4.7 treats 0 as its default threshold of 1. A negative,
// truthy threshold retains every splat, including those quantized to alpha 0.
const SPLAT_ALPHA_THRESHOLD = -1;
const DRAG_THRESHOLD_PX = 6;
const SCENE_SCALE = 1;
const INITIAL_CAMERA_DISTANCE_FACTOR = 3.2;
const CAMERA_NEAR_RADIUS_FACTOR = 0.01;
const DEFAULT_CAMERA_DIRECTION = [0, 0.0625, 1];
// Contact point diameter as a fraction of the point-cloud bounding radius.
const CONTACT_POINT_SIZE = 0.03;

const stage = document.getElementById("demo-stage");
const viewerHost = document.getElementById("demo-viewer");
const moveHint = document.getElementById("demo-move-hint");
const moveHintText = document.getElementById("demo-move-hint-text");
const impactMoveHintText = moveHintText.textContent.trim();
const impactMarker = document.getElementById("impact-marker");
const modeButtons = Array.from(
  document.querySelectorAll(".demo-switcher-button[data-demo-mode]"),
);
const contactModeButton = modeButtons.find(
  (button) => button.dataset.demoMode === "contact",
);
const soundEditModeButton = modeButtons.find(
  (button) => button.dataset.demoMode === "editing",
);
const contactControls = document.getElementById("contact-impact-controls");
const contactButtons = Array.from(
  document.querySelectorAll(".contact-impact-button"),
);
const soundEditLegend = document.getElementById("sound-edit-legend");
const soundEditSourceMaterial = document.getElementById(
  "sound-edit-source-material",
);
const soundEditTargetMaterial = document.getElementById(
  "sound-edit-target-material",
);

let demo = null;
let viewer = null;
let viewerStarted = false;
let loadingPromise = null;
let activeMode = "impact";
let selectedCard = null;
let audioContext = null;
let audioWorkletPromise = null;
let activeAudioNode = null;
let activeRecordedAudio = null;
let THREE = null;
let GaussianSplats3D = null;
let libraryPromise = null;
let ContactPLYLoader = null;
let contactLibraryPromise = null;
let contactPoints = null;
let contactBounds = null;
let contactAbortController = null;
let contactRequestVersion = 0;
let contactLoading = false;
let activeContactImpactId = null;

impactMarker.addEventListener("animationend", () => {
  impactMarker.hidden = true;
  impactMarker.classList.remove("pulse");
});

function ensureViewerLibraries() {
  if (!libraryPromise) {
    libraryPromise = Promise.all([
      import("three"),
      import("@mkkellogg/gaussian-splats-3d"),
    ]).then(([threeModule, splatModule]) => {
      THREE = threeModule;
      GaussianSplats3D = splatModule;
    });
  }
  return libraryPromise;
}

function ensureContactLibrary() {
  if (!contactLibraryPromise) {
    contactLibraryPromise = Promise.all([
      ensureViewerLibraries(),
      import("three/addons/loaders/PLYLoader.js"),
    ]).then(([, plyModule]) => {
      ContactPLYLoader = plyModule.PLYLoader;
      return ContactPLYLoader;
    });
  }
  return contactLibraryPromise;
}

function setViewerState(message, busy = false) {
  viewerHost.setAttribute("aria-label", message);
  viewerHost.setAttribute("aria-busy", String(busy));
}

async function fetchChecked(url, type, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    throw new Error(`${response.status} while loading ${url}`);
  }
  return type === "json" ? response.json() : response.arrayBuffer();
}

function elementCount(shape) {
  return shape.reduce((product, value) => product * value, 1);
}

function binaryView(buffer, description) {
  const count = elementCount(description.shape);
  if (description.dtype === "float32") {
    return new Float32Array(buffer, description.byte_offset, count);
  }
  if (description.dtype === "float16") {
    return new Uint16Array(buffer, description.byte_offset, count);
  }
  throw new Error(`Unsupported binary dtype: ${description.dtype}`);
}

function readValue(view, description, index) {
  return description.dtype === "float16"
    ? THREE.DataUtils.fromHalfFloat(view[index])
    : view[index];
}

function parsePicking(manifest, buffer) {
  const arrays = manifest.picking.arrays;
  const picking = {
    count: manifest.n_gaussians,
    means: binaryView(buffer, arrays.means),
    scales: binaryView(buffer, arrays.scales),
    quaternions: binaryView(buffer, arrays.quaternions),
    opacities: binaryView(buffer, arrays.opacities),
  };
  for (let index = 0; index < picking.means.length; index += 1) {
    picking.means[index] *= SCENE_SCALE;
    picking.scales[index] *= SCENE_SCALE;
  }
  return picking;
}

function parseModal(manifest, buffer) {
  const arrays = manifest.modal.arrays;
  const views = {};
  Object.entries(arrays).forEach(([name, description]) => {
    views[name] = binaryView(buffer, description);
  });
  return { manifest, arrays, views };
}

function objectBounds(means, count) {
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  for (let index = 0; index < count; index += 1) {
    for (let axis = 0; axis < 3; axis += 1) {
      const value = means[index * 3 + axis];
      min[axis] = Math.min(min[axis], value);
      max[axis] = Math.max(max[axis], value);
    }
  }
  const center = min.map((value, axis) => (value + max[axis]) * 0.5);
  const radius =
    Math.hypot(max[0] - min[0], max[1] - min[1], max[2] - min[2]) *
    0.5;
  return { center, radius: Math.max(radius, 0.05) };
}

function pointCloudBounds(geometry) {
  geometry.computeBoundingBox();
  if (!geometry.boundingBox || geometry.boundingBox.isEmpty()) {
    throw new Error("Contact PLY has no visible points");
  }
  const center = new THREE.Vector3();
  const size = new THREE.Vector3();
  geometry.boundingBox.getCenter(center);
  geometry.boundingBox.getSize(size);
  const radius = size.length() * 0.5;
  if (!Number.isFinite(radius) || radius <= 0) {
    throw new Error("Contact PLY has invalid bounds");
  }
  return { center: center.toArray(), radius };
}

async function ensureAudio() {
  if (!window.AudioContext && !window.webkitAudioContext) {
    throw new Error("This browser does not support Web Audio.");
  }
  if (!audioContext || audioContext.state === "closed") {
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    audioContext = new AudioContextClass({
      latencyHint: "interactive",
      sampleRate: 44100,
    });
    const workletUrl = new URL("impact-audio-worklet.js", import.meta.url);
    audioWorkletPromise = audioContext.audioWorklet.addModule(workletUrl);
  }
  await Promise.all([audioContext.resume(), audioWorkletPromise]);
  return audioContext;
}

function cameraPosition(bounds, direction) {
  const length = Math.hypot(...direction);
  const distance = bounds.radius * INITIAL_CAMERA_DISTANCE_FACTOR;
  return bounds.center.map(
    (value, axis) => value + (direction[axis] * distance) / length,
  );
}

function createViewer(bounds, cameraDirection) {
  const [x, y, z] = bounds.center;
  const initialCameraPosition = cameraPosition(bounds, cameraDirection);
  const currentViewer = new GaussianSplats3D.Viewer({
    rootElement: viewerHost,
    cameraUp: [0, 1, 0],
    initialCameraLookAt: [x, y, z],
    initialCameraPosition,
    sharedMemoryForWorkers: false,
    gpuAcceleratedSort: false,
    integerBasedSort: false,
    sphericalHarmonicsDegree: 2,
    inMemoryCompressionLevel: 0,
    freeIntermediateSplatData: true,
    sceneRevealMode: GaussianSplats3D.SceneRevealMode.Instant,
  });
  // The built-in viewer treats a click as a request to move the orbit target.
  // Impact clicks must leave the camera unchanged; dragging still uses the
  // viewer's OrbitControls.
  currentViewer.onMouseClick = () => {};
  return currentViewer;
}

function configureCamera(currentViewer, bounds) {
  const camera = currentViewer.camera;
  const width = Math.max(viewerHost.clientWidth, 1);
  const height = Math.max(viewerHost.clientHeight, 1);
  camera.near = Math.max(
    bounds.radius * CAMERA_NEAR_RADIUS_FACTOR,
    0.0001,
  );
  if (camera.isPerspectiveCamera) {
    camera.aspect = width / height;
  }
  camera.updateProjectionMatrix();
  if (currentViewer.controls) {
    currentViewer.controls.minDistance = 0;
    currentViewer.controls.maxDistance = Infinity;
    currentViewer.controls.enableZoom = true;
    currentViewer.controls.enablePan = true;
    currentViewer.controls.update();
  }
  currentViewer.forceRenderNextFrame();
}

function frameViewer(
  currentViewer,
  bounds,
  cameraDirection = DEFAULT_CAMERA_DIRECTION,
) {
  const [x, y, z] = bounds.center;
  const position = cameraPosition(bounds, cameraDirection);
  configureCamera(currentViewer, bounds);
  currentViewer.camera.position.fromArray(position);
  currentViewer.camera.up.set(0, 1, 0);
  currentViewer.camera.lookAt(x, y, z);
  currentViewer.camera.updateMatrixWorld(true);
  if (currentViewer.controls) {
    currentViewer.controls.target.set(x, y, z);
    currentViewer.controls.update();
  }
  currentViewer.forceRenderNextFrame();
}

function setGaussianVisible(visible) {
  if (!viewer) return;
  const splatMesh = viewer.getSplatMesh?.() || viewer.splatMesh;
  if (splatMesh) splatMesh.visible = visible;
  viewer.forceRenderNextFrame();
}

function disposeContactPoints(points) {
  if (!points) return;
  points.removeFromParent();
  points.geometry.dispose();
  points.material.dispose();
}

function removeContactPoints() {
  disposeContactPoints(contactPoints);
  contactPoints = null;
  contactBounds = null;
  setGaussianVisible(true);
}

function stopRecordedAudio() {
  if (!activeRecordedAudio) return;
  activeRecordedAudio.pause();
  activeRecordedAudio.currentTime = 0;
  activeRecordedAudio = null;
}

function cancelContactRequest() {
  contactRequestVersion += 1;
  contactAbortController?.abort();
  contactAbortController = null;
  contactLoading = false;
}

function resetContactView({ frameGaussian = false } = {}) {
  cancelContactRequest();
  stopRecordedAudio();
  stopActiveImpact();
  removeContactPoints();
  activeContactImpactId = null;
  if (frameGaussian && viewer && demo) {
    frameViewer(viewer, demo.bounds, demo.cameraDirection);
  }
}

function stopActiveImpact() {
  if (!activeAudioNode) return;
  activeAudioNode.port.postMessage({ type: "stop" });
  activeAudioNode.disconnect();
  activeAudioNode = null;
}

async function loadObject(
  name,
  basePath,
  requestedVariant,
  cameraDirection,
) {
  if (!basePath) return;
  if (
    demo?.basePath === basePath &&
    demo?.variantName === requestedVariant
  ) {
    syncDemoModeUI();
    return;
  }
  if (loadingPromise) return loadingPromise;

  let loadFailed = false;
  viewerHost.hidden = false;
  moveHint.hidden = true;
  setViewerState(`Loading ${name}`, true);
  cancelContactRequest();
  stopActiveImpact();
  stopRecordedAudio();
  activeContactImpactId = contactPoints?.userData.impactId || null;
  const librariesReady = ensureViewerLibraries();

  loadingPromise = (async () => {
    const rootUrl = new URL(`${basePath}/manifest.json`, document.baseURI);
    const rootManifest = await fetchChecked(rootUrl, "json");
    const variantName = requestedVariant || rootManifest.default_variant;
    if (!rootManifest.variants[variantName]) {
      throw new Error(`${name} does not provide modal variant ${variantName}`);
    }
    const variantUrl = new URL(
      rootManifest.variants[variantName].manifest,
      rootUrl,
    );
    const variantManifest = await fetchChecked(variantUrl, "json");
    const pickingUrl = new URL(rootManifest.picking.uri, rootUrl);
    const modalUrl = new URL(variantManifest.modal.uri, variantUrl);
    const sceneUrl = new URL(rootManifest.scene.uri, rootUrl);

    const [pickingBuffer, modalBuffer] = await Promise.all([
      fetchChecked(pickingUrl, "buffer"),
      fetchChecked(modalUrl, "buffer"),
    ]);
    await librariesReady;
    const picking = parsePicking(rootManifest, pickingBuffer);
    const modal = parseModal(variantManifest, modalBuffer);
    const bounds = objectBounds(picking.means, picking.count);
    const oldSceneCount = viewer?.getSceneCount() || 0;
    if (!viewer) {
      viewer = createViewer(bounds, cameraDirection);
    }

    await viewer.addSplatScene(sceneUrl.href, {
      splatAlphaRemovalThreshold: SPLAT_ALPHA_THRESHOLD,
      showLoadingUI: true,
      progressiveLoad: false,
      scale: [SCENE_SCALE, SCENE_SCALE, SCENE_SCALE],
    });
    if (contactPoints) setGaussianVisible(false);
    if (oldSceneCount) {
      await viewer.removeSplatScenes(
        Array.from({ length: oldSceneCount }, (_, index) => index),
        false,
      );
    }
    if (!viewerStarted) {
      viewer.start();
      viewerStarted = true;
      installPointerInteraction(viewer);
    }
    demo = {
      name,
      basePath,
      variantName,
      rootManifest,
      rootUrl,
      viewer,
      picking,
      modal,
      bounds,
      cameraDirection,
    };
    removeContactPoints();
    activeContactImpactId = null;
    frameViewer(viewer, bounds, cameraDirection);
    viewer.forceRenderNextFrame();
    requestAnimationFrame(() => {
      viewer.update();
      viewer.render();
      viewer.forceRenderNextFrame();
    });
    syncDemoModeUI({ announce: false });
  })()
    .catch((error) => {
      loadFailed = true;
      console.error(error);
      syncDemoModeUI({ announce: false });
      if (demo) {
        viewerHost.hidden = false;
        setViewerState(
          `Unable to load ${name}: ${error.message} · ${demo.name} remains loaded`,
        );
      } else {
        setViewerState(`Unable to load ${name}: ${error.message}`);
      }
      throw error;
    })
    .finally(() => {
      loadingPromise = null;
      syncDemoModeUI({ announce: !loadFailed });
    });
  syncDemoModeUI({ announce: false });
  return loadingPromise;
}

async function loadModalVariant(variantName) {
  const currentDemo = demo;
  if (!currentDemo || loadingPromise) return false;
  if (currentDemo.variantName === variantName) return true;
  const variantRecord = currentDemo.rootManifest.variants[variantName];
  if (!variantRecord) return false;

  stopActiveImpact();
  setViewerState(`Loading ${currentDemo.name} ${variantName}`, true);
  loadingPromise = (async () => {
    const variantUrl = new URL(
      variantRecord.manifest,
      currentDemo.rootUrl,
    );
    const variantManifest = await fetchChecked(variantUrl, "json");
    const modalUrl = new URL(variantManifest.modal.uri, variantUrl);
    const modalBuffer = await fetchChecked(modalUrl, "buffer");
    if (demo !== currentDemo) return false;
    currentDemo.modal = parseModal(variantManifest, modalBuffer);
    currentDemo.variantName = variantName;
    return true;
  })();

  try {
    return await loadingPromise;
  } catch (error) {
    console.error(error);
    setViewerState(
      `Unable to load ${currentDemo.name} ${variantName}: ${error.message}`,
    );
    return false;
  } finally {
    loadingPromise = null;
    syncDemoModeUI({ announce: false });
  }
}

// Transform a world-space vector by R(q)^T, where q is stored as wxyz.
function inverseRotate(vx, vy, vz, w, x, y, z) {
  return [
    (1 - 2 * (y * y + z * z)) * vx +
      2 * (x * y + z * w) * vy +
      2 * (x * z - y * w) * vz,
    2 * (x * y - z * w) * vx +
      (1 - 2 * (x * x + z * z)) * vy +
      2 * (y * z + x * w) * vz,
    2 * (x * z + y * w) * vx +
      2 * (y * z - x * w) * vy +
      (1 - 2 * (x * x + y * y)) * vz,
  ];
}

function firstGaussianHit(ray, picking) {
  let bestIndex = -1;
  let bestDistance = Infinity;
  const sigmaSquared = PICK_SIGMA * PICK_SIGMA;

  for (let index = 0; index < picking.count; index += 1) {
    if (picking.opacities[index] < PICK_OPACITY_THRESHOLD) continue;
    const i3 = index * 3;
    const i4 = index * 4;
    const ox = ray.origin.x - picking.means[i3];
    const oy = ray.origin.y - picking.means[i3 + 1];
    const oz = ray.origin.z - picking.means[i3 + 2];
    const w = picking.quaternions[i4];
    const x = picking.quaternions[i4 + 1];
    const y = picking.quaternions[i4 + 2];
    const z = picking.quaternions[i4 + 3];
    const localOrigin = inverseRotate(ox, oy, oz, w, x, y, z);
    const localDirection = inverseRotate(
      ray.direction.x,
      ray.direction.y,
      ray.direction.z,
      w,
      x,
      y,
      z,
    );
    const sx = Math.max(picking.scales[i3], 1e-7);
    const sy = Math.max(picking.scales[i3 + 1], 1e-7);
    const sz = Math.max(picking.scales[i3 + 2], 1e-7);
    const px = localOrigin[0] / sx;
    const py = localOrigin[1] / sy;
    const pz = localOrigin[2] / sz;
    const dx = localDirection[0] / sx;
    const dy = localDirection[1] / sy;
    const dz = localDirection[2] / sz;
    const a = dx * dx + dy * dy + dz * dz;
    if (a <= 0) continue;
    // This closest-point form stays stable when a Gaussian axis is near zero.
    const closest = -(px * dx + py * dy + pz * dz) / a;
    const cx = px + closest * dx;
    const cy = py + closest * dy;
    const cz = pz + closest * dz;
    const closestRadiusSquared = cx * cx + cy * cy + cz * cz;
    if (closestRadiusSquared > sigmaSquared) continue;
    const halfSpan = Math.sqrt((sigmaSquared - closestRadiusSquared) / a);
    const near = closest - halfSpan;
    const far = closest + halfSpan;
    const distance = near > 1e-5 ? near : far > 1e-5 ? far : Infinity;
    if (distance < bestDistance) {
      bestDistance = distance;
      bestIndex = index;
    }
  }
  return bestIndex < 0 ? null : { index: bestIndex, distance: bestDistance };
}

function modalRow(modal, name, gaussianIndex) {
  const description = modal.arrays[name];
  const view = modal.views[name];
  const width = description.shape.at(-1);
  const row = description.shape.length === 1 ? 0 : gaussianIndex;
  const output = new Float32Array(width);
  for (let mode = 0; mode < width; mode += 1) {
    output[mode] = readValue(view, description, row * width + mode);
  }
  return output;
}

async function playImpact(modal, gaussianIndex) {
  const context = await ensureAudio();
  const frequencies = modalRow(modal, "frequencies", gaussianIndex);
  const gains = modalRow(modal, "gains", gaussianIndex);
  const dampings = modalRow(modal, "dampings", gaussianIndex);
  let gainBound = 0;
  gains.forEach((gain) => {
    gainBound += Math.abs(gain);
  });

  stopActiveImpact();
  const node = new AudioWorkletNode(context, "avmsf-impact-synth", {
    numberOfInputs: 0,
    numberOfOutputs: 1,
    outputChannelCount: [1],
    processorOptions: {
      frequencies,
      gains,
      dampings,
      duration: modal.manifest.audio.duration_seconds,
      outputGain: 0.8 / Math.max(gainBound, 1e-4),
    },
  });
  node.port.onmessage = (event) => {
    if (event.data === "ended") {
      node.disconnect();
      if (activeAudioNode === node) activeAudioNode = null;
    }
  };
  node.connect(context.destination);
  activeAudioNode = node;
}

function showImpactMarker(event) {
  const bounds = stage.getBoundingClientRect();
  impactMarker.style.left = `${event.clientX - bounds.left}px`;
  impactMarker.style.top = `${event.clientY - bounds.top}px`;
  impactMarker.hidden = false;
  impactMarker.classList.remove("pulse");
  void impactMarker.offsetWidth;
  impactMarker.classList.add("pulse");
}

function installPointerInteraction(currentViewer) {
  const canvas = currentViewer.renderer.domElement;
  const raycaster = new THREE.Raycaster();
  const pointer = new THREE.Vector2();
  let press = null;
  const impactPlaybackMode = () =>
    activeMode === "impact" || activeMode === "editing";

  canvas.addEventListener("pointerdown", (event) => {
    if (!impactPlaybackMode()) return;
    if (!event.isPrimary || event.button !== 0) return;
    press = {
      pointerId: event.pointerId,
      x: event.clientX,
      y: event.clientY,
      dragged: false,
    };
  });
  canvas.addEventListener("pointermove", (event) => {
    if (!press || event.pointerId !== press.pointerId) return;
    if (Math.hypot(event.clientX - press.x, event.clientY - press.y) > DRAG_THRESHOLD_PX) {
      press.dragged = true;
    }
  });
  canvas.addEventListener("pointercancel", () => {
    press = null;
  });
  canvas.addEventListener("pointerup", async (event) => {
    if (!impactPlaybackMode()) {
      press = null;
      return;
    }
    if (!press || event.pointerId !== press.pointerId) return;
    const dragged = press.dragged;
    press = null;
    if (dragged) return;
    const currentDemo = demo;
    if (!currentDemo || loadingPromise) return;

    const bounds = canvas.getBoundingClientRect();
    pointer.x = ((event.clientX - bounds.left) / bounds.width) * 2 - 1;
    pointer.y = -((event.clientY - bounds.top) / bounds.height) * 2 + 1;
    currentViewer.camera.updateMatrixWorld(true);
    raycaster.setFromCamera(pointer, currentViewer.camera);
    const hit = firstGaussianHit(raycaster.ray, currentDemo.picking);
    if (!hit) {
      setViewerState("No object hit · Click directly on the visible object");
      return;
    }

    try {
      await playImpact(currentDemo.modal, hit.index);
      showImpactMarker(event);
      const soundType = activeMode === "editing" ? "Edited impact" : "Impact";
      setViewerState(
        `${currentDemo.name} · ${soundType} at Gaussian ${hit.index}`,
      );
    } catch (error) {
      console.error(error);
      setViewerState(`Audio error: ${error.message}`);
    }
  });
}

const cards = Array.from(document.querySelectorAll(".object-card"));

function contactImpactIds(card) {
  const value = card?.dataset.contactImpacts?.trim();
  if (!value) return [];
  const ids = value
    .split(",")
    .map((value) => value.trim())
    .filter(Boolean);
  if (ids.length !== 2 || ids.some((value) => !/^\d{3}$/.test(value))) {
    throw new Error(
      `${card.dataset.name} must define two three-digit contact impacts`,
    );
  }
  return ids;
}

function syncContactButtons(ids) {
  const objectReady = Boolean(
    demo &&
      selectedCard &&
      demo.basePath === selectedCard.dataset.demoPath &&
      !loadingPromise,
  );
  contactButtons.forEach((button, index) => {
    const impactId = ids[index];
    button.hidden = !impactId;
    if (!impactId) return;
    button.dataset.impactId = impactId;
    button.querySelector(".contact-impact-label").textContent =
      `Impact ${index + 1}`;
    button.setAttribute(
      "aria-label",
      `Play Impact ${index + 1} and show its contact localization`,
    );
    const active = activeContactImpactId === impactId;
    const loading = active && contactLoading;
    button.classList.toggle("active", active);
    button.classList.toggle("loading", loading);
    button.setAttribute("aria-pressed", String(active));
    button.setAttribute("aria-busy", String(loading));
    button.disabled = !objectReady;
  });
}

function soundEditingAvailable() {
  return Boolean(
    demo &&
      selectedCard &&
      demo.basePath === selectedCard.dataset.demoPath &&
      demo.rootManifest.variants.sound_edit &&
      selectedCard.dataset.originalMaterial &&
      selectedCard.dataset.editedMaterial,
  );
}

function syncDemoModeUI({ announce = true } = {}) {
  const contactMode = activeMode === "contact";
  const editingMode = activeMode === "editing";
  const contactIds = contactImpactIds(selectedCard);
  const contactAvailable = contactIds.length === contactButtons.length;
  const editingAvailable = soundEditingAvailable();
  contactModeButton.disabled = !contactAvailable;
  if (contactAvailable) {
    contactModeButton.removeAttribute("title");
  } else {
    contactModeButton.title =
      "Contact localization is unavailable for this object";
  }
  soundEditModeButton.disabled = !editingAvailable;
  if (editingAvailable) {
    soundEditModeButton.removeAttribute("title");
  } else {
    soundEditModeButton.title = "Sound editing is unavailable for this object";
  }
  modeButtons.forEach((button) => {
    const active = button.dataset.demoMode === activeMode;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", String(active));
  });
  contactControls.hidden = !contactMode || !contactAvailable;
  soundEditLegend.hidden = !editingMode || !editingAvailable;
  if (editingAvailable) {
    soundEditSourceMaterial.textContent =
      selectedCard.dataset.originalMaterial;
    soundEditTargetMaterial.textContent = selectedCard.dataset.editedMaterial;
  }
  moveHintText.textContent = contactMode
    ? "Drag to rotate · Select an impact"
    : impactMoveHintText;
  moveHint.hidden = !demo;
  if (contactMode) impactMarker.hidden = true;
  syncContactButtons(contactIds);

  if (!announce || !demo || loadingPromise) return;
  if (contactLoading) {
    setViewerState(
      `Loading ${demo.name} contact localization for Impact ${activeContactImpactId}`,
      true,
    );
  } else if (contactMode && contactPoints) {
    setViewerState(
      `${demo.name} · Contact localization for Impact ${activeContactImpactId}`,
    );
  } else if (contactMode) {
    setViewerState(`${demo.name} ready · Select an impact on the left`);
  } else if (editingMode) {
    setViewerState(
      `${demo.name} ready · ${selectedCard.dataset.originalMaterial} to ` +
        `${selectedCard.dataset.editedMaterial} · Click the object to strike`,
    );
  } else {
    setViewerState(`${demo.name} ready · Click the object to strike`);
  }
}

async function setDemoMode(mode) {
  if (mode !== "impact" && mode !== "contact" && mode !== "editing") return;
  if (mode === "contact" && contactImpactIds(selectedCard).length === 0) return;
  if (mode === "editing" && !soundEditingAvailable()) return;
  if (loadingPromise) return;
  if (mode === activeMode) {
    syncDemoModeUI();
    return;
  }
  const shouldFrameGaussian = Boolean(
    demo && activeMode === "contact" && activeContactImpactId,
  );
  resetContactView({ frameGaussian: shouldFrameGaussian });
  if (mode !== "contact") {
    const variantName =
      mode === "editing" ? "sound_edit" : selectedCard.dataset.demoVariant;
    const cardBeingLoaded = selectedCard;
    const showLoading = demo?.variantName !== variantName;
    if (showLoading) {
      cardBeingLoaded.classList.add("loading");
      setObjectControlsBusy(true);
    }
    let loaded = false;
    try {
      loaded = await loadModalVariant(variantName);
    } finally {
      if (showLoading) {
        cardBeingLoaded.classList.remove("loading");
        setObjectControlsBusy(false);
      }
    }
    if (!loaded) {
      syncDemoModeUI();
      return;
    }
  }
  activeMode = mode;
  impactMarker.hidden = true;
  syncDemoModeUI();
}

function isCurrentContactRequest(version, expectedDemo, expectedCard, impactId) {
  return (
    version === contactRequestVersion &&
    activeMode === "contact" &&
    demo === expectedDemo &&
    selectedCard === expectedCard &&
    activeContactImpactId === impactId
  );
}

async function activateContactImpact(button) {
  const impactId = button.dataset.impactId;
  const expectedCard = selectedCard;
  const expectedDemo = demo;
  if (
    activeMode !== "contact" ||
    !impactId ||
    !expectedCard ||
    !expectedDemo ||
    loadingPromise ||
    expectedDemo.basePath !== expectedCard.dataset.demoPath
  ) {
    return;
  }

  cancelContactRequest();
  stopActiveImpact();
  stopRecordedAudio();
  activeContactImpactId = impactId;
  contactLoading = true;
  const requestVersion = ++contactRequestVersion;
  const abortController = new AbortController();
  contactAbortController = abortController;
  syncDemoModeUI();

  const contactBaseUrl = new URL(
    `${expectedCard.dataset.demoPath}/contact/`,
    document.baseURI,
  );
  const plyUrl = new URL(`impact${impactId}.ply`, contactBaseUrl);
  const wavUrl = new URL(`impact${impactId}.wav`, contactBaseUrl);
  const recording = new Audio(wavUrl.href);
  recording.preload = "auto";
  activeRecordedAudio = recording;
  recording.addEventListener(
    "ended",
    () => {
      if (activeRecordedAudio === recording) activeRecordedAudio = null;
    },
    { once: true },
  );

  let geometry = null;
  let material = null;
  try {
    const playbackPromise = recording.play();
    const [, , plyBuffer] = await Promise.all([
      playbackPromise,
      ensureContactLibrary(),
      fetchChecked(plyUrl, "buffer", { signal: abortController.signal }),
    ]);
    if (
      !isCurrentContactRequest(
        requestVersion,
        expectedDemo,
        expectedCard,
        impactId,
      )
    ) {
      return;
    }

    geometry = new ContactPLYLoader().parse(plyBuffer);
    const position = geometry.getAttribute("position");
    const color = geometry.getAttribute("color");
    if (!position?.count || !color || color.count !== position.count) {
      throw new Error("Contact PLY must contain matching XYZ and RGB vertices");
    }
    const bounds = pointCloudBounds(geometry);
    geometry.computeBoundingSphere();
    material = new THREE.PointsMaterial({
      size: bounds.radius * CONTACT_POINT_SIZE,
      sizeAttenuation: true,
      vertexColors: true,
    });
    const points = new THREE.Points(geometry, material);
    points.name = `${expectedDemo.name} contact Impact ${impactId}`;
    points.userData.impactId = impactId;

    if (
      !isCurrentContactRequest(
        requestVersion,
        expectedDemo,
        expectedCard,
        impactId,
      )
    ) {
      geometry.dispose();
      material.dispose();
      return;
    }

    const previousPoints = contactPoints;
    expectedDemo.viewer.threeScene.add(points);
    contactPoints = points;
    contactBounds = bounds;
    geometry = null;
    material = null;
    setGaussianVisible(false);
    disposeContactPoints(previousPoints);
    frameViewer(expectedDemo.viewer, bounds, expectedDemo.cameraDirection);
    contactAbortController = null;
    contactLoading = false;
    syncDemoModeUI();
  } catch (error) {
    geometry?.dispose();
    material?.dispose();
    abortController.abort();
    if (
      error.name === "AbortError" ||
      !isCurrentContactRequest(
        requestVersion,
        expectedDemo,
        expectedCard,
        impactId,
      )
    ) {
      return;
    }
    console.error(error);
    contactAbortController = null;
    contactLoading = false;
    stopRecordedAudio();
    activeContactImpactId = contactPoints?.userData.impactId || null;
    syncDemoModeUI({ announce: false });
    setViewerState(`Unable to load Impact ${impactId}: ${error.message}`);
  }
}

function setObjectControlsBusy(busy) {
  cards.forEach((card) => {
    card.disabled = busy;
  });
}

function selectCard(cardToSelect) {
  const switchingObjects = Boolean(
    selectedCard && selectedCard !== cardToSelect,
  );
  if (switchingObjects) {
    resetContactView();
    activeMode = "impact";
  } else if (
    activeMode === "contact" &&
    contactImpactIds(cardToSelect).length === 0
  ) {
    resetContactView({ frameGaussian: true });
    activeMode = "impact";
  }
  selectedCard = cardToSelect;
  cards.forEach((card) => {
    const active = card === cardToSelect;
    card.classList.toggle("active", active);
    card.setAttribute("aria-pressed", String(active));
  });
  syncDemoModeUI({ announce: false });
}

async function activateCard(card, selectImmediately = false) {
  if (!card || loadingPromise) return;
  if (
    card === selectedCard &&
    demo?.basePath === card.dataset.demoPath
  ) {
    if (activeMode !== "impact") await setDemoMode("impact");
    return;
  }
  let activated = false;
  if (selectImmediately) selectCard(card);
  card.classList.add("loading");
  setObjectControlsBusy(true);
  try {
    await loadObject(
      card.dataset.name,
      card.dataset.demoPath,
      card.dataset.demoVariant,
      card.dataset.cameraDirection
        ? card.dataset.cameraDirection.split(",").map(Number)
        : DEFAULT_CAMERA_DIRECTION,
    );
    selectCard(card);
    activated = true;
  } catch (_error) {
    // loadObject reports the error through the viewer's accessible label.
  } finally {
    card.classList.remove("loading");
    setObjectControlsBusy(false);
    syncDemoModeUI({ announce: activated });
  }
}

cards.forEach((card) => {
  card.addEventListener("click", () => activateCard(card));
});

modeButtons.forEach((button) => {
  button.addEventListener("click", () => setDemoMode(button.dataset.demoMode));
});

contactButtons.forEach((button) => {
  button.addEventListener("pointerdown", (event) => event.stopPropagation());
  button.addEventListener("click", (event) => {
    event.stopPropagation();
    activateContactImpact(button);
  });
});

window.addEventListener("resize", () => {
  if (viewer && demo) configureCamera(viewer, contactBounds || demo.bounds);
});
activateCard(cards[0], true);
