(() => {
  "use strict";

  const state = { capabilities: null, result: null, nextEmitter: 1, audioUrl: null };
  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));
  const svgNS = "http://www.w3.org/2000/svg";
  const palette = ["#f0b45c", "#79e5ee", "#ae98ff", "#7ae2a5", "#ff858b"];
  const controls = {
    mic: $("#mic-preset"), room: $("#room-preset"), processing: $("#processing"),
    duration: $("#duration"), noise: $("#noise"), run: $("#run-button"),
    reset: $("#reset-button"), export: $("#export-button"), add: $("#add-emitter"),
  };

  function formatSigned(value, digits = 0) {
    const number = Number(value);
    return `${number < 0 ? "−" : ""}${Math.abs(number).toFixed(digits)}`;
  }

  function svgElement(name, attributes = {}) {
    const element = document.createElementNS(svgNS, name);
    Object.entries(attributes).forEach(([key, value]) => element.setAttribute(key, String(value)));
    return element;
  }

  function emitterCards() { return $$(".emitter-card", $("#emitter-list")); }

  function addEmitter(seed = {}) {
    if (emitterCards().length >= 8) return;
    const fragment = $("#emitter-template").content.cloneNode(true);
    const card = $(".emitter-card", fragment);
    const index = state.nextEmitter++;
    $(".emitter-name", card).value = seed.id || (index === 1 ? "speaker" : `source_${index}`);
    $(".emitter-sound", card).value = seed.sound || (index === 1 ? "broadband" : "alarm");
    $(".emitter-bearing", card).value = seed.bearing_deg ?? (index === 1 ? 38 : -55);
    $(".emitter-distance", card).value = seed.distance_m ?? (index === 1 ? 2 : 2.8);
    $(".emitter-gain", card).value = seed.gain_db ?? (index === 1 ? -3 : -8);
    $(".emitter-onset", card).value = seed.onset_s ?? (index === 1 ? .1 : .5);
    $(".emitter-enabled", card).checked = seed.enabled ?? true;
    $(".remove-emitter", card).addEventListener("click", () => { card.remove(); updateScene(); });
    $$("input, select", card).forEach((input) => input.addEventListener("input", updateScene));
    $("#emitter-list").appendChild(fragment);
    updateScene();
  }

  function readEmitter(card, index) {
    return {
      id: $(".emitter-name", card).value.trim() || `source_${index + 1}`,
      sound: $(".emitter-sound", card).value,
      directivity: $(".emitter-directivity", card).value,
      bearing_deg: Number($(".emitter-bearing", card).value),
      distance_m: Number($(".emitter-distance", card).value),
      gain_db: Number($(".emitter-gain", card).value),
      onset_s: Number($(".emitter-onset", card).value),
      duration_s: null, loop: true, enabled: $(".emitter-enabled", card).checked,
    };
  }

  function collectScenario() {
    return {
      duration_s: Number(controls.duration.value), noise_std: Number(controls.noise.value),
      mic_preset: controls.mic.value, room: controls.room.value, processing: controls.processing.value,
      emitters: emitterCards().map(readEmitter),
    };
  }

  function scenePoint(bearing, distance) {
    const radius = Math.min(260, 90 + Number(distance) * 48);
    const angle = Number(bearing) * Math.PI / 180;
    return { x: 600 + Math.cos(angle) * radius, y: 390 - Math.sin(angle) * radius };
  }

  function updateMicrophones() {
    const layer = $("#microphones-layer");
    layer.replaceChildren();
    const preset = state.capabilities?.microphone_presets.find((item) => item.id === controls.mic.value);
    const positions = preset?.positions_m || [[.05,.03,0],[-.045,.035,0],[-.05,-.038,0],[.042,-.047,0]];
    positions.forEach((position) => layer.appendChild(svgElement("circle", { class: "microphone", cx: position[0] * 520, cy: -position[1] * 520, r: 5 })));
  }

  function renderScene() {
    const layer = $("#emitters-layer");
    layer.replaceChildren();
    emitterCards().map(readEmitter).forEach((emitter, index) => {
      if (!emitter.enabled) return;
      const point = scenePoint(emitter.bearing_deg, emitter.distance_m);
      const group = svgElement("g");
      [18, 31, 44].forEach((radius) => group.appendChild(svgElement("circle", { class: "sound-ring", cx: point.x, cy: point.y, r: radius })));
      group.appendChild(svgElement("circle", { class: "emitter-source", cx: point.x, cy: point.y, r: 9, fill: palette[index % palette.length] }));
      const name = svgElement("text", { class: "source-name", x: point.x + 17, y: point.y - 5 });
      name.textContent = emitter.id;
      const detail = svgElement("text", { class: "source-detail", x: point.x + 17, y: point.y + 13 });
      detail.textContent = `${formatSigned(emitter.bearing_deg)}° · ${emitter.distance_m.toFixed(1)} m`;
      group.append(name, detail);
      layer.appendChild(group);
    });
    updateMicrophones();
  }

  function markStale() {
    $("#estimate-layer").replaceChildren();
    $("#scene-status").textContent = state.result ? "Scene changed · run again" : "Ready to render";
    $("#compass-needle").style.opacity = state.result ? ".16" : ".24";
  }

  function updateScene() {
    $("#duration-value").textContent = `${Number(controls.duration.value).toFixed(1)} s`;
    $("#noise-value").textContent = Number(controls.noise.value).toFixed(3);
    emitterCards().forEach((card) => {
      $(".emitter-bearing-value", card).textContent = `${formatSigned($(".emitter-bearing", card).value)}°`;
      $(".emitter-distance-value", card).textContent = `${Number($(".emitter-distance", card).value).toFixed(1)} m`;
    });
    renderScene();
    markStale();
  }

  function drawEstimate(spatial) {
    const layer = $("#estimate-layer");
    layer.replaceChildren();
    const needle = $("#compass-needle");
    if (!spatial.valid || spatial.bearing_deg === null) { needle.style.opacity = ".2"; return; }
    const point = scenePoint(spatial.bearing_deg, 4.2);
    layer.appendChild(svgElement("line", { class: "estimate-line", x1: 600, y1: 390, x2: point.x, y2: point.y, "marker-end": "url(#arrow-estimate)" }));
    needle.style.transform = `rotate(${90 - spatial.bearing_deg}deg)`;
    needle.style.opacity = "1";
  }

  function drawWaveform(waveform) {
    const canvas = $("#waveform");
    const ratio = window.devicePixelRatio || 1;
    const width = Math.max(280, canvas.clientWidth);
    const height = Math.max(60, canvas.clientHeight);
    canvas.width = width * ratio; canvas.height = height * ratio;
    const context = canvas.getContext("2d");
    context.scale(ratio, ratio); context.clearRect(0, 0, width, height);
    const combined = waveform[0] || [];
    context.strokeStyle = "#79e5ee"; context.lineWidth = 1.2; context.beginPath();
    combined.forEach((value, index) => {
      const x = index / Math.max(1, combined.length - 1) * width;
      const y = height / 2 - value / Math.max(.001, state.result.audio.peak) * height * .38;
      if (index === 0) context.moveTo(x, y); else context.lineTo(x, y);
    });
    context.stroke();
  }

  function setAudio(encoded) {
    if (state.audioUrl) URL.revokeObjectURL(state.audioUrl);
    const binary = atob(encoded); const bytes = new Uint8Array(binary.length);
    for (let index = 0; index < binary.length; index++) bytes[index] = binary.charCodeAt(index);
    state.audioUrl = URL.createObjectURL(new Blob([bytes], { type: "audio/wav" }));
    $("#audio-player").src = state.audioUrl;
  }

  function updateResult(result) {
    state.result = result;
    const spatial = result.spatial;
    const bearing = spatial.valid ? `${formatSigned(spatial.bearing_deg, 1)}°` : "—";
    const confidence = `${Math.round(spatial.confidence * 100)}%`;
    $("#bearing-result").textContent = bearing;
    $("#confidence-result").textContent = spatial.valid ? `${confidence} confidence` : "No valid detection";
    $("#direction-result").textContent = spatial.valid ? `${bearing} relative bearing` : "No estimate";
    $("#confidence-detail").textContent = confidence;
    $("#instrument-state").textContent = spatial.valid ? "Listening · estimate valid" : "Listening · no detection";
    $("#scene-status").textContent = spatial.valid ? `PCM estimate ${bearing}` : "No valid acoustic detection";
    const semantic = result.semantic;
    $("#semantic-result").textContent = semantic.valid && semantic.events.length
      ? `${semantic.events[0].label} · ${Math.round(semantic.events[0].score * 100)}%`
      : semantic.status === "not_configured" ? "BEATs not configured" : semantic.status.replaceAll("_", " ");
    $("#integrity").textContent = spatial.association_valid
      ? `Single-source evaluation error: ${spatial.evaluation_error_deg.toFixed(1)}°. Semantic labels remain whole-window only.`
      : "Direction is audio-derived. Source association is unresolved; labels describe the whole window.";
    $("#instrument").classList.add("running");
    drawEstimate(spatial); drawWaveform(result.audio.waveform); setAudio(result.audio.wav_base64);
    controls.export.disabled = false;
  }

  async function runScenario() {
    controls.run.disabled = true; controls.run.classList.add("loading"); $("#error-box").hidden = true;
    try {
      const response = await fetch("/api/simulate", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(collectScenario()) });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || `Request failed with ${response.status}`);
      updateResult(data);
    } catch (error) {
      $("#error-box").textContent = error.message; $("#error-box").hidden = false;
    } finally {
      controls.run.disabled = false; controls.run.classList.remove("loading");
    }
  }

  function exportReport() {
    if (!state.result) return;
    const clean = structuredClone(state.result); delete clean.audio.wav_base64;
    const url = URL.createObjectURL(new Blob([JSON.stringify(clean, null, 2)], { type: "application/json" }));
    const link = document.createElement("a"); link.href = url; link.download = `onda-scene-${new Date().toISOString().replace(/[:.]/g, "-")}.json`; link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  function reset() {
    controls.mic.value = "onda_reference_4"; controls.room.value = "direct"; controls.processing.value = "raw";
    controls.duration.value = "2.4"; controls.noise.value = ".002"; state.nextEmitter = 1; state.result = null;
    $("#emitter-list").replaceChildren();
    addEmitter({ id: "speaker", sound: "broadband", bearing_deg: 38, distance_m: 2, gain_db: -3, onset_s: .1 });
    controls.export.disabled = true; $("#instrument").classList.remove("running");
    $("#instrument-state").textContent = "Ready"; $("#bearing-result").textContent = "—"; $("#confidence-result").textContent = "Run a scene";
    $("#direction-result").textContent = "No estimate"; $("#confidence-detail").textContent = "—"; $("#semantic-result").textContent = "BEATs not configured";
    $("#integrity").textContent = "Direction is audio-derived. Labels describe the whole window and are not associated with a direction.";
    const context = $("#waveform").getContext("2d"); context.clearRect(0, 0, $("#waveform").width, $("#waveform").height);
  }

  function toggleInstrument() {
    const details = $("#instrument-details"); const open = details.hidden;
    details.hidden = !open; $("#instrument").classList.toggle("expanded", open); $("#instrument-toggle").setAttribute("aria-expanded", String(open));
  }

  function bind() {
    [controls.mic, controls.room, controls.processing, controls.duration, controls.noise].forEach((input) => input.addEventListener("input", updateScene));
    controls.add.addEventListener("click", () => addEmitter()); controls.run.addEventListener("click", runScenario);
    controls.reset.addEventListener("click", reset); controls.export.addEventListener("click", exportReport);
    $("#instrument-toggle").addEventListener("click", toggleInstrument);
    window.addEventListener("resize", () => { if (state.result) drawWaveform(state.result.audio.waveform); });
  }

  async function initialize() {
    bind(); reset();
    try {
      const response = await fetch("/api/capabilities");
      if (!response.ok) throw new Error("SDK capability endpoint unavailable");
      state.capabilities = await response.json();
      $("#runtime-label").textContent = "SDK connected"; $(".status-dot").classList.add("online");
      if (!state.capabilities.semantic_backend.available) $("#semantic-result").textContent = "BEATs not configured";
      updateScene();
    } catch (error) {
      $("#runtime-label").textContent = "SDK unavailable"; $("#error-box").textContent = error.message; $("#error-box").hidden = false;
    }
  }

  initialize();
})();
