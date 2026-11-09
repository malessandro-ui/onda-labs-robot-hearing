(() => {
  "use strict";

  const state = {
    capabilities: null, result: null, nextEmitter: 1, audioUrl: null,
    live: false, request: null, restartTimer: null, audioContext: null, monitoring: false,
    monitorSources: new Set(), nextAudioTime: 0, latestAudio: null,
  };
  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));
  const svgNS = "http://www.w3.org/2000/svg";
  const palette = ["#f0b45c", "#79e5ee", "#ae98ff", "#7ae2a5", "#ff858b"];
  const controls = {
    mic: $("#mic-preset"), room: $("#room-preset"), processing: $("#processing"),
    duration: $("#duration"), noise: $("#noise"), run: $("#run-button"),
    reset: $("#reset-button"), export: $("#export-button"), add: $("#add-emitter"),
    live: $("#live-button"), monitor: $("#monitor-button"),
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
    if (emitterCards().length >= (state.capabilities?.max_emitters ?? 8)) return;
    const fragment = $("#emitter-template").content.cloneNode(true);
    const card = $(".emitter-card", fragment);
    const index = state.nextEmitter++;
    $(".emitter-name", card).value = seed.id || `Phone ${index}`;
    $(".emitter-sound", card).value = seed.sound || "telephone";
    $(".emitter-bearing", card).value = seed.bearing_deg ?? ((38 - 60 * (index - 1) + 540) % 360 - 180);
    $(".emitter-distance", card).value = seed.distance_m ?? (index === 1 ? 2 : 2.8);
    $(".emitter-gain", card).value = seed.gain_db ?? (index === 1 ? -3 : -8);
    $(".emitter-onset", card).value = seed.onset_s ?? (index === 1 ? .1 : .5);
    $(".emitter-enabled", card).checked = seed.enabled ?? true;
    $(".remove-emitter", card).addEventListener("click", () => { card.remove(); updateScene(); });
    $$("input:not([type=file]), select", card).forEach((input) => input.addEventListener("input", updateScene));
    $(".emitter-wav", card).addEventListener("change", (event) => attachWav(card, event.target));
    $("#emitter-list").appendChild(fragment);
    updateScene();
  }

  async function attachWav(card, input) {
    const file = input.files[0];
    if (!file) return;
    try {
      if (file.size > 330000) throw new Error("Choose a WAV file up to 330 KB.");
      const bytes = new Uint8Array(await file.arrayBuffer());
      if (!card.isConnected || input.files[0] !== file) return;
      let binary = "";
      for (const byte of bytes) binary += String.fromCharCode(byte);
      card.wavAsset = { wav_name: file.name.slice(0, 80), wav_base64: btoa(binary) };
      const option = $('option[value="wav"]', card);
      option.disabled = false;
      option.textContent = card.wavAsset.wav_name;
      $(".emitter-sound", card).value = "wav";
      updateScene();
    } catch (error) {
      input.value = "";
      showError(error.message);
    }
  }

  function readEmitter(card, index) {
    const sound = $(".emitter-sound", card).value;
    return {
      id: $(".emitter-name", card).value.trim() || `source_${index + 1}`,
      sound,
      ...(sound === "wav" ? card.wavAsset : {}),
      directivity: $(".emitter-directivity", card).value,
      bearing_deg: Number($(".emitter-bearing", card).value),
      distance_m: Number($(".emitter-distance", card).value),
      gain_db: Number($(".emitter-gain", card).value),
      onset_s: Number($(".emitter-onset", card).value),
      duration_s: null, loop: $(".emitter-loop", card).checked, enabled: $(".emitter-enabled", card).checked,
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
    const radius = 90 + Number(distance) * 25;
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

  function clearResult(status, hint) {
    state.result = null;
    state.latestAudio = null;
    controls.export.disabled = true;
    $("#instrument").classList.remove("running");
    $("#instrument-state").textContent = status;
    $("#scene-status").textContent = status;
    $("#bearing-result").textContent = "—";
    $("#confidence-result").textContent = hint;
    $("#direction-result").textContent = "No estimate";
    $("#confidence-detail").textContent = "—";
    updateSemantic(null);
    drawEstimate({ valid: false });
    drawWaveform(null);
    const player = $("#audio-player");
    player.removeAttribute("src");
    player.load();
    player.hidden = true;
    if (state.audioUrl) URL.revokeObjectURL(state.audioUrl);
    state.audioUrl = null;
    $("#integrity").textContent = "Bearing comes from microphone audio. Sound labels cover the whole audio window.";
  }

  function updateScene() {
    $("#duration-value").textContent = `${Number(controls.duration.value).toFixed(1)} s`;
    $("#noise-value").textContent = Number(controls.noise.value).toFixed(3);
    const cards = emitterCards();
    $("#object-count").textContent = `${cards.length} object${cards.length === 1 ? "" : "s"}`;
    controls.add.disabled = cards.length >= (state.capabilities?.max_emitters ?? 8);
    cards.forEach((card) => {
      $(".emitter-bearing-value", card).textContent = `${formatSigned($(".emitter-bearing", card).value)}°`;
      $(".emitter-distance-value", card).textContent = `${Number($(".emitter-distance", card).value).toFixed(1)} m`;
    });
    renderScene();
    const wasRendering = !state.live && state.request !== null;
    cancelRequest();
    $("#error-box").hidden = true;
    if (state.live) {
      if (state.latestAudio || $("#instrument-state").textContent !== "Updating scene") {
        clearResult("Updating scene", "Listening resumes shortly");
      }
      state.restartTimer = setTimeout(startLive, 180);
    } else if (wasRendering || state.result || state.latestAudio) {
      clearResult("Scene changed", "Start live or render a clip");
    }
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

  function drawWaveform(audio) {
    const canvas = $("#waveform");
    const ratio = window.devicePixelRatio || 1;
    const width = Math.max(280, canvas.clientWidth);
    const height = Math.max(60, canvas.clientHeight);
    canvas.width = width * ratio; canvas.height = height * ratio;
    const context = canvas.getContext("2d");
    context.scale(ratio, ratio); context.clearRect(0, 0, width, height);
    if (!audio) return;
    const combined = audio.waveform[0] || [];
    context.strokeStyle = "#79e5ee"; context.lineWidth = 1.2; context.beginPath();
    combined.forEach((value, index) => {
      const x = index / Math.max(1, combined.length - 1) * width;
      const y = height / 2 - value / Math.max(.001, audio.peak) * height * .38;
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
    $("#audio-player").hidden = false;
  }

  function updateReadout(spatial, status) {
    const bearing = spatial.valid ? `${formatSigned(spatial.bearing_deg, 1)}°` : "—";
    const confidence = `${Math.round(spatial.confidence * 100)}%`;
    $("#bearing-result").textContent = bearing;
    $("#confidence-result").textContent = spatial.valid ? `${confidence} confidence` : "No valid detection";
    $("#direction-result").textContent = spatial.valid ? `${bearing} relative bearing` : "No estimate";
    $("#confidence-detail").textContent = spatial.valid ? confidence : "—";
    $("#instrument-state").textContent = status;
    $("#scene-status").textContent = spatial.valid ? `${status} · PCM estimate ${bearing}` : `${status} · no valid detection`;
    drawEstimate(spatial);
  }

  function updateSemantic(semantic) {
    $("#semantic-result").textContent = !semantic
      ? state.capabilities?.semantic_backend.available ? "Waiting for audio" : "BEATs not configured"
      : semantic.valid && semantic.events.length
      ? `${semantic.events[0].label} · ${Math.round(semantic.events[0].score * 100)}%`
      : semantic.status === "not_configured" ? "BEATs not configured" : semantic.status.replaceAll("_", " ");
  }

  function updateResult(result) {
    state.result = result;
    state.latestAudio = result.audio;
    const spatial = result.spatial;
    updateReadout(spatial, "Rendered clip");
    updateSemantic(result.semantic);
    $("#integrity").textContent = spatial.evaluation_error_deg !== null
      ? `Single-source evaluation error: ${spatial.evaluation_error_deg.toFixed(1)}°. Semantic labels remain whole-window only.`
      : "Direction is audio-derived. Source association is unresolved; labels describe the whole window.";
    drawWaveform(result.audio); setAudio(result.audio.wav_base64);
    controls.export.disabled = false;
  }

  async function runScenario() {
    cancelRequest();
    clearResult("Rendering clip", "Waiting for audio");
    const request = new AbortController();
    state.request = request;
    controls.run.disabled = true; controls.run.classList.add("loading"); $("#error-box").hidden = true;
    try {
      const response = await postScene("/api/simulate", request);
      const data = await response.json();
      if (request.signal.aborted) return;
      updateResult(data);
    } catch (error) {
      if (!request.signal.aborted) {
        clearResult("Render failed", "Check scene setup");
        showError(error.message);
      }
    } finally {
      if (state.request === request) state.request = null;
      controls.run.disabled = state.live; controls.run.classList.remove("loading");
    }
  }

  function showError(message) {
    const box = $("#error-box");
    box.textContent = message;
    box.hidden = false;
  }

  async function postScene(url, request) {
    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(collectScenario()),
      signal: request.signal,
    });
    if (!response.ok) {
      const data = await response.json();
      throw new Error(data.error || `Request failed with ${response.status}`);
    }
    return response;
  }

  function stopMonitor() {
    for (const source of state.monitorSources) source.stop();
    state.monitorSources.clear();
    state.nextAudioTime = 0;
  }

  function cancelRequest() {
    clearTimeout(state.restartTimer);
    state.request?.abort();
    state.request = null;
    stopMonitor();
  }

  function queueMonitor(encoded, sampleRate) {
    const context = state.audioContext;
    if (!state.monitoring || context?.state !== "running") return;
    if (state.nextAudioTime > context.currentTime + .4) return;
    if (state.nextAudioTime < context.currentTime) state.nextAudioTime = context.currentTime + .10;
    const bytes = atob(encoded);
    const buffer = context.createBuffer(1, bytes.length / 2, sampleRate);
    const samples = buffer.getChannelData(0);
    for (let index = 0; index < samples.length; index++) {
      const value = bytes.charCodeAt(index * 2) | (bytes.charCodeAt(index * 2 + 1) << 8);
      samples[index] = (value >= 32768 ? value - 65536 : value) / 32768;
    }
    const source = context.createBufferSource();
    source.buffer = buffer;
    source.connect(context.destination);
    source.onended = () => state.monitorSources.delete(source);
    state.monitorSources.add(source);
    source.start(state.nextAudioTime);
    state.nextAudioTime += buffer.duration;
  }

  async function toggleMonitor() {
    if (state.monitoring) {
      setMonitoring(false);
      return;
    }
    try {
      state.audioContext ??= new AudioContext();
      await state.audioContext.resume();
      if (state.live) setMonitoring(true);
    } catch (error) {
      showError(`Audio monitor unavailable: ${error.message}`);
    }
  }

  function setMonitoring(enabled) {
    state.monitoring = enabled;
    controls.monitor.textContent = enabled ? "Mute monitor" : "Monitor audio";
    controls.monitor.setAttribute("aria-pressed", String(enabled));
    if (!enabled) stopMonitor();
  }

  async function startLive() {
    if (!state.live) return;
    const request = new AbortController();
    state.request = request;
    $("#error-box").hidden = true;
    let watchdog;
    const armWatchdog = () => {
      clearTimeout(watchdog);
      watchdog = setTimeout(() => {
        if (state.request !== request) return;
        setLive(false);
        clearResult("Stream interrupted", "Start live to retry");
        showError("No audio arrived from the SDK.");
      }, 3000);
    };
    armWatchdog();
    try {
      const response = await postScene("/api/live", request);
      const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
      let buffer = "";
      while (!request.signal.aborted) {
        const { value, done } = await reader.read();
        if (request.signal.aborted) return;
        if (done) throw new Error("The SDK stopped sending audio.");
        buffer += value;
        let boundary = buffer.indexOf("\n");
        while (boundary !== -1) {
          const frame = JSON.parse(buffer.slice(0, boundary));
          buffer = buffer.slice(boundary + 1);
          state.latestAudio = frame.audio;
          updateReadout(frame.spatial, "Live simulation");
          updateSemantic(frame.semantic);
          $("#instrument").classList.add("running");
          if (!$("#instrument-details").hidden) drawWaveform(frame.audio);
          queueMonitor(frame.audio.pcm16_base64, frame.sample_rate_hz);
          armWatchdog();
          boundary = buffer.indexOf("\n");
        }
      }
    } catch (error) {
      if (!request.signal.aborted) {
        cancelRequest();
        clearResult("Live paused", "Edit the scene or stop live");
        showError(error.message);
      }
    } finally {
      clearTimeout(watchdog);
      if (state.request === request) state.request = null;
    }
  }

  function setLive(enabled) {
    cancelRequest();
    state.live = enabled;
    controls.live.textContent = enabled ? "Stop live" : "Start live";
    controls.run.disabled = enabled;
    controls.monitor.hidden = !enabled;
    $("#error-box").hidden = true;
    if (!enabled) setMonitoring(false);
    clearResult(enabled ? "Starting live" : "Live stopped", enabled ? "Waiting for audio" : "Open to listen");
    if (enabled) startLive();
  }

  function exportReport() {
    if (!state.result) return;
    const clean = structuredClone(state.result); delete clean.audio.wav_base64;
    const url = URL.createObjectURL(new Blob([JSON.stringify(clean, null, 2)], { type: "application/json" }));
    const link = document.createElement("a"); link.href = url; link.download = `onda-scene-${new Date().toISOString().replace(/[:.]/g, "-")}.json`; link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  function reset() {
    setLive(false);
    controls.mic.value = "onda_reference_4"; controls.room.value = "direct"; controls.processing.value = "raw";
    controls.duration.value = "2.4"; controls.noise.value = ".002"; state.nextEmitter = 1; state.result = null;
    $("#emitter-list").replaceChildren();
    addEmitter({ id: "speaker", sound: "broadband", bearing_deg: 38, distance_m: 2, gain_db: -3, onset_s: .1 });
    clearResult("Ready", "Open to listen");
  }

  function toggleInstrument() {
    const details = $("#instrument-details"); const open = details.hidden;
    details.hidden = !open; $("#instrument").classList.toggle("expanded", open); $("#instrument-toggle").setAttribute("aria-expanded", String(open));
    if (open) drawWaveform(state.latestAudio);
  }

  function bind() {
    [controls.mic, controls.room, controls.processing, controls.duration, controls.noise].forEach((input) => input.addEventListener("input", updateScene));
    controls.add.addEventListener("click", () => {
      addEmitter();
      $("#scene-settings").open = true;
      $(".emitter-name", emitterCards().at(-1)).focus();
    });
    controls.run.addEventListener("click", runScenario);
    controls.live.addEventListener("click", () => setLive(!state.live));
    controls.monitor.addEventListener("click", toggleMonitor);
    controls.reset.addEventListener("click", reset); controls.export.addEventListener("click", exportReport);
    $("#instrument-toggle").addEventListener("click", toggleInstrument);
    window.addEventListener("resize", () => drawWaveform(state.latestAudio));
  }

  async function initialize() {
    bind(); reset();
    try {
      const response = await fetch("/api/capabilities");
      if (!response.ok) throw new Error("SDK capability endpoint unavailable");
      state.capabilities = await response.json();
      $("#runtime-label").textContent = "SDK connected"; $(".status-dot").classList.add("online");
      updateSemantic(null);
      updateMicrophones();
    } catch (error) {
      $("#runtime-label").textContent = "SDK unavailable"; $("#error-box").textContent = error.message; $("#error-box").hidden = false;
    }
  }

  initialize();
})();
