(() => {
  "use strict";

  const state = { capabilities: null, result: null, nextEmitter: 1, audioUrl: null };
  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));
  const svgNS = "http://www.w3.org/2000/svg";
  const palette = ["#f5b65d", "#78dce8", "#ad93ff", "#72d49a", "#ff7b82", "#7fb0ff"];

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
    $(".emitter-directivity", card).value = seed.directivity || "omni";
    $(".emitter-bearing", card).value = seed.bearing_deg ?? (index === 1 ? 38 : -55);
    $(".emitter-distance", card).value = seed.distance_m ?? (index === 1 ? 2 : 2.8);
    $(".emitter-gain", card).value = seed.gain_db ?? (index === 1 ? -3 : -8);
    $(".emitter-onset", card).value = seed.onset_s ?? (index === 1 ? .1 : .5);
    $(".emitter-onset", card).max = controls.duration.value;
    $(".emitter-enabled", card).checked = seed.enabled ?? true;
    $(".remove-emitter", card).addEventListener("click", () => {
      card.remove();
      updateAll();
    });
    $$(`input, select`, card).forEach((input) => input.addEventListener("input", updateAll));
    $("#emitter-list").appendChild(fragment);
    updateAll();
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
      duration_s: null,
      loop: true,
      enabled: $(".emitter-enabled", card).checked,
    };
  }

  function collectScenario() {
    return {
      duration_s: Number(controls.duration.value),
      noise_std: Number(controls.noise.value),
      mic_preset: controls.mic.value,
      room: controls.room.value,
      processing: controls.processing.value,
      emitters: emitterCards().map(readEmitter),
    };
  }

  function updateLabels() {
    $("#duration-value").textContent = `${Number(controls.duration.value).toFixed(1)} s`;
    $("#noise-value").textContent = Number(controls.noise.value).toFixed(3);
    $("#timeline-duration").textContent = `${Number(controls.duration.value).toFixed(1)} seconds`;
    const cards = emitterCards();
    const active = cards.filter((card) => $(".emitter-enabled", card).checked).length;
    $("#emitter-count").textContent = `${active} active`;
    cards.forEach((card) => {
      const enabled = $(".emitter-enabled", card).checked;
      card.classList.toggle("disabled", !enabled);
      $(".emitter-bearing-value", card).textContent = `${formatSigned($(".emitter-bearing", card).value)}°`;
      $(".emitter-distance-value", card).textContent = `${Number($(".emitter-distance", card).value).toFixed(1)} m`;
      $(".emitter-gain-value", card).textContent = `${formatSigned($(".emitter-gain", card).value)} dB`;
      const onset = $(".emitter-onset", card);
      onset.max = controls.duration.value;
      if (Number(onset.value) > Number(onset.max)) onset.value = onset.max;
      $(".emitter-onset-value", card).textContent = `${Number(onset.value).toFixed(1)} s`;
    });
  }

  function scenePoint(bearing, distance) {
    const radius = Math.min(156, 55 + Number(distance) * 35);
    const angle = Number(bearing) * Math.PI / 180;
    return { x: 360 + Math.cos(angle) * radius, y: 232 - Math.sin(angle) * radius };
  }

  function updateMicrophones() {
    const layer = $("#microphones-layer");
    layer.replaceChildren();
    const preset = state.capabilities?.microphone_presets.find((item) => item.id === controls.mic.value);
    const positions = preset?.positions_m || [[.05, .03, 0], [-.045, .035, 0], [-.05, -.038, 0], [.042, -.047, 0]];
    positions.forEach((position) => {
      layer.appendChild(svgElement("circle", { class: "microphone", cx: position[0] * 420, cy: -position[1] * 420, r: 4.4 }));
    });
  }

  function renderScene() {
    const emittersLayer = $("#emitters-layer");
    const truthLayer = $("#truth-layer");
    emittersLayer.replaceChildren();
    truthLayer.replaceChildren();
    const emitters = emitterCards().map(readEmitter);
    emitters.forEach((emitter, index) => {
      if (!emitter.enabled) return;
      const point = scenePoint(emitter.bearing_deg, emitter.distance_m);
      const group = svgElement("g");
      [18, 29].forEach((radius) => group.appendChild(svgElement("circle", { class: "sound-ring", cx: point.x, cy: point.y, r: radius })));
      group.appendChild(svgElement("circle", { class: "emitter-source", cx: point.x, cy: point.y, r: 8, fill: palette[index % palette.length] }));
      const name = svgElement("text", { class: "source-name", x: point.x + 14, y: point.y - 5 });
      name.textContent = emitter.id;
      const detail = svgElement("text", { class: "source-detail", x: point.x + 14, y: point.y + 11 });
      detail.textContent = `${formatSigned(emitter.bearing_deg)}° · ${emitter.distance_m.toFixed(1)} m`;
      group.append(name, detail);
      emittersLayer.appendChild(group);
      if (emitters.filter((item) => item.enabled).length === 1) {
        truthLayer.appendChild(svgElement("line", { class: "truth-line", x1: 360, y1: 232, x2: point.x, y2: point.y, "marker-end": "url(#arrow-truth)" }));
      }
    });
    updateMicrophones();
  }

  function renderTimeline() {
    const track = $("#timeline-track");
    const playhead = $("#timeline-playhead");
    track.replaceChildren(playhead);
    const duration = Math.max(.1, Number(controls.duration.value));
    emitterCards().map(readEmitter).forEach((emitter) => {
      if (!emitter.enabled) return;
      const event = document.createElement("div");
      event.className = "timeline-event";
      event.style.left = `${Math.min(100, emitter.onset_s / duration * 100)}%`;
      event.style.width = `${Math.max(0, 100 - emitter.onset_s / duration * 100)}%`;
      event.setAttribute("aria-label", `${emitter.id} active from ${emitter.onset_s} seconds`);
      track.insertBefore(event, playhead);
    });
  }

  function clearEstimate() {
    $("#estimate-layer").replaceChildren();
    $("#stage-chip").textContent = "Configuration changed · run again";
    $("#chain-spatial").textContent = "Waiting";
    $("#chain-association").textContent = "Not claimed";
    if (state.result) $("#valid-badge").textContent = "Stale";
  }

  function updateAll() {
    updateLabels();
    renderScene();
    renderTimeline();
    clearEstimate();
  }

  function drawEstimate(spatial) {
    const layer = $("#estimate-layer");
    layer.replaceChildren();
    if (!spatial.valid || spatial.bearing_deg === null) return;
    const point = scenePoint(spatial.bearing_deg, 3.2);
    layer.appendChild(svgElement("line", { class: "estimate-line", x1: 360, y1: 232, x2: point.x, y2: point.y, "marker-end": "url(#arrow-estimate)" }));
  }

  function drawWaveform(waveform) {
    const canvas = $("#waveform");
    const ratio = window.devicePixelRatio || 1;
    const width = Math.max(400, canvas.clientWidth);
    const height = Math.max(140, canvas.clientHeight);
    canvas.width = Math.floor(width * ratio);
    canvas.height = Math.floor(height * ratio);
    const context = canvas.getContext("2d");
    context.scale(ratio, ratio);
    const styles = getComputedStyle(document.documentElement);
    context.clearRect(0, 0, width, height);
    context.strokeStyle = styles.getPropertyValue("--border");
    context.lineWidth = 1;
    const laneHeight = height / Math.max(1, waveform.length);
    waveform.forEach((values, channel) => {
      const center = laneHeight * (channel + .5);
      context.beginPath();
      context.moveTo(0, center);
      context.lineTo(width, center);
      context.stroke();
      const color = styles.getPropertyValue(`--channel-${channel + 1}`) || styles.getPropertyValue("--accent");
      context.strokeStyle = color;
      context.lineWidth = 1.3;
      context.beginPath();
      values.forEach((value, index) => {
        const x = index / Math.max(1, values.length - 1) * width;
        const amplitude = Math.min(.9, value / Math.max(.001, state.result.audio.peak));
        const y = center - amplitude * laneHeight * .38;
        if (index === 0) context.moveTo(x, y); else context.lineTo(x, y);
      });
      for (let index = values.length - 1; index >= 0; index--) {
        const x = index / Math.max(1, values.length - 1) * width;
        const amplitude = Math.min(.9, values[index] / Math.max(.001, state.result.audio.peak));
        context.lineTo(x, center + amplitude * laneHeight * .38);
      }
      context.closePath();
      context.globalAlpha = .72;
      context.stroke();
      context.globalAlpha = 1;
      context.strokeStyle = styles.getPropertyValue("--border");
    });
    const legend = $("#channel-legend");
    legend.replaceChildren();
    waveform.forEach((_, index) => {
      const item = document.createElement("span");
      const swatch = document.createElement("i");
      swatch.style.background = styles.getPropertyValue(`--channel-${index + 1}`);
      item.append(swatch, document.createTextNode(`Microphone ${index + 1}`));
      legend.appendChild(item);
    });
  }

  function setAudio(encoded) {
    if (state.audioUrl) URL.revokeObjectURL(state.audioUrl);
    const binary = atob(encoded);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    state.audioUrl = URL.createObjectURL(new Blob([bytes], { type: "audio/wav" }));
    $("#audio-player").src = state.audioUrl;
  }

  function updateSemantic(semantic) {
    const badge = $("#semantic-badge");
    const events = $("#semantic-events");
    $("#semantic-brain").textContent = semantic.brain_name;
    events.replaceChildren();
    if (semantic.valid && semantic.events.length) {
      badge.textContent = "Interpreted";
      badge.className = "badge valid";
      $("#chain-semantic").textContent = semantic.events[0].label;
      semantic.events.forEach((event) => {
        const row = document.createElement("div");
        row.className = "semantic-event";
        const label = document.createElement("strong");
        label.textContent = event.label;
        const score = document.createElement("output");
        score.textContent = `${Math.round(event.score * 100)}%`;
        const bar = document.createElement("div");
        bar.className = "semantic-bar";
        const fill = document.createElement("span");
        fill.style.width = `${Math.max(0, Math.min(100, event.score * 100))}%`;
        bar.appendChild(fill);
        row.append(label, score, bar);
        events.appendChild(row);
      });
    } else {
      const configured = semantic.status !== "not_configured";
      badge.textContent = configured ? semantic.status.replaceAll("_", " ") : "Not configured";
      badge.className = `badge ${semantic.status === "brain_error" ? "invalid" : "muted"}`;
      $("#chain-semantic").textContent = configured ? semantic.status.replaceAll("_", " ") : "BEATs offline";
      const empty = document.createElement("div");
      empty.className = "semantic-empty";
      empty.textContent = semantic.reason;
      events.appendChild(empty);
    }
    $("#semantic-scope").textContent = "Labels describe the whole audio window. Direction-to-label association is not claimed.";
  }

  function updateResult(result) {
    state.result = result;
    const spatial = result.spatial;
    const telemetry = result.telemetry;
    const validBadge = $("#valid-badge");
    validBadge.textContent = spatial.valid ? "Valid" : "No detection";
    validBadge.className = `badge ${spatial.valid ? "valid" : "invalid"}`;
    $("#bearing-result").textContent = spatial.valid ? `${formatSigned(spatial.bearing_deg, 1)}°` : "—";
    $("#error-result").textContent = spatial.evaluation_error_deg === null ? "Not associated" : `${spatial.evaluation_error_deg.toFixed(1)}°`;
    $("#confidence-result").textContent = `${Math.round(spatial.confidence * 100)}%`;
    $("#render-result").textContent = `${result.render_ms.toFixed(1)} ms`;
    $("#peak-result").textContent = result.audio.peak.toFixed(3);
    const policyRows = $$("#policy-values > div");
    result.policy.values.forEach((value, index) => {
      $("code", policyRows[index]).textContent = index === 3 ? String(Math.round(value)) : Number(value).toFixed(3);
    });
    const telemetryValues = [
      result.samples_per_channel.toLocaleString(),
      telemetry.overrange_samples.toLocaleString(),
      telemetry.clipped_samples.toLocaleString(),
      spatial.association_valid ? "Single source" : "Unresolved",
    ];
    $$("#telemetry > div").forEach((row, index) => $("strong", row).textContent = telemetryValues[index]);
    $("#stage-chip").textContent = spatial.valid ? `PCM estimate · ${formatSigned(spatial.bearing_deg, 1)}°` : "No valid acoustic detection";
    $("#chain-pcm").textContent = `${result.channels} ch · ${(result.samples_per_channel / result.sample_rate_hz).toFixed(1)} s`;
    $("#chain-spatial").textContent = spatial.valid ? `${formatSigned(spatial.bearing_deg, 1)}° · ${Math.round(spatial.confidence * 100)}%` : "No detection";
    $("#chain-association").textContent = spatial.association_valid ? "Single-source evaluation" : "Unresolved";
    updateSemantic(result.semantic);
    drawEstimate(spatial);
    drawWaveform(result.audio.waveform);
    setAudio(result.audio.wav_base64);
    controls.export.disabled = false;
  }

  async function runScenario() {
    controls.run.disabled = true;
    controls.run.classList.add("loading");
    $("#error-box").hidden = true;
    try {
      const response = await fetch("/api/simulate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(collectScenario()),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || `Request failed with ${response.status}`);
      updateResult(data);
    } catch (error) {
      const box = $("#error-box");
      box.textContent = error.message;
      box.hidden = false;
    } finally {
      controls.run.disabled = false;
      controls.run.classList.remove("loading");
    }
  }

  function exportReport() {
    if (!state.result) return;
    const clean = structuredClone(state.result);
    delete clean.audio.wav_base64;
    const url = URL.createObjectURL(new Blob([JSON.stringify(clean, null, 2)], { type: "application/json" }));
    const link = document.createElement("a");
    link.href = url;
    link.download = `onda-acoustic-scene-${new Date().toISOString().replace(/[:.]/g, "-")}.json`;
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  function reset() {
    controls.mic.value = "onda_reference_4";
    controls.room.value = "direct";
    controls.processing.value = "raw";
    controls.duration.value = "2.4";
    controls.noise.value = ".002";
    state.nextEmitter = 1;
    $("#emitter-list").replaceChildren();
    addEmitter({ id: "speaker", sound: "broadband", bearing_deg: 38, distance_m: 2, gain_db: -3, onset_s: .1 });
    state.result = null;
    controls.export.disabled = true;
    $("#valid-badge").className = "badge muted";
    $("#valid-badge").textContent = "Not run";
    $("#bearing-result").textContent = "—";
    $("#error-result").textContent = "—";
    $("#confidence-result").textContent = "—";
    $("#render-result").textContent = "—";
    $("#peak-result").textContent = "—";
    $("#estimate-layer").replaceChildren();
    $("#stage-chip").textContent = "Ready to render";
    $("#chain-pcm").textContent = "Ready";
    $("#chain-spatial").textContent = "Waiting";
    $("#chain-association").textContent = "Not claimed";
    $("#semantic-badge").className = "badge muted";
    $("#semantic-badge").textContent = "Checking";
    $("#semantic-events").replaceChildren(Object.assign(document.createElement("div"), { className: "semantic-empty", textContent: "Run a scene to inspect whole-window sound meaning." }));
    const context = $("#waveform").getContext("2d");
    context.clearRect(0, 0, $("#waveform").width, $("#waveform").height);
  }

  function bind() {
    [controls.mic, controls.room, controls.processing, controls.duration, controls.noise].forEach((input) => input.addEventListener("input", updateAll));
    controls.add.addEventListener("click", () => addEmitter());
    controls.run.addEventListener("click", runScenario);
    controls.reset.addEventListener("click", reset);
    controls.export.addEventListener("click", exportReport);
    window.addEventListener("resize", () => { if (state.result) drawWaveform(state.result.audio.waveform); });
  }

  async function initialize() {
    bind();
    reset();
    try {
      const response = await fetch("/api/capabilities");
      if (!response.ok) throw new Error("SDK capability endpoint unavailable");
      state.capabilities = await response.json();
      $("#runtime-label").textContent = state.capabilities.renderer;
      $(".status-dot").classList.add("online");
      const semantic = state.capabilities.semantic_backend;
      $("#chain-semantic").textContent = semantic.available ? "BEATs ready" : "BEATs offline";
      $("#semantic-badge").textContent = semantic.available ? "Ready" : "Not configured";
      $("#semantic-badge").className = `badge ${semantic.available ? "valid" : "muted"}`;
      if (!semantic.available) {
        $("#semantic-events .semantic-empty").textContent = semantic.reason;
      }
      updateAll();
    } catch (error) {
      $("#runtime-label").textContent = "SDK unavailable";
      const box = $("#error-box");
      box.textContent = error.message;
      box.hidden = false;
    }
  }

  initialize();
})();
