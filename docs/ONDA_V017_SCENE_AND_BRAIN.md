# Onda v0.17 — Acoustic Scene + Sound Brain (additive developer drop-in)

## Scope / design

**Unified SDK, not a separate product.** Adds `onda_hearing.acoustic_scene` and
`onda_hearing.semantics` namespaces to the existing `onda_hearing` package.
Does not change v0.16 `StreamingHearing`, its PCM buffer, `Pose`, or its existing
21 soak/regression tests. Simulated acoustic coordinates are hidden from policy.

**Acoustic scene:** world-fixed and object-bound emitters, offsets and
quaternions (XYZW), 16 kHz mono PCM WAV assets, sample-indexed onset,
looping, duration, explicit trigger and stop, gain, simple cardioid pattern,
multisource accumulation, per-microphone direct-path propagation delay,
inverse-distance pressure, deterministic noise, optional channel FIR, moving
source retarded-time approximation, reset/isolation per renderer instance.

**Semantic layer:** `SoundBrain.classify(mono,sample_rate) -> tuple[EventScore,...]`
with an asynchronous bounded-window worker reading a **second, independently
configured v0.16 PCMBuffer**. The unified producer writes the same PCM to both
rings, because the existing spatial ring has a 2048-sample fixed window while
BEATs needs a longer frame. The
standalone default `OndaSoundBrain` BEATs integration uses the provided local
checkpoint. Companies supply their own implementation of `SoundBrain` to swap
in their own classifier. A stable `PolicyEncoder` exposes a fixed-size float32
vector and separate valid/age flags. A semantic classification belongs to the
**entire audio window**, NOT a localized source; no semantic-spatial association
is inferred or invented.

### Architecture

```
USD/MuJoCo scene -> pose snapshots -> PoseTimeline -> Onda AcousticSceneRenderer
                                                  |  [samples,channels] float32
                                                  v
                                          v0.16 StreamingHearing
                                            |      |
                                     spatial SRP  PCM -> separate PCMBuffer -> async SoundBrain
                                            |                     |
                                            +------UnifiedHearing-+
                                                      |
                                      policy feature vector + metadata
```

### Installation

On Shadeform or your Mac (on `develop/v0.16-pcm-streaming`):

```bash
cd /ephemeral/onda-nvidia/onda-v015
python /ephemeral/onda-nvidia/onda_v017_dropin/INSTALL.py "$PWD"
python -m pytest -q tests/test_acoustic_scene_v017.py tests/test_semantics_v017.py
python -m pytest -q tests/test_streaming_pcm.py tests/test_streaming_soak.py
```

If you extracted the ZIP elsewhere, change the install path accordingly. The
installer is **additive and fail-closed**: byte-different files with matching
names abort installation, and it refuses non-v0.16 branches unless explicitly
overridden. Existing Git history and files are not modified. The original 30-min
soak does not need to be rerun for every code addition, but native smoke and
integration testing remain required.

### Sound Brain: exact environment settings

```bash
export ONDA_BEATS_CHECKPOINT="$HOME/Developer/onda-foundation/models/pretrained/beats/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt"
export ONDA_BEATS_LABELS="$HOME/Developer/onda-foundation/models/pretrained/beats/beats_cpt2_labels.json"
export ONDA_BEATS_MODULE_DIR="/path/to/trusted/microsoft/unilm/beats"
```

**Important:** The checkpoint path given by the user is a **local Mac path**.
The Shadeform instance probably does not have that file. You must transfer
it to a location on Shadeform and set the environment variable to its remote
path before running BEATs. The checkpoint (hundreds of MB) is **NOT in the ZIP**
and should not be committed to GitHub. The labels file must have 527 entries
in the exact predictor-head index order; inspect/reconcile its format if
needed. An arbitrary 527-name list without matching order produces wrong
labels. The official `BEATs.py` module and dependencies must be installed or
on `ONDA_BEATS_MODULE_DIR`.

**Naming vs training provenance:** 'Onda Sound Brain' names Onda's recognition
interface/backend. The named `BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt`
is the public AudioSet-finetuned BEATs cpt2 naming convention; this change does
NOT prove that your checkpoint includes further Onda training. For an Onda-
custom head, implement a separate `SoundBrain` backend and test label mapping.

This loader requires explicit trusted-file opt-in (`allow_unsafe_checkpoint=True`)
and uses `torch.load(weights_only=True)` to avoid arbitrary pickle object
construction where supported. It loads `model`/`cfg` from the checkpoint and
uses **the probabilities output by official `BEATs.forward()`**, which has
already applied sigmoid; it does not apply sigmoid twice. No silent fallback
on load/inference failures; errors are visible to the caller or worker telemetry.

### Full example

```bash
python examples/v017_telephone_scene.py --wav /absolute/path/phone_ring_16k_mono.wav --duration 4
```

The example is wall-clock paced but does not depend on Isaac Sim or MuJoCo. It
shows: source attached to a phone object path, four microphones attached to a
robot-head object, rendered PCM fed into v0.16, spatial localization alongside
BEATs outputs, and a fixed-shape policy vector.

### API: plug-and-play brains (default Onda Sound Brain)

```python
from onda_hearing.semantics import attach_sound_brain

# Default loads ONDA_BEATS_CHECKPOINT, ONDA_BEATS_LABELS,
# and ONDA_BEATS_MODULE_DIR; failure is explicit, not a silent stub.
hearing = attach_sound_brain(existing_v016_stream,
                             allow_unsafe_checkpoint=True)

# Or supply any company-specific model with a stable output vocabulary:
hearing = attach_sound_brain(existing_v016_stream,
                             brain=custom_brain,
                             vocabulary=('Telephone', 'Trumpet', 'Alarm'))
```

`attach_sound_brain` automatically makes an independent bounded 2-second PCM
ring for semantic windows (0.5-second hop by default) and keeps the existing
v0.16 2048-sample spatial ring intact. The same producer feeds both. Models
may be slow without blocking the robot's policy thread.

### API: object-based and coordinate-based emitters

```python
from onda_hearing.acoustic_scene import *

phone = Emitter('phone_1','ring',
    Binding(object_path='/World/Desk/Phone',local_offset_m=(0,0,.05)),
    schedule=Schedule(onset_s=None,loop=True))

alarm = Emitter('alarm_1','alarm',
    Binding(world=Pose3((5.0,2.0,1.0))),
    schedule=Schedule(onset_s=3.,duration_s=4.))

# `assets` maps asset IDs to AudioAsset(...); source files must be 16k mono.
# `resolver` supplies Pose3 for object paths; MuJoCo and USD adapters capture
# their current poses into a PoseTimeline.
# renderer = AcousticSceneRenderer(RendererConfig(), mic_array, [phone,alarm], assets,resolver)
# renderer.trigger('phone_1', sample_index)  # event-driven start
# renderer.stop('phone_1', sample_index)     # emission stops; acoustic delay remains
# renderer.set_gain_db('phone_1',-6.0)
# pcm = renderer.render(start_sample,frames)
# stream.push_audio(pcm,start_sample)
```

**Motion snapshot rule:** IsaacUSDSceneSnapshot and MuJoCoPoseSnapshot read the
*current* simulator state, not a past state. Sample their object transforms on
simulation ticks and append to PoseTimeline; the renderer interpolates that
history. The last pose may be held for **at most 10 ms** to accommodate a fixed
pose grid; larger missing-future queries fail explicitly. For larger source
velocities or tight physical accuracy requirements increase pose frequency and
validate the temporal approximation. Never use wall time as simulation time.

**Batching:** Create one scene renderer, timeline and semantic state per parallel
Isaac Lab/MuJoCo environment. Do not share a mutable renderer across envs;
`reset()` it at episode reset. USD/MuJoCo pose snapshot classes are lazy imports,
so core tests do not require Isaac or MuJoCo installed.

## Current limitations / required validation before customer deployment

- Reference, direct-path only. Optional channel FIR approximates *stationary*
  room responses; not geometrically accurate 3D reflections/occlusion.
- Moving-source propagation uses a small number of fixed-point iterations and
  timestamped vector-interpolated poses; no acceleration/relativistic/shock-wave guarantee.
- Mono PCM WAV is the supported file path; explicit conversion required for
  other sample rates / stereo / compressed formats. No resampling in SDK.
- Rendering exposes overrange/clipping telemetry, with optional explicit
  `clip_output=True`. The default preserves linear signals without silently
  distorting the waveform. Monitor signal range before BEATs ingestion.
- Scheduling uses integer sample indices for trigger/stop; instantaneous
  gain changes and non-seamless source loops can produce clicks. Add smooth
  gain envelopes/crossfade loops before shipping audio authoring tools.
- Simulation reference renderer is NumPy CPU code, not a GPU ray tracer.
  Multi-env GPU scaling, allocation peaks, and high source counts UNBENCHMARKED.
- BEATs model weights and 527-label index mapping are NOT bundled and were
  not available for testing in the generation environment; test in your Mac
  environment with the real checkpoint. Production model latency, quality,
  device memory, and event calibration remain unverified.
- `OndaSoundBrain` accepts trusted official BEATs cpt2 AudioSet checkpoint
  only. A custom independently fine-tuned head needs a custom brain adapter.
- Mixing all microphones to mono for semantics means multiple simultaneous
  sound labels have **no source-direction association**. This must be
  implemented separately before advertising sound-specific localization.
- Current `UnifiedHearing` is a library glue class, not a complete
  Isaac Sim GUI extension or an RL observation manager plugin.
- Python threads are best-effort, not hard real time. Headless native Isaac
  Lab/MuJoCo motion tests and 30-min integration soak must be performed on
  v0.17; old v0.16 results don't automatically carry over.

## Recommended merge gates

1. All 39 additive regression tests, v0.16 regressions, and native Isaac Lab
   motion tests pass on the target branch.
2. Prove source emission path and correct frames by running attached telephone
   with world pose changing in both Isaac and MuJoCo; ground truth never in policy.
3. Verify onset/stop/loop exactly on sample boundaries, including chunk splits.
4. Validate propagation delays against geometry analytically and against a
   trusted independent implementation; include rotating and translating arrays.
5. Profile P95/P99 overhead and allocations under increasing source/env counts.
6. Independently validate Onda Sound Brain checkpoint+527 labels and test
   correct semantic class on labeled, held-out audio; never invent scores.
7. Expand to semantic-spatial source association and calibrated room rendering
   as separate, measurable subsequent milestones.

### Extra validation commands

```bash
bash tools/run_v017_validation.sh
python tools/validate_sound_brain_v017.py --wav /path/to/real_sound_16k_mono.wav
python examples/v017_mujoco_object_binding.py --wav /path/to/real_sound_16k_mono.wav
```

The MuJoCo smoke example exercises actual changing body transforms; it does
not render a GUI or implement a reinforcement-learning task. Both the BEATs
checkpoint test and native MuJoCo test require target-platform dependencies
and were **not run** by the archive author.
