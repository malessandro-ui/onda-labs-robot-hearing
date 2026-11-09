Onda Robot Hearing
==================

Spatial Audio Intelligence for Robotics

Onda Robot Hearing is a software development kit developed by Onda Labs to enable robots to perceive, localize, and respond to sound.

The SDK connects multichannel acoustic processing with robot state, spatial awareness, and physical control. It provides a common foundation for integrating hearing into robotic systems, from acoustic simulation and localization to sound-directed motion.

Onda's architecture is simulator-independent, supports configurable microphone arrays, and is designed to integrate auditory perception with existing robotic sensing and control systems.

Current version: 0.14.0-alpha

Simulation integrations: NVIDIA Isaac Sim 5.1 and MuJoCo

Acoustic Scene GUI
------------------

The v0.17 branch includes a local robot acoustic intelligence observatory for
building fixed-world acoustic scenes, rendering real multichannel PCM, seeing
where sounds originate, inspecting optional Onda Sound Brain / BEATs results,
reviewing confidence and policy observations, listening to the result, and
exporting a validation report.

    python -m pip install -e ".[dev]"
    onda-hearing-gui

See [Onda Hearing Workbench](docs/GUI_WORKBENCH.md) for the supported workflow
and current limits.

Primary capabilities: Multichannel acoustics, sound localization, motion-aware spatial processing, acoustic simulation, and auditory control


Overview
--------

Robots must operate in environments where important information is not always visible.

A person may speak from outside a camera's field of view. A mechanical failure may produce an identifiable sound before it becomes visually apparent. An impact, alarm, or other acoustic event may require a robot to redirect its attention.

Hearing provides a complementary source of information about the physical environment.

Onda Labs is developing the acoustic perception infrastructure needed to make sound a useful sensing modality for robotics.

Onda Robot Hearing provides the software components for processing microphone signals, estimating sound direction, incorporating robot motion, and connecting acoustic observations to robot behavior.


Core Capabilities
-----------------

Multichannel Audio Processing

The SDK supports configurable microphone-array geometries and synchronized multichannel audio.

Developers can define microphone positions, generate channel-specific acoustic signals, and evaluate array behavior using known source and robot configurations.

The acoustic runtime provides structured audio representations that can be used independently of a particular robot or simulation platform.


Sound Source Localization

Onda includes spatial processing components for estimating sound direction from multichannel microphone signals.

The SDK provides GCC-PHAT and SRP-based processing, spatial estimator interfaces, and acoustic candidate tracking components.

Localization outputs are represented as spatial observations with associated evidence and timing information.

These estimates can be consumed by downstream control and perception systems without requiring access to ground-truth sound-source positions.


Motion-Aware Spatial Perception

Robotic hearing requires accounting for the movement and orientation of the microphone array.

Onda provides coordinate-frame transformations and motion-aware spatial processing components for interpreting sound directions relative to robot pose.

The system distinguishes between microphone-relative and world-frame directions, allowing acoustic measurements to be interpreted consistently as a robot changes orientation.

This architecture supports integration with robot state estimation and external pose sources.


Auditory Control

Onda connects acoustic observations to robot control interfaces.

The hearing-to-motion pipeline uses estimated sound direction to determine a desired head orientation and applies the corresponding command to a simulated robot articulation.

The controller operates on acoustic-derived observations rather than privileged source-position information.

This integration has been validated in native NVIDIA Isaac Sim and MuJoCo environments.


Acoustic Simulation

The SDK includes acoustic propagation, microphone geometry, and room-modeling components.

These components support controlled simulation of sound propagation, multichannel audio generation, and acoustic evaluation.

The software includes direct-path propagation, room impulse response utilities, and tools for constructing repeatable acoustic benchmark scenarios.


Robotics Simulation Integration

Onda separates acoustic processing from simulator-specific functionality.

NVIDIA Isaac Sim and MuJoCo adapters provide access to robot state, geometry, and articulation control while using a common hearing runtime.

This allows the acoustic processing architecture to remain consistent across simulation platforms.


System Architecture
-------------------

The Onda hearing pipeline connects robot state, multichannel audio, acoustic perception, and physical control.

    ROBOT / SIMULATION ENVIRONMENT
                 |
                 v
         Robot Pose and State
                 |
                 v
          Microphone Array
                 |
                 v
        Multichannel Audio PCM
                 |
                 v
        Acoustic Perception
                 |
                 v
        Spatial Observations
      Bearing / Evidence / Time
                 |
                 v
        Robot Control Interface
                 |
                 v
          Robot Actuation
                 |
                 v
         Updated Robot State

The simulator supplies robot state and source geometry to the acoustic runtime.

The runtime generates microphone signals, processes acoustic information, and produces spatial observations.

Those observations are passed to the robot control interface, which applies commands to the articulated system.

The updated robot state becomes available to subsequent perception and control cycles.

This separation enables Onda's hearing components to operate across different simulators without embedding simulator-specific physics APIs into the core runtime.


Engineering Principles
----------------------

Simulator Independence

Acoustic processing and robot control interfaces are separated from simulator-specific APIs.


Configurable Hardware Geometry

Microphone arrays are described using explicit spatial specifications rather than fixed hardware assumptions.


Coordinate-Frame Consistency

Robot orientation and microphone pose are incorporated when interpreting spatial acoustic observations.


Observation-Driven Control

Control decisions use acoustic measurements rather than direct access to known sound-source coordinates.


Explicit Timing

Audio sampling, acoustic perception updates, and robot physics steps are treated as distinct timing domains.


Reproducible Evaluation

Validation runs retain audio recordings, simulation telemetry, numerical results, and acceptance criteria.


NVIDIA Isaac Sim Integration
----------------------------

Onda Robot Hearing v0.14 includes a native NVIDIA Isaac Sim 5.1 hearing-to-motion integration.

The implementation connects a four-channel microphone array to an articulated robot head controlled through NVIDIA PhysX.

The acoustic runtime generates synchronized microphone signals from a simulated sound source.

Onda processes those signals to estimate sound direction and produces spatial observations for the control interface.

The controller converts the acoustic estimate into a head-yaw command, which is applied to the robot articulation through Isaac Sim.

The resulting motion is evaluated against the known acoustic source position.


Native Isaac Sim Validation Results
-----------------------------------

The integration was validated using NVIDIA Isaac Sim 5.1 on an NVIDIA RTX 6000 Ada GPU.

Three acceptance experiments were completed.

| Experiment | Final Head Yaw | Alignment Error | Result |
|---|---:|---:|---|
| Sound source at +38 degrees | +35.997 degrees | 2.003 degrees | PASS |
| Sound source at -38 degrees | -39.997 degrees | 1.997 degrees | PASS |
| Silence | 0.000 degrees | No movement | PASS |

All three native Isaac Sim acceptance tests passed.

Both directional experiments achieved approximately 2 degrees of final head alignment error.

The silence experiment produced no accepted directional acoustic observations and no head movement.


Validation Configuration
------------------------

| Parameter | Configuration |
|---|---|
| Simulator | NVIDIA Isaac Sim 5.1 |
| GPU | NVIDIA RTX 6000 Ada |
| Robot articulation | Revolute head-yaw joint |
| Microphone array | Four channels |
| Sample rate | 16 kHz |
| Experiment duration | 2.4 seconds |
| Acoustic scenario | Single-source, direct-path |
| Tested source bearings | +38 and -38 degrees |
| Control input | Audio-derived spatial observations |
| Source ground truth | Excluded from controller |

The native integration demonstrates a complete acoustic perception-to-motion loop within a physics-based robot simulation.

The results establish performance under the specified test conditions. They do not constitute a general accuracy guarantee for arbitrary acoustic environments or physical microphone hardware.


Reproducing the Isaac Sim Validation
------------------------------------

The repository includes a native Isaac Sim validation runner:

    tools/validate_isaac_v014.py

From a compatible Isaac Sim 5.1 Python environment, run:

    python -u tools/validate_isaac_v014.py

The validator automatically:

1. Generates the required Isaac Sim USD scenes.
2. Initializes the simulated robot articulations.
3. Runs the positive-bearing acoustic experiment.
4. Runs the negative-bearing acoustic experiment.
5. Runs the silence control.
6. Evaluates the experiment acceptance criteria.
7. Writes a consolidated validation report.

Each experiment records:

- Multichannel microphone audio
- Robot joint telemetry
- Acoustic perception observations
- Experiment summary
- Execution logs
- Validation status

Validation documentation:

[Native Isaac Sim Validation Report](docs/validation/V014_NATIVE_ISAAC_VALIDATION.md)

[Machine-Readable Validation Results](docs/validation/V014_NATIVE_ISAAC_VALIDATION.json)

[Isaac Sim Integration Guide](docs/V014_ISAAC_HEAR_AND_TURN.md)


MuJoCo Integration
------------------

Onda also provides native MuJoCo hearing-to-motion integration.

The MuJoCo implementation uses an articulated robot head, a configurable microphone array, and Onda's spatial processing and control components.

The integration enables acoustic-derived head motion within a MuJoCo physics environment.

The underlying hearing and control architecture is shared with the Isaac Sim implementation.

To install the MuJoCo dependencies:

    python -m pip install -e ".[mujoco,dev]"

Run the standard demonstration:

    python examples/mujoco_hear_and_turn.py --strict

Run the opposite-bearing demonstration:

    python examples/mujoco_hear_and_turn.py --source-bearing -38 --strict

Run the silence control:

    python examples/mujoco_hear_and_turn.py --mode silence --strict

Further documentation:

[MuJoCo Closed-Loop Integration](docs/V013_CLOSED_LOOP_MUJOCO.md)


Installation
------------

Onda Robot Hearing is implemented in Python.

Package dependencies and optional simulator integrations are defined in pyproject.toml.

To install from the local repository:

    cd onda-robot-hearing

    python3 -m venv .venv

    source .venv/bin/activate

    python -m pip install -e ".[dev]"

Run the test suite:

    python -m pytest -q

The core SDK can be tested independently of optional simulator installations where supported.

Native NVIDIA Isaac Sim execution requires an Isaac Sim-compatible Python environment.

Native MuJoCo execution requires the corresponding MuJoCo dependencies.


SDK Components
--------------

The SDK is organized into distinct processing and integration modules.

| Module | Responsibility |
|---|---|
| contracts/ | Audio, geometry, localization, timing, and observation data contracts |
| runtime/ | Scene processing, audio scheduling, streaming, and hearing execution |
| physics/ | Acoustic propagation and reference simulation |
| rooms/ | Room modeling and impulse response utilities |
| spatial/ | Sound localization, estimation, and tracking interfaces |
| motion/ | Coordinate transforms and motion-aware spatial processing |
| control/ | Robot orientation and motion control |
| simulators/ | Isaac Sim and MuJoCo integration |
| experiments/ | Hearing-to-motion execution and evaluation |

Additional directories include:

| Directory | Purpose |
|---|---|
| examples/ | Simulator and acoustic demonstrations |
| tests/ | Automated software tests |
| benchmarks/ | Spatial and acoustic evaluation |
| tools/ | Validation utilities |
| docs/ | Engineering and integration documentation |


Robotics Integration
--------------------

Onda is designed to make acoustic observations accessible to robotic software systems.

A robotics application can provide microphone geometry, synchronized audio, robot orientation, and relevant state information to the hearing runtime.

The perception layer can return structured spatial observations for downstream processing.

These observations can support:

- Sound-directed robot head orientation
- Auditory attention and sensor steering
- Spatial awareness beyond a camera's field of view
- Acoustic event monitoring
- Multimodal perception and sensor fusion
- Robot learning and policy evaluation
- Integration with higher-level planning systems

Integration depends on the robot platform, microphone hardware, available state interfaces, and application requirements.

Hardware-specific deployment and real-time performance require validation on the target platform.


Validation Scope
----------------

Onda Robot Hearing v0.14 has demonstrated native hearing-to-motion integration in NVIDIA Isaac Sim 5.1, including positive-bearing localization, negative-bearing localization, and silence control.

The current native Isaac Sim validation covers a four-channel simulated microphone array and a single direct-path acoustic source.

Additional conditions, including reverberation, multiple simultaneous sources, robot locomotion noise, and physical microphone arrays, require separate evaluation.

The SDK includes spatial processing, room-modeling, and tracking components beyond the validated native demonstration. These components have their own implementation and testing scope.

Speech understanding, acoustic event semantics, persistent physical source identity, and hardware deployment are not established by the v0.14 native acceptance results.


Development Roadmap
-------------------

NVIDIA Isaac Lab Integration

The next development phase focuses on making Onda's acoustic observations available to robot learning environments built with NVIDIA Isaac Lab.

Planned components include:

- Isaac Lab acoustic observation adapters
- Policy-ready directional representations
- Confidence and detection-validity signals
- Timestamped hearing observations
- Multi-environment execution
- Hearing-conditioned control tasks


Advanced Acoustic Environments

Further development will extend acoustic evaluation across:

- Reverberant indoor environments
- Multiple simultaneous acoustic sources
- Moving sound sources
- Moving microphone arrays
- Robot-generated ego-noise
- Acoustic domain randomization
- Simulation-to-hardware evaluation

These developments support Onda Labs' broader objective of integrating acoustic intelligence into embodied robotic systems.


Documentation
-------------

[Architecture Report](docs/ARCHITECTURE_REPORT.md)

[Implementation Gates](docs/IMPLEMENTATION_GATES.md)

[Isaac Sim Integration](docs/V014_ISAAC_HEAR_AND_TURN.md)

[Native Isaac Sim Validation](docs/validation/V014_NATIVE_ISAAC_VALIDATION.md)

[MuJoCo Closed-Loop Integration](docs/V013_CLOSED_LOOP_MUJOCO.md)

[Runtime and Simulator Architecture](docs/V012_RUNTIME_SIMULATORS.md)

[Spatial Engineering](docs/V05_SPATIAL_ENGINEERING.md)

[Room Acoustics and Motion](docs/V07_ROOM_MOTION.md)

Previous release documentation and engineering notes are retained in the docs directory.


Licensing
---------

The source repository is maintained without a public redistribution license.

Redistribution, commercial use, and third-party integration require appropriate authorization from Onda Labs and compliance with applicable third-party licenses.


Onda Labs
---------

Spatial Audio Intelligence for Robotics.
