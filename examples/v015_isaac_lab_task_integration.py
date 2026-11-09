"""Onda Isaac Lab integration recipe for an EXISTING Isaac Lab task.

This is an example, not a standalone NVIDIA task. It intentionally does not
create a robot, modify an RL reward, or hardcode an Isaac Lab release. Invoke
`attach_onda` AFTER constructing an existing Isaac Lab ManagerBasedRLEnv or
DirectRLEnv. A caller still owns and launches the Isaac Sim/Isaac Lab app.
"""
import numpy as np

from onda_hearing.contracts import MicrophoneArray
from onda_hearing.runtime import HearingRuntime, ScheduledSource
from onda_hearing.policy import WindowedHearingPipeline
from onda_hearing.experiments.auditory_control import AcousticOnlySelector, AcousticSelectionConfig
from onda_hearing.spatial.motion_srp import MotionAwareSRPLocalizer, MotionSRPConfig
from onda_hearing.simulators.isaac_lab_runtime import IsaacLabAcousticConfig, IsaacLabHearingManager
from onda_hearing.simulators.isaac_lab_wrapper import (
    IsaacLabPoseRef, IsaacLabSceneReader, OndaIsaacLabEnvWrapper,
)


def attach_onda(env, *, array_asset: str, source_asset: str,
                source_audio: np.ndarray, array_body_index: int | None = None,
                max_cpu_envs: int = 16):
    """Returns a wrapper whose observation dict adds `onda_hearing`: [N,5].

    `array_asset`: existing rigid robot asset key in env.scene, e.g. `robot`.
    `array_body_index`: optional index of the array-bearing head body.
    `source_asset`: existing source rigid object asset key, e.g. `phone`.
    Both world poses must correspond to actual acoustic source / mic origin.
    Waveform is not a semantic label or privileged observation.
    """
    if int(env.num_envs) > max_cpu_envs:
        raise ValueError('CPU reference adapter is not appropriate for this many environments')
    dt_ns = round(float(env.step_dt) * 1e9)
    if not (0 < dt_ns <= 1_000_000_000):
        raise ValueError('invalid Isaac Lab step_dt')
    sample_rate_hz = 16000
    audio = np.asarray(source_audio,dtype=np.float32)
    if audio.ndim != 1 or audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError('source_audio must be finite nonempty mono PCM')
    array = MicrophoneArray.from_positions(
        [[.05,.03,0],[-.045,.035,0],[-.05,-.038,0],[.042,-.047,0]],
        array_id='head_array',frame_id='robot/head',geometry_id='asym90mm_v015')

    def make_pipeline(env_id):
        runtime=HearingRuntime(array,(ScheduledSource('sound',audio,0),),
                               sample_rate_hz=sample_rate_hz)
        selector=AcousticOnlySelector(MotionAwareSRPLocalizer(MotionSRPConfig(
            block_samples=320,hop_samples=160,angular_resolution_deg=2,
            max_candidates=3,min_relative_score=.65,min_prominence=.012)),
            AcousticSelectionConfig(min_score=.08,ambiguous_ratio=.88,min_consistent_windows=2))
        return WindowedHearingPipeline(runtime, selector)

    manager=IsaacLabHearingManager(
        IsaacLabAcousticConfig(int(env.num_envs),dt_ns,max_cpu_envs=max_cpu_envs),
        make_pipeline)
    poses=IsaacLabSceneReader(array=IsaacLabPoseRef(array_asset,array_body_index),
                             sources={'sound':IsaacLabPoseRef(source_asset)})
    return OndaIsaacLabEnvWrapper(env,manager,poses)


# Usage inside an already launched Isaac Lab SimulationApp:
#   env = gym.make(task_id, cfg=task_cfg).unwrapped
#   env = attach_onda(env, array_asset='robot', source_asset='phone',
#                     source_audio=waveform_at_16k, array_body_index=<head_body_index>)
#   observations, info = env.reset()
#   observations, rewards, terminated, truncated, extras = env.step(actions)
#   spatial_features = observations['onda_hearing']  # float32 [num_envs,5]
#   # The policy, not Onda, decides actions and rewards.
