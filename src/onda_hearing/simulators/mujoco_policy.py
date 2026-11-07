"""MuJoCo policy bridge. Does not own the physics step or a robot policy.

Use this after `mujoco.mj_step` with the existing MuJoCoSceneAdapter,
HearingRuntime and a configured perception backend. No ground-truth position
is delivered to policy features. No mujoco import is needed in this module.
"""
from __future__ import annotations

from onda_hearing.policy import PolicyHearingConfig, PolicyHearingEncoder


class MuJoCoPolicyBridge:
    def __init__(self, runtime, pose_adapter, *, config: PolicyHearingConfig | None = None):
        self.runtime = runtime
        self.pose_adapter = pose_adapter
        self.encoder = PolicyHearingEncoder(config)
        self._previous_time_ns = None

    def reset(self):
        self.runtime.reset()
        self.encoder.reset()
        self._previous_time_ns = None

    def observe(self, time_ns: int):
        """Advance acoustic simulation at a monotonically increasing physics time.

        First call only initializes the runtime's epoch and returns invalid features.
        """
        if type(time_ns) is not int or time_ns < 0:
            raise ValueError("time_ns must be nonnegative int")
        if self._previous_time_ns is not None and time_ns <= self._previous_time_ns:
            raise ValueError("physics time must advance; reset after rewind")
        result = self.runtime.advance(self.pose_adapter.snapshot(time_ns))
        self._previous_time_ns = time_ns
        if result is not None:
            self.encoder.ingest_hearing(result.observation)
        return self.encoder.read(time_ns)


class WindowedMuJoCoPolicyBridge:
    """Fully audio-derived v0.15 policy adapter for native MuJoCo stepping.

    This class does not own physics, actuation, or the customer's policy.
    The caller advances mj_step, then calls observe(time_ns). It exposes
    PolicyFrame including the five-element spatial features. The selector
    sees acoustic PCM and array pose history only, never simulator source truth.

    Unlike the earlier MuJoCoPolicyBridge, this explicitly incorporates the
    proven windowed MotionAwareSRPLocalizer / AcousticOnlySelector pipeline.
    """
    def __init__(self, runtime, pose_adapter, selector, *, pipeline_config=None,
                 policy_config: PolicyHearingConfig | None = None):
        from onda_hearing.policy import WindowedHearingPipeline
        self.pose_adapter = pose_adapter
        self.pipeline = WindowedHearingPipeline(
            runtime, selector, config=pipeline_config, policy_config=policy_config)

    def reset(self):
        self.pipeline.reset()

    def observe(self, time_ns: int):
        if type(time_ns) is not int or time_ns < 0:
            raise ValueError('time_ns must be nonnegative int')
        return self.pipeline.advance(self.pose_adapter.snapshot(time_ns))
