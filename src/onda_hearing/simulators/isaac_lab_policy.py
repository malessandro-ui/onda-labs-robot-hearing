"""Isaac Lab observation terms for Onda hearing, with lazy torch dependencies.

The original `HearingObservationBank` term remains supported for existing
v0.15 applications. New applications should use `IsaacLabHearingManager`, whose
spatial observations come from the proven WindowedHearingPipeline.
"""
from __future__ import annotations

from onda_hearing.policy import HearingObservationBank
from .isaac_lab_runtime import (
    IsaacLabAcousticConfig, IsaacLabHearingManager,
    onda_acoustic_policy_observation, onda_reset_hearing,
)


def acoustic_policy_observation(env, *, bank_attribute: str = 'onda_hearing_bank',
                                time_attribute: str = 'onda_time_ns'):
    """Legacy 5-element ObservationManager term for event-based banks."""
    bank = getattr(env, bank_attribute)
    if not isinstance(bank, HearingObservationBank):
        raise TypeError(f'{bank_attribute} must be a HearingObservationBank')
    if int(env.num_envs) != bank.num_envs:
        raise ValueError('environment/bank batch sizes differ')
    now_ns = getattr(env, time_attribute)
    if hasattr(now_ns, 'detach'):
        now_ns = now_ns.detach().cpu().numpy()
    return bank.torch(now_ns, device=getattr(env, 'device', None))
