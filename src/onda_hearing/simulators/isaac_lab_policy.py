"""Isaac Lab ObservationManager-compatible observation term.

Integration contract: an application attaches `onda_hearing_bank` to its
manager-based environment, ingests new acoustic events after physics updates,
and exposes integer `onda_time_ns` (scalar or per-environment array).
The term only reads policy features. It does not run audio generation inside
ObservationManager and does not import Isaac Lab at SDK import time.
"""
from __future__ import annotations

from onda_hearing.policy import HearingObservationBank


def acoustic_policy_observation(env, *, bank_attribute: str = "onda_hearing_bank",
                                time_attribute: str = "onda_time_ns"):
    bank = getattr(env, bank_attribute)
    if not isinstance(bank, HearingObservationBank):
        raise TypeError(f"{bank_attribute} must be a HearingObservationBank")
    if int(env.num_envs) != bank.num_envs:
        raise ValueError("environment/bank batch sizes differ")
    now_ns = getattr(env, time_attribute)
    if hasattr(now_ns, "detach"):
        now_ns = now_ns.detach().cpu().numpy()
    device = getattr(env, "device", None)
    return bank.torch(now_ns, device=device)
