"""Isaac Sim integration pattern: run from Isaac Sim's Python environment.

This file intentionally does not start SimulationApp. Attach the adapter to the
stage belonging to your running app and call on_physics_step AFTER a simulation
step. Your stage must contain the indicated USD prim paths. The example shows
usage without claiming to create a complete Isaac Sim environment.
"""
from onda_hearing.simulators import IsaacSimSceneAdapter


def make_adapter(stage):
    return IsaacSimSceneAdapter(stage,array_prim='/World/Robot/head_array',
          source_prims={'phone':'/World/Phone', 'knock':'/World/Door'})


def on_physics_step(runtime,adapter,simulation_time_seconds):
    snapshot=adapter.snapshot(round(simulation_time_seconds*1e9))
    return runtime.advance(snapshot)
