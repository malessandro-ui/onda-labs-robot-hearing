"""Author an EXPERIMENTAL minimal USD yaw-head scene for Isaac Sim 5.1.

Run with Isaac Sim's ./python.sh, which supplies pxr USD bindings. This script
only authors a stage; it does not run the physics or validate PhysX drives.
Check articulation creation and joint behavior inside your Isaac installation.

./python.sh examples/isaac_make_demo_stage.py --output /tmp/onda_isaac_head.usd
"""
from __future__ import annotations
import argparse
from pathlib import Path
import math


def make_stage(path: Path, source_azimuth_deg: float = 38.0) -> None:
    from pxr import Usd, UsdGeom, UsdPhysics, Sdf, Gf

    if not math.isfinite(source_azimuth_deg) or abs(source_azimuth_deg) > 75:
        raise ValueError('source angle must be finite and within ±75 degrees')
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(path.resolve()))
    stage.SetMetadata('metersPerUnit', 1.0)
    stage.SetMetadata('upAxis', 'Z')
    root = UsdGeom.Xform.Define(stage, '/World')
    stage.SetDefaultPrim(root.GetPrim())
    scene = UsdPhysics.Scene.Define(stage, '/World/PhysicsScene')
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
    scene.CreateGravityMagnitudeAttr(0.0)  # one-axis kinematic comparison scene
    robot = UsdGeom.Xform.Define(stage, '/World/Robot')

    # Fixed-base articulation: Base -> head_yaw -> Head.
    base = UsdGeom.Xform.Define(stage, '/World/Robot/Base')
    UsdPhysics.RigidBodyAPI.Apply(base.GetPrim())
    base_mass = UsdPhysics.MassAPI.Apply(base.GetPrim())
    base_mass.CreateMassAttr(5.0)

    # Anchor the base to the world without an extra controllable DOF.
    anchor = UsdPhysics.FixedJoint.Define(
        stage, '/World/Robot/base_fixed'
    )
    anchor.CreateBody1Rel().SetTargets([
        Sdf.Path('/World/Robot/Base')
    ])

    head = UsdGeom.Xform.Define(stage, '/World/Robot/Head')
    head.AddTranslateOp().Set(Gf.Vec3d(0, 0, 1))
    rigid = UsdPhysics.RigidBodyAPI.Apply(head.GetPrim())
    UsdPhysics.ArticulationRootAPI.Apply(robot.GetPrim())
    mass = UsdPhysics.MassAPI.Apply(head.GetPrim())
    mass.CreateMassAttr(1.)
    sphere = UsdGeom.Sphere.Define(stage, '/World/Robot/Head/Visual')
    sphere.CreateRadiusAttr(.08)
    # Collision intentionally absent: test the revolute joint and drive first.
    UsdGeom.Xform.Define(stage, '/World/Robot/Head/MicrophoneArray')
    beacon = UsdGeom.Xform.Define(stage, '/World/Beacon')
    a = math.radians(source_azimuth_deg)
    beacon.AddTranslateOp().Set(Gf.Vec3d(1.5 * math.cos(a), 1.5 * math.sin(a), 1))
    joint = UsdPhysics.RevoluteJoint.Define(stage, '/World/Robot/head_yaw')
    joint.CreateBody0Rel().SetTargets([
        Sdf.Path('/World/Robot/Base')
    ])
    joint.CreateBody1Rel().SetTargets([
        Sdf.Path('/World/Robot/Head')
    ])
    joint.CreateAxisAttr('Z')
    joint.CreateLocalPos0Attr(Gf.Vec3f(0, 0, 1))  # static world pivot
    joint.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
    # USD angle attributes are degrees, unlike Isaac Core Python joint targets.
    joint.CreateLowerLimitAttr(-90.)
    joint.CreateUpperLimitAttr(90.)
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), 'angular')
    drive.CreateTypeAttr('force')
    drive.CreateStiffnessAttr(400.)
    drive.CreateDampingAttr(40.)
    drive.CreateMaxForceAttr(1000.)
    drive.CreateTargetPositionAttr(0.)
    stage.GetRootLayer().Save()
    print(f'Authored experimental USD stage: {path.resolve()}')
    print('Robot articulation prim: /World/Robot')
    print('Head yaw joint: head_yaw')
    print('Array prim: /World/Robot/Head/MicrophoneArray')
    print('Source prim: /World/Beacon')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('/tmp/onda_isaac_head.usd'))
    parser.add_argument('--source-bearing', type=float, default=38.)
    args = parser.parse_args()
    make_stage(args.output, args.source_bearing)
