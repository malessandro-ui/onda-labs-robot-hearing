"""Isaac Lab interface contract tests, runnable WITHOUT Kit/Isaac Lab."""
import math
from types import SimpleNamespace
import numpy as np
import pytest

from onda_hearing.contracts import MicrophoneArray
from onda_hearing.runtime import HearingRuntime, ScheduledSource
from onda_hearing.policy import WindowedHearingPipeline
from onda_hearing.experiments.auditory_control import AcousticOnlySelector, AcousticSelectionConfig
from onda_hearing.spatial.motion_srp import MotionAwareSRPLocalizer, MotionSRPConfig
from onda_hearing.simulators.isaac_lab_runtime import (
    IsaacLabAcousticConfig, IsaacLabHearingManager, onda_acoustic_policy_observation,
    onda_reset_hearing,
)
from onda_hearing.simulators.isaac_lab_wrapper import (
    IsaacLabPoseRef, IsaacLabSceneReader, OndaIsaacLabEnvWrapper,
)


def factory(signal):
    array = MicrophoneArray.from_positions(
        [[.05,.03,0],[-.045,.035,0],[-.05,-.038,0],[.042,-.047,0]],
        array_id='array', frame_id='robot/head', geometry_id='isaac_lab_test')
    def make(_):
        runtime = HearingRuntime(array, (ScheduledSource('beacon', signal, 0),), sample_rate_hz=16000)
        selector = AcousticOnlySelector(MotionAwareSRPLocalizer(MotionSRPConfig(
            block_samples=320, hop_samples=160, angular_resolution_deg=2,
            max_candidates=3, min_relative_score=.65, min_prominence=.012)),
            AcousticSelectionConfig(min_score=.08, ambiguous_ratio=.88, min_consistent_windows=2))
        return WindowedHearingPipeline(runtime, selector)
    return make


def batch(n=2, bearings=(38, -38)):
    pos = np.zeros((n,3)); pos[:,2] = 1
    quat = np.zeros((n,4)); quat[:,0] = 1  # Isaac WXYZ identity
    target = np.array([[1.5*math.cos(math.radians(x)),
                        1.5*math.sin(math.radians(x)),1] for x in bearings[:n]])
    return pos, quat, {'beacon': target}


def test_time_and_partial_reset_and_no_data_leakage():
    manager = IsaacLabHearingManager(IsaacLabAcousticConfig(2,20_000_000), factory(np.zeros(32000)))
    inputs = batch()
    assert manager.read().shape == (2,5)
    manager.seed(*inputs)
    for step in range(1, 17):
        manager.advance(*inputs, step_token=step)
    assert all(x.audio_samples == 320 for x in manager.frames())
    assert all(x.new_decision for x in manager.frames()) is False  # 20ms step, 100ms cadence
    assert not manager.read()[:,3].any()
    old_ticks = manager._ticks.copy()
    manager.reset([0]); manager.seed(*inputs, env_ids=[0])
    np.testing.assert_array_equal(manager.read()[0], 0)
    manager.advance(*inputs, step_token=17)
    assert manager._ticks[0] == 1 and manager._ticks[1] == old_ticks[1] + 1
    assert manager.frames()[0].audio_samples == 320
    with pytest.raises(ValueError, match='duplicate'):
        manager.advance(*inputs, step_token=17)
    with pytest.raises(ValueError):
        manager.seed(*inputs, env_ids=[1])
    with pytest.raises(ValueError):
        manager.advance(*batch(), step_token=10)
    out = manager.read()
    out[0, 0] = 1
    assert manager.read()[0,0] == 0


def test_wxyz_quaternion_and_input_rejection():
    manager = IsaacLabHearingManager(IsaacLabAcousticConfig(1,20_000_000), factory(np.zeros(32000)))
    p, q, sources = batch(1)
    q[:] = [[math.sqrt(.5),0,0,math.sqrt(.5)]]
    manager.seed(p,q,sources)
    assert manager.pipelines[0].runtime._history[0].array_orientation_xyzw == pytest.approx((0,0,math.sqrt(.5),math.sqrt(.5)))
    with pytest.raises(ValueError, match='quaternions'):
        manager.advance(p, q*2, sources)
    with pytest.raises(ValueError, match='source:beacon'):
        manager.advance(p,q,{'beacon': np.ones((2,3))})
    with pytest.raises(ValueError, match='source:beacon'):
        manager.advance(p,q,{'beacon': np.full((1,3), np.nan)})
    with pytest.raises(ValueError, match='capacity'):
        IsaacLabAcousticConfig(17,20_000_000)
    with pytest.raises(ValueError, match='independent'):
        same = factory(np.zeros(32000))(0)
        IsaacLabHearingManager(IsaacLabAcousticConfig(2,20_000_000), lambda _: same)


def test_genuine_audio_direction_no_privileged_features():
    rng = np.random.default_rng(202613)
    audio = rng.normal(size=16000*2).astype(np.float32)
    spectrum = np.fft.rfft(audio)
    freqs = np.fft.rfftfreq(audio.size, 1/16000)
    spectrum[(freqs<300)|(freqs>3800)] = 0
    audio = np.fft.irfft(spectrum, n=audio.size).astype(np.float32)
    audio *= .085/max(1.,float(np.std(audio))*3)
    manager = IsaacLabHearingManager(IsaacLabAcousticConfig(2,20_000_000),factory(audio))
    inputs=batch()
    manager.seed(*inputs)
    for tick in range(1, 71):
        manager.advance(*inputs, step_token=tick)
    outputs=manager.read()
    assert outputs.shape == (2,5) and outputs.dtype == np.float32
    assert outputs[:,3].tolist() == [1,1]
    actual=np.degrees(np.arctan2(outputs[:,0],outputs[:,1]))
    assert abs(actual[0]-38) <= 10
    assert abs(actual[1]+38) <= 10
    assert all(f.observation is None or not hasattr(f.observation,'source_positions_m') for f in manager.frames())


def test_observation_term_is_read_only():
    torch = pytest.importorskip('torch')
    manager = IsaacLabHearingManager(IsaacLabAcousticConfig(2,20_000_000), factory(np.zeros(32000)))
    env=SimpleNamespace(num_envs=2,device='cpu',onda_hearing_manager=manager)
    a=onda_acoustic_policy_observation(env)
    b=onda_acoustic_policy_observation(env)
    assert a.shape==(2,5) and a.dtype==torch.float32
    assert torch.equal(a,b)
    assert manager.frames()==(None,None)
    onda_reset_hearing(env, torch.tensor([1]))
    assert manager.frames()==(None,None)


class FakeEnv:
    num_envs=2
    device='cpu'
    def __init__(self):
        torch=pytest.importorskip('torch')
        self.torch=torch
        def asset(positions):
            return SimpleNamespace(data=SimpleNamespace(root_pos_w=torch.tensor(positions,dtype=torch.float64),
               root_quat_w=torch.tensor([[1,0,0,0],[1,0,0,0]],dtype=torch.float64)))
        self.scene={'robot':asset([[0,0,1],[0,0,1]]),
                    'phone':asset([[1.5,0,1],[0,1.5,1]])}
        self.counter=0
    def reset(self):
        self.counter=0
        return {'policy':self.torch.zeros((2,3))},{'ok':True}
    def step(self,action):
        self.counter+=1
        done=self.torch.tensor([self.counter==4,False])
        return {'policy':self.torch.ones((2,3))},self.torch.zeros(2),done,self.torch.zeros(2,dtype=self.torch.bool),{}
    def close(self): pass


def test_existing_task_wrapper_preserves_policy_and_auto_resets():
    env=FakeEnv()
    poses=IsaacLabSceneReader(array=IsaacLabPoseRef('robot'),sources={'beacon':IsaacLabPoseRef('phone')})
    manager=IsaacLabHearingManager(IsaacLabAcousticConfig(2,20_000_000),factory(np.zeros(32000)))
    wrapper=OndaIsaacLabEnvWrapper(env,manager,poses)
    obs, info=wrapper.reset()
    assert set(obs)=={'policy','onda_hearing'}
    assert obs['policy'].shape==(2,3) and obs['onda_hearing'].shape==(2,5)
    for j in range(4):
        obs,*_=wrapper.step(env.torch.zeros((2,1)))
        assert obs['policy'].shape==(2,3)
    assert manager._ticks.tolist()==[0,4]
    assert manager.frames()[0].audio_samples==0
    assert manager.frames()[1].audio_samples==320
    assert obs['onda_hearing'].shape==(2,5)
    assert manager.read()[0,3]==0
    obs,_=wrapper.reset()
    obs,*_=wrapper.step(env.torch.zeros((2,1)))
    assert obs["onda_hearing"].shape == (2,5)
    wrapper.close()
