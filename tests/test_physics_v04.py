import numpy as np
import pytest
from onda_hearing.physics import (AcousticConfig,PoseTrajectory,PhysicalSource,DirectPathRenderer,
                                   convolve_rir,sample_acoustic_profile)

FS=16000
CFG=AcousticConfig(sample_rate_hz=FS, interpolation_half_width=12)

def click(length=2400):
    a=np.zeros(length);a[800]=.1
    return a

def render(offsets,source_position,signal=None,**kw):
    s=PhysicalSource('a',click() if signal is None else signal,PoseTrajectory.stationary(source_position))
    return DirectPathRenderer(CFG).render(offsets,PoseTrajectory.stationary([0,0,0]),[s],.14,**kw)

def test_delay_and_tdoa_independent_geometry():
    x=render([[0,0,0],[.2,0,0]], [2,0,0])
    # expected geometric delays, independent of the renderer implementation
    expected=((2-.2)-2)/343
    actual=(np.argmax(np.abs(x[:,1]))-np.argmax(np.abs(x[:,0])))/FS
    assert abs(actual-expected)<1.1/FS
    assert x.shape==(2240,2)

def test_distance_inverse_amplitude():
    tone=.1*np.sin(2*np.pi*250*np.arange(4000)/FS)
    x=render([[0,0,0]],[1,0,0],tone); y=render([[0,0,0]],[2,0,0],tone)
    assert np.std(x[500:1500]) / np.std(y[500:1500]) == pytest.approx(2,rel=.005)

def test_multisource_additivity():
    r=DirectPathRenderer(CFG)
    a=PoseTrajectory.stationary([0,0,0])
    s1=PhysicalSource('a',click(),PoseTrajectory.stationary([2,0,0]))
    s2=PhysicalSource('b',click(),PoseTrajectory.stationary([0,2,0]),start_s=.01)
    args=([[0,0,0],[.1,0,0],[0,.1,0]],a)
    both=r.render(*args,[s1,s2],.14)
    summed=r.render(*args,[s1],.14).astype(np.float64)+r.render(*args,[s2],.14)
    np.testing.assert_allclose(both,summed,atol=2e-8)

def test_fractional_delay_sine_phase():
    # Steady sinusoid with prefix; compare phase against analytic fractional delay.
    f=437.0
    t=np.arange(4096)/FS
    signal=np.sin(2*np.pi*f*t)*.1
    x=render([[0,0,0]],[1.075,0,0],signal)
    te=np.arange(len(x))/FS-1.075/343
    expected=np.sin(2*np.pi*f*te)*.1/1.075
    mask=(te>.01)&(te<(len(signal)/FS-.01))
    assert np.max(abs(x[mask,0]-expected[mask]))<1e-4

def test_moving_microphone_continuity():
    t=np.linspace(0,.1,6)
    positions=np.column_stack((np.linspace(0,.1,6),np.zeros(6),np.zeros(6)))
    traj=PoseTrajectory(t,positions)
    tone=np.sin(2*np.pi*300*np.arange(4000)/FS)*.1
    src=PhysicalSource('tone',tone,PoseTrajectory.stationary([2,0,0]))
    x=DirectPathRenderer(CFG).render([[0,0,0]],traj,[src],.1)[:,0]
    # No discontinuity at trajectory segment boundaries. In-band signal should be smooth.
    mask=np.arange(len(x))/FS>.02
    assert np.max(np.abs(np.diff(x[mask]))) < .015

def test_moving_source_retarded_time():
    src_traj=PoseTrajectory([0,.1],[[2,0,0],[1.9,0,0]])
    source=PhysicalSource('moving',click(),src_traj)
    x=DirectPathRenderer(CFG).render([[0,0,0]],PoseTrajectory.stationary([0,0,0]),[source],.14)
    assert np.isfinite(x).all() and np.max(abs(x))>0

def test_rir_known_convolution_and_sum():
    x=np.array([[1.,2.],[0,0],[0,0]])
    h=np.zeros((3,2,2));h[0,0,0]=1;h[1,1,0]=.5;h[2,0,1]=2
    y=convolve_rir(x,h)
    np.testing.assert_allclose(y[:,0],[1,1,0,0,0])
    np.testing.assert_allclose(y[:,1],[0,0,2,0,0])
    assert convolve_rir(x,h,output_mode='same').shape==(3,2)

def test_randomization_deterministic_and_bounded():
    a=sample_acoustic_profile(6,seed=17)
    b=sample_acoustic_profile(6,seed=17)
    assert a.rt60_s==b.rt60_s and a.noise_std==b.noise_std
    np.testing.assert_array_equal(a.channel_gains,b.channel_gains)
    assert .2<=a.rt60_s<=.6 and 0<=a.noise_std<=.002
    assert np.all(a.channel_gains>=10**(-1/20)) and np.all(a.channel_gains<=10**(1/20))

def test_fail_closed_validation():
    with pytest.raises(ValueError): PoseTrajectory([0,0],[[0,0,0],[1,0,0]])
    with pytest.raises(ValueError): AcousticConfig(interpolation_half_width=1)
    with pytest.raises(ValueError): DirectPathRenderer(CFG).render([[0,0,0]],PoseTrajectory.stationary([0,0,0]),[], -1)
    with pytest.raises(ValueError): convolve_rir(np.zeros((5,2)),np.zeros((3,3,2)))
    with pytest.raises(ValueError): sample_acoustic_profile(2,seed=1,rt60_range_s=(.8,.1))

def test_legacy_still_runs():
    from onda_hearing import FreeFieldSimulator,MicrophoneArray,SoundSource
    a=MicrophoneArray([[0,0,0]])
    x=FreeFieldSimulator(16000).render(a,[0,0,0],[SoundSource('test',click(),[2,0,0])],.14)
    assert x.pcm.shape==(2240,1)
