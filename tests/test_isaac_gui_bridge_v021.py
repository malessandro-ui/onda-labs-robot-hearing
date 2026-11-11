from threading import Thread
import time
import pytest
from fastapi.testclient import TestClient
from onda_hearing.simulators.isaac_gui_bridge import IsaacBridgeState, create_app, BridgeError


def test_snapshot_is_isolated_and_events():
    state=IsaacBridgeState()
    state.set_scene({'loaded':True})
    state.set_selection({'prim_path':'/Root/forklift'})
    copy=state.snapshot('scene')
    copy['data']['loaded']=False
    assert state.snapshot('scene')['data']['loaded'] is True
    assert state.events(0)['events'][-1]['event']=='selection'
    assert state.events(9999)['events']==[]


def test_api_get_and_post_uses_main_thread_queue():
    state=IsaacBridgeState()
    app=create_app(state)
    client=TestClient(app)
    assert client.get('/api/isaac/status').status_code==200
    assert client.get('/api/isaac/scene').json()['data']['loaded'] is False
    assert client.post('/api/isaac/emitters',json={}).status_code==403
    assert client.post('/api/isaac/emitters',json={},headers={'X-Onda-Client':'OndaGUI'}).status_code==503
    state.set_status(connected_to_isaac=True)

    def kit_worker():
        until=time.monotonic()+3
        while time.monotonic()<until:
            if state.drain(lambda op,payload: {'attached':payload['prim_path']},max_count=1):return
            time.sleep(.005)

    worker=Thread(target=kit_worker)
    worker.start()
    r=client.post('/api/isaac/emitters',json={'prim_path':'/Root/forklift'},headers={'X-Onda-Client':'OndaGUI'})
    worker.join(3)
    assert r.status_code==201
    assert r.json()=={'attached':'/Root/forklift'}
    assert client.get('/api/isaac/events?since=0').status_code==200


def test_server_cannot_access_kit_stage_directly():
    state=IsaacBridgeState()
    assert not hasattr(state, 'stage')
    state.close()
    with pytest.raises(BridgeError): state.submit('add_emitter',{})


def test_render_file_endpoint_never_escapes_directory(tmp_path):
    state=IsaacBridgeState()
    folder=tmp_path/'renders'
    folder.mkdir()
    token='a'*32
    (folder/(token+'.wav')).write_bytes(b'RIFFfake')
    c=TestClient(create_app(state,render_root=folder))
    assert c.get('/api/isaac/renders/'+token+'.wav').status_code==200
    assert c.get('/api/isaac/renders/notatoken.wav').status_code==404
    assert c.get('/api/isaac/renders/'+('b'*32)+'.wav').status_code==404
