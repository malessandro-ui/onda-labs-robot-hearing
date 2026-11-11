#!/usr/bin/env python3
"""Offline benchmark of Onda finite image geometry against PRA 0.10.1 images.

No direct RIR sum: compares wall-image locations and propagation distances.
PRA adds a fractional-delay synthesis offset to its audio RIR; don't compare
absolute waveform indexes without compensating that implementation detail.
"""
import json
import numpy as np
from onda_hearing.acoustic_scene.physical import (
    TransmissionMaterial, AcousticGeometry, AcousticSurface)
from onda_hearing.acoustic_scene.unified_physics import find_specular_path


def run():
    import pyroomacoustics as pra
    room_dims=np.array([4.,3.,2.5])
    src=np.array([1.,.8,1.])
    mic=np.array([2.2,1.5,1.1])
    material=TransmissionMaterial((125.,8000.),(50.,50.),'synthetic_geometry_check')
    surfaces=[]
    for axis,size in enumerate(room_dims):
        for side in (0.,size):
            other=[i for i in range(3) if i!=axis]
            pts=[]
            for u,v in ((0,0),(1,0),(1,1),(0,1)):
                p=np.zeros(3);p[axis]=side
                p[other[0]]=u*room_dims[other[0]]
                p[other[1]]=v*room_dims[other[1]]
                pts.append(tuple(p))
            surfaces.append(AcousticSurface.quad(f'axis{axis}_side{side}',pts,material))
    geom=AcousticGeometry(tuple(surfaces))
    room=pra.ShoeBox(room_dims, fs=16000, materials=pra.Material(.2), max_order=1)
    room.add_source(src);room.add_microphone_array(mic.reshape(3,1))
    room.image_source_model()
    images=room.sources[0].images.T
    orders=room.sources[0].orders
    first_images=images[orders==1]
    report=[]
    for surface in surfaces:
        face=np.asarray(surface.vertices_m)
        a,b,c=face[:3]
        n=np.cross(b-a,c-a);n/=np.linalg.norm(n)
        mirror=src-2*np.dot(src-a,n)*n
        error=float(np.min(np.linalg.norm(first_images-mirror,axis=1)))
        path=find_specular_path(src,mic,(surface,),geom)
        if path is None:
            raise AssertionError(f'missing rectangular shoebox reflection: {surface.surface_id}')
        expected=float(np.linalg.norm(mic-mirror))
        assert abs(path.length_m-expected)<1e-8
        report.append({'surface_id':surface.surface_id,
                       'image_position_error_m':error,
                       'path_distance_error_m':abs(path.length_m-expected),
                       'delay_s':path.length_m/343.})
    assert max(x['image_position_error_m'] for x in report)<1e-5
    print(json.dumps({'test':'first_order_shoebox_vs_pyroomacoustics_0.10.1',
                      'status':'PASS','n':len(report),
                      'comparison':'image locations and path distance (NOT RIR amplitude)',
                      'paths':report},indent=2))

if __name__=='__main__':
    run()
