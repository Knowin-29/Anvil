"""Experimental geometry-preserving DoF representation; NOT an active model.

A prospective model would predict12 independent numbers, then this explicit
model decoder would construct the original native output. This does NOT repair
stored predictions or alter any score. Valid geometry alone does not establish
pose/contact accuracy or teacher eligibility. MOPD integration remains pending.
"""
import json
import math
import numpy as np

def encode(target,context):
    pose=np.asarray(target['camera_from_gripper'],dtype=float)
    w,h=context['image_size_wh'];u,v=target['contact_point_px'];z=target['contact_point_camera'][2]
    assert z>0
    def logit(x):
        x=min(1-1e-10,max(1e-10,x));return math.log(x)-math.log1p(-x)
    return np.concatenate([pose[:3,0],pose[:3,1],pose[:3,3],[logit(u/w),logit(v/h),math.log(z)]]).tolist()

def decode(latent,context):
    values=np.asarray(latent,dtype=float)
    if values.shape!=(12,) or not np.isfinite(values).all():raise ValueError('Expected12 finite latent values')
    x=values[:3];nx=np.linalg.norm(x)
    if nx<1e-8:raise ValueError('Degenerate first orientation vector')
    x=x/nx;y=values[3:6]-x*np.dot(x,values[3:6]);ny=np.linalg.norm(y)
    if ny<1e-8:raise ValueError('Collinear orientation vectors')
    y=y/ny;zaxis=np.cross(x,y);pose=np.eye(4);pose[:3,:3]=np.stack([x,y,zaxis],axis=1);pose[:3,3]=values[6:9]
    def sigmoid(x):return 1/(1+math.exp(-max(-30,min(30,float(x)))))
    w,h=context['image_size_wh'];pixel=np.array([sigmoid(values[9])*w,sigmoid(values[10])*h])
    if abs(values[11])>30:raise ValueError('Unrepresentable contact depth')
    depth=math.exp(values[11]);ray=np.linalg.solve(np.asarray(context['intrinsics'],dtype=float),[pixel[0],pixel[1],1.])
    point=ray*(depth/ray[2]);entity=context['entity']
    return dict(contract='anvil-prism-gripper-se3-v1',entity_id=entity['entity_id'],label=entity['label'],
        frame='camera_x_right_y_down_z_forward',length_unit='prism_source_length_unit_unscaled',pixel_coordinates='zero_based_xy',
        camera_from_gripper=pose.tolist(),approach_axis_camera=zaxis.tolist(),jaw_axis_camera=x.tolist(),
        contact_point_camera=point.tolist(),contact_point_px=pixel.tolist())
