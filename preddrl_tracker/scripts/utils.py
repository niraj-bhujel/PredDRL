#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 20:14:11 2021

@author: loc
"""
import numpy as np

def motion_kinematics(p, dt, rel_idx=0):
    r = p - p[int(rel_idx)]
    r_norm = np.linalg.norm(r, axis=-1, keepdims=True)

    v = np.gradient(p, dt, axis=0)
    v_norm = np.linalg.norm(v, axis=-1, keepdims=True)
 
    a = np.gradient(v, dt, axis=0)
    a_norm = np.linalg.norm(a, axis=-1, keepdims=True)

    h = np.divide(v, v_norm, out=np.zeros_like(v), where=(v_norm > 0.))

    d = np.divide(r, r_norm, out=np.zeros_like(r), where=(r_norm > 0.))

    return p, r, v, v_norm, a, a_norm, h, d

def node_sequence(node_pos, dt, pad_front=0, pad_end=20, seq_len=20, rel_idx=0, frames=None):
    
    p, r, v, v_norm, a, a_norm, h, d = motion_kinematics(node_pos, dt, rel_idx)
    
    mask = np.zeros((seq_len,))
    fid = np.zeros((seq_len,))
    
    state_dict = {}
    for s in ['pos', 'rel', 'vel', 'acc', 'hed', 'dir']:
        state_dict[s] = np.zeros((2, seq_len))
        
    for s in ['vnorm', 'anorm']:
        state_dict[s] = np.zeros((1, seq_len))
        
    state_dict['pos'][:, pad_front:pad_end] = p.T
    state_dict['rel'][:, pad_front:pad_end] = r.T
    state_dict['vel'][:, pad_front:pad_end] = v.T
    state_dict['vnorm'][:, pad_front:pad_end] = v_norm.T
    state_dict['acc'][:, pad_front:pad_end] = a.T
    state_dict['anorm'][:, pad_front:pad_end] = a_norm.T
    state_dict['hed'][:, pad_front:pad_end] = h.T
    state_dict['hed'][:, pad_front:pad_end] = d.T

    fid[pad_front:pad_end] = frames
    mask[pad_front:pad_end] = 1
    
    return state_dict, fid, mask

