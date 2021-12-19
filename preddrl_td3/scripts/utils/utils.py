import os
import shutil

import numpy as np
import joblib
import matplotlib.pyplot as plt
from matplotlib import animation

import torch

def model_parameters(model, verbose=0):
    if verbose>0:
        print('{:<30} {:<10} {:}'.format('Parame Name', 'Total Param', 'Param Shape'))
    total_params=0
    for name, param in model.named_parameters():
        if param.requires_grad:
            if verbose>0:
                print('{:<30} {:<10} {:}'.format(name, param.numel(), tuple(param.shape)))
            total_params+=param.numel()
    if verbose>0:
        print('Total Trainable Parameters :{:<10}'.format(total_params))
    return total_params
    
def model_attributes(model, verbose=0):
    attributes = {k:v for k, v in model.__dict__.items() if not k.startswith('_')}
    
    if verbose>0:
        print(sorted(attributes.items()))
        
    return attributes


def create_new_dir(new_dir, clean=False):
    if clean:
        shutil.rmtree(new_dir, ignore_errors=True)
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    return new_dir


def save_ckpt(model, save_ckpt_dir, step):
    state = {
        'step': step,
        'model_state': model.state_dict(),
        'actor_optimizer': model.actor_optimizer.state_dict(),
        'critic_optimizer': model.critic_optimizer.state_dict(),
    }
    print('Saving checkpoint .. ', save_ckpt_dir)
    torch.save(state, os.path.join(save_ckpt_dir, 'model.pth'))

def load_ckpt(model, load_ckpt_dir):
    print('Loading checkpoint .. ', load_ckpt_dir)
    checkpoint = torch.load(os.path.join(load_ckpt_dir, 'model.pth'))

    model.load_state_dict(checkpoint['model_state'])
    model.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
    model.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
    model.iteration = checkpoint['step']
    return model

def copy_src(root_src_dir, root_dst_dir, overwrite=True):
    for src_dir, dirs, files in os.walk(root_src_dir):
        dst_dir = src_dir.replace(root_src_dir, root_dst_dir, 1)
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        for file in files:
            if 'cpython' in file:
                continue
            src_file = os.path.join(src_dir, file)
            dst_file = os.path.join(dst_dir, file)
            if os.path.exists(dst_file):
                if overwrite:
                    shutil.copy(src_file, dst_file)
            else:
                shutil.copy(src_file, dst_file)

def init_weights_kaiming(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
        torch.nn.init.zeros_(m.bias.data)

def init_weights_xavier(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight.data, gain=1.4142135623730951) # gain for relu
        torch.nn.init.zeros_(m.bias.data)
