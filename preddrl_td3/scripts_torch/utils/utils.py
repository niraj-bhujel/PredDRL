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
    print('Total Trainable Parameters :{:<10}'.format(total_params))
    return total_params
    
def model_attributes(model, verbose=0):
    attributes = {k:v for k, v in model.__dict__.items() if not k.startswith('_')}
    
    if verbose>0:
        print(sorted(attributes.items()))
        
    return attributes

def save_path(samples, filename):
    joblib.dump(samples, filename, compress=3)


def restore_latest_n_traj(dirname, n_path=10, max_steps=None):
    assert os.path.isdir(dirname)
    filenames = get_filenames(dirname, n_path)
    return load_trajectories(filenames, None)


def get_filenames(dirname, n_path=None):
    import re
    itr_reg = re.compile(
        r"step_(?P<step>[0-9]+)_epi_(?P<episodes>[0-9]+)_return_(-?)(?P<return_u>[0-9]+).(?P<return_l>[0-9]+).pkl")

    itr_files = []
    for _, filename in enumerate(os.listdir(dirname)):
        m = itr_reg.match(filename)
        if m:
            itr_count = m.group('step')
            itr_files.append((itr_count, filename))

    n_path = n_path if n_path is not None else len(itr_files)
    itr_files = sorted(itr_files, key=lambda x: int(
        x[0]), reverse=True)[:n_path]
    filenames = []
    for itr_file_and_count in itr_files:
        filenames.append(os.path.join(dirname, itr_file_and_count[1]))
    return filenames


def load_trajectories(filenames, max_steps=None):
    assert len(filenames) > 0
    paths = []
    for filename in filenames:
        paths.append(joblib.load(filename))

    def get_obs_and_act(path):
        obses = path['obs'][:-1]
        next_obses = path['obs'][1:]
        actions = path['act'][:-1]
        if max_steps is not None:
            return obses[:max_steps], next_obses[:max_steps], actions[:max_steps-1]
        else:
            return obses, next_obses, actions

    for i, path in enumerate(paths):
        if i == 0:
            obses, next_obses, acts = get_obs_and_act(path)
        else:
            obs, next_obs, act = get_obs_and_act(path)
            obses = np.vstack((obs, obses))
            next_obses = np.vstack((next_obs, next_obses))
            acts = np.vstack((act, acts))
    return {'obses': obses, 'next_obses': next_obses, 'acts': acts}


def frames_to_gif(frames, prefix, save_dir, interval=50, fps=30):
    """
    Convert frames to gif file
    """
    assert len(frames) > 0
    plt.figure(figsize=(frames[0].shape[1] / 72.,
                        frames[0].shape[0] / 72.), dpi=72)
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    # TODO: interval should be 1000 / fps ?
    anim = animation.FuncAnimation(
        plt.gcf(), animate, frames=len(frames), interval=interval)
    output_path = "{}/{}.gif".format(save_dir, prefix)
    anim.save(output_path, writer='imagemagick', fps=fps)


def save_ckpt(model, save_ckpt_dir, step):
    state = {

        'model_state': model.state_dict(),
        'actor_optimizer': model.actor_optimizer.state_dict(),
        'critic_optimizer': model.critic_optimizer.state_dict(),
    }

    torch.save(state, os.path.join(save_ckpt_dir, 'model_step_' + str(step) + '.pth'))

def load_ckpt(model, load_ckpt_dir, last_step):
    checkpoint = torch.load(os.path.join(load_ckpt_dir, 'model_step_' + str(last_step) + '.pth'))

    model.load_state_dict(checkpoint['model_state'])
    model.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
    model.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])

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