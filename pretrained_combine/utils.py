import torch
import tempfile
import os
import shutil

class WarmupLRSchedule(object):
    '''
    Implement the learning rate schedule from Attention is All You Need
    This needs to be a top-level class in order to pickle it, even though a nested function would
    otherwise work.
    '''
    def __init__(self, warmup_steps=4000):
        ''' Initialize the learning rate schedule '''
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        ''' The actual learning rate schedule '''
        # the schedule doesn't allow for step to be zero (it's raised to the negative power),
        # but the input step is zero-based so just do a max with 1

        step = max(1, step)
        return min(step ** -0.5, step * self.warmup_steps ** -1.5)


def checkpoint(epoch, step, modules, directory, filename='checkpoint.pt', max_checkpoints=5):
    '''
    Save a checkpoint
    Args:
        epoch - current epoch
        step - current step
        modules - a dict of name to object that supports the method state_dict
        directory - the directory to save the checkpoint file
        filename - the filename of the checkpoint
        max_checkpoints - how many checkpoints to keep
    '''
    if not os.path.isdir(directory):
        os.makedirs(directory)

    state = {
        'step': step,
        'epoch': epoch,
    }

    for name, obj in modules.items():
        state[name] = obj.state_dict()

    with tempfile.NamedTemporaryFile() as temp_checkpoint_file:
        torch.save(state, temp_checkpoint_file)
        checkpoint_path = os.path.join(directory, filename)
        if os.path.exists(checkpoint_path):
            root, ext = os.path.splitext(filename)
            for i in range(max_checkpoints - 2, -1, -1):
                previous_path = os.path.join(directory, f'{root}{i}{ext}') if i else checkpoint_path
                if os.path.exists(previous_path):
                    backup_path = os.path.join(directory, f'{root}{i+1}{ext}')
                    if os.path.exists(backup_path):
                        os.replace(previous_path, backup_path)
                    else:
                        os.rename(previous_path, backup_path)

        shutil.copy(temp_checkpoint_file.name, f'{checkpoint_path}.incomplete')
        os.rename(f'{checkpoint_path}.incomplete', checkpoint_path)

    return checkpoint_path