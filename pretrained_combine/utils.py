import torch
import tempfile
import os
import shutil
import torch.nn as nn


def compute_metrics_accumulate(pred, label):
    n = pred.shape[0]
    tp = sum((pred == 1) * (label == 1))
    tn = sum((pred == 0) * (label == 0))
    fp = sum((pred == 1) * (label == 0))
    fn = sum((pred == 0) * (label == 1))
    return tp, tn, fp, fn, n


def compute_metrics_total(tp, tn, fp, fn, n):
    accuracy = (tp + tn) / n
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1: {f1}')
    return {'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1}


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


def restore(path, modules, num_checkpoints=1, map_location=None, strict=True):
    '''
    Restore from a checkpoint
    Args:
        path - path to restore from
        modules - a dict of name to object that supports the method load_state_dict
    '''
    if not os.path.isfile(path):
        print(f'Cannot find checkpoint: {path}')
        return 0, 0

    print(f'Loading checkpoint {path}')
    state = torch.load(path, map_location=map_location)

    if 'model' in modules:
        model_state = state['model']
        root, ext = os.path.splitext(path)

        # strip any trailing digits
        base = root.rstrip(''.join(str(i) for i in range(10)))

        # determine the integer representation of the trailing digits
        idx = root[len(base):]
        start_idx = int(idx) if idx else 0

        count = 1
        for idx in range(1, num_checkpoints):
            # use the digits as the start index for loading subsequent checkpoints for averaging
            path = f'{base}{start_idx + idx}{ext}'
            if not os.path.isfile(path):
                print(f'Cannot find checkpoint: {path} Skipping it!')
                continue

            print(f'Averaging with checkpoint {path}')
            previous_state = torch.load(path, map_location=map_location)
            previous_model_state = previous_state['model']
            for name, param in model_state.items():
                param.mul_(count).add_(previous_model_state[name]).div_(count + 1)

            count += 1

    for name, obj in modules.items():
        if isinstance(obj, nn.Module):
            obj.load_state_dict(state[name], strict=strict)
        else:
            obj.load_state_dict(state[name])

    return state['epoch'], state['step']


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