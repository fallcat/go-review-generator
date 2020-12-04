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