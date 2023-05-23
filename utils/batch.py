import torch
import numpy as np


class Batch(object):
    """Suggested keys: [obs, act, rew, done, obs_next, info]"""

    def __init__(self, **kwargs):
        super().__init__()
        self.update(**kwargs)

    def __len__(self):
        _size = min([
            len(v) for k,v in self.__dict__.items() if self.__dict__[k] is not None
        ])
        return _size

    def __getitem__(self, index):
        b = Batch()
        for k in self.__dict__.keys():
            if self.__dict__[k] is not None:
                b.update(**{k: self.__dict__[k][index]})
        return b

    def update(self, **kwargs):
        self.__dict__.update(kwargs)

    # sample from batch without traversing
    def sample(self, batch_size):
        N = len(self)
        assert 0 < batch_size <= N
        idx = np.random.choice(N, batch_size)
        return self[idx]
    
    # yield mini batch by traversing
    def traverse_sampler(self, size=None, permute=True):
        length = self.__len__()
        if size is None:
            size = length
        temp = 0
        if permute:
            index = np.random.permutation(length)
        else:
            index = np.arange(length)
        while temp < length:
            yield self[index[temp:temp + size]]
            temp += size