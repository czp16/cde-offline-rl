import os
import numpy as np
import torch
import yaml
import argparse

class Config:
    DEVICE = torch.device('cpu')

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.hyperparams = {}

    def load_hyperparams(self):
        self.args = self.parser.parse_args()
        
        assert hasattr(self.args, 'hyperparams'), 'The file of hyperparameters is not loaded.'
        with open(self.args.hyperparams, 'r') as f:
            self.hyperparams = yaml.safe_load(f)

        for key, value in self.args.__dict__.items():
            if key == 'env_name' and value:
                self.hyperparams['env']['name'] = self.args.__dict__['env_name']
            elif key in self.hyperparams['misc']:
                self.hyperparams['misc'][key] = self.args.__dict__[key]
            elif key in self.hyperparams['DICE']:
                self.hyperparams['DICE'][key] = self.args.__dict__[key]

    def select_device(self):
        cudaid = self.hyperparams['misc']['cudaid']
        if cudaid >= 0:
            Config.DEVICE = torch.device('cuda:%d' % (cudaid))
        else:
            Config.DEVICE = torch.device('cpu')
            torch.set_num_threads(4)

def to_tensor(x):
    if isinstance(x, torch.Tensor):
        return x.to(device=Config.DEVICE)
    x = np.asarray(x, dtype=float)
    x = torch.tensor(x, device=Config.DEVICE, dtype=torch.float32)
    return x

def to_numpy(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return x
    

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)