import torch
import torch.nn as nn
import torch.distributions as D

from utils.misc import to_tensor

MEAN_MIN = -7.24 # atanh(-0.999999)
MEAN_MAX = 7.24 # atanh(0.999999)
LOG_SIG_MAX = 2
LOG_SIG_MIN = -5


class Critic(nn.Module):
    def __init__(self, s_dim, a_dim=0, hidden_dims=[]):
        super().__init__()
        self.model = []
        last_dim = s_dim + a_dim
        for next_dim in hidden_dims:
            self.model += [nn.Linear(last_dim, next_dim), nn.ReLU()]
            last_dim = next_dim

        last_layer = nn.Linear(last_dim, 1)
        # init to avoid instability issue
        nn.init.uniform_(last_layer.weight, -3e-3, 3e-3)
        nn.init.uniform_(last_layer.bias, -3e-3, 3e-3)

        self.model += [last_layer]
        self.model = nn.Sequential(*self.model)

    def forward(self, s, a=None):
        s = to_tensor(s)
        s = s.view(s.shape[0], -1)

        if a is None:
            value = self.model(s)
        else:
            a = to_tensor(a)
            a = a.view(a.shape[0], -1)
            value = self.model(torch.cat([s, a], dim=1))
        return value

class NormalPolicy(nn.Module):
    def __init__(self, s_dim, a_dim, hidden_dims=[256, 128]):
        super().__init__()
        
        self.model = []
        last_dim = s_dim
        for next_dim in hidden_dims:
            self.model += [nn.Linear(last_dim, next_dim), nn.ReLU()]
            last_dim = next_dim
        self.model = nn.Sequential(*self.model)
        self.mu = nn.Linear(last_dim, a_dim)
        self.sigma = nn.Linear(last_dim, a_dim)
        # nn.init.uniform_(self.mu.weight, -1e-3, 1e-3)
        nn.init.uniform_(self.sigma.weight, -1e-3, 1e-3)
    
    def forward(self, s):
        s = to_tensor(s)
        s = s.view(s.shape[0], -1)
        logits = self.model(s)
        mu = self.mu(logits)
        sigma = torch.clamp(self.sigma(logits), LOG_SIG_MIN, LOG_SIG_MAX)
        sigma = torch.exp(sigma)

        dist = D.Independent(D.Normal(mu, sigma), 1)

        action = dist.rsample()
        return dist, action, action
    
    def deterministic_action(self, s):
        s = to_tensor(s).view(s.shape[0], -1)
        logits = self.model(s)
        mu = self.mu(logits)
        return mu
    
    @staticmethod
    def log_prob(dist, action, action2=None):
        log_p = dist.log_prob(action)
        return log_p, log_p
    
class MixtureNormalPolicy(nn.Module):
    def __init__(self, n_comp, s_dim, a_dim, temperature=1.0, hidden_dims=[256]):
        super().__init__()
        self.n_comp = n_comp
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.temperature = temperature

        self.model = []
        last_dim = s_dim
        for next_dim in hidden_dims:
            self.model += [nn.Linear(last_dim, next_dim), nn.ReLU()]
            last_dim = next_dim
        self.model = nn.Sequential(*self.model)
        self.mu = nn.Linear(last_dim, n_comp*a_dim)
        self.sigma = nn.Linear(last_dim, n_comp*a_dim)
        self.category = nn.Linear(last_dim, n_comp)

        nn.init.uniform_(self.sigma.weight, -1e-3, 1e-3)
        nn.init.uniform_(self.sigma.bias, -1e-3, 1e-3)

    def forward(self, s):
        s = to_tensor(s)
        s = s.view(s.shape[0], -1)
        h = self.model(s)
        mu = self.mu(h)
        sigma = torch.clamp(self.sigma(h), LOG_SIG_MIN, LOG_SIG_MAX)
        sigma = torch.exp(sigma)
        
        mu = mu.view(-1, self.n_comp, self.a_dim)
        sigma = sigma.view(-1, self.n_comp, self.a_dim)
        logits = self.category(h) / self.temperature
        
        comp_dist = D.Independent(D.Normal(mu, sigma), 1)
        mix_dist = D.Categorical(logits=logits)

        gmm = D.MixtureSameFamily(mix_dist, comp_dist)

        action = gmm.sample()

        return gmm, action, action

    @staticmethod
    def log_prob(dist, action, action2=None):
        log_p = dist.log_prob(action).unsqueeze(-1)
        return log_p, log_p


class TanhNormalPolicy(nn.Module):
    def __init__(self, s_dim, a_dim, hidden_dims=[256, 128]):
        super().__init__()
        
        self.model = []
        last_dim = s_dim
        for next_dim in hidden_dims:
            self.model += [nn.Linear(last_dim, next_dim), nn.ReLU()]
            last_dim = next_dim
        self.model = nn.Sequential(*self.model)
        self.mu = nn.Linear(last_dim, a_dim)
        self.sigma = nn.Linear(last_dim, a_dim)
        # nn.init.uniform_(self.mu.weight, -1e-3, 1e-3)
        nn.init.uniform_(self.sigma.weight, -1e-3, 1e-3)
    
    def forward(self, s):
        s = to_tensor(s)
        s = s.view(s.shape[0], -1)
        logits = self.model(s)
        mu = torch.clamp(self.mu(logits), MEAN_MIN, MEAN_MAX)
        sigma = torch.clamp(self.sigma(logits), LOG_SIG_MIN, LOG_SIG_MAX)
        sigma = torch.exp(sigma)

        dist = D.Independent(D.Normal(mu, sigma), 1)

        pretanh_action = dist.rsample()
        action = torch.tanh(pretanh_action)
        return dist, action, pretanh_action
    
    def deterministic_action(self, s):
        s = to_tensor(s).view(s.shape[0], -1)
        logits = self.model(s)
        mu = torch.clamp(self.mu(logits), MEAN_MIN, MEAN_MAX)
        return torch.tanh(mu)
    
    @staticmethod
    def log_prob(dist, action=None, pretanh_action=None):
        eps = 1e-6
        # x: pretanh action, y: action; y = tanh(x)
        x = torch.atanh(action.clamp(-1.0+eps, 1.0-eps)) if pretanh_action is None else pretanh_action
        y = torch.tanh(pretanh_action) if action is None else action

        log_px = dist.log_prob(x).unsqueeze(-1)
        log_py = log_px - torch.log(1.0 - y.pow(2) + eps).sum(-1,keepdim=True)
        return log_py, log_px


class TanhMixtureNormalPolicy(nn.Module):
    def __init__(self, n_comp, s_dim, a_dim, temperature=1.0, hidden_dims=[256]):
        super().__init__()
        self.n_comp = n_comp
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.temperature = temperature

        self.model = []
        last_dim = s_dim
        for next_dim in hidden_dims:
            self.model += [nn.Linear(last_dim, next_dim), nn.ReLU()]
            last_dim = next_dim
        self.model = nn.Sequential(*self.model)
        self.mu = nn.Linear(last_dim, n_comp*a_dim)
        self.sigma = nn.Linear(last_dim, n_comp*a_dim)
        self.category = nn.Linear(last_dim, n_comp)

        nn.init.uniform_(self.sigma.weight, -1e-3, 1e-3)
        nn.init.uniform_(self.sigma.bias, -1e-3, 1e-3)

    def forward(self, s):
        s = to_tensor(s)
        s = s.view(s.shape[0], -1)
        h = self.model(s)
        mu = self.mu(h)
        sigma = torch.clamp(self.sigma(h), LOG_SIG_MIN, LOG_SIG_MAX)
        sigma = torch.exp(sigma)
        
        mu = mu.view(-1, self.n_comp, self.a_dim)
        sigma = sigma.view(-1, self.n_comp, self.a_dim)
        logits = self.category(h) / self.temperature
        
        comp_dist = D.Independent(D.Normal(mu, sigma), 1)
        mix_dist = D.Categorical(logits=logits)

        gmm = D.MixtureSameFamily(mix_dist, comp_dist)

        pretanh_action = gmm.sample()
        action = torch.tanh(pretanh_action)

        return gmm, action, pretanh_action
    
    def get_action_std(self, s):
        s = to_tensor(s)
        s = s.view(s.shape[0], -1)
        h = self.model(s)
        sigma = torch.clamp(self.sigma(h), LOG_SIG_MIN, LOG_SIG_MAX)
        sigma = torch.exp(sigma)
        sigma = sigma.view(-1, self.n_comp, self.a_dim).mean(1)
        return sigma

    @staticmethod
    def log_prob(dist, action=None, pretanh_action=None):
        eps = 1e-6
        # x: pretanh action, y: action; y = tanh(x)
        x = torch.atanh(action.clamp(-1.0+eps, 1.0-eps)) if pretanh_action is None else pretanh_action
        y = torch.tanh(pretanh_action) if action is None else action

        log_px = dist.log_prob(x).unsqueeze(-1)
        log_py = log_px - torch.log(1.0 - y.pow(2) + eps).sum(-1,keepdim=True)

        return log_py, log_px