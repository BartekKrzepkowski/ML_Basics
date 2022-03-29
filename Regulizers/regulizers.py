import torch


class Dropout(torch.nn.Module):
    
    def __init__(self, p=0.5):
        super(Dropout, self).__init__()
        assert 0 <= p and p < 1, 'p out of range'
        self.bernoulli = torch.distributions.bernoulli.Bernoulli(probs=1-p)
        self.multipler = 1. / (1. - p)
        self.p = p
        
    def forward(self, x):
        if self.training:
#             mask = self.bernoulli.sample(x.size())
            mask = torch.rand_like(x) > self.p
            x = x * mask * self.multipler
        return x
    
class BatchNorm1d(torch.nn.Module):
    def __init__(self, num_features, eps=1e-05, momentum=0.1):
        super(BatchNorm1d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.m = momentum
        self.beta = torch.nn.Parameter(torch.zeros(1,num_features), requires_grad=True)
        self.gamma = torch.nn.Parameter(torch.ones(1,num_features), requires_grad=True)
        self.register_buffer(name='mu', tensor=torch.zeros(1,num_features))
        self.register_buffer(name='sigma', tensor=torch.ones(1,num_features))

    def forward(self, x):
        if self.training:
            mean = x.mean(dim=0, keepdim=True)
            var = x.var(dim=0, keepdim=True, unbiased=False)
            self.mu = (1 - self.m) * self.mu + self.m * mean
            self.sigma = (1 - self.m) * self.sigma + self.m * var
        else:
            mean = self.mu
            var = self.sigma
            
        z = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * z + self.beta

class BatchNorm2d(torch.nn.Module):
    def __init__(self, n_feat, eps=1e-5, momentum=0.1):
        super(BatchNorm2d, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.eps = eps
        self.m = momentum
        self.beta = torch.nn.Parameter(torch.zeros(n_feat)).to(self.device)
        self.gamma = torch.nn.Parameter(torch.ones(n_feat)).to(self.device)
        self.register_buffer('mu', torch.zeros(n_feat).to(self.device))
        self.register_buffer('sigma2', torch.ones(n_feat).to(self.device))

    def forward(self, x):
        if self.training:
            n = x.numel() / x.size(1)
            mean = x.mean(dim=[0,2,3])
            var = x.var(dim=[0,2,3], unbiased=False)
            with torch.no_grad():
                self.mu = (1-self.m) * self.mu + self.m * mean
                self.sigma2 = (1-self.m) * self.sigma2 * n / (n-1) + self.m * var
        else:
            mean = self.mu
            var = self.sigma2

        z = (x - mean[None, :, None, None]) / torch.sqrt(var[None, :, None, None] + self.eps)
        return self.gamma[None, :, None, None] * z + self.beta[None, :, None, None]
    
    

