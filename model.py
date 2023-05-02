import math

import torch
import torch.nn as nn
import torch.distributions as dist
import torch.nn.functional as F


class VerySimple(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.l1 = nn.Linear(input_dim, 1024)
        self.l2 = nn.Linear(1024, 256)
        self.l3 = nn.Linear(256, output_dim)

    def forward(self, input):
        # input (n, in)
        n, input_dim = input.shape

        # (n, in) ->  (n, out)

        x = F.relu(self.l1(input))
        x = F.relu(self.l2(x))
        x = self.l3(x)

        # (n, out)
        return x


class Simple(nn.Module):

    def __init__(self, input_dim, output_dim, num_task, mode='shared', l1_dim=1024, l2_dim=256, var_size=None,
                 num_f=None,
                 temperature=0.01, bias=True):
        super().__init__()

        if var_size is None:
            var_size = [2048, 64, 1]

        self.mode = mode
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_task = num_task

        if mode == 'shared':
            self.l1 = LinearShared(input_dim, l1_dim, num_task, bias=bias)
            self.l2 = LinearShared(l1_dim, l2_dim, num_task, bias=bias)
            #self.l3 = LinearShared(l2_dim, output_dim, num_task, bias=bias)
            self.l3 = LinearIndependent(l2_dim, output_dim, num_task, bias=bias)

        elif mode == 'independent':
            self.l1 = LinearIndependent(input_dim, l1_dim, num_task, bias=bias)
            self.l2 = LinearIndependent(l1_dim, l2_dim, num_task, bias=bias)
            self.l3 = LinearIndependent(l2_dim, output_dim, num_task, bias=bias)

        elif mode == 'rps':
            self.l1 = LinearRPS(input_dim, l1_dim, num_task, num_f=num_f, temperature=temperature, bias=bias)
            self.l2 = LinearRPS(l1_dim, l2_dim, num_task, num_f=num_f, temperature=temperature, bias=bias)
            self.l3 = LinearRPS(l2_dim, output_dim, num_task, num_f=num_f, temperature=temperature, bias=bias)

        elif mode == 'pps':
            self.l1 = LinearPps(input_dim, l1_dim, num_task, num_cat=num_f, var_size=var_size[0], bias=bias,
                                temperature=temperature,
                                )
            self.l2 = LinearPps(l1_dim, l2_dim, num_task, num_cat=num_f, var_size=var_size[1], temperature=temperature,
                                bias=bias)
            self.l3 = LinearPps(l2_dim, output_dim, num_task, num_cat=num_f, var_size=var_size[2],
                                temperature=temperature,
                                bias=bias)

        else:
            raise Exception('undefined mode!')

    def forward(self, input):

        # input (n, in)

        # (n, in) -> (n, 1, in) -> (n, num_task, in)
        n, input_dim = input.shape
        x = input.unsqueeze(1).expand(n, self.num_task, input_dim)

        losses = 0

        if self.mode == 'pps':
            x, loss1 = self.l1(x)
            x = F.relu(x)
            x, loss2 = self.l2(x)
            x = F.relu(x)
            x, loss3 = self.l3(x)
            losses = loss1 + loss2 + loss3

        else:
            x = F.relu(self.l1(x))
            x = F.relu(self.l2(x))
            x = self.l3(x)

        # (n, num_task, out)
        return x, losses


class LinearShared(nn.Module):
    def __init__(self, input_dim, output_dim, num_task, bias=True):
        super(LinearShared, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_task = num_task

        self.weight = nn.Parameter(torch.empty((output_dim, input_dim)))
        if bias:
            self.bias = nn.Parameter(torch.empty(output_dim))
        else:
            self.register_parameter('bias', None)

        self.init_params()

    def init_params(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def forward(self, input):

        # input (n, num_task, in)

        # (1, 1, out, in) @ (n, num_task, in, 1) -> (n, num_task, out, 1) -> (n, num_task, out)
        # x = torch.matmul(self.weight.unsqueeze(0).unsqueeze(0), input.unsqueeze(-1)).squeeze(-1)
        # x = torch.tensordot(input, self.weight, dims=([2], [1]))

        # (out, in) @ (n, num_task, in) -> (n, num_task, out)
        #print(input.shape, self.weight.shape)
        x = torch.einsum('nti,ji->ntj', input, self.weight)

        if self.bias is not None:
            # (n, num_task, out) + (1, 1, out) -> (n, num_task, out)
            x = x + self.bias

        # (n, num_task, out)
        return x


class LinearIndependent(nn.Module):
    def __init__(self, input_dim, output_dim, num_task, bias=True):
        super(LinearIndependent, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_task = num_task

        self.weight = nn.Parameter(torch.empty((self.num_task, output_dim, input_dim)))
        if bias:
            self.bias = nn.Parameter(torch.empty((self.num_task, output_dim)))
        else:
            self.register_parameter('bias', None)

        self.init_params()

    def init_params(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def forward(self, input):

        # input (n, num_task, in)

        # (1, num_task, out, in) @ (n, num_task, in, 1) -> (n, num_task, out, 1) -> (n, num_task, out)
        # x = torch.matmul(self.weight.unsqueeze(0), input.unsqueeze(-1)).squeeze(-1)

        # (n, num_task, in)  (num_task, out, in) -> (n, num_task, out)
        x = torch.einsum('nti,tji->ntj', input, self.weight)

        if self.bias is not None:
            # (n, num_task, out) + (1, num_task, out) -> (n, num_task, out)
            x = x + self.bias.unsqueeze(0)

        # (n, num_task, out)
        return x


class LinearRPS(nn.Module):

    def __init__(self, input_dim, output_dim, num_task, num_f=None, bias=True, temperature=0.01):
        super(LinearRPS, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_task = num_task
        self.num_f = num_f
        if self.num_f is None:
            self.num_f = self.num_task

        self.rel = nn.Parameter(torch.empty(self.num_task, self.num_f))
        self.weight = nn.Parameter(torch.empty((output_dim, input_dim, self.num_f)))
        if bias:
            self.bias = nn.Parameter(torch.empty((output_dim, self.num_f)))
        else:
            self.register_parameter('bias', None)

        self.temperature = temperature
        self.eps = torch.finfo(torch.float32).eps
        self.init_params()
        # self.dist = dist.RelaxedOneHotCategorical(temperature, probs=self.rel)

    def init_params(self):

        nn.init.constant_(self.rel, 1 / self.num_f)

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def sample_weights(self, shape):

        uniforms = torch.rand(shape + (self.num_task, self.num_f), device='cuda')
        uniforms = torch.clamp(uniforms, min=self.eps, max=1 - self.eps)

        gumbels = -((-(uniforms.log())).log())
        scores = (self.rel + gumbels) / self.temperature
        return (scores - scores.logsumexp(dim=-1, keepdim=True)).exp()

    def forward(self, input):

        # input (n, num_task, in)

        # (out, in, num_task, num_f)

        # d1 = dist.RelaxedOneHotCategorical(self.temperature, logits=self.rel)
        # d2 = dist.RelaxedOneHotCategorical(self.temperature, logits=self.rel)

        # s_weight = d1.rsample((self.output_dim, self.input_dim))
        s_weight = self.sample_weights((self.output_dim, self.input_dim))

        # (out, in, 1, num_f) * (out, in, num_task, num_f)
        # -> (out, in, num_task) -> (num_task, out, in)=
        # weight_sampled = torch.sum(self.weight.unsqueeze(2) * s_weight, dim=-1).permute(2, 0, 1)

        # (out, in, num_f) * (out, in, num_task, num_f) -> (out, in, num_task)
        weight_sampled = torch.einsum('jif,jitf->jit', self.weight, s_weight)

        # (1, num_task, out, in) @ (n, num_task, in, 1) -> (n, num_task, out, 1) -> (n, num_task, out)
        # x = torch.matmul(weight_sampled.unsqueeze(0), input.unsqueeze(-1)).squeeze(-1)

        # (n, num_task, in)  (num_task, out, in) -> (n, num_task, out)
        x = torch.einsum('nti,jit->ntj', input, weight_sampled)

        if self.bias is not None:
            # (out, num_task, num_f)
            # s_bias = d2.rsample([self.output_dim])
            # s_bias = torch.randn((self.output_dim, self.num_task, self.num_f), device='cuda')
            s_bias = self.sample_weights((self.output_dim,))

            # (out, 1, num_f) * (out, num_task, num_f)
            # -> (out, num_task) -> (num_task, out)
            # bias_sampled = torch.sum(self.bias.unsqueeze(1) * s_bias, dim=-1).permute(1, 0)

            bias_sampled = torch.einsum('jf,jtf->tj', self.bias, s_bias)

            # (n, num_task, out) + (1, num_task, out) -> (n, num_task, out)
            x = x + bias_sampled.unsqueeze(0)

        # (n, num_task, out)
        return x


class LinearPps(nn.Module):

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 num_tasks: int,
                 num_cat: int,
                 var_size: int,
                 bias: bool = True,
                 temperature: float = 0.1):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.num_features = in_features * out_features

        self.num_tasks = num_tasks

        # define other hyperparameters required:
        self.num_cat = num_cat  # number of categories (default to number of tasks)
        self.var_size = var_size  # size of each variable (vector)
        self.num_vars = self.num_features // self.var_size  # number of variables

        # Ensure num_features is divisible by var_size
        assert self.num_features % self.var_size == 0

        self.temperature = nn.Parameter(torch.tensor(temperature))
        self.dist_probs = nn.Parameter(torch.Tensor(self.num_tasks, self.num_vars, self.num_cat))

        # the weights in pps lienar layers are all flattened, and then reshaped later if needed.
        self.weight = nn.Parameter(torch.Tensor(self.num_vars, self.var_size, self.num_cat))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.num_vars, self.num_cat, self.out_features))
        else:
            self.register_parameter('bias', None)

        # used for permuting the weight matrix in a deterministic way. Later used for scatter operation
        self.perm_pattern = torch.randperm(self.num_features).cuda()
        self.eps = torch.finfo(torch.float32).eps

        self.init_params()
        # self.dist = dist.RelaxedOneHotCategorical(temperature, probs=self.rel)

        print(
            f'Number of variables: {self.num_vars}, variable size: {self.var_size}, number of categories: {self.num_cat}')

    def init_params(self):

        nn.init.constant_(self.dist_probs, 1 / self.num_cat)

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def sample_weights(self):

        uniforms = torch.rand_like(self.dist_probs)  # torch.rand(shape + list(self.dist_probs.shape), device='cuda')
        uniforms = torch.clamp(uniforms, min=self.eps, max=1 - self.eps)

        gumbels = -((-(uniforms.log())).log())
        scores = (self.dist_probs + gumbels) / self.temperature
        return (scores - scores.logsumexp(dim=-1, keepdim=True)).exp()

    def forward(self, input: torch.Tensor) -> (torch.Tensor, torch.Tensor):

        # Inputs:
        # input (N, T, I)
        num_tasks = input.shape[1]

        assert num_tasks == self.num_tasks

        # Sample from relaxed categorical distribution
        # dist = torch.distributions.relaxed_categorical.RelaxedOneHotCategorical(
        #     self.temperature, logits=self.dist_probs, validate_args=False)

        # Generate one-hot vectors
        # (T, num_vars, C)
        vars_onehot = self.sample_weights()  # dist.rsample()

        # Expand and reshape one-hot vectors
        # (T, num_vars, C) -> (T, num_vars, var_size, C)
        vars_onehot_exp = vars_onehot.unsqueeze(-2).expand(-1, -1, self.var_size, -1)

        # print(vars_onehot_exp.shape)

        # (T, num_vars, var_size, C) * (1, num_vars, var_size, C) -> (T, num_vars, var_size, C)
        sampled_weight = vars_onehot_exp * self.weight.unsqueeze(0)

        # (T, num_vars, var_size, C) -> (T, num_features)
        sampled_weight = torch.sum(sampled_weight, dim=-1).view(num_tasks, -1)

        # sampled_weight = torch.einsum('tvsc,vsc->tvs', vars_onehot_exp, self.weight).view(num_tasks, -1)

        # Permute the sampled_weight
        # (T, num_features) -> (T, num_features)
        sampled_weight = torch.index_select(sampled_weight, dim=-1, index=self.perm_pattern)

        # now we need to reshape the sampled_weight
        # (T, num_features) -> (T, in_features, out_features)
        sampled_weight = sampled_weight.view(num_tasks, self.in_features, self.out_features)

        # (N, T, in_features) @ (T, in_features, out_features) -> (N, T, out_features)
        output = torch.einsum('nti,tio->nto', input, sampled_weight)

        # Sample bias if bias is enabled
        # bias is linear, so we just add all sampled variables together
        if self.bias is not None:
            # (T, num_vars, C, 1) * (1, num_vars, C, out_feature) -> (T, num_vars, C, out_features)
            sampled_bias = vars_onehot.unsqueeze(-1) * self.bias.unsqueeze(0)
            # (T, num_vars, C, out_features)  -> (T, out_features)
            sampled_bias = torch.sum(sampled_bias, dim=(1, 2))

            output = output + sampled_bias

        # make weights orthogonal. calculate loss
        # (num_vars, C, var_size) * (num_vars, var_size, C) *  -> (num_vars, C, C)
        dp = torch.bmm(self.weight.transpose(1, 2), self.weight)
        # print(torch.diag_embed(dp, dim1=-2, dim2=-1).shape)

        dp = dp - torch.diag_embed(torch.diagonal(dp, dim1=-2, dim2=-1))
        loss = torch.sum(dp ** 2)

        # (N, T, out_features)
        return output, loss
