import torch

temperature = torch.tensor(1e-3)
probs = torch.tensor([[0.0, 0.0, 0.1, 0.9], [0.9, 0.1, 0.0, 0.0]])

dist = torch.distributions.relaxed_categorical.RelaxedOneHotCategorical(temperature,
                                                                        probs=probs)


print(dist.rsample((10, )))
