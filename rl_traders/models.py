import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_utils.models import FeedForward


class EIIEFeedForwarad(nn.Module):
    def __init__(self, model_params, cash_bias):
        super(EIIEFeedForwarad, self).__init__()
        self.lower_model = FeedForward(model_params['lower_params'])
        self.upper_model = FeedForward(model_params['upper_params'])
        self.cash_bias = nn.Parameter(cash_bias)

    def forward(self, states, prev_actions):
        n_batch = states.shape[0]
        outputs = self.lower_model(states)
        # We do not use cash actions as input, prev_actions[:, 0]
        prev_actions = prev_actions[:,  None, None, 1:]
        # Concatenation with channel dimension
        outputs = torch.cat((outputs, prev_actions), dim=1)
        prev_softmax = self.upper_model(outputs)
        _cash_bias = self.cash_bias.repeat(n_batch, 1)
        prev_softmax = torch.cat((_cash_bias, prev_softmax), dim=-1)
        actions = F.softmax(prev_softmax, dim=-1)
        return actions

    def predict(self, state, prev_action):
        states = state[None, :]
        prev_actions = prev_action[None, :]
        return self.forward(states, prev_actions)[0].detach().numpy()
