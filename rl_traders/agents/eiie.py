import os
import shutil
from tensorboardX import SummaryWriter

import numpy as np
import torch
from torch_utils.optim import get_optimizer

from .core import BaseAgent
from ..processors import Processor
from ..configs import EIIE_CONFIG, LR_SPEC, SCHEDULER_SPEC
from ..models import EIIEFeedForwarad
from ..memories import PortfolioVectorMemory


class EIIEAgent(BaseAgent):
    def __init__(self, action_spec, state_spec=None,
                 processor=Processor(),
                 is_debug=False, model_params=EIIE_CONFIG,
                 lr_spec=LR_SPEC, scheduler_spec=SCHEDULER_SPEC,
                 memory_limit=100000, window_length=50,
                 batch_size=50, beta=5.0e-5, device='cpu', log_dir='./logs',
                 log_hist_freq=None, *args, **kwargs):
        super(EIIEAgent, self).__init__(action_spec=action_spec,
                                        state_spec=state_spec,
                                        processor=processor,
                                        is_debug=is_debug)
        cash_bias = torch.zeros((1,), dtype=torch.float)
        self.model = EIIEFeedForwarad(model_params, cash_bias)
        self.memory = PortfolioVectorMemory(memory_limit, window_length,
                                            beta=beta, *args, **kwargs)
        self.optimizer, self.scheduler = get_optimizer(self.model.parameters(),
                                                       lr_spec, scheduler_spec)
        self.batch_size = batch_size
        self.device = device
        # Delete old logs if any
        if os.path.isdir(log_dir):
            print('Delete old tensorboard log')
            shutil.rmtree(log_dir)
        self.writer = SummaryWriter(log_dir)
        self.log_hist_freq = log_hist_freq
        self.actions_record = []

    def _observe(self, observation, action, reward, terminal, info, is_store):
        self.memory.append(observation, action, reward,
                           terminal, info, is_store=is_store)

    def fit(self, n_epochs, step, training=True, *args, **kwargs):
        self.model.train()
        losses = []
        for i in range(n_epochs):
            experiences = self.memory.sample(batch_size=self.batch_size)
            states = np.stack([exp.state for exp in experiences])
            # Shape = (n_batch, n_currencies)
            rewards = np.stack([exp.reward for exp in experiences])
            terminals = np.stack([exp.terminal for exp in experiences])
            indices = np.stack([exp.index for exp in experiences])
            prev_as = np.stack([exp.previous_action for exp in experiences])
            # Training with PyTorch
            self.optimizer.zero_grad()
            states_tensor = torch.tensor(states, dtype=torch.float,
                                         device=self.device)
            prev_as_tensor = torch.tensor(prev_as, dtype=torch.float,
                                          device=self.device)
            actions = self.model(states_tensor, prev_as_tensor)
            # Define objective function
            rewards_tensor = torch.tensor(rewards, dtype=torch.float,
                                          device=self.device)
            rewards_sum = torch.sum(rewards_tensor * actions, dim=1)
            rewards_log = torch.log(1. + rewards_sum)
            rewards_log_mean = torch.mean(rewards_log, dim=0)
            loss = -rewards_log_mean
            loss.backward()
            self.optimizer.step()
            self.scheduler.step(loss.item())
            losses.append(loss.item())
            # Update Portfolio Vector
            actions_np = actions.detach().numpy()
            for idx, action in zip(indices, actions_np):
                self.update_portfolio_vector(action, idx)
            if training:
                self.record(loss.item(), actions_np, i, training=True)
        self.record(np.mean(losses), actions_np, step, training=False)

    def record(self, loss, actions, idx, training=True):
        lr = self.optimizer.param_groups[0]['lr']
        if training:
            name = 'training'
        else:
            name = 'test'
        self.writer.add_scalar(f'data/{name}_loss', loss, idx)
        self.writer.add_scalar(f'data/{name}_lr', lr, idx)
        action_dim = actions.shape[1]
        if self.log_hist_freq is not None:
            self.actions_record += list(actions)
        if self.log_hist_freq and\
                len(self.actions_record) >= self.log_hist_freq * len(actions):
            actions_record = np.array(self.actions_record)
            for ai in range(action_dim):
                self.writer.add_histogram(f'data/{name}_action_{ai}',
                                          actions_record[:, ai], idx,
                                          bins='auto')
            self.actions_record = []

    def _predict(self, observation, prev_action, *args, **kwargs):
        self.model.eval()
        state = self.get_recent_state(observation)
        state = torch.tensor(state, dtype=torch.float, device=self.device)
        prev_action = torch.tensor(prev_action, dtype=torch.float,
                                   device=self.device)
        states = state[None, :]
        prev_actions = prev_action[None, :]
        action = self.model(states, prev_actions).detach().numpy()[0]
        return action

    def generate_action(self, alphas=None, size=None):
        if alphas is None:
            alphas = np.ones(self.action_shape)
        return np.random.dirichlet(alphas, size=size)

    def get_recent_state(self, current_observation):
        return self.memory.get_recent_state(current_observation)

    def update_portfolio_vector(self, action, idx):
        """Update action memory for Portfolio Vector Memory"""
        self.memory.update_portfolio_vector(action, idx)
