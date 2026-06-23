import numpy as np
from collections import namedtuple
import os
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')





np.random.seed(0)
torch.manual_seed(0)





def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)



class ActorNet(nn.Module):
    def __init__(self, state_dim, action_dim, h_dim = 200):
        super(ActorNet, self).__init__()
        self.state_dim = state_dim 
        self.action_dim = action_dim 
        self.h_dim = h_dim
        self.fc = nn.Sequential(nn.Linear(self.state_dim, self.h_dim),
                                nn.ReLU6(),
                                nn.Linear(self.h_dim, self.h_dim),
                                nn.ReLU6(),
                                nn.Linear(self.h_dim, self.h_dim))
        
        self.out = nn.Linear(self.h_dim, self.action_dim)

        self.fc.apply(init_weights)
        self.out.apply(init_weights)

    def forward(self, s):
        x = F.relu(self.fc(s))
        x = x/0.1
        output = F.softmax(self.out(x), dim=1)
        output = torch.clamp(output, min=1e-10, max=1-1e-10)
        return output


class CriticNet(nn.Module):
    def __init__(self, state_dim, action_dim, h_dim = 200):
        super(CriticNet, self).__init__()
        self.state_dim = state_dim 
        self.action_dim = action_dim 
        self.h_dim = h_dim 
        self.fc = nn.Sequential(nn.Linear(self.state_dim + 1, self.h_dim),
                                nn.ReLU6(),
                                nn.Linear(self.h_dim, self.h_dim),
                                nn.ReLU6(),
                                nn.Linear(self.h_dim, self.h_dim))
        self.v_head = nn.Linear(self.h_dim, 1)
        self.fc.apply(init_weights)
        self.v_head.apply(init_weights)

    def forward(self, sa):
        x = F.relu(self.fc(sa))
        state_value = self.v_head(x)
        return state_value


class Memory():

    data_pointer = 0
    isfull = False

    def __init__(self, capacity, batch_size):
        self.memory = np.empty(capacity, dtype=object)
        self.capacity = capacity
        self.batch_size = batch_size

    def update(self, transition):
        index = self.data_pointer % self.capacity
        self.memory[index] = transition
        self.data_pointer += 1
        if self.data_pointer > self.capacity:
            self.isfull = True


    def sample(self):
        if self.isfull: 
            sample_index = np.random.choice(self.capacity, self.batch_size)
        else: 
            sample_index = np.random.choice(self.data_pointer, self.batch_size)

        return self.memory[sample_index]


class Agent():

    max_grad_norm = 0.5

    def __init__(self, state_dim, action_dim, h_dim = 200, epsilon = 0.9, LR_A = 1e-4, LR_C = 1e-4, gamma = 0.9, replace_iter_a = 1100, replace_iter_c = 1000, memory_capacity = 20000, batch_size = 64):
        self.training_step = 0
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.epsilon = epsilon
        self.h_dim = h_dim
        self.eval_cnet, self.target_cnet = CriticNet(self.state_dim, self.action_dim, self.h_dim).float(), CriticNet(self.state_dim, self.action_dim, self.h_dim).float()
        self.eval_anet, self.target_anet = ActorNet(self.state_dim, self.action_dim, self.h_dim).float(), ActorNet(self.state_dim, self.action_dim, self.h_dim).float()
        self.memory = Memory(memory_capacity, batch_size)
        self.gamma = gamma
        self.replace_iter_a = replace_iter_a 
        self.replace_iter_c = replace_iter_c
        self.lr_a = LR_A 
        self.lr_c = LR_C
        self.optimizer_c = optim.RMSprop(self.eval_cnet.parameters(), lr=self.lr_c)
        self.optimizer_a = optim.RMSprop(self.eval_anet.parameters(), lr=self.lr_a)
        self.cast = lambda x: x

    def select_action(self, state):
        
        state = torch.from_numpy(state).float().unsqueeze(0)
        action_probs = self.eval_anet(state)
        action_probs = action_probs.detach().squeeze()
        calc_action = torch.multinomial(action_probs, 1)
        calc_action = calc_action.numpy()[0]
        return calc_action

    def store_transition(self, transition):
        self.memory.update(transition)

    def update(self):
        self.training_step += 1

        transitions = self.memory.sample()

        s = np.array([t.s for t in transitions]) 

        s = torch.tensor(s, dtype=torch.float)
        a = torch.tensor([t.a for t in transitions], dtype=torch.float)
        r = torch.tensor([t.r for t in transitions], dtype=torch.float).view(-1, 1)
        
        s_ = np.array([t.s_ for t in transitions]) 

        s_ = torch.tensor(s_, dtype=torch.float)

        with torch.no_grad():
            action_probs = self.target_anet(s_)
            action_probs = action_probs.detach().squeeze()
            action = torch.multinomial(action_probs, 1)
            q_target = r + self.gamma * self.target_cnet(torch.cat((s_, action), dim=1))
        
        a = torch.unsqueeze(a, dim=-1)
        q_eval = self.eval_cnet(torch.cat((s, a), dim=1))

        self.optimizer_c.zero_grad()
        c_loss = F.smooth_l1_loss(q_eval, q_target)
        c_loss.backward()
        self.optimizer_c.step()

        self.optimizer_a.zero_grad()
        action_probs = self.eval_anet(s)
        action_probs = action_probs.squeeze()
        action = torch.multinomial(action_probs, 1)
        log_policy = torch.log(action_probs + 1e-6)
        a_loss = -(log_policy * self.eval_cnet(torch.cat((s, action), dim=1))).mean()
        a_loss.backward()
        self.optimizer_a.step()

        if self.training_step % self.replace_iter_c == 0:
            self.target_cnet.load_state_dict(self.eval_cnet.state_dict())
        if self.training_step % self.replace_iter_a == 0:
            self.target_anet.load_state_dict(self.eval_anet.state_dict())


        return q_eval.mean().item()

    @property
    def networks(self):
        return [self.eval_anet, self.target_anet, self.eval_cnet, self.target_cnet]

    def prepare_update(self):
        if DEVICE == 'cuda':
            self.cast = lambda x: x.cuda()
        else:
            self.cast = lambda x: x.cpu()

        for network in self.networks:
            network = self.cast(network)
            network.train()

    #def prepare_eval(self):
    #    self.cast = lambda x: x.cpu()
    #    for network in self.networks:
    #        network = self.cast(network)
    #        network.eval()

    def save_param(self, path='./param/ac.pkl'):
        print("Saving")
        save_dict = {'networks': [network.state_dict() for network in self.networks]}
        torch.save(save_dict, path)

    def load_param(self, path='./param/ac.pkl'):
        print("loading")
        save_dict = torch.load(path)
        for network, params in zip(self.networks, save_dict['networks']):
            network.load_state_dict(params)


