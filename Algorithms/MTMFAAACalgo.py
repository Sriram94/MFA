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





class MTMFAAAC(nn.Module): 
    def __init__(self, state_dim, action_dim, action_dim2, num_elements, batch_size, observation_space, action_space, hidden_dim=200, norm_in=False, attend_heads=4):
        super(MTMFAAAC, self).__init__()
        assert (hidden_dim % attend_heads) == 0
        
        idim = state_dim + action_dim
        odim = action_dim
        odim2 = action_dim2
        self.observation_space = observation_space
        self.action_space = action_space
        
        self.state_dim = state_dim 
        self.batch_size = batch_size
        self.action_dim = action_dim
        self.num_elements = num_elements 
        
        self.attend_heads = attend_heads

        self.critic_encoders = nn.ModuleList()
        

        self.critics = nn.Sequential()
        self.critics.add_module('critic_fc1', nn.Linear(4 * hidden_dim,
                                                  hidden_dim))
        self.critics.add_module('critic_nl', nn.LeakyReLU())
        self.critics.add_module('critic_fc2', nn.Linear(hidden_dim, odim))

        self.state_encoders = nn.Sequential()
        
        if norm_in:
            self.state_encoders.add_module('s_enc_bn', nn.BatchNorm1d(
                                        state_dim, affine=False))
        self.state_encoders.add_module('s_enc_fc1', nn.Linear(state_dim,
                                                        hidden_dim))
        self.state_encoders.add_module('s_enc_nl', nn.LeakyReLU())

        
        self.meanfield_encoders = nn.Sequential()

        if norm_in:
            self.meanfield_encoders.add_module('s_enc_meanfield_bn', nn.BatchNorm1d(
                                        odim, affine=False))

        self.meanfield_encoders.add_module('s_enc_meanfield_fc1', nn.Linear(odim,
                                                        hidden_dim))
        self.meanfield_encoders.add_module('s_enc_meanfield_nl', nn.LeakyReLU())

        
        self.meanfield2_encoders = nn.Sequential()

        if norm_in:
            self.meanfield2_encoders.add_module('s_enc_meanfield2_bn', nn.BatchNorm1d(
                                        odim2, affine=False))

        self.meanfield2_encoders.add_module('s_enc_meanfield2_fc1', nn.Linear(odim2,
                                                        hidden_dim))
        self.meanfield2_encoders.add_module('s_enc_meanfield2_nl', nn.LeakyReLU())


        for j in range(num_elements):
            state_dim = self.observation_space[j]
            action_dim = self.action_space[j]
            idim = state_dim + action_dim
            encoder = nn.Sequential()
            if norm_in:
                encoder.add_module('enc_bn', nn.BatchNorm1d(idim,
                                                            affine=False))
            encoder.add_module('enc_fc1', nn.Linear(idim, hidden_dim))
            encoder.add_module('enc_nl', nn.LeakyReLU())
            self.critic_encoders.append(encoder)



        attend_dim = hidden_dim // attend_heads
        self.key_extractors = nn.ModuleList()
        self.selector_extractors = nn.ModuleList()
        self.value_extractors = nn.ModuleList()
        for i in range(attend_heads):
            self.key_extractors.append(nn.Linear(hidden_dim, attend_dim, bias=False))
            self.selector_extractors.append(nn.Linear(hidden_dim, attend_dim, bias=False))
            self.value_extractors.append(nn.Sequential(nn.Linear(hidden_dim,
                                                                attend_dim),
                                                       nn.LeakyReLU()))

        self.shared_modules = [self.key_extractors, self.selector_extractors,
                               self.value_extractors, self.critic_encoders]


    def shared_parameters(self):
        """
        Parameters shared across agents and reward heads
        """
        return chain(*[m.parameters() for m in self.shared_modules])

    def scale_shared_grads(self):
        """
        Scale gradients for parameters that are shared since they accumulate
        gradients from the critic loss function multiple times
        """
        for p in self.shared_parameters():
            p.grad.data.mul_(1. / self.nagents)

    def forward(self, inps, state_agent, meanfield, meanfield2, return_q=False, return_all_q=True,
                regularize=False, return_attend=False, logger=None, niter=0):
        """
        Inputs:
            inps (list of PyTorch Matrices): Inputs to each agents' encoder
                                             (batch of obs + ac)
            agents (int): indices of agents to return Q for
            return_q (bool): return Q-value
            return_all_q (bool): return Q-value for all actions
            regularize (bool): returns values to add to loss function for
                               regularization
            return_attend (bool): return attention weights per agent
            logger (TensorboardX SummaryWriter): If passed in, important values
                                                 are logged
        """
        agents = range(1)
        states = [s for s, a in inps]
        actions = [a for s, a in inps]

        list_state = [[] for i in range(len(states[0]))] 
        list_action = [[] for i in range(len(states[0]))] 

        for i in range(len(states[0])):
            for j in range(len(states)):
                list_state[i].append(states[j][i])
                list_action[i].append(actions[j][i])



        

        for i in range(len(list_state)):
            b = torch.Tensor(self.batch_size, self.observation_space[i])
            list_state[i] = torch.cat(list_state[i], out=b)
            b = torch.Tensor(self.batch_size, self.action_space[i])
            list_action[i] = torch.cat(list_action[i], out=b)

        for i in range(len(list_action)):
            list_action[i] = list_action[i].reshape(self.batch_size, self.action_space[i])


        new_inps = [torch.cat((s.reshape(self.batch_size, i), a.reshape(self.batch_size, j)), dim=1) for s, a, i, j in zip(list_state, list_action, self.observation_space, self.action_space)]
        
        sa_encodings = [encoder(inp) for encoder, inp in zip(self.critic_encoders, new_inps)]
        s_encodings = [self.state_encoders(state_agent) for a_i in agents]
        meanfield_encodings = [self.meanfield_encoders(meanfield) for a_i in agents]
        meanfield2_encodings = [self.meanfield2_encoders(meanfield2) for a_i in agents]
        all_head_keys = [[k_ext(enc) for enc in sa_encodings] for k_ext in self.key_extractors]
        all_head_values = [[v_ext(enc) for enc in sa_encodings] for v_ext in self.value_extractors]
        all_head_selectors = [[sel_ext(enc) for i, enc in enumerate(s_encodings) if i in agents]
                              for sel_ext in self.selector_extractors]

        other_all_values = [[] for _ in range(len(agents))]
        all_attend_logits = [[] for _ in range(len(agents))]
        all_attend_probs = [[] for _ in range(len(agents))]
        for curr_head_keys, curr_head_values, curr_head_selectors in zip(
                all_head_keys, all_head_values, all_head_selectors):
            for i, a_i, selector in zip(range(len(agents)), agents, curr_head_selectors):
                keys = [k for j, k in enumerate(curr_head_keys) if j != a_i]
                values = [v for j, v in enumerate(curr_head_values) if j != a_i]
                attend_logits = torch.matmul(selector.view(selector.shape[0], 1, -1),
                                             torch.stack(keys).permute(1, 2, 0))
                scaled_attend_logits = attend_logits / np.sqrt(keys[0].shape[1])
                attend_weights = F.softmax(scaled_attend_logits, dim=2)
                other_values = (torch.stack(values).permute(1, 2, 0) *
                                attend_weights).sum(dim=2)
                other_all_values[i].append(other_values)
                all_attend_logits[i].append(attend_logits)
                all_attend_probs[i].append(attend_weights)
        
        all_rets = []
        for i, a_i in enumerate(agents):
            head_entropies = [(-((probs + 1e-8).log() * probs).squeeze().sum(1)
                               .mean()) for probs in all_attend_probs[i]]
            agent_rets = []
            critic_in = torch.cat((s_encodings[i], meanfield_encodings[i], meanfield2_encodings[i], *other_all_values[i]), dim=1)
            all_q = self.critics(critic_in)
            
            if return_q:
                agent_rets.append(q)
            if return_all_q:
                agent_rets.append(all_q)
            if regularize:
                attend_mag_reg = 1e-3 * sum((logit**2).mean() for logit in
                                            all_attend_logits[i])
                regs = (attend_mag_reg,)
                agent_rets.append(regs)
            if return_attend:
                agent_rets.append(np.array(all_attend_probs[i]))
            if logger is not None:
                logger.add_scalars('agent%i/attention' % a_i,
                                   dict(('head%i_entropy' % h_i, ent) for h_i, ent
                                        in enumerate(head_entropies)),
                                   niter)
            if len(agent_rets) == 1:
                all_rets.append(agent_rets[0])
            else:
                all_rets.append(agent_rets)
        
        if len(all_rets) == 1:
            return all_rets[0]
        else:
            return all_rets



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

    def __init__(self, state_dim, action_dim, action_dim2, num_elements, observation_space, action_space, h_dim = 200, epsilon = 0.9, LR_A = 1e-4, LR_C = 1e-4, gamma = 0.9, replace_iter_a = 1100, replace_iter_c = 1000, memory_capacity = 20000, batch_size = 64, temperature=0.1):
        self.training_step = 0
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_dim2 = action_dim2
        self.epsilon = epsilon
        self.num_elements = num_elements 
        self.temperature = temperature

        self.h_dim = h_dim
        self.eval_cnet, self.target_cnet = MTMFAAAC(self.state_dim, self.action_dim, self.action_dim2, self.num_elements, batch_size, observation_space, action_space, hidden_dim=self.h_dim).float(), MTMFAAAC(self.state_dim, self.action_dim, self.action_dim2, self.num_elements, batch_size, observation_space, action_space, hidden_dim=self.h_dim).float()
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
        action_probs = F.softmax(action_probs/self.temperature, dim=1)
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
        a = np.array([t.a for t in transitions]) 
        a = torch.tensor(a, dtype=torch.float)

        meanfield = np.array([t.meanfield for t in transitions])

        meanfield = torch.tensor(meanfield, dtype=torch.float)
        
        meanfield2 = np.array([t.meanfield2 for t in transitions])

        meanfield2 = torch.tensor(meanfield2, dtype=torch.float)


        r = torch.tensor([t.r for t in transitions], dtype=torch.float).view(-1, 1)
        
        s_ = np.array([t.s_ for t in transitions]) 

        s_ = torch.tensor(s_, dtype=torch.float)

        sothers = [t.sothers for t in transitions]


        aothers = [t.aothers for t in transitions]
        s_others = [t.s_others for t in transitions]
        merged_list = [(sothers[i], aothers[i]) for i in range(0, len(sothers))]

        merged_list2 = [(s_others[i], aothers[i]) for i in range(0, len(s_others))]

        meanfield = meanfield.squeeze()
        meanfield2 = meanfield2.squeeze()

        with torch.no_grad():
            q_values = self.target_cnet(merged_list2, s_, meanfield, meanfield2)
            max_value = torch.max(q_values)
            q_target = r + self.gamma * max_value
        
        
        q_eval = self.eval_cnet(merged_list, s, meanfield, meanfield2)
        int_acs = a.max(dim=1, keepdim=True)[1]
        q_eval = q_eval.gather(1, int_acs)


        self.optimizer_c.zero_grad()
        loss = nn.MSELoss()
        c_loss = loss(q_eval, q_target)
        c_loss.backward()
        self.optimizer_c.step()

        self.optimizer_a.zero_grad()

        action_probs = self.eval_anet(s)
        action_probs = action_probs.squeeze()
        
        log_policy = torch.log(action_probs + 1e-6)
        
        with torch.no_grad():
            action_probs = self.target_anet(s_)
            action_probs = action_probs.detach().squeeze()
            action = torch.multinomial(action_probs, 1)
            action = F.one_hot(action, num_classes=self.action_dim)
            
            action = torch.squeeze(action)
            
            int_acs = action.max(dim=1, keepdim=True)[1]
            q_target = q_values.gather(1, int_acs)
        
        a_loss = -(log_policy * q_target).mean()
        
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

    def save_param(self, path='./param/mfaaac.pkl'):
        print("Saving")
        save_dict = {'networks': [network.state_dict() for network in self.networks]}
        torch.save(save_dict, path)

    def load_param(self, path='./param/mfaaac.pkl'):
        print("loading")
        save_dict = torch.load(path)
        for network, params in zip(self.networks, save_dict['networks']):
            network.load_state_dict(params)


