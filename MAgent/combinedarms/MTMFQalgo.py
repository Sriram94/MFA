import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym

torch.manual_seed(0)
np.random.seed(0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("The device is", device)



class Net(nn.Module):
    def __init__(self, n_states, n_actions, n_actions1, n_actions2):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_states, 50)
        self.fc1.weight.data.normal_(0, 0.1)   
        self.fcm1 = nn.Linear(n_actions1, 50)
        self.fcm1.weight.data.normal_(0, 0.1)   
        self.fcm2 = nn.Linear(n_actions2, 50)
        self.fcm2.weight.data.normal_(0, 0.1)   
        self.fc2 = nn.Linear(150,50)
        self.fc2.weight.data.normal_(0, 0.1)   
        self.fc3 = nn.Linear(50,50)
        self.fc3.weight.data.normal_(0, 0.1)   
        self.out = nn.Linear(50, n_actions)
        self.out.weight.data.normal_(0, 0.1)   

    def forward(self, x, mean_a1, mean_a2):
        x = self.fc1(x)
        x = F.relu(x)
        y1 = self.fcm1(mean_a1)
        y1 = F.relu(y1)
        y2 = self.fcm2(mean_a2)
        y2 = F.relu(y2)
        z = torch.cat((x,y1, y2), dim = 1)
        z = self.fc2(z)
        z = F.relu(z)
        z = self.fc3(z)
        z = F.relu(z)
        actions_value = self.out(z)
        return actions_value


class MTMFQ(object):
    def __init__(self, n_states, n_actions, n_actions1, n_actions2, batch_size=64, lr=0.01, epsilon=0.9, gamma = 0.9, target_replace = 1, memory_capacity=20000):
        self.eval_net, self.target_net = Net(n_states, n_actions, n_actions1, n_actions2).to(device), Net(n_states, n_actions, n_actions1, n_actions2).to(device)
        self.batch_size = batch_size
        self.n_states = n_states
        self.n_actions = n_actions
        self.n_actions1 = n_actions1
        self.n_actions2 = n_actions2
        self.memory_capacity = memory_capacity
        self.target_replace = target_replace
        self.memory_capacity = memory_capacity
        self.temperature = 0.1

        self.learn_step_counter = 0                                     
        self.gamma = gamma
        self.lr = lr
        self.memory_counter = 0                                         
        self.epsilon = epsilon
        self.memory = np.zeros((memory_capacity, n_states * 2 + 2 + n_actions1 + n_actions2))     
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.lr)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x, meanfield1, meanfield2):
        
        x = torch.unsqueeze(torch.FloatTensor(x), 0).to(device)
        meanfield1 = torch.FloatTensor(meanfield1).to(device)
        meanfield2 = torch.FloatTensor(meanfield2).to(device)

            
        if np.random.uniform() < self.epsilon:   
            actions_value = self.eval_net.forward(x, meanfield1, meanfield2).to(device)
            actions_value = F.softmax(actions_value/self.temperature, dim=1)

            action = torch.max(actions_value, 1)[1].cpu().data.numpy()
            action = action[0] 
        else:   
            action = np.random.randint(0, self.n_actions)
        
        return action

    def store_transition(self, s, a, r, meanfield1, meanfield2, s_):
        meanfield1 = meanfield1[0]
        meanfield2 = meanfield2[0]
        transition = np.hstack((s, [a, r], meanfield1, meanfield2, s_))
        index = self.memory_counter % self.memory_capacity 
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        if self.learn_step_counter % self.target_replace == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        if self.memory_counter > self.memory_capacity:
            sample_index = np.random.choice(self.memory_capacity, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :self.n_states]).to(device)
        b_a = torch.LongTensor(b_memory[:, self.n_states:self.n_states+1].astype(int)).to(device)
        b_r = torch.FloatTensor(b_memory[:, self.n_states+1:self.n_states+2]).to(device)
        b_mean1 = torch.FloatTensor(b_memory[:, self.n_states+2:self.n_states+2+self.n_actions1]).to(device)
        b_mean2 = torch.FloatTensor(b_memory[:, self.n_states+2+self.n_actions1:self.n_states+2+self.n_actions1+self.n_actions2]).to(device)
        b_s_ = torch.FloatTensor(b_memory[:, -self.n_states:]).to(device)



        q_eval = self.eval_net(b_s, b_mean1, b_mean2).gather(1, b_a)  
        q_next = self.target_net(b_s_, b_mean1, b_mean2).detach()     
        q_target = b_r + self.gamma * q_next.max(1)[0].view(self.batch_size, 1)   
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save(self, PATH):
        torch.save(self.eval_net.state_dict(), PATH)
        print("Model saved")

    def restore(self, PATH):
        self.eval_net.load_state_dict(torch.load(PATH))
        self.eval_net.eval()
        self.eval_net.to(device)
        print("Model restored")



