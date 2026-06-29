from pettingzoo.magent import tiger_deer_v4
from collections import namedtuple
import os
from WMFRLalgo import Agent
import random
import csv
import numpy as np 


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

np.random.seed(0)
random.seed(0)
torch.manual_seed(0)
Transition = namedtuple('Transition', ['s', 'a', 'r', 's_', 'sothers', 'aothers', 's_others'])


def change_observation(observation):
    observation = observation.tolist()
    new_list = []
    for i in range(len(observation)):
        for j in range(len(observation[i])):
            for k in range(len(observation[i][j])):
                new_list.append(observation[i][j][k])
    new_observation = np.array(new_list)
    return new_observation



def run_tiger(parallel_env):
    
    step = 0
    deer_n_actions = 5
    tiger_n_actions = 9
    observation_space = 2349
    action_space = 9

    with open('pettingzoomagentmaac.csv', 'w+') as myfile:
        myfile.write('{0},{1},{2}\n'.format("Episode", "sumofrewards(MAAC)", "sumofrewards(MAAC)"))
    
    num_episode = 0 

    n_action = tiger_n_actions
    team_size = parallel_env.team_size()


    actual_team_size = team_size[1]

    while num_episode < 5:
        observation = parallel_env.reset()
        accumulated_reward = 0
        max_cycles = 100
        actions = {}
        for step in range(max_cycles):
        
            for agent in parallel_env.agents:
                if "deer" in agent:
                    action = random.randint(0, (deer_n_actions-1))
                    actions[agent] = action
                else: 
                    agent_observation = observation[agent]
                    agent_observation = change_observation(agent_observation)               
                    action = agent1.select_action(agent_observation) 
                    actions[agent] = action 
                    


            new_observation, rewards, dones, infos = parallel_env.step(actions)   
            
            sothers = [] 
            aothers = [] 
            s_others = [] 

            team_size = parallel_env.team_size()
            team_size = team_size[1]
            
            for agent_new in parallel_env.agents:
                if "tiger" in agent_new: 
                    sothers.append(torch.FloatTensor(change_observation(observation[agent_new])))
                    s_others.append(torch.FloatTensor(change_observation(new_observation[agent_new])))
                    action_new = actions[agent_new]
                    
                    action_new = torch.from_numpy(np.eye(n_action)[action_new])
                    aothers.append(action_new)
            
            if team_size < actual_team_size: 
                s = team_size

                for i in range(s, actual_team_size):
                    obs = np.zeros(observation_space)
                    sothers.append(torch.FloatTensor(obs))

                    s_others.append(torch.FloatTensor(obs))
                    action_new = np.zeros(action_space)
                    
                    action_new = torch.from_numpy(action_new)
                    aothers.append(action_new)

                
            
            for agent in parallel_env.agents: 
                if "tiger" in agent: 
                    accumulated_reward = accumulated_reward + rewards[agent]
                    agent_observation = observation[agent]
                    agent_observation = change_observation(agent_observation)
                    agent_nextobservation = new_observation[agent]
                    agent_nextobservation = change_observation(agent_nextobservation)

                    action = actions[agent]
                    action = np.eye(n_action)[action]
                    agent1.store_transition(Transition(agent_observation, action, rewards[agent], agent_nextobservation, sothers, aothers, s_others))

            
            observation = new_observation
            print("the step is", step)
            
            
            if not parallel_env.agents:  
                break
            
            
            
            
                
                
            
        print("learning") 
        agent1.prepare_update()
        agent1.update()
        

        print("The episode is", num_episode)
        
        with open('pettingzoomagentmaac.csv', 'a') as myfile:
            myfile.write('{0},{1}\n'.format(num_episode, accumulated_reward))
        

        num_episode = num_episode + 1            

        team_size = parallel_env.team_size()
        print("The number of deers alive is", team_size[0])
        print("The number of tigers alive is", team_size[1]) 
         

    agent1.save_param()
    
    agent1.load_param()
    
    
    print('game over')


if __name__ == "__main__":
    parallel_env = tiger_deer_v4.parallel_env(map_size = 74, max_cycles=500, minimap_mode = True, extra_features=True)
    parallel_env.seed(1)
    if not os.path.exists('./param'):
        print('param dir doesnt exit')

    team_size = parallel_env.team_size()
    print("The number of deers alive is", team_size[0])
    print("The number of tigers alive is", team_size[1]) 

    
    
    agent1 = Agent(2349, 9, team_size[1])
    agent2 = Agent(2349, 9, team_size[1])
    run_tiger(parallel_env)
