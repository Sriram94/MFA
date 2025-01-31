from pettingzoo.magent import tiger_deer_v4
from collections import namedtuple
import os
from MFAQLalgo import Agent
import csv
import random
import numpy as np 


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

np.random.seed(0)
random.seed(0)
torch.manual_seed(0)

Transition = namedtuple('Transition', ['s', 'a', 'r', 's_', 'sothers', 'aothers', 's_others', 'a_others', 'meanfield'])


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
    max_elements = 10


    with open('pettingzoomagentmfaa.csv', 'w+') as myfile:
        myfile.write('{0},{1}\n'.format("Episode", "sumofrewards(MFAA)"))
    
    num_episode = 0 



    while num_episode < 5:
        observation = parallel_env.reset()
        accumulated_reward = 0
        max_cycles = 100
        actions = {}

        
        meanfield = [np.zeros((1, action_space))]

        
        for agent in parallel_env.agents: 
            if "tiger" in agent:
                actions[agent] = 0 


        for step in range(max_cycles):
            list_actions = []
            curr_actions = {} 
            
            for agent in parallel_env.agents: 
                if "tiger" in agent:
                    curr_actions[agent] = actions[agent]


            neighbours = parallel_env.get_neighbour_tiger_list(curr_actions, max_elements)



            sothers_dict = {}
            aothers_dict = {} 
            
            
            for agent in parallel_env.agents:
                if "tiger" in agent:
                    neighbour_list = neighbours[agent]
                    j = 0
                    sothers = []
                    aothers = [] 

                    for agent_new in neighbour_list:

                        j = j + 1
                        sothers.append(torch.FloatTensor(change_observation(observation[agent_new])))
                        action_new = actions[agent_new]
                        
                        action_new = torch.from_numpy(np.eye(action_space)[action_new])
                        aothers.append(action_new)

                    if j < max_elements:  
                
                        for i in range(j, max_elements):
                            obs = np.zeros(observation_space)
                            sothers.append(torch.FloatTensor(obs))

                            action_new = np.zeros(action_space)
                            
                            action_new = torch.from_numpy(action_new)
                            aothers.append(action_new)

                        
                    sothers_dict[agent] = sothers
                    aothers_dict[agent] = aothers


        
            for agent in parallel_env.agents:
                
                if "deer" in agent:
                    action = random.randint(0, (deer_n_actions-1))
                    actions[agent] = action
                else: 
                    agent_observation = observation[agent]
                    agent_observation = change_observation(agent_observation)
                   
                    action = agent1.select_action(agent_observation, sothers_dict[agent], aothers_dict[agent], meanfield[0]) 
                    actions[agent] = action 
                    list_actions.append(action)
                    


            new_observation, rewards, dones, infos = parallel_env.step(actions)   

            meanfield[0] = np.mean(list(map(lambda x: np.eye(tiger_n_actions)[x], list_actions)), axis=0, keepdims=True)
            
            

            
                   
            s_others_dict = {}
            a_others_dict = {} 
            
            
            
            for agent in parallel_env.agents:
                if "tiger" in agent:
                    neighbour_list = neighbours[agent]
                    j = 0
                    s_others = []
                    a_others = [] 

                    for agent_new in neighbour_list:

                        j = j + 1
                        s_others.append(torch.FloatTensor(change_observation(new_observation[agent_new])))
                        action_new = actions[agent_new]
                        
                        action_new = torch.from_numpy(np.eye(action_space)[action_new])
                        a_others.append(action_new)

                    if j < max_elements:  
                
                        for i in range(j, max_elements):
                            obs = np.zeros(observation_space)
                            s_others.append(torch.FloatTensor(obs))

                            action_new = np.zeros(action_space)
                            
                            action_new = torch.from_numpy(action_new)
                            a_others.append(action_new)

                        
                    s_others_dict[agent] = s_others
                    a_others_dict[agent] = a_others
                
            
            for agent in parallel_env.agents: 
                if "tiger" in agent:
                    accumulated_reward = accumulated_reward + rewards[agent]
                    agent_observation = observation[agent]
                    agent_observation = change_observation(agent_observation)
                    agent_nextobservation = new_observation[agent]
                    agent_nextobservation = change_observation(agent_nextobservation)

                    action = actions[agent]
                    action = np.eye(action_space)[action]
                    agent1.store_transition(Transition(agent_observation, action, rewards[agent], agent_nextobservation, sothers_dict[agent], aothers_dict[agent], s_others_dict[agent], a_others_dict[agent], meanfield[0]))
                    


            
            observation = new_observation
            print("the step is", step)
            
            
            if not parallel_env.agents:  
                break             
                
            
        print("learning") 
        agent1.prepare_update()
        agent1.update()
    
        print("The episode is", num_episode)
        
        with open('pettingzoomagentmfaa.csv', 'a') as myfile:
            myfile.write('{0},{1}\n'.format(num_episode, accumulated_reward))
        

        num_episode = num_episode + 1            
        team_size = parallel_env.team_size()
        print("The number of deers alive is", team_size[0])
        print("The number of tigers alive is", team_size[1])



    agent1.save_param()
    
    agent1.load_param()
    
    
    # end of game
    print('game over')


if __name__ == "__main__":
    parallel_env = tiger_deer_v4.parallel_env(map_size = 74, max_cycles=500, minimap_mode = True, extra_features=True)
    parallel_env.seed(1)
    if not os.path.exists('./param'):
        print('param dir doesnt exit')
    

    num_elements = 10
    size = len(parallel_env.agents) 
    observation_space = {}
    action_space = {}
    
    agent1 = Agent(2349, 9, num_elements)
    
    team_size = parallel_env.team_size()
    
    print("The number of deers alive is", team_size[0])
    print("The number of tigers alive is", team_size[1])
    run_tiger(parallel_env)
