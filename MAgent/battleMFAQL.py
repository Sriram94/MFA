from pettingzoo.magent import battle_v4
from collections import namedtuple
import os
from MFAQLalgo import Agent
import csv
import numpy as np 


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

np.random.seed(0)
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





def run_battle(parallel_env):
    
    step = 0
    observation_space = 6929
    action_space = 21
    max_elements = 10


    with open('pettingzoomagentmfaa.csv', 'w+') as myfile:
        myfile.write('{0},{1},{2}\n'.format("Episode", "sumofrewards(MFAA)", "sumofrewards(MFAA)"))
    
    num_episode = 0 



    while num_episode < 5:
        observation = parallel_env.reset()
        accumulated_reward = [0,0]
        max_cycles = 100
        actions = {}

        n_groups = 2
        n_action = [[] for i in range(n_groups)]
        
        n_action[0] = 21  
        n_action[1] = 21  
        meanfield = [np.zeros((1, n_action[0])), np.zeros((1, n_action[1]))]

        
        for agent in parallel_env.agents: 
            actions[agent] = 0 


        for step in range(max_cycles):
            list_actions = [[] for i in range(n_groups)]
            curr_actions = {} 
            
            for agent in parallel_env.agents: 
                curr_actions[agent] = actions[agent]


            neighbours = parallel_env.get_neighbour_list(curr_actions, max_elements)



            sothers_dict = {}
            aothers_dict = {} 
            
            
            for agent in parallel_env.agents:
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
                agent_observation = observation[agent]
                agent_observation = change_observation(agent_observation)
               
                if "red" in agent: 
                    action = agent1.select_action(agent_observation, sothers_dict[agent], aothers_dict[agent], meanfield[0]) 
                    actions[agent] = action 
                    list_actions[0].append(action)
                    
                else: 
                    action = agent2.select_action(agent_observation, sothers_dict[agent], aothers_dict[agent], meanfield[1]) 
                    
                    actions[agent] = action 
                    list_actions[1].append(action)


            new_observation, rewards, dones, infos = parallel_env.step(actions)   
            
            
            for i in range(n_groups):
                meanfield[i] = np.mean(list(map(lambda x: np.eye(n_action[i])[x], list_actions[i])), axis=0, keepdims=True)

            
                   
            s_others_dict = {}
            a_others_dict = {} 
            
            
            
            for agent in parallel_env.agents:
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
                if "red" in agent: 
                    team = 0
                else: 
                    team = 1
                
                accumulated_reward[team] = accumulated_reward[team] + rewards[agent]
                agent_observation = observation[agent]
                agent_observation = change_observation(agent_observation)
                agent_nextobservation = new_observation[agent]
                agent_nextobservation = change_observation(agent_nextobservation)

                if team == 0: 
                    action = actions[agent]
                    action = np.eye(action_space)[action]
                    agent1.store_transition(Transition(agent_observation, action, rewards[agent], agent_nextobservation, sothers_dict[agent], aothers_dict[agent], s_others_dict[agent], a_others_dict[agent], meanfield[0]))
                
                else: 
                    action = actions[agent]
                    action = np.eye(action_space)[action]
                    agent2.store_transition(Transition(agent_observation, action, rewards[agent], agent_nextobservation, sothers_dict[agent], aothers_dict[agent], s_others_dict[agent], a_others_dict[agent], meanfield[1]))



            
            observation = new_observation
            print("the step is", step)
            
            
            if not parallel_env.agents:  
                break
            
            
            
            
                
                
            
        print("learning") 
        if agent1.memory.isfull:
            agent1.prepare_update()
            agent1.update()
        
        if agent2.memory.isfull:
            agent2.prepare_update()
            agent2.update()

        print("The episode is", num_episode)
        
        with open('pettingzoomagentmfaa.csv', 'a') as myfile:
            myfile.write('{0},{1},{2}\n'.format(num_episode, accumulated_reward[0], accumulated_reward[1]))
        

        num_episode = num_episode + 1            
        team_size = parallel_env.team_size()
        print("The number of red agents alive is", team_size[0])
        print("The number of blue agents alive is", team_size[1])

    agent1.save_param()
    agent2.save_param()
    
    agent1.load_param()
    agent2.load_param()
    
    
    # end of game
    print('game over')


if __name__ == "__main__":
    parallel_env = battle_v4.parallel_env(map_size = 12, max_cycles=500, minimap_mode = True, extra_features=True)
    parallel_env.seed(1)
    if not os.path.exists('./param'):
        print('param dir doesnt exit')
    

    num_elements = 10
    size = len(parallel_env.agents) 
    observation_space = {}
    action_space = {}
    
    agent1 = Agent(6929, 21, num_elements)
    agent2 = Agent(6929, 21, num_elements)
    print("The total number of agents are", size)
    run_battle(parallel_env)
