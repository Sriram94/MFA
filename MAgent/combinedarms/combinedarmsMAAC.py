from pettingzoo.magent import combined_arms_v6
from collections import namedtuple
import os
from MAACalgo import Agent
import csv
import numpy as np 


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

np.random.seed(0)
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



def run_combinedarms(parallel_env): 
    step = 0
    observation_space = {}
    action_space = {}
    observation_space_mele = 5915 
    action_space_mele = 9 
    observation_space_ranged = 8619 
    action_space_ranged = 25

    with open('pettingzoomagentmaac.csv', 'w+') as myfile:
        myfile.write('{0},{1},{2}\n'.format("Episode", "sumofrewards(MAAC)", "sumofrewards(MAAC)"))
    
    num_episode = 0 

    for agent in parallel_env.agents:
        if 'mele' in agent:
            observation_space[agent] = observation_space_mele
            action_space[agent] = action_space_mele
        else:
            observation_space[agent] = observation_space_ranged
            action_space[agent] = action_space_ranged

    actual_team_size = parallel_env.team_size()

    while num_episode < 5:
        observation = parallel_env.reset()
        accumulated_reward = [0,0]
        max_cycles = 100
        actions = {}
        for step in range(max_cycles):
        
            for agent in parallel_env.agents:
                agent_observation = observation[agent]
                agent_observation = change_observation(agent_observation)
               
                if "red" in agent: 
                    if "mele" in agent:
                        action = agent1.select_action(agent_observation) 
                    else: 
                        action = agent2.select_action(agent_observation) 
                    actions[agent] = action 
                    
                else: 
                    if "mele" in agent:
                        action = agent3.select_action(agent_observation) 
                    else: 
                        action = agent4.select_action(agent_observation) 

                    actions[agent] = action 


            new_observation, rewards, dones, infos = parallel_env.step(actions)   
            
            sothers = [] 
            aothers = [] 
            s_others = [] 

            team_size = parallel_env.team_size()
            for agent_new in parallel_env.agents:
                if "red" in agent_new: 
                    if "mele" in agent_new:
                        sothers.append(torch.FloatTensor(change_observation(observation[agent_new])))
                        s_others.append(torch.FloatTensor(change_observation(new_observation[agent_new])))
                        action_new = actions[agent_new]
                        
                        action_new = torch.from_numpy(np.eye(action_space[agent_new])[action_new])
                        aothers.append(action_new)
            
            if team_size[0] < actual_team_size[0]: 
                s = team_size[0]

                for i in range(s, actual_team_size[0]):
                    obs = np.zeros(observation_space_mele)
                    sothers.append(torch.FloatTensor(obs))

                    s_others.append(torch.FloatTensor(obs))
                    action_new = np.zeros(action_space_mele)
                    
                    action_new = torch.from_numpy(action_new)
                    aothers.append(action_new)

            for agent_new in parallel_env.agents:
                if "red" in agent_new: 
                    if "ranged" in agent_new:
                        sothers.append(torch.FloatTensor(change_observation(observation[agent_new])))
                        s_others.append(torch.FloatTensor(change_observation(new_observation[agent_new])))
                        action_new = actions[agent_new]
                        
                        action_new = torch.from_numpy(np.eye(action_space[agent_new])[action_new])
                        aothers.append(action_new)
            
            if team_size[1] < actual_team_size[1]: 
                s = team_size[1]

                for i in range(s, actual_team_size[1]):
                    obs = np.zeros(observation_space_ranged)
                    sothers.append(torch.FloatTensor(obs))

                    s_others.append(torch.FloatTensor(obs))
                    action_new = np.zeros(action_space_ranged)
                    
                    action_new = torch.from_numpy(action_new)
                    aothers.append(action_new)
            
            
            for agent_new in parallel_env.agents:
                if "blue" in agent_new: 
                    if "mele" in agent_new:
                        sothers.append(torch.FloatTensor(change_observation(observation[agent_new])))
                        s_others.append(torch.FloatTensor(change_observation(new_observation[agent_new])))
                        action_new = actions[agent_new]
                        
                        action_new = torch.from_numpy(np.eye(action_space[agent_new])[action_new])
                        aothers.append(action_new)
            
            if team_size[2] < actual_team_size[2]: 
                s = team_size[2]

                for i in range(s, actual_team_size[2]):
                    obs = np.zeros(observation_space_mele)
                    sothers.append(torch.FloatTensor(obs))

                    s_others.append(torch.FloatTensor(obs))
                    action_new = np.zeros(action_space_mele)
                    
                    action_new = torch.from_numpy(action_new)
                    aothers.append(action_new)
                   
            for agent_new in parallel_env.agents:
                if "blue" in agent_new: 
                    if "ranged" in agent_new:
                        sothers.append(torch.FloatTensor(change_observation(observation[agent_new])))
                        s_others.append(torch.FloatTensor(change_observation(new_observation[agent_new])))
                        action_new = actions[agent_new]
                        
                        action_new = torch.from_numpy(np.eye(action_space[agent_new])[action_new])
                        aothers.append(action_new)
            
            if team_size[3] < actual_team_size[3]: 
                s = team_size[3]

                for i in range(s, actual_team_size[3]):
                    obs = np.zeros(observation_space_ranged)
                    sothers.append(torch.FloatTensor(obs))

                    s_others.append(torch.FloatTensor(obs))
                    action_new = np.zeros(action_space_ranged)
                    
                    action_new = torch.from_numpy(action_new)
                    aothers.append(action_new)
                
            
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
                    if "mele" in agent:
                        action = actions[agent]
                        action = np.eye(action_space_mele)[action]
                        agent1.store_transition(Transition(agent_observation, action, rewards[agent], agent_nextobservation, sothers, aothers, s_others))
                    else: 
                        action = actions[agent]
                        action = np.eye(action_space_ranged)[action]
                        agent2.store_transition(Transition(agent_observation, action, rewards[agent], agent_nextobservation, sothers, aothers, s_others))
                else: 
                    if "mele" in agent:
                        action = actions[agent]
                        action = np.eye(action_space_mele)[action]
                        agent3.store_transition(Transition(agent_observation, action, rewards[agent], agent_nextobservation, sothers, aothers, s_others))

                    else:
                        action = actions[agent]
                        action = np.eye(action_space_ranged)[action]
                        agent4.store_transition(Transition(agent_observation, action, rewards[agent], agent_nextobservation, sothers, aothers, s_others))
            
            observation = new_observation
            print("the step is", step)
            
            
            if not parallel_env.agents:  
                break
            
            
            
            
                
                
            
        print("learning") 
        agent1.prepare_update()
        agent1.update()
        
        agent2.prepare_update()
        agent2.update()

        agent3.prepare_update()
        agent3.update()

        agent4.prepare_update()
        agent4.update()
        
        print("The episode is", num_episode)
        
        with open('pettingzoomagentmaac.csv', 'a') as myfile:
            myfile.write('{0},{1},{2}\n'.format(num_episode, accumulated_reward[0], accumulated_reward[1]))
        

        num_episode = num_episode + 1            
         
        team_size = parallel_env.team_size()
        
        print("The number of red melee agents alive is", team_size[0])
        print("The number of red ranged agents alive is", team_size[1])
        print("The number of blue melee agents alive is", team_size[2])
        print("The number of blue ranged agents alive is", team_size[3])


    agent1.save_param('./param/agent1maac.pkl')
    agent2.save_param('./param/agent2maac.pkl')
    agent3.save_param('./param/agent3maac.pkl')
    agent4.save_param('./param/agent4maac.pkl')
    
    agent1.load_param('./param/agent1maac.pkl')
    agent2.load_param('./param/agent2maac.pkl')
    agent3.load_param('./param/agent3maac.pkl')
    agent4.load_param('./param/agent4maac.pkl')
    
    
    # end of game
    print('game over')


if __name__ == "__main__":
    parallel_env = combined_arms_v6.parallel_env(map_size = 20, max_cycles=500, minimap_mode = True, extra_features=True) # 50 gives 100 agents in each team.
    parallel_env.seed(1)
    if not os.path.exists('./param'):
        print('param dir doesnt exit')

    size = len(parallel_env.agents) 
    observation_space = []
    action_space = []
    
    for agent in parallel_env.agents:
        if 'mele' in agent: 
            observation_space.append(5915)
            action_space.append(9)
        else: 
            observation_space.append(8619)
            action_space.append(25)
    
    
    agent1 = Agent(5915, 9, size, observation_space, action_space)
    agent2 = Agent(8619, 25, size, observation_space, action_space)
    agent3 = Agent(5915, 9, size, observation_space, action_space)
    agent4 = Agent(8619, 25, size, observation_space, action_space)
    team_size = parallel_env.team_size()
    
    print("The number of red melee agents alive is", team_size[0])
    print("The number of red ranged agents alive is", team_size[1])
    print("The number of blue melee agents alive is", team_size[2])
    print("The number of blue ranged agents alive is", team_size[3])
    run_combinedarms(parallel_env)
