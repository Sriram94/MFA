from pettingzoo.magent import combined_arms_v6
from collections import namedtuple
from ILalgo import DQN
import os
from MTMFAAACalgo import Agent
import csv
import numpy as np 


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

np.random.seed(0)
torch.manual_seed(0)
Transition = namedtuple('Transition', ['s', 'a', 'r', 's_', 'sothers', 'aothers', 's_others', 'a_others', 'meanfield', 'meanfield2'])

def linear_decay(epoch, x, y):
    min_v, max_v = y[0], y[-1]
    start, end = x[0], x[-1]

    if epoch == start:
        return min_v

    eps = min_v

    for i, x_i in enumerate(x):
        if epoch <= x_i:
            interval = (y[i] - y[i - 1]) / (x_i - x[i - 1])
            eps = interval * (epoch - x[i - 1]) + y[i - 1]
            break

    return eps



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
    eps = 1
    observation_space_mele = 5915 
    action_space_mele = 9 
    observation_space_ranged = 8619 
    action_space_ranged = 25
    max_elements = 10

    with open('pettingzoomagentmtmfaaac.csv', 'w+') as myfile:
        myfile.write('{0},{1},{2}\n'.format("Episode", "sumofrewards(MTMFMAAAC)", "sumofrewards(MTMFMAAAC)"))
    
    num_episode = 0 


    while num_episode < 5:
        observation = parallel_env.reset()
        accumulated_reward = [0,0]
        max_cycles = 100
        actions = {}
        n_groups = 4
        n_action = [[] for i in range(n_groups)]


        n_action[0] = 9
        n_action[1] = 25
        n_action[2] = 9
        n_action[3] = 25
        meanfield = [np.zeros((1, n_action[0])), np.zeros((1, n_action[1])), np.zeros((1, n_action[2])), np.zeros((1, n_action[3]))]

        for agent in parallel_env.agents:
            actions[agent] = 0

        for step in range(max_cycles):
            list_actions = [[] for i in range(n_groups)]
            curr_actions = {}

            for agent in parallel_env.agents:
                curr_actions[agent] = actions[agent]


            neighbourtype1, neighbourtype2 = parallel_env.get_type_neighbour_list(curr_actions, max_elements)


            sothers_dict = {}
            aothers_dict = {}


            for agent in parallel_env.agents:
                neighbour_list = neighbourtype1[agent]
                j = 0
                sothers = []
                aothers = []

                for agent_new in neighbour_list:

                    j = j + 1
                    sothers.append(torch.FloatTensor(change_observation(observation[agent_new])))
                    action_new = actions[agent_new]

                    action_new = torch.from_numpy(np.eye(action_space_mele)[action_new])
                    aothers.append(action_new)

                if j < (max_elements/2):

                    for i in range(j, int(max_elements/2)):
                        obs = np.zeros(observation_space_mele)
                        sothers.append(torch.FloatTensor(obs))

                        action_new = np.zeros(action_space_mele)

                        action_new = torch.from_numpy(action_new)
                        aothers.append(action_new)

                neighbour_list = neighbourtype2[agent]

                j = 0

                for agent_new in neighbour_list:

                    j = j + 1
                    sothers.append(torch.FloatTensor(change_observation(observation[agent_new])))
                    action_new = actions[agent_new]

                    action_new = torch.from_numpy(np.eye(action_space_ranged)[action_new])
                    aothers.append(action_new)

                if j < (max_elements/2):

                    for i in range(j, int(max_elements/2)):
                        obs = np.zeros(observation_space_ranged)
                        sothers.append(torch.FloatTensor(obs))

                        action_new = np.zeros(action_space_ranged)

                        action_new = torch.from_numpy(action_new)
                        aothers.append(action_new)

                sothers_dict[agent] = sothers
                aothers_dict[agent] = aothers


        
            for agent in parallel_env.agents:
                agent_observation = observation[agent]
                agent_observation = change_observation(agent_observation)
               
                if "red" in agent: 
                    if "mele" in agent:
                        if (np.random.uniform() <= eps):
                            action = RL1.choose_action(agent_observation)
                        else: 
                            action = agent1.select_action(agent_observation) 

                        actions[agent] = action
                        list_actions[0].append(action)
                    
                    else:
                        if (np.random.uniform() <= eps):
                            action = RL2.choose_action(agent_observation)
                        else: 
                            action = agent2.select_action(agent_observation) 

                        actions[agent] = action
                        list_actions[1].append(action)

                    
                else: 
                    if "mele" in agent:
                        if (np.random.uniform() <= eps):
                            action = RL3.choose_action(agent_observation)
                        else: 
                            action = agent3.select_action(agent_observation) 

                        actions[agent] = action
                        list_actions[2].append(action)
                    
                    else:
                        if (np.random.uniform() <= eps):
                            action = RL4.choose_action(agent_observation)
                        else: 
                            action = agent4.select_action(agent_observation) 

                        actions[agent] = action
                        list_actions[3].append(action)


            new_observation, rewards, dones, infos = parallel_env.step(actions)   
            
            for i in range(n_groups):
                meanfield[i] = np.mean(list(map(lambda x: np.eye(n_action[i])[x], list_actions[i])), axis=0, keepdims=True)

            
            s_others_dict = {}            
            a_others_dict = {}

            for agent in parallel_env.agents:
                neighbour_list = neighbourtype1[agent]
                j = 0
                s_others = []
                a_others = []

                for agent_new in neighbour_list:

                    j = j + 1
                    s_others.append(torch.FloatTensor(change_observation(new_observation[agent_new])))
                    action_new = actions[agent_new]

                    action_new = torch.from_numpy(np.eye(action_space_mele)[action_new])
                    a_others.append(action_new)

                if j < max_elements/2:

                    for i in range(j, int(max_elements/2)):
                        obs = np.zeros(observation_space_mele)
                        s_others.append(torch.FloatTensor(obs))

                        action_new = np.zeros(action_space_mele)

                        action_new = torch.from_numpy(action_new)
                        a_others.append(action_new)

                neighbour_list = neighbourtype2[agent]
                j = 0
                for agent_new in neighbour_list:

                    j = j + 1
                    s_others.append(torch.FloatTensor(change_observation(new_observation[agent_new])))
                    action_new = actions[agent_new]

                    action_new = torch.from_numpy(np.eye(action_space_ranged)[action_new])
                    a_others.append(action_new)

                if j < max_elements/2:

                    for i in range(j, int(max_elements/2)):
                        obs = np.zeros(observation_space_ranged)
                        s_others.append(torch.FloatTensor(obs))

                        action_new = np.zeros(action_space_ranged)

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
                    if "mele" in agent:
                        action = actions[agent]
                        action = np.eye(action_space_mele)[action]
                        agent1.store_transition(Transition(agent_observation, action, rewards[agent], agent_nextobservation, sothers_dict[agent], aothers_dict[agent], s_others_dict[agent], a_others_dict[agent], meanfield[0], meanfield[1]))
                    else: 
                        action = actions[agent]
                        action = np.eye(action_space_ranged)[action]
                        agent2.store_transition(Transition(agent_observation, action, rewards[agent], agent_nextobservation, sothers_dict[agent], aothers_dict[agent], s_others_dict[agent], a_others_dict[agent], meanfield[1], meanfield[0]))
                else: 
                    if "mele" in agent:
                        action = actions[agent]
                        action = np.eye(action_space_mele)[action]
                        agent3.store_transition(Transition(agent_observation, action, rewards[agent], agent_nextobservation, sothers_dict[agent], aothers_dict[agent], s_others_dict[agent], a_others_dict[agent], meanfield[2], meanfield[3]))

                    else:
                        action = actions[agent]
                        action = np.eye(action_space_ranged)[action]
                        agent4.store_transition(Transition(agent_observation, action, rewards[agent], agent_nextobservation, sothers_dict[agent], aothers_dict[agent], s_others_dict[agent], a_others_dict[agent], meanfield[3], meanfield[2]))
            
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
        
        with open('pettingzoomagentmtmfaaac.csv', 'a') as myfile:
            myfile.write('{0},{1},{2}\n'.format(num_episode, accumulated_reward[0], accumulated_reward[1]))
        

        num_episode = num_episode + 1            
        eps = linear_decay(num_episode, [0, int(500 * 0.95), 500], [1, 0.2, 0])
         
        team_size = parallel_env.team_size()
        
        print("The number of red melee agents alive is", team_size[0])
        print("The number of red ranged agents alive is", team_size[1])
        print("The number of blue melee agents alive is", team_size[2])
        print("The number of blue ranged agents alive is", team_size[3])


    agent1.save_param('./param/agent1mtmfaaac.pkl')
    agent2.save_param('./param/agent2mtmfaaac.pkl')
    agent3.save_param('./param/agent3mtmfaaac.pkl')
    agent4.save_param('./param/agent4mtmfaaac.pkl')
    
    agent1.load_param('./param/agent1mtmfaaac.pkl')
    agent2.load_param('./param/agent2mtmfaaac.pkl')
    agent3.load_param('./param/agent3mtmfaaac.pkl')
    agent4.load_param('./param/agent4mtmfaaac.pkl')
    
    
    # end of game
    print('game over')


if __name__ == "__main__":
    parallel_env = combined_arms_v6.parallel_env(map_size = 20, max_cycles=500, minimap_mode = True, extra_features=True) # 50 gives 100 agents in each team.
    parallel_env.seed(1)
    if not os.path.exists('./param'):
        print('param dir doesnt exit')

    num_elements = 10

    size = len(parallel_env.agents) 
    observation_space = []
    action_space = []


    for i in range(0, int(num_elements/2)):
            observation_space.append(5915)
            action_space.append(9)

    for i in range(0, int(num_elements/2)):
            observation_space.append(8619)
            action_space.append(25)
    
    
    
    agent1 = Agent(5915, 9, 25, num_elements, observation_space, action_space)
    agent2 = Agent(8619, 25, 9, num_elements, observation_space, action_space)
    agent3 = Agent(5915, 9, 25, num_elements, observation_space, action_space)
    agent4 = Agent(8619, 25, 9, num_elements, observation_space, action_space)


    RL1 = DQN(5915, 9, epsilon = 1)
    RL2 = DQN(8619, 25, epsilon = 1)
    RL3 = DQN(5915, 9, epsilon = 1)
    RL4 = DQN(8619, 25, epsilon = 1)


    RL1.restore("./dqnmodel1.ckpt")
    RL2.restore("./dqnmodel2.ckpt")
    RL3.restore("./dqnmodel3.ckpt")
    RL4.restore("./dqnmodel4.ckpt")


    team_size = parallel_env.team_size()
    
    print("The number of red melee agents alive is", team_size[0])
    print("The number of red ranged agents alive is", team_size[1])
    print("The number of blue melee agents alive is", team_size[2])
    print("The number of blue ranged agents alive is", team_size[3])
    run_combinedarms(parallel_env)
