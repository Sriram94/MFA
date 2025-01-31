from pettingzoo.magent import combined_arms_v6
from collections import namedtuple
import os
from ACalgo import Agent
import csv
import numpy as np 

np.random.seed(0)
Transition = namedtuple('Transition', ['s', 'a', 'r', 's_'])


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
    with open('pettingzoomagentac.csv', 'w+') as myfile:
        myfile.write('{0},{1},{2}\n'.format("Episode", "sumofrewards(AC)", "sumofrewards(AC)"))
    
    num_episode = 0 
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
                    
                else: 
                    if "mele" in agent:
                        action = agent3.select_action(agent_observation) 
                    else: 
                        action = agent4.select_action(agent_observation) 
                
                actions[agent] = action 


            new_observation, rewards, dones, infos = parallel_env.step(actions)   
            
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
                        agent1.store_transition(Transition(agent_observation, actions[agent], rewards[agent], agent_nextobservation))
                    else: 
                        agent2.store_transition(Transition(agent_observation, actions[agent], rewards[agent], agent_nextobservation))

                else: 
                    if "mele" in agent:
                        agent3.store_transition(Transition(agent_observation, actions[agent], rewards[agent], agent_nextobservation))
                    else: 
                        agent4.store_transition(Transition(agent_observation, actions[agent], rewards[agent], agent_nextobservation))

            
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
        
        with open('pettingzoomagentac.csv', 'a') as myfile:
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
    
    
    print('game over')


if __name__ == "__main__":
    parallel_env = combined_arms_v6.parallel_env(map_size = 50, max_cycles=500, minimap_mode = True, extra_features=True) # 50 gives 100 agents in each team. 
    parallel_env.seed(1)
    if not os.path.exists('./param'):
        print('param dir doesnt exit')

    agent1 = Agent(5915, 9)
    agent2 = Agent(8619, 25)
    agent3 = Agent(5915, 9)
    agent4 = Agent(8619, 25)
    team_size = parallel_env.team_size()
    print("The number of red melee agents alive is", team_size[0])
    print("The number of red ranged agents alive is", team_size[1])
    print("The number of blue melee agents alive is", team_size[2])
    print("The number of blue ranged agents alive is", team_size[3])

    run_combinedarms(parallel_env)
