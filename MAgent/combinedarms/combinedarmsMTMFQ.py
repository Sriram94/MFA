from pettingzoo.magent import combined_arms_v6
from MTMFQalgo import MTMFQ
import csv
import numpy as np 

np.random.seed(0)


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
    with open('pettingzoomagentMTMFQ.csv', 'w+') as myfile:
        myfile.write('{0},{1},{2}\n'.format("Episode", "sumofrewards(MTMFQ)", "sumofrewards(MTMFQ)"))
    
    num_episode = 0 
    while num_episode < 5:
        agent_num = 0
        observation = parallel_env.reset()
        list_tmp = [0]
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

        for step in range(max_cycles):
            list_actions = [[] for i in range(n_groups)]
        
            for agent in parallel_env.agents:
                agent_observation = observation[agent]
                        
                agent_observation = change_observation(agent_observation)
                if "red" in agent: 
                    if "mele" in agent:
                        action = RL1.choose_action(agent_observation, meanfield[0], meanfield[1])
                        list_actions[0].append(action)
                    else: 
                        action = RL2.choose_action(agent_observation, meanfield[0], meanfield[1])
                        list_actions[1].append(action)
                else: 
                    if "mele" in agent:
                        action = RL3.choose_action(agent_observation, meanfield[2], meanfield[3])
                        list_actions[2].append(action)
                    else: 
                        action = RL4.choose_action(agent_observation, meanfield[2], meanfield[3])
                        list_actions[3].append(action)

                actions[agent] = action 
            
            new_observation, rewards, dones, infos = parallel_env.step(actions)   

            for i in range(n_groups):
                meanfield[i] = np.mean(list(map(lambda x: np.eye(n_action[i])[x], list_actions[i])), axis=0, keepdims=True)

            
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
                        RL1.store_transition(agent_observation, actions[agent], rewards[agent], meanfield[0], meanfield[1], agent_nextobservation)
                    else: 
                        RL2.store_transition(agent_observation, actions[agent], rewards[agent], meanfield[0], meanfield[1], agent_nextobservation)

                else: 
                    if "mele" in agent:
                        RL3.store_transition(agent_observation, actions[agent], rewards[agent], meanfield[2], meanfield[3], agent_nextobservation)
                    else: 
                        RL4.store_transition(agent_observation, actions[agent], rewards[agent], meanfield[2], meanfield[3], agent_nextobservation)

            
            observation = new_observation
            print("the step is", step)
            
            if not parallel_env.agents:  
                break
             
            
        print("learning") 
        RL1.learn()
        RL2.learn()
        RL3.learn()
        RL4.learn()
        

        print("The episode is", num_episode)
        
        with open('pettingzoomagentMTMFQ.csv', 'a') as myfile:
            myfile.write('{0},{1},{2}\n'.format(num_episode, accumulated_reward[0], accumulated_reward[1]))
        

        num_episode = num_episode + 1            

        team_size = parallel_env.team_size()
        print("The number of red melee agents alive is", team_size[0])
        print("The number of red ranged agents alive is", team_size[1])
        print("The number of blue melee agents alive is", team_size[2])
        print("The number of blue ranged agents alive is", team_size[3])
         

    RL1.save("./mtmfqmodel1.ckpt")
    RL2.save("./mtmfqmodel2.ckpt")
    RL3.save("./mtmfqmodel3.ckpt")
    RL4.save("./mtmfqmodel4.ckpt")

    RL1.restore("./mtmfqmodel1.ckpt")
    RL2.restore("./mtmfqmodel2.ckpt")
    RL3.restore("./mtmfqmodel3.ckpt")
    RL4.restore("./mtmfqmodel4.ckpt")
    
    
    # end of game
    print('game over')


if __name__ == "__main__":
    parallel_env = combined_arms_v6.parallel_env(map_size = 50, max_cycles=500, minimap_mode = True, extra_features=True) # 50 gives 100 agents in each team.
    parallel_env.seed(1)
    RL1 = MTMFQ(5915, 9, 9, 25)
    RL2 = MTMFQ(8619, 25, 9, 25)
    RL3 = MTMFQ(5915, 9, 9, 25)
    RL4 = MTMFQ(8619, 25, 9, 25)
    
    team_size = parallel_env.team_size()
    print("The number of red melee agents alive is", team_size[0])
    print("The number of red ranged agents alive is", team_size[1])
    print("The number of blue melee agents alive is", team_size[2])
    print("The number of blue ranged agents alive is", team_size[3])

    run_combinedarms(parallel_env) 
