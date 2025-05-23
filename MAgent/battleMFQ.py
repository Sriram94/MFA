from pettingzoo.magent import battle_v4
from MFQalgo import MFQ
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



def run_battle(parallel_env):
    
    step = 0
    with open('pettingzoomagentMFQ.csv', 'w+') as myfile:
        myfile.write('{0},{1},{2}\n'.format("Episode", "sumofrewards(MFQ)", "sumofrewards(MFQ)"))
    
    num_episode = 0 
    while num_episode < 5:
        agent_num = 0
        observation = parallel_env.reset()
        list_tmp = [0]
        accumulated_reward = [0,0]
        max_cycles = 100
        actions = {}
        n_groups = 2
        n_action = [[] for i in range(n_groups)]
        
        n_action[0] = 21  
        n_action[1] = 21  
        meanfield = [np.zeros((1, n_action[0])), np.zeros((1, n_action[1]))]

        for step in range(max_cycles):
            list_actions = [[] for i in range(n_groups)]
        
            for agent in parallel_env.agents:
                agent_observation = observation[agent]
                        
                agent_observation = change_observation(agent_observation)
                if "red" in agent: 
                    action = RL1.choose_action(agent_observation, meanfield[0])
                    actions[agent] = action 
                    list_actions[0].append(action)
                else: 
                    action = RL2.choose_action(agent_observation, meanfield[1])
                    actions[agent] = action 
                    list_actions[1].append(action)
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
                    RL1.store_transition(agent_observation, actions[agent], rewards[agent], meanfield[0], agent_nextobservation)
                else: 
                    RL2.store_transition(agent_observation, actions[agent], rewards[agent], meanfield[1], agent_nextobservation)

            
            observation = new_observation
            print("the step is", step)
            
            if not parallel_env.agents:  
                break
             
            
        print("learning") 
        RL1.learn()
        RL2.learn()
        

        print("The episode is", num_episode)
        
        with open('pettingzoomagentMFQ.csv', 'a') as myfile:
            myfile.write('{0},{1},{2}\n'.format(num_episode, accumulated_reward[0], accumulated_reward[1]))
        

        num_episode = num_episode + 1            
         
        team_size = parallel_env.team_size()
        print("The number of red agents alive is", team_size[0])
        print("The number of blue agents alive is", team_size[1])

    RL1.save("./mfqmodel1.ckpt")
    RL2.save("./mfqmodel2.ckpt")

    RL1.restore("./mfqmodel1.ckpt")
    RL2.restore("./mfqmodel2.ckpt")
    
    
    # end of game
    print('game over')


if __name__ == "__main__":
    parallel_env = battle_v4.parallel_env(map_size = 40, max_cycles=500, minimap_mode = True, extra_features=True)
    parallel_env.seed(1)
    RL1 = MFQ(6929, 21)
    RL2 = MFQ(6929, 21)
    size = len(parallel_env.agents) 
    print("The total number of agents are", size)
    run_battle(parallel_env)
