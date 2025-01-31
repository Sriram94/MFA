from pettingzoo.magent import tiger_deer_v4
import random
from MFQalgo import MFQ
import csv
import numpy as np 

random.seed(0)
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



def run_tiger(parallel_env):
    
    step = 0
    deer_n_actions = 5
    tiger_n_actions = 9
    with open('pettingzoomagentMFQ.csv', 'w+') as myfile:
        myfile.write('{0},{1}\n'.format("Episode", "sumofrewards(MFQ)"))
    
    num_episode = 0 
    while num_episode < 5:
        agent_num = 0
        observation = parallel_env.reset()
        list_tmp = [0]
        accumulated_reward = 0
        max_cycles = 100
        actions = {}

        meanfield = [np.zeros((1, tiger_n_actions))]

        for step in range(max_cycles):
            list_actions = []
        
            for agent in parallel_env.agents:
                agent_observation = observation[agent]
                        
                agent_observation = change_observation(agent_observation)
                if "deer" in agent: 
                    action = random.randint(0, (deer_n_actions-1))
                    actions[agent] = action
                else: 
                    action = RL1.choose_action(agent_observation, meanfield[0])
                    actions[agent] = action 
                    list_actions.append(action)

            new_observation, rewards, dones, infos = parallel_env.step(actions)   

            meanfield[0] = np.mean(list(map(lambda x: np.eye(tiger_n_actions)[x], list_actions)), axis=0, keepdims=True)

            
            for agent in parallel_env.agents: 
                if "tiger" in agent:
                    accumulated_reward = accumulated_reward + rewards[agent]
                    agent_observation = observation[agent]
                    agent_observation = change_observation(agent_observation)
                    agent_nextobservation = new_observation[agent]
                    agent_nextobservation = change_observation(agent_nextobservation)
                    RL1.store_transition(agent_observation, actions[agent], rewards[agent], meanfield[0], agent_nextobservation)

            
            observation = new_observation
            print("the step is", step)
            
            if not parallel_env.agents:  
                break
             
            
        print("learning") 
        RL1.learn()
        

        print("The episode is", num_episode)
        
        with open('pettingzoomagentMFQ.csv', 'a') as myfile:
            myfile.write('{0},{1}\n'.format(num_episode, accumulated_reward))
        

        num_episode = num_episode + 1            
         
        team_size = parallel_env.team_size()
        print("The number of deers alive is", team_size[0])
        print("The number of tigers alive is", team_size[1])

    RL1.save("./mfqmodel1.ckpt")

    RL1.restore("./mfqmodel1.ckpt")
    
    
    # end of game
    print('game over')


if __name__ == "__main__":

    parallel_env = tiger_deer_v4.parallel_env(map_size = 74, max_cycles=500, minimap_mode = True, extra_features=True) # 74 gives 54 tigers, 100 gives 100 tigers.    
    
    parallel_env.seed(1)

    RL1 = MFQ(2349, 9)
    team_size = parallel_env.team_size()
    print("The number of deers alive is", team_size[0])
    print("The number of tigers alive is", team_size[1])

    run_tiger(parallel_env) 
