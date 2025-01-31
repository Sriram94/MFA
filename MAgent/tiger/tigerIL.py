from pettingzoo.magent import tiger_deer_v4
from ILalgo import DQN
import csv
import random
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
    with open('pettingzoomagentDQN.csv', 'w+') as myfile:
        myfile.write('{0},{1}\n'.format("Episode", "sumofrewards(DQN)"))
    
    num_episode = 0 
    while num_episode < 5:
        agent_num = 0
        observation = parallel_env.reset()
        accumulated_reward = 0
        max_cycles = 100
        actions = {}
        for step in range(max_cycles):
        
            for agent in parallel_env.agents:
                agent_observation = observation[agent]
                        
                agent_observation = change_observation(agent_observation)
                if "deer" in agent: 
                    action = random.randint(0, (deer_n_actions-1))
                    actions[agent] = action 
                else: 
                    action = RL1.choose_action(agent_observation)
                    actions[agent] = action 
            new_observation, rewards, dones, infos = parallel_env.step(actions)   
            
            for agent in parallel_env.agents: 
                if "tiger" in agent:
                    accumulated_reward = accumulated_reward + rewards[agent]
                    agent_observation = observation[agent]
                    agent_observation = change_observation(agent_observation)
                    agent_nextobservation = new_observation[agent]
                    agent_nextobservation = change_observation(agent_nextobservation)
                    RL1.store_transition(agent_observation, actions[agent], rewards[agent], agent_nextobservation)

            
            observation = new_observation
            print("the step is", step)
            
            if not parallel_env.agents:  
                break
            
            
            
            
                
                
            
        print("learning") 
        RL1.learn()
        

        print("The episode is", num_episode)
        
        with open('pettingzoomagentDQN.csv', 'a') as myfile:
            myfile.write('{0},{1}\n'.format(num_episode, accumulated_reward))
        

        num_episode = num_episode + 1            
         
        team_size = parallel_env.team_size()
        print("The number of deers alive is", team_size[0])
        print("The number of tigers alive is", team_size[1])

    RL1.save("./dqnmodel1.ckpt")

    RL1.restore("./dqnmodel1.ckpt")
    
    
    # end of game
    print('game over')


if __name__ == "__main__":
    parallel_env = tiger_deer_v4.parallel_env(map_size = 74, max_cycles=500, minimap_mode = True, extra_features=True) # 74 gives 54 tigers, 100 gives 100 tigers. 

    parallel_env.seed(1)

    RL1 = DQN(2349, 9)
    team_size = parallel_env.team_size()
    print("The number of deers alive is", team_size[0])
    print("The number of tigers alive is", team_size[1])
    
    run_tiger(parallel_env)
