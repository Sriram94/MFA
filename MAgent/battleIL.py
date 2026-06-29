from pettingzoo.magent import battle_v4
from ILalgo import DQN
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
    with open('pettingzoomagentDQN.csv', 'w+') as myfile:
        myfile.write('{0},{1},{2}\n'.format("Episode", "sumofrewards(DQN)", "sumofrewards(DQN)"))
    
    num_episode = 0 
    while num_episode < 5:
        agent_num = 0
        observation = parallel_env.reset()
        list_tmp = [0]
        accumulated_reward = [0,0]
        max_cycles = 500
        actions = {}
        for step in range(max_cycles):
        
            for agent in parallel_env.agents:
                agent_observation = observation[agent]
                        
                agent_observation = change_observation(agent_observation)
                if "red" in agent: 
                    action = RL1.choose_action(agent_observation)
                    actions[agent] = action 
                else: 
                    action = RL2.choose_action(agent_observation)
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
                    RL1.store_transition(agent_observation, actions[agent], rewards[agent], agent_nextobservation)
                else: 
                    RL2.store_transition(agent_observation, actions[agent], rewards[agent], agent_nextobservation)

            
            observation = new_observation
            print("the step is", step)
            
            if not parallel_env.agents:  
                break
            
            
            
            
                
                
            
        print("learning") 
        RL1.learn()
        RL2.learn()
        

        print("The episode is", num_episode)
        
        with open('pettingzoomagentDQN.csv', 'a') as myfile:
            myfile.write('{0},{1},{2}\n'.format(num_episode, accumulated_reward[0], accumulated_reward[1]))
        

        num_episode = num_episode + 1            
         
        team_size = parallel_env.team_size()
        print("The number of red agents alive is", team_size[0])
        print("The number of blue agents alive is", team_size[1])

    RL1.save("./dqnmodel1.ckpt")
    RL2.save("./dqnmodel2.ckpt")

    RL1.restore("./dqnmodel1.ckpt")
    RL2.restore("./dqnmodel2.ckpt")
    
    
    # end of game
    print('game over')


if __name__ == "__main__":
    parallel_env = battle_v4.parallel_env(map_size = 40, max_cycles=500, minimap_mode = True, extra_features=True) # 40 gives 64 agents, 50 gives 100 agents, 60 gives 144 agents. 
    parallel_env.seed(1)
    RL1 = DQN(6929, 21)
    RL2 = DQN(6929, 21)
    team_size = parallel_env.team_size()
    print("The number of red agents alive is", team_size[0])
    print("The number of blue agents alive is", team_size[1])
    run_battle(parallel_env)
