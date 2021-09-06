import gym
import torch as T
import numpy as np
from tqdm import tqdm
from networks import DQNAgent

k_size = 4

def step_env(env, observation_space, action, total_reward):
    ''' Concat k number of steps '''

    k_state = np.zeros(observation_space)
    k_reward = 0
    for k in range(k_size):
        state_next, reward, terminal, _ = env.step(int(action[0]))
        total_reward += reward

        k_state[k_size*k:k_size*(k+1)] = state_next
        k_reward += reward

        if terminal:
            break

    return k_state, k_reward, total_reward, terminal


def run(train, load, num_episodes=1000):
    env = gym.make("CartPole-v1")

    observation_space = np.array(env.observation_space.shape) * k_size
    action_space = env.action_space.n
    agent = DQNAgent(
        state_space=observation_space,
        action_space=action_space,
        max_memory_size=30000,
        batch_size=32,
        gamma=0.9,
        lr=0.00025,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.99
    )

    # Restart the environment for each episode
    env.reset()

    total_rewards = []
    if load:
        agent.load_models()
        if not train:
            env.render(mode="human")

    for ep_num in tqdm(range(num_episodes)):
        state = env.reset().tolist()*k_size
        state = T.Tensor([state])
        total_reward = 0
        steps = 0
        terminal = False
        
        while not terminal:
            action = agent.select_action(state)
            steps += 1

            next_state, reward, total_reward, terminal = \
                step_env(env, observation_space, action, total_reward)
            
            state_next = T.Tensor([next_state])
            reward = T.tensor([reward]).unsqueeze(0)

            terminal = T.tensor([int(terminal)]).unsqueeze(0)

            if train:
                agent.memory.store_transition(state, action, reward, state_next, terminal)
                agent.learn()

            state = state_next
        
        total_rewards.append(total_reward)

        if ep_num != 0 and ep_num % 100 == 0:
            print("\nEpisode {} total score = {}, average score = {}".format(ep_num + 1, np.sum(total_rewards), np.mean(total_rewards)))
        num_episodes += 1
        
    print("\nEpisode {} total score = {}, average score = {}".format(ep_num + 1, np.sum(total_rewards), np.mean(total_rewards)))

    if train:
       agent.save_models()

    env.close()

    if not load:
        # Plot graph
        pass

if __name__ == "__main__":
    run(True, False, num_episodes=2000)
    
