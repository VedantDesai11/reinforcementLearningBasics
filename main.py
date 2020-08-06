import gym
import numpy as np

env = gym.make("MountainCar-v0")
env.reset()

learning_rate = 0.1
discount = 0.95
episodes = 25000

discrete_os_size = [20] * len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / discrete_os_size

q_table = np.random.uniform(low=-2, high=0, size=(discrete_os_size + [env.action_space.n]))

print(discrete_os_size)