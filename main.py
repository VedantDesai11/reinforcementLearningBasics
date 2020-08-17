import gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

env = gym.make("MountainCar-v0")

learning_rate = 0.1
discount = 0.95
episodes = 25000

show = 1000

discrete_os_size = [20] * len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / discrete_os_size

epsilon = 0.5
start_epsilon_decay = 1
end_epsilon_decay = episodes // 2

epsilon_decay_value = epsilon / (end_epsilon_decay - start_epsilon_decay)

q_table = np.random.uniform(low=-2, high=0, size=(discrete_os_size + [env.action_space.n]))

episode_rewards = []
aggrigate_episode_rewards = {'ep': [], 'avg': [], 'min': [], 'max': []}


def get_discrete_state(state):
	discrete_state = (state - env.observation_space.low) / discrete_os_win_size
	return tuple(discrete_state.astype(np.int))


for episode in range(episodes):

	episode_reward = 0
	"""
	if episode % show == 0:
		render = True
	else:
		render = False
	"""
	render = True

	discrete_state = get_discrete_state(env.reset())

	done = False

	while not done:

		if np.random.random() > epsilon:
			action = np.argmax(q_table[discrete_state])
		else:
			action = np.random.randint(0, env.action_space.n)

		new_state, reward, done, _ = env.step(action)
		episode_reward += reward

		new_discrete_state = get_discrete_state(new_state)

		if render:
			env.render()

		if not done:
			max_future_q = np.max(q_table[new_discrete_state])
			current_q = q_table[discrete_state + (action,)]

			new_q = (1 - learning_rate) * current_q + learning_rate * (reward + discount * max_future_q)

			q_table[discrete_state + (action,)] = new_q

		elif new_state[0] >= env.goal_position:
			#print(f"We made it on episode {episode}")
			q_table[discrete_state + (action,)] = 0

		discrete_state = new_discrete_state

	if end_epsilon_decay >= episode >= start_epsilon_decay:
		epsilon -= epsilon_decay_value

	episode_rewards.append(episode_reward)

	if episode % 10 == 0:
		average_reward = sum(episode_rewards[-show:]) / len(episode_rewards[-show:])
		aggrigate_episode_rewards['ep'].append(episode)
		aggrigate_episode_rewards['avg'].append(average_reward)
		aggrigate_episode_rewards['min'].append(min(episode_rewards[-show:]))
		aggrigate_episode_rewards['max'].append(max(episode_rewards[-show:]))
		np.save(f"qtables/{episode}-qtable.npy", q_table)

env.close()

plt.plot(aggrigate_episode_rewards['ep'], aggrigate_episode_rewards['avg'], label="Average")
plt.plot(aggrigate_episode_rewards['ep'], aggrigate_episode_rewards['min'], label="Minimun")
plt.plot(aggrigate_episode_rewards['ep'], aggrigate_episode_rewards['max'], label="Maximum")

plt.legend(loc=2)
plt.savefig(f"Episode_VS_Rewards.png")




