import gymnasium as gym

env = gym.make('CartPole-v1')
print(env.observation_space)
print(env.action_space)
print(env.reset())

print(env.step(action=0))
print(env.step(action=1))