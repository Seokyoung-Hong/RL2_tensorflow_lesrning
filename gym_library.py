import gym


env = gym.make("MountainCar-v0")
# print(env.observation_space)
# print(env.observation_space.low)
# print(env.observation_space.high)

# print(env._max_episode_steps)

# print(env.action_space)
env.reset()


step = 0
score = 0

while True :
    env.render()
    
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    score += reward
    step += 1
    if done :
        break

print(f"Filan Score : {score}")
print(f'step : {step}')

env.close()