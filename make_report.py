import gym
import numpy as np
import pickle

env = gym.make("MountainCar-v0")

scores = []
training_data = []
accepted_scores = []
required_score = -198

for i in range(20000):
    print(f'{i} | ', end='')
    env.reset()
    score = 0
    game_memory = []
    previous_obs = []
    
    while True :
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        if len(previous_obs) > 0 :
            game_memory.append([previous_obs, action])
        previous_obs = obs
        if obs[0] > -0.2 :
            reward = 1
        score += reward
        if done :
            break
        # env.render()
            
    scores.append(score)
    if score > required_score :
        accepted_scores.append(score)
        for data in game_memory :
            training_data.append(data)
print()
with open('test_result.pickle','wb') as f :
    pickle.dump(training_data,f)

# print(training_data)

print("finished!")
print(f'mean of scores {np.mean(scores)}')
print(f"length of accepted_scores {len(accepted_scores)}")
print(f'mean of accepted_scores {np.mean(accepted_scores)}')
