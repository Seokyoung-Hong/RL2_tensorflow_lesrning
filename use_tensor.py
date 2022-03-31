import tensorflow as tf
import gym
import random
import numpy as np
import csv

env = gym.make("MountainCar-v0")
f = open("test_result.txt",'a')
'''
scores = []
training_data = []
accepted_scores = []
required_score = -198
for i in range(20000):
    # print(f'{i} | ', end='')
    # print()
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
            f.write(f"{data},")
f.close()

print(training_data)

print("finished!")
print(f'mean of scores {np.mean(scores)}')
print(f"length of accepted_scores {len(accepted_scores)}")
print(f'mean of accepted_scores {np.mean(accepted_scores)}')
'''

traning_data.append


t_X = np.array([i[0] for i in training_data]).reshape(-1,2)
t_Y = np.array([i[1] for i in training_data]).reshape(-1,1)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128,input_shape=(2, ), activation = 'relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(3,activation='softmax')
])

model.compile(optimizer=tf.optimizers.Adam(), loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
# history = model.fit(t_X, t_Y, epochs=30, callbacks=[callback], batch_size=16, validation_split=0.25)
history = model.fit(t_X, t_Y, epochs=30, callbacks=[callback], batch_size=16)

import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'], 'b-', label='accuracy')
plt.plot(history.history['val_accuracy'], 'k--', label='val_accuracy')
plt.legend()
plt.show()

env.close()
env.reset()

score = 0
step = 0
previous_obs = []
while True:
    env.render()
    if len(previous_obs) == 0:
        action = env.action_space.sample()
    else:
#         logit = model.predict(np.expand_dims(previous_obs, axis=0))[0]
#         action = np.argmax(logit)
#     obs, reward, done, _ = env.step(action)
#     previous_obs = obs
#     score += reward
#     step += 1

#     if done:
#         break

# print('score : ', score)
# print('step : ', step)
# env.close()
