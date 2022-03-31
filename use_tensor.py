import tensorflow as tf
import gym
import numpy as np
import pickle

env = gym.make("MountainCar-v0")



with open('test_result.pickle','rb') as f :
    training_data = pickle.load(f)

print(training_data)

t_X = np.array([i[0] for i in training_data]).reshape(-1,2)
t_Y = np.array([i[1] for i in training_data]).reshape(-1,1)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128,input_shape=(2, ), activation = 'relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(3,activation='softmax')
])

model.compile(optimizer=tf.optimizers.Adam(), loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
history = model.fit(t_X, t_Y, epochs=10, callbacks=[callback], batch_size=16, validation_split=0.25)
# history = model.fit(t_X, t_Y, epochs=30, callbacks=[callback], batch_size=16)

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
        logit = model.predict(np.expand_dims(previous_obs, axis=0))[0]
        action = np.argmax(logit)
    obs, reward, done, _ = env.step(action)
    previous_obs = obs
    score += reward
    step += 1

    if done:
        break

print('score : ', score)
print('step : ', step)
env.close()
