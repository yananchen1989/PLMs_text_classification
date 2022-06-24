
##### main reference: https://keras.io/examples/rl/actor_critic_cartpole/

import gym,math
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd 

gpus = tf.config.list_physical_devices('GPU')
print('======>',gpus,'<=======')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)


# seed = 42
gamma = 0.99  # Discount factor for past rewards
max_steps_per_episode = 10000
env = gym.make("CartPole-v1")  # Create the environment
# env.seed(seed)
eps = np.finfo(np.float32).eps.item()  # Smallest number such that 1.0 + eps != 1.0

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

temperature = 1.5
num_inputs = 4
num_actions = 2
num_hidden = 256

inputs = layers.Input(shape=(num_inputs,))
common1 = layers.Dense(256, activation="relu")(inputs)
common2 = layers.Dense(512, activation="relu")(common1)
common = layers.Dense(256, activation="relu")(common2)
# shared
action = layers.Dense(num_actions, activation="softmax")(common)
critic = layers.Dense(1)(common)

model = keras.Model(inputs=inputs, outputs=[action, critic])

# to mitigate overconfidence with only one model,
# One possibility is to train K different models with different datasets. 
# For each episode, we randomly select one of the models to be used throughout the whole episode. 
# So we will have a more coherent strategy for our actions rather than introducing some random actions. 
# Since different episodes may use different models, 
# we open ourselves to a different perspective and being less overconfident to a single perspective (a training model).

#   training part
optimizer = keras.optimizers.Adam(learning_rate=0.01)
huber_loss = keras.losses.Huber()
action_probs_history = []
critic_value_history = []
rewards_history = []
running_reward = 0
episode_count = 0

reward_last_100 = []

episode_reward_records = []
while True:  # Run until solved
    state = env.reset()
    episode_reward = 0
    with tf.GradientTape() as tape:

        for timestep in range(1, max_steps_per_episode):
            # env.render(); Adding this line would show the attempts
            # of the agent in a pop up window.

            state = tf.convert_to_tensor(state)
            state = tf.expand_dims(state, 0)

            # Predict action probabilities and estimated future rewards
            # from environment state
            action_probs, critic_value = model(state)
            critic_value_history.append(critic_value[0, 0])

            # Sample action from action probability distribution
            # randomly select based on actions distribution
            # temperature can also be used, inverse proportional to the number of episode trained
            # temperature = math.exp(-episode_count * alpha)
            action_probs_softmax = np.exp(action_probs / temperature) / np.sum(np.exp(action_probs/temperature))
            action = np.random.choice(num_actions, p=np.squeeze(action_probs_softmax))
            


            action_probs_history.append(tf.math.log(action_probs[0, action]))

            # Apply the sampled action in our environment
            state, reward, done, _ = env.step(action)
            rewards_history.append(reward)
            episode_reward += reward

            if done:
                break

        # Update running reward to check condition for solving
        running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward

        # Calculate expected value from rewards
        # - At each timestep what was the total reward received after that timestep
        # - Rewards in the past are discounted by multiplying them with gamma
        # - These are the labels for our critic
        returns = []
        discounted_sum = 0
        for r in rewards_history[::-1]:
            discounted_sum = r + gamma * discounted_sum
            returns.insert(0, discounted_sum)

        # Normalize
        returns = np.array(returns)
        returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
        returns = returns.tolist()

        # Calculating loss values to update our network
        history = zip(action_probs_history, critic_value_history, returns)
        actor_losses = []
        critic_losses = []
        for log_prob, value, ret in history:
            # At this point in history, the critic estimated that we would get a
            # total reward = `value` in the future. We took an action with log probability
            # of `log_prob` and ended up recieving a total reward = `ret`.
            # The actor must be updated so that it predicts an action that leads to
            # high rewards (compared to critic's estimate) with high probability.
            diff = ret - value
            actor_losses.append(-log_prob * diff)  # actor loss

            # The critic must be updated so that it predicts a better estimate of
            # the future rewards.
            critic_losses.append(
                huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0))
            )

        # Backpropagation
        loss_value = sum(actor_losses) + sum(critic_losses)
        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # Clear the loss and reward history
        action_probs_history.clear()
        critic_value_history.clear()
        rewards_history.clear()

    # Log details
    episode_count += 1
    if episode_count % 10 == 0:
        template = "running reward: {:.2f} at episode {}"
        print(template.format(episode_reward, episode_count))

    episode_reward_records.append((episode_count, episode_reward))
    if len(episode_reward_records) % 100 == 0:


    reward_last_100.append(episode_reward)

    if len(reward_last_100) >= 100 and sum(reward_last_100[-100:]) / len(reward_last_100[-100:]) > 475:  # Condition to consider the task solved
        print("Solved at episode {}!".format(episode_count))
        break

# plot 
df = pd.DataFrame(episode_reward_records, columns=['episode','reward'])
sns.lineplot(data=df, x="episode", y="reward", markers=True, dashes=False)
plt.show()


# sometimes, ,odel parameters oscillate and do not converge.


