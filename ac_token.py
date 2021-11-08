"""
Title: Actor Critic Method
Author: [Apoorv Nandan](https://twitter.com/NandanApoorv)
Date created: 2020/05/13
Last modified: 2020/05/13
Description: Implement Actor Critic Method in CartPole environment.
"""
"""
## Introduction

This script shows an implementation of Actor Critic method on CartPole-V0 environment.

### Actor Critic Method

As an agent takes actions and moves through an environment, it learns to map
the observed state of the environment to two possible outputs:

1. Recommended action: A probability value for each action in the action space.
   The part of the agent responsible for this output is called the **actor**.
2. Estimated rewards in the future: Sum of all rewards it expects to receive in the
   future. The part of the agent responsible for this output is the **critic**.

Agent and Critic learn to perform their tasks, such that the recommended actions
from the actor maximize the rewards.

### CartPole-V0

A pole is attached to a cart placed on a frictionless track. The agent has to apply
force to move the cart. It is rewarded for every time step the pole
remains upright. The agent, therefore, must learn to keep the pole from falling over.

### References

- [CartPole](http://www.derongliu.org/adp/adp-cdrom/Barto1983.pdf)
- [Actor Critic Method](https://hal.inria.fr/hal-00840470/document)
"""
"""
## Setup
"""

import argparse,os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

parser = argparse.ArgumentParser()
parser.add_argument("--future_steps", default=32, type=int)
parser.add_argument("--beams", default=256, type=int)
parser.add_argument("--gpu", default="5,6,7", type=str)
args = parser.parse_args()
print('args==>', args)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import tensorflow as tf 
gpus = tf.config.experimental.list_physical_devices('GPU')
print('======>',gpus,'<=======')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
      # tf.config.experimental.set_virtual_device_configuration(gpu, \
      #      [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
  except RuntimeError as e:
    print(e)
    
from transformers import top_k_top_p_filtering
from torch.nn import functional as F
import os,string,torch,math,time

from transformers import pipeline
from utils.load_data import * 
ds = load_data(dataset='uci', samplecnt= 128)
ds.df_train['content'] = ds.df_train['content'].map(lambda x: x.strip(string.punctuation))

# gpt2
from transformers import GPT2Tokenizer, GPT2LMHeadModel #TFGPT2LMHeadModel, TFGPT2Model, TFAutoModelForCausalLM
tokenizer_gpt2 = GPT2Tokenizer.from_pretrained('gpt2', cache_dir="./cache", local_files_only=True)
#tokenizer_gpt2.padding_side = "left" 
tokenizer_gpt2.pad_token = tokenizer_gpt2.eos_token # to avoid an error "<|endoftext|>": 50256
tokenizer_gpt2.sep_token = '<|sep|>'
#tokenizer_gpt2.add_tokens(tokenizer_gpt2.sep_token)
print(tokenizer_gpt2)
# no
gpt2 = GPT2LMHeadModel.from_pretrained('gpt2', cache_dir="./cache", local_files_only=True)
# tc pp
#gpt2 = GPT2LMHeadModel.from_pretrained('ft_model_{}_{}'.format('t5', 'ep') )
gpt2.trainable = False
gpt2.config.pad_token_id = 50256

gen_nlp_gpt2  = pipeline("text-generation", model=gpt2, tokenizer=tokenizer_gpt2, device=1, return_full_text=True)

device_i = torch.device("cuda:{}".format(1) if torch.cuda.is_available() else "cpu")
gpt2.to(device_i)


from utils.transblock import * 
with tf.distribute.MirroredStrategy().scope():
#with tf.device('/gpu:0'):
    model_cls = get_model_bert(ds.df_test.label.unique().shape[0])
model_cls.load_weights("./model_cls/model_full_uci.h5")          
model_cls.trainable = False

print("model cls loaded")

eps = np.finfo(np.float32).eps.item()  # Smallest number such that 1.0 + eps != 1.0

"""
## Implement Actor Critic network

This network learns two functions:

1. Actor: This takes as input the state of our environment and returns a
probability value for each action in its action space.
2. Critic: This takes as input the state of our environment and returns
an estimate of total rewards in the future.

In our implementation, they share the initial layer.
"""
with tf.distribute.MirroredStrategy().scope():
#with tf.device('/gpu:0'):
    model = get_model_bert_ac(gpt2.config.vocab_size)

print("model ac loaded")



def sample_action_gpt(sent, action_probs):
    input_ids = tokenizer_gpt2.encode(sent, return_tensors="pt").to(device_i)
    # get logits of last hidden state
    next_token_logits = gpt2(input_ids).logits[:, -1, :] / 1.0
    next_token_logits_a = torch.mul(torch.tensor(action_probs.numpy()).to(device_i), next_token_logits)
    # filter
    filtered_next_token_logits = top_k_top_p_filtering(next_token_logits_a, top_p=0.9) # , top_p=1

    probs = F.softmax(filtered_next_token_logits, dim=-1) # 50257

    next_token = torch.multinomial(probs, num_samples=1)

    return next_token


def get_future_score(sent, label, future_steps, beams):
    # ori
    ori_ids = tokenizer_gpt2.encode(sent, return_tensors="pt")
    tokens_len_ori = ori_ids.shape[1]
    result = gen_nlp_gpt2([sent], max_length=tokens_len_ori+future_steps, do_sample=True, top_p=0.9, top_k=0, temperature=1,\
                            repetition_penalty=1.0, num_return_sequences=beams, clean_up_tokenization_spaces=True)
    
    x = tf.convert_to_tensor([ ii['generated_text'].strip() for ii in result ])
    y = np.array([label] * x.shape[0])
    #preds = model_cls(x)
    future_loss = model_cls.evaluate(x, y, batch_size=64, verbose=0) 
    #future_loss = tf.keras.losses.SparseCategoricalCrossentropy()(tf.convert_to_tensor([label] * preds.shape[0]), preds)
    return future_loss



def next_sent_reward(sent, label, next_token, future_steps=32, beams=512):
    score_ori = get_future_score(sent, label, future_steps, beams)

    next_state_ids = torch.cat([tokenizer_gpt2.encode(sent, return_tensors="pt"), next_token.cpu()], dim=-1)

    sent_next = tokenizer_gpt2.decode(next_state_ids[0], clean_up_tokenization_spaces=True, skip_special_tokens=True)
    
    score_next  = get_future_score(sent_next, label, future_steps, beams)

    return score_ori - score_next, sent_next


optimizer = keras.optimizers.Adam(learning_rate=5e-5)
huber_loss = keras.losses.Huber()
action_probs_history = []
critic_value_history = []
rewards_history = []
returns_history = []


for epoch in range(100):
    ds.df_train = ds.df_train.sample(frac=1)
    for ix, row in ds.df_train.reset_index().iterrows():
        torch.cuda.empty_cache()

        sent = row['content']
        label = row['label']
        label_name = row['label_name']  
         
        episode_reward = 0
        with tf.GradientTape(watch_accessed_variables=True) as tape:
            #tape.watch(model.trainable_weights)

            for step in range(64):
                # env.render(); Adding this line would show the attempts
                # of the agent in a pop up window.

                state = tf.convert_to_tensor(sent)
                state = tf.expand_dims(state, 0) #  shape=(1, 4)

                # Predict action probabilities and estimated future rewards
                # from environment state

                action_probs, critic_value = model(state)
            
                # Sample action from action probability distribution with GPT2
                with tape.stop_recording():
                    next_token = sample_action_gpt(sent, action_probs)# action
                    #action = np.random.choice(gpt2.config.vocab_size, p=np.squeeze(action_probs))
                    
                    # Apply the sampled action in our environment
                    reward, state = next_sent_reward(sent, label, next_token, args.future_steps, args.beams)
                    print("step reward:", reward)

                critic_value_history.append(critic_value[0, 0])
                action_probs_history.append(tf.math.log(action_probs[0, next_token.cpu().numpy()[0][0] ]) )
                rewards_history.append(reward)
                episode_reward += reward

            # Calculate expected value from rewards
            # - At each timestep what was the total reward received after that timestep
            # - Rewards in the past are discounted by multiplying them with gamma
            # - These are the labels for our critic
            print("episode_reward:", episode_reward)
            returns = []
            discounted_sum = 0
            for r in rewards_history[::-1]:
                discounted_sum = r + gamma * discounted_sum
                returns.insert(0, discounted_sum)

            # Normalize
            returns = np.array(returns)
            returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
            returns = returns.tolist()
            returns_history.extend(returns)



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



