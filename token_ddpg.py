import argparse,os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

parser = argparse.ArgumentParser()
parser.add_argument("--future_steps", default=32, type=int)
parser.add_argument("--beams", default=256, type=int)
parser.add_argument("--gpu", default="0,1", type=str)
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
tokenizer_gpt2 = GPT2Tokenizer.from_pretrained('gpt2', cache_dir="./cache", local_files_only=False)
#tokenizer_gpt2.padding_side = "left" 
tokenizer_gpt2.pad_token = tokenizer_gpt2.eos_token # to avoid an error "<|endoftext|>": 50256
tokenizer_gpt2.sep_token = '<|sep|>'
#tokenizer_gpt2.add_tokens(tokenizer_gpt2.sep_token)
print(tokenizer_gpt2)
# no
gpt2 = GPT2LMHeadModel.from_pretrained('gpt2', cache_dir="./cache", local_files_only=False)
# tc pp
#gpt2 = GPT2LMHeadModel.from_pretrained('ft_model_{}_{}'.format('t5', 'ep') )
gpt2.trainable = False
gpt2.config.pad_token_id = 50256

gen_nlp_gpt2  = pipeline("text-generation", model=gpt2, tokenizer=tokenizer_gpt2, device=len(gpus)-1, return_full_text=True)

device_i = torch.device("cuda:{}".format(1) if torch.cuda.is_available() else "cpu")
gpt2.to(device_i)


from utils.transblock import * 
with tf.distribute.MirroredStrategy().scope():
#with tf.device('/gpu:0'):
    model_cls = get_model_bert(ds.df_test.label.unique().shape[0])
model_cls.load_weights("./model_cls/model_full_uci.h5")          
model_cls.trainable = False

print("model cls loaded")

from collections import deque

num_actions = gpt2.config.vocab_size
print("Size of Action Space ->  {}".format(num_actions))

upper_bound = 2
lower_bound = -2

print("Max Value of Action ->  {}".format(upper_bound))
print("Min Value of Action ->  {}".format(lower_bound))

class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)



class Buffer:
    def __init__(self, buffer_capacity=100000, batch_size=64):
        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size

        # Its tells us num of times record() was called.
        self.buffer_counter = 0

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.state_buffer = deque(maxlen=self.buffer_capacity )
        self.action_buffer = deque(maxlen=self.buffer_capacity ) 
        self.reward_buffer = deque(maxlen=self.buffer_capacity ) 
        self.next_state_buffer = deque(maxlen=self.buffer_capacity )

    # Takes (s,a,r,s') obervation tuple as input
    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        self.state_buffer.append(obs_tuple[0])
        self.action_buffer.append(obs_tuple[1])
        self.reward_buffer.append(obs_tuple[2])
        self.next_state_buffer.append(obs_tuple[3])
        self.buffer_counter += 1

    # Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows
    # TensorFlow to build a static graph out of the logic and computations in our function.
    # This provides a large speed up for blocks of code that contain many small TensorFlow operations such as this one.
    @tf.function
    def update(
        self, state_batch, action_batch, reward_batch, next_state_batch,
    ):
        # Training and updating Actor & Critic networks.
        # See Pseudo Code.
        with tf.GradientTape() as tape:
            target_actions = target_actor(next_state_batch, training=True)
            y = reward_batch + gamma * target_critic(
                [next_state_batch, target_actions], training=True
            )
            critic_value = critic_model([state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
        critic_optimizer.apply_gradients(
            zip(critic_grad, critic_model.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = actor_model(state_batch, training=True)
            critic_value = critic_model([state_batch, actions], training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
        actor_optimizer.apply_gradients(
            zip(actor_grad, actor_model.trainable_variables)
        )

    # We compute the loss and update parameters
    def learn(self):
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        # Convert to tensors
        state_batch = tf.convert_to_tensor(np.array([self.state_buffer[i] for i in batch_indices]))
        action_batch = tf.convert_to_tensor(np.array([self.action_buffer[i] for i in batch_indices]))
        reward_batch = tf.convert_to_tensor(np.array([self.reward_buffer[i] for i in batch_indices]))
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(np.array([self.next_state_buffer[i] for i in batch_indices]))

        self.update(state_batch, action_batch, reward_batch, next_state_batch)


# This update target parameters slowly
# Based on rate `tau`, which is much less than one.
@tf.function
def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))


"""
Here we define the Actor and Critic networks. These are basic Dense models
with `ReLU` activation.

Note: We need the initialization for last layer of the Actor to be between
`-0.003` and `0.003` as this prevents us from getting `1` or `-1` output values in
the initial stages, which would squash our gradients to zero,
as we use the `tanh` activation.
"""
preprocessor_file = "./resource/albert_en_preprocess_3" # https://tfhub.dev/tensorflow/albert_en_preprocess/3
preprocessor_layer = hub.KerasLayer(preprocessor_file)
encoder = hub.KerasLayer("./resource/albert_en_base_2", trainable=True)


def get_actor():
    # Initialize weights between -3e-3 and 3-e3
    last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string) 
    encoder_inputs = preprocessor_layer(text_input)
    outputs = encoder(encoder_inputs)
    embed = outputs["pooled_output"]  

    outputs = layers.Dense(num_actions, activation="tanh", kernel_initializer=last_init)(embed)

    # Our upper bound is 2.0 for Pendulum.
    outputs = outputs * upper_bound
    model = tf.keras.Model(text_input, outputs)
    return model


def get_critic():
    # State as input
    state_input = tf.keras.layers.Input(shape=(), dtype=tf.string) 
    encoder_inputs = preprocessor_layer(state_input)
    outputs = encoder(encoder_inputs)
    embed = outputs["pooled_output"]  

    # Action as input
    action_input = layers.Input(shape=(num_actions))
    action_out = layers.Dense(1024, activation="relu")(action_input)

    # Both are passed through seperate layer before concatenating
    concat = layers.Concatenate()([embed, action_out])

    outputs = layers.Dense(1)(concat)

    # Outputs single value for give state-action
    model = tf.keras.Model([state_input, action_input], outputs)

    return model


"""
`policy()` returns an action sampled from our Actor network plus some noise for
exploration.
"""


def policy(state, noise_object):
    sampled_actions = tf.squeeze(actor_model(state))
    noise = noise_object()
    # Adding noise to action
    sampled_actions = sampled_actions.numpy() + noise

    # We make sure action is within bounds
    legal_action = np.clip(sampled_actions, lower_bound, upper_bound)

    return np.squeeze(legal_action)


"""
## Training hyperparameters
"""

std_dev = 0.2
ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))

actor_model = get_actor()
critic_model = get_critic()

target_actor = get_actor()
target_critic = get_critic()

# Making the weights equal initially
target_actor.set_weights(actor_model.get_weights())
target_critic.set_weights(critic_model.get_weights())

# Learning rate for actor-critic models
critic_lr = 0.002
actor_lr = 0.001

critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

total_episodes = 100
# Discount factor for future rewards
gamma = 0.99
# Used to update target networks
tau = 0.005

buffer = Buffer(100000, 64)

"""
Now we implement our main training loop, and iterate over episodes.
We sample actions using `policy()` and train with `learn()` at each time step,
along with updating the Target networks at a rate `tau`.
"""
def sample_action_gpt(sent, action):
    input_ids = tokenizer_gpt2.encode(sent, return_tensors="pt").to(device_i)
    # get logits of last hidden state
    next_token_logits = gpt2(input_ids).logits[:, -1, :] / 1.0
    next_token_logits_a = torch.mul(torch.tensor(action.reshape(1, -1)).to(device_i), next_token_logits)
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
    return future_loss[0]

def next_sent_reward(sent_ori, sent, label, next_token, future_steps=32, beams=512):
    score_ori = get_future_score(sent_ori, label, future_steps, beams)

    next_state_ids = torch.cat([tokenizer_gpt2.encode(sent, return_tensors="pt"), next_token.cpu()], dim=-1)

    sent_next = tokenizer_gpt2.decode(next_state_ids[0], clean_up_tokenization_spaces=True, skip_special_tokens=True)
    
    score_next  = get_future_score(sent_next, label, future_steps, beams)

    return tf.convert_to_tensor(score_ori - score_next), sent_next

# To store reward history of each episode

# To store average reward history of last few episodes
def env_reward(loss_ori, sent, label, next_token):
    next_state_ids = torch.cat([tokenizer_gpt2.encode(sent, return_tensors="pt"), next_token.cpu()], dim=-1)
    sent_next = tokenizer_gpt2.decode(next_state_ids[0], clean_up_tokenization_spaces=True, skip_special_tokens=True)
    cls_score_next = model_cls.predict([sent_next], steps=1, verbose=0)[0]

    loss_next = tf.keras.losses.sparse_categorical_crossentropy(np.array([label]), cls_score_next)[0]
    return loss_ori - loss_next, sent_next

reward_offset = 0.01

for epoch in range(100):
    ds.df_train = ds.df_train.sample(frac=1)
    ep_reward_list = []
    for ix, row in ds.df_train.reset_index().iterrows():
        torch.cuda.empty_cache()
        sent = row['content']
        label = row['label']
        label_name = row['label_name']
        cls_score_ori = model_cls.predict([row['content']], steps=1, verbose=0)[0]
        loss_ori = tf.keras.losses.sparse_categorical_crossentropy(np.array([label]), cls_score_ori)[0]

        avg_reward_list = []
        for step in range(64):
            action = policy(tf.expand_dims(tf.convert_to_tensor(sent), 0), ou_noise)
            next_token = sample_action_gpt(sent, action)
            #reward, sent_next = next_sent_reward(row['content'], sent, label, next_token, args.future_steps, args.beams)
            reward, sent_next = env_reward(loss_ori, sent, label, next_token)

            buffer.record((sent, list(action), float(reward.numpy()), sent_next))
            sent = sent_next
            ep_reward_list.append(reward.numpy())
            avg_reward_list.append(reward.numpy())

        print(ix, label_name, "instance reward==>", np.array(avg_reward_list).mean(), "==>", sent_next)

        if ix > 0 and ix % 8 == 0:
            buffer.learn()
            update_target(target_actor.variables, actor_model.variables, tau)
            update_target(target_critic.variables, critic_model.variables, tau)

        
    print("epoch reward==>", np.array(ep_reward_list).mean())


