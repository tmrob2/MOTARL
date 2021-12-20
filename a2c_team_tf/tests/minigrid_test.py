# thoughts: we can engineer the reward further by constructing a reward function
# rewarding steps in multitask completion, i.e. if there are two tasks then the
# reward function looks like 0.5 - 0.9 * steps / (max_steps / 2). This is a one off
# reward for completing one task and two task completion will obviously be
# 1 - 0.9 * steps / max_steps, i.e. a reward of one for accomplishing both tasks -
# a penalty for the number of steps taken to get here.

"""
Consecutive task DFA environment
"""

import collections
import copy
import numpy as np
import tensorflow as tf
import gym
import tqdm
from a2c_team_tf.utils import obs_wrapper
from a2c_team_tf.nets.base import Actor, Critic, ActorCrticLSTM
from a2c_team_tf.lib.tf2_a2c_base import Agent
from a2c_team_tf.utils.dfa import DFAStates, DFA, CrossProductDFA
from abc import ABC
from a2c_team_tf.envs.minigrid_fetch_mult import MultObjNoGoal
from a2c_team_tf.utils.parallel_envs import ParallelEnv
from a2c_team_tf.utils.env_utils import make_env


# env = gym.make()
env_key = 'Mult-obj-4x4-v0'
seed = 44
max_steps_per_update = 12
np.random.seed(seed)
tf.random.set_seed(seed)
min_episode_criterion = 100
max_episodes = 80000
num_tasks = 2
reward_threshold = 0.95
running_reward = 0
num_procs = 20
recurrence = 4
recurrent = recurrence > 1

# construct DFAs
class PickupObj(DFAStates, ABC):
    def __init__(self):
        self.init = "I"
        self.carrying = "C"

def pickup_ball(env: MultObjNoGoal, _):
    if env.carrying:
        if env.carrying.type == "ball":
            return "C"
        else:
            return "I"
    else:
        return "I"

def pickup_key(env: MultObjNoGoal, _):
    if env.carrying:
        if env.carrying.type == "key":
            return "C"
        else:
            return "I"
    else:
        return "I"

def finished(a, b):
    return "C"

def make_pickup_ball_dfa():
    dfa = DFA(start_state="I", acc=["C"], rej=[])
    states = PickupObj()
    dfa.add_state(states.init, pickup_ball)
    dfa.add_state(states.carrying, finished)
    return dfa

def make_pickup_key_dfa():
    dfa = DFA(start_state="I", acc=["C"], rej=[])
    states = PickupObj()
    dfa.add_state(states.init, pickup_key)
    dfa.add_state(states.carrying, finished)
    return dfa
#############################################################################
#  Construct Environments
#############################################################################
envs = []
for i in range(num_procs):
    envs.append(make_env(env_key=env_key, max_steps_per_episode=max_steps_per_update, seed=seed + 1000 * i))
#############################################################################
#  Initialise data structures
#############################################################################
episodes_reward: collections.deque = collections.deque(maxlen=min_episode_criterion)
model = ActorCrticLSTM(num_actions=envs[0].action_space.n, num_tasks=num_tasks, recurrent=True)
ball = make_pickup_ball_dfa()
key = make_pickup_key_dfa()
xdfa = CrossProductDFA(num_tasks=num_tasks, dfas=[copy.deepcopy(obj) for obj in [key, ball]], agent=0)
e, c, mu, chi, lam = 0.8, 0.85, 1.0, 1.0, 1.0
agent = Agent(envs, model, num_tasks=num_tasks, xdfa=xdfa, one_off_reward=1.0,
              e=e, c=c, mu=mu, chi=chi, lam=lam, gamma=1.0, lr=1e-4,
              num_procs=num_procs, num_frames_per_proc=max_steps_per_update, recurrence=recurrence)

state = agent.tf_reset2()
log_reward = tf.zeros([num_procs, num_tasks + 1], dtype=tf.float32)
actions, observations, values, rewards, masks, state, running_rewards, log_reward = agent.collect_batch2(initial_obs=state, log_reward=log_reward)
returns = agent.get_expected_return2(rewards)
ini_values = values[0, :, :]
advantages = agent.compute_advantages(returns, values, ini_values, ini_values)
# concatenate masks together => masks reshape: T x S -> S x T -> S * T
masks = tf.reshape(tf.transpose(masks), [-1])
actions = tf.reshape(tf.transpose(actions), [-1])
# Concatenate the samples together =>
#   values/rewards/returns reshape: T x S x D -> S x T x D -> (S * T) x D
values = tf.reshape(tf.transpose(values, perm=[1, 0, 2]), [-1, values.shape[-1]])
returns = tf.reshape(tf.transpose(returns, perm=[1, 0, 2]), [-1, values.shape[-1]])
observations = tf.reshape(tf.transpose(observations, perm=[1, 0, 2, 3]), [-1, observations.shape[-2], observations.shape[-1]])
advantages = tf.reshape(advantages, [-1, advantages.shape[-1]])
# destroy flow
advantages = tf.convert_to_tensor(advantages)
returns = tf.convert_to_tensor(returns)
observations = tf.convert_to_tensor(observations)
actions = tf.convert_to_tensor(actions)
print(f"Shapes => observations: {observations.shape}, actions: {actions.shape}"
      f"values: {values.shape}, returns: {returns.shape}, "
      f"masks: {masks.shape}, advantages: {advantages.shape}")

#ii = agent.tf_starting_indexes()
#print(f"indices: {ii}")
#loss = 0
#for t in range(recurrence):
#    ix = ii + t
#    sub_batch_obs = tf.gather(observations, indices=ix)
#    print("sub batch of observations: ", sub_batch_obs.shape)
#    # compute loss
#    mask = tf.expand_dims(tf.cast(tf.gather(masks, indices=ix), tf.bool), 1)
#    action_logits_t, value = agent.model(sub_batch_obs, mask=mask)
#    sb_advantage = tf.gather(advantages, indices=ix)
#    print(f"Action probs: {action_logits_t.shape}, sb_adv: {sb_advantage.shape}")
#    sb_actions = tf.gather(actions, indices=ix)
#    print("actions: ", sb_actions)
#    action_probs_t = tf.nn.softmax(action_logits_t)
#    z = tf.range(action_probs_t.shape[0])
#    print("action probabilities shape ", action_probs_t.shape, "z shape ", z.shape, z)
#    jj = tf.transpose([z, sb_actions])
#    action_probs = tf.gather_nd(action_probs_t, indices=jj)
#    print("action probs: ", action_probs.shape, "advantages: ", tf.squeeze(sb_advantage).shape)
#    actor_loss = tf.math.reduce_mean(tf.math.log(action_probs) * tf.squeeze(sb_advantage))
#    print("actor loss: ", actor_loss)
#    # value = agent.critic(sub_batch_obs, mask=mask)
#    sb_returns = tf.gather(returns, indices=ix)
#    print("value shape: ", tf.squeeze(value).shape, "batch returns shape: ", sb_returns.shape)
#    critic_sb_losses = agent.huber(tf.squeeze(value), sb_returns)
#    print("critic loss shape ", critic_sb_losses.shape, "\ncritic losses ", critic_sb_losses, "\ncritic loss ", tf.nn.compute_average_loss(critic_sb_losses))
#    loss += actor_loss + tf.nn.compute_average_loss(critic_sb_losses)
#loss /= recurrence
#print("loss: ", loss)
