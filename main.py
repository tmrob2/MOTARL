import collections
import motaplib
import tqdm
from utils import Action1, Coord
from environment import Environment
from abc import ABC
from model import ActorCritic
import tensorflow as tf
import statistics

step_rew0 = 10
task_prob0 = 0.8
NUMAGENTS, TASKS, GRIDSIZE, N_OBJS = 2, 1, (5, 5), 3
# Discount factor for future rewards
gamma = 1.0
mu = 1.0 / NUMAGENTS # fixed even probability of allocating each task to each agent
lam = 1.0
chi = 1.0
c = step_rew0
e = task_prob0 # task reward threshold


class Env1(Environment, ABC):
    # todo convert the output of these into probability distributions
    # todo act should also return a word
    def act(self, action):
        x, y = self.position.x, self.position.y
        if action == Action1.FORWARD:
            self.position.set(x, y + 1)
        elif action == Action1.LEFT:
            self.position.set(x - 1, y)
        elif action == Action1.DOWN:
            self.position.set(x, y - 1)
        elif action == Action1.RIGHT:
            self.position.set(x + 1, y)
        elif action == Action1.PICK:
            self.position.set(x, y)
        elif action == Action1.PLACE:
            self.position.set(x, y)
        self.energy += self.STEP_VALUE


env1 = Env1(grid_size=GRIDSIZE, n_objs=N_OBJS, start_energy=1.0, start_location=Coord(2, 2), num_tasks=TASKS, name="env1")
env2 = Env1(grid_size=GRIDSIZE, n_objs=N_OBJS, start_energy=2.0, start_location=Coord(2, 3), num_tasks=TASKS, name="env2")
num_actions = len(Action1) # 2
num_hidden_units = 128
models = [ActorCritic(num_actions, num_hidden_units, TASKS, name="AC{}".format(i)) for i in range(NUMAGENTS)]
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

min_episodes_criterion = 1
max_episodes = 2
max_steps_per_episode = 1000
envs = [env1, env2]
motap = motaplib.TfObsEnv(envs, models)


# Cartpole-v0 is considered solved if average reward is >= 195 over 100
# consecutive trials
reward_threshold = 195
running_reward = 0

# Keep last episodes reward
episodes_reward: collections.deque = collections.deque(maxlen=min_episodes_criterion)

with tqdm.trange(max_episodes) as t:
  for i in t:
    #initial_state = tf.constant(env.reset(), dtype=tf.float32)
    #episode_reward = int(train_step(
    #    initial_state, models, optimizer, gamma, max_steps_per_episode))
    episode_reward = int(motap.train_step(
        optimizer,
        gamma,
        max_steps_per_episode,
        TASKS,
        lam,
        chi,
        mu,
        e
    ))

    episodes_reward.append(episode_reward)
    running_reward = statistics.mean(episodes_reward)

    t.set_description(f'Episode {i}')
    t.set_postfix(
        episode_reward=episode_reward, running_reward=running_reward)

    # Show average episode reward every 10 episodes
    #if i % 10 == 0:
      #print(f'Episode {i}: average reward: {running_reward}')

    if running_reward > reward_threshold and i >= min_episodes_criterion:
        break

print(f'\nSolved at episode {i}: average reward: {running_reward:.2f}!')