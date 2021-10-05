from environment import Environment
from utils import Action1, Coord
from abc import ABC
import unittest
from model import ActorCritic
import tensorflow as tf
import motaplib
import numpy as np


#seed = 103
#tf.random.set_seed(seed)
#np.random.seed(seed)
step_rew0 = 10
task_prob0 = 0.8
NUMAGENTS, TASKS, GRIDSIZE, N_OBJS = 2, 1, (10, 10), 3
max_steps_per_episode = 1000
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


env1 = Env1(grid_size=GRIDSIZE, n_objs=N_OBJS, start_energy=1.0, start_location=Coord(4, 4), num_tasks=TASKS)
env2 = Env1(grid_size=GRIDSIZE, n_objs=N_OBJS, start_energy=2.0, start_location=Coord(7, 3), num_tasks=TASKS)
num_actions = len(Action1) # 2
num_hidden_units = 128
models = [ActorCritic(num_actions, num_hidden_units, TASKS, name="AC{}".format(i)) for i in range(NUMAGENTS)]
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)


class TestModelMethods(unittest.TestCase):
    # implement a test for converting training data to correct tensor shapes
    # todo implement a test for an instance of calculating a loss function
    # todo test that there is no flow between A and Advantage grad_theta π(a_t|s_t)*(G - A)
    # two environment step test, and extract v_pi,ini
    # todo test tf function implementation of KL divergence function
    def test_environment(self):
        return True

    def test_tf_action_sampling(self):
        return True

    def _test_episode(self):
        num_models = len(models)
        envs = [env1, env2]
        motap = motaplib.TfObsEnv(envs, models)
        for i in range(num_models):
            initial_state = tf.constant(envs[i].reset(), dtype=tf.float32)
            action_probs, values, rewards = \
                motap.run_episode(initial_state, i, max_steps_per_episode)

        print("action probs: \n{}".format(action_probs))
        print("values: \n{}".format(values))
        #print("rewards: \{}".format(rewards))
        return True

    def test_compute_h(self):
        pass

    def test_compute_f(self):
        pass

    def _expected_rewards(self):
        """function to test what the output of the expected rewards for an episode is"""
        num_models = len(models)
        envs = [env1, env2]
        motap = motaplib.TfObsEnv(envs, models)
        #for i in range(num_models):
        initial_state = tf.constant(envs[0].reset(), dtype=tf.float32)
        action_probs, values, rewards = \
            motap.run_episode(initial_state, 0, max_steps_per_episode)
        returns = motap.get_expected_returns(rewards, gamma, TASKS, False)
        #print("values: \n".format(values))
        #print("returns: \n{}".format(returns))
        return True

    def _compute_loss(self):
        return True

    def test_train_step(self):
        envs = [env1, env2]
        motap = motaplib.TfObsEnv(envs, models)
        motap.train_step(optimizer, gamma, max_steps_per_episode, TASKS, lam, chi, mu, e)


if __name__ == '__main__':
    unittest.main()
