from abc import ABC
import copy
import unittest
from a2c_team_tf.nets.base import ActorCritic
import tensorflow as tf
import tqdm
from a2c_team_tf.lib import motaplib
from gym_minigrid.minigrid import *
from gym.envs.registration import register
from a2c_team_tf.utils.dfa import DFA, DFAStates, CrossProductDFA
from a2c_team_tf.environments.minigrid_wrapper import convert_to_flat_and_full, ObjRoom
import collections
import statistics

max_steps_per_episode = 1000
seed = 103
render_env = False
print_rewards = False
# tf.random.set_seed(seed)
# np.random.seed(seed)
register(
    id="empty-room-5x5-v0",
    entry_point='a2c_team_tf.environments.minigrid_wrapper:EmptyRoom5x5',
    max_episode_steps=max_steps_per_episode
)

register(
    id="obj1-room-3x3-v0",
    entry_point='a2c_team_tf.environments.minigrid_wrapper:OneKeyRoom3x3',
    max_episode_steps=max_steps_per_episode
)

register(
    id="obj1-room-2x2-v0",
    entry_point='a2c_team_tf.environments.minigrid_wrapper:OneKeyRoom2x2',
    max_episode_steps=max_steps_per_episode
)
env1 = gym.make('obj1-room-2x2-v0')
env1_ = convert_to_flat_and_full(env1)
env2 = gym.make('obj1-room-2x2-v0')
env2_ = convert_to_flat_and_full(env2)
envs = [env1_, env2_]


# Very simple task to pick up a key
class PickupObjStates(DFAStates, ABC):
    def __init__(self):
        self.init = "I"
        self.carrying = "C"


class MoveKeyStates(DFAStates, ABC):
    def __init__(self):
        self.init = "I"
        self.carrying = "C"
        self.deliver = "D"
        self.fail = "F"


def pickup_key(env: ObjRoom):
    """If the agent is not carrying a key, then picks up a key"""
    if env is not None:
        next_state = "C" if isinstance(env.carrying, Key) else "I"
    else:
        next_state = "I"
    return next_state


def pickup_ball(env: ObjRoom):
    """If the agent is not carrying a ball, then it picks up a ball"""
    if env is not None:
        next_state = "C" if isinstance(env.carrying, Ball) else "I"
    else:
        next_state = "I"
    return next_state


def carrying(env: ObjRoom):
    """If the agent is carrying a key then an agent must continue to carry the key, unless it is
    at the drop off coordinate"""
    if env.carrying is None:
        if np.array_equal(env.agent_pos, np.ndarray([1, 1])):
            return "D"
    else:
        return "C"


def finish(_):
    return "C"


def deliver(_):
    return "D"


def fail(_):
    return "F"


def make_key_dfa():
    """Task: Pick up a key move it to 1,1, then go to the goal state"""
    dfa = DFA(start_state="I", acc=["D"], rej=["F"])
    dfa.states = MoveKeyStates()
    dfa.add_state(dfa.states.init, pickup_key)
    dfa.add_state(dfa.states.carrying, carrying)
    dfa.add_state(dfa.states.deliver, deliver)
    dfa.add_state(dfa.states.fail, fail)
    dfa.start()
    return dfa


def make_pickup_key_dfa():
    """Task pick up a key"""
    dfa = DFA(start_state="I", acc=["C"], rej=[])
    dfa.states = PickupObjStates()
    dfa.add_state(dfa.states.init, pickup_key)
    dfa.add_state(dfa.states.carrying, finish)
    dfa.start()
    return dfa


def make_pickup_ball_dfa():
    """Task: pick up a ball"""
    dfa = DFA(start_state="I", acc=["C"], rej=[])
    dfa.states = PickupObjStates()
    dfa.add_state(dfa.states.init, pickup_ball)
    dfa.add_state(dfa.states.carrying, finish)
    dfa.start()
    return dfa


# Parameters
step_rew0 = 10
task_prob0 = 0.8
N_AGENTS, N_TASKS, N_OBJS = 2, 2, 2
gamma = 1.0
mu = 1.0 / N_AGENTS  # fixed even probability of allocating each task to each agent
lam = 1.0
chi = 1.0
c = step_rew0
e = task_prob0  # task reward threshold
num_actions = env1.action_space.n
num_hidden_units = 128
models = [ActorCritic(num_actions, num_hidden_units, N_TASKS, name="AC{}".format(i)) for i in range(N_AGENTS)]
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

"""
Tests can be run as is, or can be modified to render environments, or print certain characteristics
of the test. 
"""


class TestModelMethods(unittest.TestCase):
    # implement a test for converting training data to correct tensor shapes
    # two environment step test, and extract v_pi,ini
    def _test_environment_init(self):
        state = env1_.reset()
        print("state", state)
        assert env1_.action_space.n == 7
        return True

    def _test_dfa(self):
        assert make_key_dfa().handlers['C'] == carrying

    def _test_action_sample_from_model(self):
        # randomly select an initial action
        print()
        print("-----------------------------------")
        print("     testing action selection      ")
        print("-----------------------------------")
        num_models = N_AGENTS

        task1 = make_pickup_key_dfa()
        task2 = make_pickup_ball_dfa()
        prod_dfa = CrossProductDFA(num_tasks=N_TASKS, dfas=[task1, task2])
        dfas = [copy.deepcopy(prod_dfa) for _ in range(N_AGENTS)]

        # Set the env config

        # Construct the MOTAP environment
        motap = motaplib.TfObsEnv(envs, models, dfas, N_TASKS, N_AGENTS, render_env, print_rewards)
        # select a random action from the model
        for i in range(num_models):
            state = envs[i].reset()
            done = False
            while not done:
                #env1_.render('human')
                state = tf.expand_dims(state, 0)
                action_logits_t, values = models[i](state)
                action = tf.random.categorical(action_logits_t, 1)[0, 0]
                state, reward, done = motap.tf_env_step(action, i)
                # print("state", state['image'])
        return True

    def test_episode(self):
        """
        Given a set of tasks, complete all the tasks in one episode. This will
        be a randomly accomplished because the model has not yet been trained
        """
        print()
        print("-----------------------------------")
        print("           testing episode         ")
        print("-----------------------------------")
        # Set a single environment
        num_models = 1

        # Construct a set of tasks for the agent to complete
        task1 = make_pickup_key_dfa()
        task2 = make_pickup_ball_dfa()
        prod_dfa = CrossProductDFA(num_tasks=N_TASKS, dfas=[task1, task2])
        dfas = [prod_dfa] * num_models

        # Set the configuration of the test

        # Construct the MOTAP environment
        motap = motaplib.TfObsEnv(envs, models, dfas, N_TASKS, N_AGENTS, render_env, print_rewards)

        # Set the initial state of the environment
        i_state = motap.envs[0].reset()
        action_probs, values, rewards = motap.run_episode(i_state, 0, max_steps_per_episode)

        # Print the returns for the episode
        # print("values: \n{}".format(values))
        # print("rewards: \n{}".format(rewards))
        return True

    def _test_move_key_episode(self):
        num_models = 1  # len(models)
        dfas = [make_key_dfa()]
        motap = motaplib.TfObsEnv(envs, models, dfas, True, False)
        for i in range(num_models):
            i_state = motap.envs[i].reset()
            print(i_state)
            action_probs, values, rewards = motap.run_episode(i_state, i, max_steps_per_episode)
            # print("action probs: \n{}".format(action_probs))
            print("values: \n{}".format(values[0]))
            print("rewards: \{}".format(rewards[-1]))
        return True

    def test_compute_h(self):
        pass

    def test_compute_f(self):
        pass

    def test_expected_returns(self):
        """function to test the expected returns for an episode, an important step before
        calculating the loss"""
        print()
        print("-----------------------------------")
        print("           testing returns         ")
        print("-----------------------------------")
        num_models = 2
        task1 = make_pickup_key_dfa()
        task2 = make_pickup_ball_dfa()
        prod_dfa = CrossProductDFA(num_tasks=N_TASKS, dfas=[task1, task2])
        dfas = [copy.deepcopy(prod_dfa) for _ in range(num_models)]

        # Construct the MOTAP environment
        motap = motaplib.TfObsEnv(envs, models, dfas, N_TASKS, N_AGENTS, render_env, print_rewards)

        for i in range(num_models):
            initial_state = tf.constant(envs[i].reset(), dtype=tf.float32)
            action_probs, values, rewards = motap.run_episode(initial_state, i, max_steps_per_episode)
            returns = motap.get_expected_returns(rewards, gamma, N_TASKS, False)
            #print("values: \n".format(values))
            print("returns: \n{}".format(returns[0]))
        return True

    def test_compute_loss(self):
        print()
        print("-----------------------------------")
        print("           testing loss            ")
        print("-----------------------------------")
        num_models = 2
        task1 = make_pickup_key_dfa()
        task2 = make_pickup_ball_dfa()
        prod_dfa = CrossProductDFA(num_tasks=N_TASKS, dfas=[task1, task2])
        dfas = [copy.deepcopy(prod_dfa) for _ in range(num_models)]

        # Set the configuration of the test

        # Construct the MOTAP environment
        motap = motaplib.TfObsEnv(envs, models, dfas, N_TASKS, N_AGENTS, render_env, print_rewards)

        # storage per model - the issue with using lists is that the don't perform the way they
        #                     are expected to in tf.function. We should look at some optimisations when the algo
        #                     is working as expected
        action_probs_l = []
        values_l = []
        rewards_l = []
        returns_l = []
        loss_l = []  # the loss storage for an agent

        for i in range(num_models):
            initial_state = tf.constant(envs[i].reset(), dtype=tf.float32)
            action_probs, values, rewards = motap.run_episode(initial_state, i, max_steps_per_episode)
            returns = motap.get_expected_returns(rewards, gamma, N_TASKS, False)
            print("returns: \n{}".format(returns[0]))
            print(f"Agent: {i} returns shape: {returns.shape}")
            print(f"Agent: {i}: values shape: {values.shape}")
            # Append tensors to respective lists
            action_probs_l.append(action_probs)
            values_l.append(values)
            rewards_l.append(rewards)
            returns_l.append(returns)

        ini_values = tf.convert_to_tensor([x[0, :] for x in values_l])

        for i in range(num_models):
            # get loss
            values = values_l[i]
            returns = returns_l[i]
            ini_values_i = ini_values[i]
            loss = motap.compute_loss(action_probs_l[i], values, returns, ini_values, ini_values_i, lam, chi, mu, e, c)
            loss_l.append(loss)
        print(f"loss: {loss_l}")
        return True

    def test_train_step(self):
        print()
        print("-----------------------------------")
        print("           testing train step      ")
        print("-----------------------------------")
        num_models = 2
        task1 = make_pickup_key_dfa()
        task2 = make_pickup_ball_dfa()
        prod_dfa = CrossProductDFA(num_tasks=N_TASKS, dfas=[task1, task2])
        dfas = [copy.deepcopy(prod_dfa) for _ in range(num_models)]

        # Set the configuration of the test

        # Construct the MOTAP environment
        motap = motaplib.TfObsEnv(envs, models, dfas, N_TASKS, N_AGENTS, render_env, print_rewards)
        motap.train_step(optimizer, gamma, max_steps_per_episode, N_TASKS, lam, chi, mu, e, c)
        return True

    def test_train(self):
        print()
        print("-----------------------------------")
        print("           testing train           ")
        print(""" 
        trains a 2 x 2 environment with a 
        key and a ball, there a two agents and 
        two tasks. An agents mission is to pick 
        up both the key and the ball        """)
        print("-----------------------------------")
        num_models = 2
        task1 = make_pickup_key_dfa()
        task2 = make_pickup_ball_dfa()
        prod_dfa = CrossProductDFA(num_tasks=N_TASKS, dfas=[task1, task2])
        dfas = [copy.deepcopy(prod_dfa) for _ in range(num_models)]

        # Set the configuration of the test
        max_episodes = 100
        min_episodes_criterion = 10
        reward_threshold = 10

        # Construct the MOTAP environment
        motap = motaplib.TfObsEnv(envs, models, dfas, N_TASKS, N_AGENTS, render_env, print_rewards)
        # Keep last episodes reward
        episodes_reward: collections.deque = collections.deque(maxlen=min_episodes_criterion)

        with tqdm.trange(max_episodes) as t:
            for tt in t:
                episode_reward = int(motap.train_step(optimizer, gamma, max_steps_per_episode, N_TASKS, lam, chi, mu, e, c))
                episodes_reward.append(episode_reward)
                running_reward = statistics.mean(episodes_reward)
                t.set_description(f"Episode {tt}")
                t.set_postfix(episode_reward=episode_reward, running_reward=running_reward)
                # Show average episode reward every 10 episodes
                # if tt % 10 == 0:
                #    print(f'Episode {tt}: average reward: {running_reward}')


                if running_reward < reward_threshold and tt >= min_episodes_criterion:
                    break
            print(f'\nSolved at episode {tt}: average reward: {running_reward:.2f}!')


if __name__ == '__main__':
    unittest.main()
