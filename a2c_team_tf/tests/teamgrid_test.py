import collections
import copy

import gym
import numpy as np
import tensorflow as tf
from a2c_team_tf.nets.base import ActorCrticLSTM
from a2c_team_tf.lib.tf2_a2c_base import MORLTAP
from a2c_team_tf.utils.dfa import DFAStates, DFA, CrossProductDFA
from abc import ABC
from a2c_team_tf.envs.team_grid_mult import TestEnv
from a2c_team_tf.utils.env_utils import make_env

env = gym.make('Team-obj-5x5-v0')



