# Multiagent Multi-Objective Reinforcement Learning

Framework to support learned task allocation to a group of agents 
cooperating to complete a set fo tasks. The framework supports two 
scenarios. 
* Agent cooperation to complete tasks in a non-interacting environment
* Agent cooperation to complete tasks in an environment where agents interact with each other

Tasks are formulated as deterministic finite automata (DFA). Agents then
learn how to complete the tasks in parallel, i.e. there is no restriction
for tasks to be completed in sequence. At the same time an allocation 
function approximation is also learned which contributes a deterministic
task allocation to an agent.

Agents are modelled using an Actor-Critic model with the (optional) addition of an 
LSTM cell to learn sub-task sequencing better. 

## Installation

Activate a new virtual environment. 

Install project with ```pip3 install -e .``` from project root directory.
A series of ```gym.Env``` environments will be installed which inherit from
the ```gym-minigrid``` environment originally found at:
https://github.com/maximecb/gym-minigrid. Additional environments can be found 
in ```a2c_team_tf/envs```, and wrappers in ```a2c_team_tf/utils/obs_wrapper.py```.

Requirements: 
* TensorFlow 2.x
* NumPy
* Matplotlib
* OpenAI Gym
* Python 3.5+

For the teamgrid example to work, the project below is also required:
* https://github.com/mila-iqia/teamgrid

## Examples

Examples of different environment implementations can be found in ```/examples``` 

The first is a cooperative ```CartPole-v0```
environment, where there are N agents and M tasks to be completed. This is an
example of where non-interacting agents learn how to complete a set of 
tasks. Real-world examples of this could be independent warehousing/plant
configurations where a set of tasks needs to be distributed to a set of agents.
This script is implemented in ```/examples/cartpole_ex.py```


The second implementation is an example of where agents interact in a single
environment to learn how to complete a set of tasks. The environment
is an augmented version of the ```gym-minigrid``` environment
In this setting there
are a number of challenges to overcome, including, how the agents learn
to resolve conflicts such as moving to same square, or trying to pick up 
the same object. This script can be found in ```/examples/team_grid_ex.py```

## Specifying Tasks

Tasks are specified using DFA. As we consider multitask allocation for concurrent tasks
we are actually interested in the cross product DFA. The base class for the product DFA (xDFA)
and DFA can be found in ```a2c_team_tf/utils/dfa.py```.

There are a number of ways to specify a DFA. The most important thing to note is how the transition 
function for the DFA is calculated. To determine the next state the DFA uses 
```self.next(data, agent)```. Here data could be anything, a ```gym.Env```, or a dictionary
of attributes for example. It is general enough to describe what ever events which need to be calculated
for task progress.

An example of DFA construction can be found in ```/examples/cartpole_ex.py```.
States of the DFA can be described using a class object which inherits the 
```DFAStates``` class. 
```python
class MoveToPos(DFAStates, ABC):
    def __init__(self):
        self.init = "I"
        self.position = "P"
        self.fail = "F"
```
A transition function, in the context of ```CartPole-v0``` which specifies the
cart going left 0.5 units, can be specified as follows:
```python
def go_left_to_pos(data, _):
    if data['state'][0] < -0.5 and not data['done']:
        return "P"
    elif data['done']:
        return "F"
    else:
        return "I"
```
The data object which the function takes as an input is a dictionary, in this
case, with attributes: 
```python
data = {"state": _, "reward": _, "done": _}
```
The DFA can be constructed with a funtion, or scripted, as follows
```python
def make_move_left_to_pos():
    dfa = DFA(start_state="I", acc=["P"], rej=["F"])
    dfa.states = MoveToPos()
    dfa.add_state(dfa.states.init, go_left_to_pos)
    dfa.add_state(dfa.states.position, finished_move)
    dfa.add_state(dfa.states.fail, failed)
    return dfa
```
Finally a cross product DFA which incorporates a number of tasks can be constructed
using
```python
left = make_move_to_left_pos()
...
dfas = [CrossProductDFA(num_tasks=num_tasks, dfas=[right, left], agent=agent) for agent in range(num_agents)]
```

## Tests

To understand the mechanics of the implementation there are a number of tests
which can be run. These are included in ```a2c_team_tf/tests```. Tests on the
```CartPole-v0``` environment using the independent environment library 
```a2c_team_tf/lib/lib_mult_env.py``` can be examined in the script
```cartpole_tests.py```. Conversely, the interacting environment library, 
```a2c_team_tf/lib/tf_a2c_base.py``` is tested in the ```teamgrid_test.py``` 
script. 

## Models

The neural network actor-critic models, are modelled using Keras layers 
in a Sub-class tensorflow model. A number of models are implemented in 
```a2c_team_tf/nets/base.py```. Depending on the architecture required,
this could included an actor-critic model with a basic shared common dense
layer and actor (action) and critic (task performance evaluation) outputs.
Additionally, there is an independent network setup where separate actor, and
critic networks can be learned. Finally, there is a network with a shared 
recurrent network layers which separates into a deep actor network, and critic
network respectively. The number of action in the actor output layer is 
defined using the ```env.action_space.n```. The number of critic outputs
is the number of tasks + 1 (agent cost/reward value).

## Visualisation

Given some learned model, a rendering of the learned allocation policy can 
be run using ```a2c_team_tf/utils/visualisation.py```. 

Data is stored asynchronously when training using ```AsyncWriter``` 
from ```a2c_team_tf/utils/data_capture```.

