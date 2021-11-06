# Installation

Activate a new virtual environment. 

Install project with ```pip3 install -e .``` from project root directory. 

To run unit tests use ```python3 a2c_team_tf/tests/unittests.py``` from project root. 

Rendering the environment, and displaying reward debug values van be controlled on 
line 17 with:
```Python
render_env = False
print_rewards = False
```

[ ] - minigrid wrapped environment still unstable needs some work
[X] - cartpole multi-agent task allocation works and appears to reasonably learn to solve 
      a task, and also learns an allocation
