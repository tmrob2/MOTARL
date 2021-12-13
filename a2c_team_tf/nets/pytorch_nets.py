import torch
import torch.nn as nn
import torch.functional as F
from torch.distributions.categorical import Categorical
from abc import abstractmethod

def init_params(m):
    class_name = m.__class__.__name__
    if class_name.find("Linear") != -1:
        m.weight.data.normal_(0, 1)
        m.weight_data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)

class ACModel:
    recurrent = False

    @abstractmethod
    def __init__(self, obs_space, action_space):
        ...

    @abstractmethod
    def forward(self, obs):
        ...

class RecurrentACModel(ACModel):
    recurrent = True

    @abstractmethod
    def forward(self, obs, memory):
        ...

    @property
    @abstractmethod
    def memory_size(self):
        ...

class ACModel_(nn.Module, RecurrentACModel):
    def __init__(self, action_space, obs_size, use_memory=False):
        super().__init__()
        self.obs_size = obs_size

        # Use memory
        self.use_memory = use_memory

        # TODO program the LSTM memory cell input

        # Define the actor's model
        self.actor = nn.Sequential(
            nn.Linear(self.obs_size, 64),
            nn.Tanh(),
            nn.Linear(64, action_space.n)
        )

        # Define the critic network
        self.critic = nn.Sequential(
            nn.Linear(self.obs_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1) # TODO update this to the number of output tasks when we are inputing multitask models
        )

        # initilise the parameters
        # self.apply(init_params)

    @property
    def memory_size(self):
        return 2 * self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.obs_size

    def forward(self, x, memory):
        if self.use_memory:
            hidden = (memory[:self.semi_memory_size], memory[:self.semi_memory_size])
            hidden = self.memory_rnn(x, hidden)
            embedding = hidden[0]
            memory = torch.cat(hidden, dim=1)
        else:
            embedding = x

        x = self.actor(embedding)
        dist = Categorical(logits=F.log_softmax(x, dim=1))
        x = self.critic(embedding)
        value = x.squeeze(1)

        return dist, value, memory


class ActorModel(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dims):
        super().__init__()
        self.num_actions = num_actions
        self.fc = nn.Sequential(
            nn.Linear(num_inputs, hidden_dims),
            nn.Tanh(),
            nn.Linear(hidden_dims, hidden_dims),
            nn.Tanh(),
            nn.Linear(hidden_dims, num_actions),
            nn.Softmax(dim=-1)
        )
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # self.to(self.device)

    def forward(self, state):
        x = self.fc(state)
        return x







