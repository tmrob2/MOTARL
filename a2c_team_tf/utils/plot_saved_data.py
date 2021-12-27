import click
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd

data = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..', 'data'))
figures = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..', 'figures'))
dfname = "data-cartpole"
title = "RL data"
df = pd.read_csv(f'{data}/{dfname}.csv', delimiter=',', header=None, names=["A1", "A1T1", "A1T2", "A2", "A2T1", "A2T2"])
x = np.arange(0, df.shape[0], 1)
plt.figure()
ax = df.plot(y=["A1", "A2"])
ax.set_xlabel('Episodes')
ax.set_ylabel('Avg Reward')
ax.set_title(f'{title}')
ax.legend()
plt.show()
#plt.savefig(f'{figures}/{sfname}', dpi='figure', format='png', padinches=0)