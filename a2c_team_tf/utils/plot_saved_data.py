import click
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd

data = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..', 'data'))
figures = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..', 'figures'))
dfname = "exp1-cartpole-learning"
dfname2 = "exp1-cartpole-alloc"
title1 = "Learning Performance"
title2 = "Agent 1 Task Allocation"
title3 = "Agent 2 Task Allocation"
title4 = "Task Performance"
sfname = "exp1-cartpole-plots"
title_size = 16
tick_font_size = 12
ival = 100  # sampling interval
f_len = 8000
df = pd.read_csv(f'{data}/{dfname}.csv', delimiter=',', header=None, names=["A1", "A1T1", "A1T2", "A2", "A2T1", "A2T2"])
df2 = pd.read_csv(f'{data}/{dfname2}.csv', delimiter=',', header=None, names=["A1T1", "A1T2", "A2T1", "A2T2"])
fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(15, 4))
df.iloc[:f_len:ival, :].filter(items=['A1', 'A2']).plot(ax=axes[0])
df2.iloc[:f_len:ival, :].filter(items=["A1T1", "A1T2"]).plot(ax=axes[1])
df2.iloc[:f_len:ival, :].filter(items=["A2T1", "A2T2"]).plot(ax=axes[2])
df.iloc[:f_len:ival, :].filter(items=['A1T2', 'A2T1']).plot(ax=axes[3])
axes[0].set_xlabel('Episodes', size=title_size)
axes[0].set_ylabel('Avg Reward', size=title_size)
axes[0].set_title(f'{title1}', size=title_size)
axes[0].legend(loc='lower right')
axes[1].set_xlabel('Episodes', size=title_size)
axes[1].set_ylabel('Allocation Probability', size=title_size)
axes[1].set_title(f'{title2}', size=title_size)
axes[1].legend(loc='lower right')
axes[2].set_xlabel('Episodes', size=title_size)
axes[2].set_ylabel('Allocation Probability', size=title_size)
axes[2].set_title(f'{title3}', size=title_size)
axes[2].legend(loc='lower right')
axes[3].set_xlabel('Episodes', size=title_size)
axes[3].set_ylabel('Success Rate (%)', size=title_size)
axes[3].set_title(f'{title4}', size=title_size)
axes[3].legend(loc='lower right')
for i in range(axes.shape[0]):
    for tick in axes[i].xaxis.get_major_ticks():
        tick.label.set_fontsize(tick_font_size)
    for tick in axes[i].yaxis.get_major_ticks():
        tick.label.set_fontsize(tick_font_size)
fig.tight_layout()
#plt.show()
plt.savefig(f'{figures}/{sfname}.png', dpi=300, format='png', padinches=0)

