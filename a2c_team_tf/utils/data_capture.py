from multiprocessing import Process, Pipe
import os
import numpy as np


def worker(conn, path1, path2, num_agents, num_tasks):
    while True:
        cmd, data = conn.recv()
        # data will be type dictionary with attributes learn, alloc
        # append the data to a file
        if cmd == 'write':
            data_2d_learn = np.reshape(data['learn'], (1, num_agents * (1 + num_tasks)))
            data_2d_alloc = np.reshape(data['alloc'].flatten(), (1, num_agents * num_tasks))
            with open(path1, "ab") as f:
                np.savetxt(f, data_2d_learn, delimiter=',')
            with open(path2, "ab") as f:
                np.savetxt(f, data_2d_alloc, delimiter=',')
        else:
            raise NotImplementedError


class AsyncWriter:
    def __init__(self, fname_learning, fname_alloc, num_agents, num_tasks):
        self.fname_learning = fname_learning
        self.fname_alloc = fname_alloc
        self.path = self.path = \
            os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..', 'data'))
        self.abs_fname_learning = f'{self.path}/{self.fname_learning}.csv'
        self.abs_fname_alloc = f'{self.path}/{self.fname_alloc}.csv'
        local, remote = Pipe()
        self.locals = []
        self.locals.append(local)
        # delete the contents of the file before beginning learning and writing
        open(self.abs_fname_learning, "w").close()
        open(self.abs_fname_alloc, "w").close()
        p = Process(target=worker,
                    args=(remote, self.abs_fname_learning, self.abs_fname_alloc, num_agents, num_tasks))
        p.daemon = True
        p.start()
        remote.close()

    def write(self, data):
        self.locals[0].send(('write', data))

