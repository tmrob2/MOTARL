from multiprocessing import Process, Pipe
import os
import numpy as np


def worker(conn, path, num_agents, num_tasks):
    while True:
        cmd, data = conn.recv()
        # append the data to a file
        if cmd == 'write':
            data_2d = np.reshape(data, (1, num_agents * (1  + num_tasks)))
            with open(path, "ab") as f:
                np.savetxt(f, data_2d, delimiter=',')
        else:
            raise NotImplementedError


class AsyncWriter:
    def __init__(self, fname, num_agents, num_tasks):
        self.fname = fname
        self.path = self.path = \
            os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..', 'data'))
        self.abs_fname = f'{self.path}/{self.fname}.csv'
        local, remote = Pipe()
        self.locals = []
        self.locals.append(local)
        # delete the contents of the file before beginning learning and writing
        open(self.abs_fname, "w").close()
        p = Process(target=worker, args=(remote, self.abs_fname, num_agents, num_tasks))
        p.daemon = True
        p.start()
        remote.close()

    def write(self, data):
        self.locals[0].send(('write', data))

