import numpy as np
from multiprocessing import Process, Queue

def config_model(name, data):

    print('Starting, process {}'.format(name))
    print('computed mean {} is {}.'.format(name, np.mean(data)))
    return None

proces = []

data = np.random.rand(100, 100)

for i in range(100):
    name = 'process{}'.format(i)
    p = Process(target=config_model, args=(name, data))
    p.start()
    proces.append(p)


for i, p in enumerate(proces):
    print('joins {}'.format(i))
    p.join()

print('finished join')
