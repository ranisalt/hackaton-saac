import numpy as np
from neupy import layers, storage, algorithms
from neupy.exceptions import StopTraining

from agent import SillyWalker, Action, create_net

net = create_net()

storage.load(
    net,
    'nets/net',
)

walker = SillyWalker()

while not walker.done:
    s = np.array([list(walker.state) + [1]])
    prediction = net.predict(s)[0]

    walker.step(Action(*prediction))

    walker._env.render()
