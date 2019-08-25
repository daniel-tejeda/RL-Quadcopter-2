import numpy as np
import matplotlib.pyplot as plt
import copy
import random
from collections import namedtuple, deque
from mpl_toolkits.mplot3d import Axes3D

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size: maximum size of buffer
            batch_size: size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self, batch_size=64):
        """Randomly sample a batch of experiences from memory."""
        return random.sample(self.memory, k=self.batch_size)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, mu, theta, sigma):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state


def plot_sim(title, data):
    fig = plt.figure(figsize = (14,14))
    #ax = fig.add_subplot(211, projection='3d')

    ax1 = plt.subplot2grid((5, 1), (0, 0), rowspan=2, projection='3d')

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title(title)
    ax1.scatter(data['x'], data['y'], data['z'])

    ax2 = plt.subplot2grid((5,1),(2,0))
    ax2.set_title("Rotor speeds")
    ax2.plot(data['time'], data['rotor_speed1'], label='Rotor 1 revolutions / second')
    ax2.plot(data['time'], data['rotor_speed2'], label='Rotor 2 revolutions / second')
    ax2.plot(data['time'], data['rotor_speed3'], label='Rotor 3 revolutions / second')
    ax2.plot(data['time'], data['rotor_speed4'], label='Rotor 4 revolutions / second')
    #ax2.legend()


    ax3 = plt.subplot2grid((5,1),(3,0))
    ax3.set_title("Rewards")
    ax3.plot(data['time'], data['reward'])


"""
    t = np.arange(0.01, 5.0, 0.01)
    s1 = np.sin(2 * np.pi * t)
    s2 = np.exp(-t)
    s3 = np.sin(4 * np.pi * t)

    ax1 = plt.subplot(311)
    plt.plot(t, s1)
    plt.setp(ax1.get_xticklabels(), fontsize=6)

    # share x only
    ax2 = plt.subplot(312, sharex=ax1)
    plt.plot(t, s2)
    # make these tick labels invisible
    plt.setp(ax2.get_xticklabels(), visible=False)

    # share x and y
    ax3 = plt.subplot(313, sharex=ax1, sharey=ax1)
    plt.plot(t, s3)
    plt.xlim(0.01, 5.0)
    plt.show()
"""
