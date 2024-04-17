"""
A simple (and very inefficient) plotting utility class. Declare a global DataVisualizer, and then you can
easily plot a rolling window of signals of your choice. Don't forget to use the right number of channels!
"""

import matplotlib.pyplot as plt
from collections import deque
import numpy as np
from functools import partial

from matplotlib.animation import FuncAnimation


class DataCollecter:

    def __init__(self, maxlen=500, nchannels=1, plot=True, min_max_scales=(1, 1)):
        """
        A lightweight wrapper around a deque for storing and visualizing data.
        :param maxlen: Number of samples after which old values are discarded (oldest at index 0)
        :param nchannels: Dimensionality of the vector stored
        :param plot: Flag to decide if a real-time plot should be made and kept up to date
        :param min_max_scales: The plotting will multiply the (minimum, maximum) values of the data to plot
        by the corresponding value in this tuple. E.g: (0, 1.1) will have 0 as the minimum always, and have a 10% margin
        on the top.
        """
        self.data_deque = deque(maxlen=maxlen)
        self.data_deque.extend([[0]*nchannels]*maxlen)
        self.plot = plot
        if not plot:
            self.fig = None
            self.lines = None
            self.ax = None
            self.min_max_scales = None
            return
        self.fig = plt.figure()
        self.lines = plt.plot(self.data_deque)
        self.ax = plt.gca()

        self.min_max_scales = min_max_scales
        ani = FuncAnimation(self.fig, partial(self.update), interval=50, save_count=0, blit=True)
        plt.show(block=False)

    def add_data(self, data_in):
        self.data_deque.append(np.array(data_in))


    def update(self, frame):
        arr = np.array(self.data_deque)
        for line, data in zip(self.lines, arr.T):
            line.set_ydata(data)
        self.fig.canvas.draw()
        self.ax.set_ylim([arr.min() * self.min_max_scales[0], arr.max() * self.min_max_scales[1] + np.finfo(float).eps])
        return self.lines

def cur_idx(data, timestep=0.005):
    return int(data.time // timestep)


def subsampled_execution(method, data, subsample_factor, timestep=0.005):
    if cur_idx(data, timestep) % subsample_factor == 0:
        method()
