import threading
import time
from threading import Thread
from collections import deque
import random

import numpy as np
from pyqtgraph import PlotWidget
import pyqtgraph as pg
import sys
from PyQt6.QtWidgets import QApplication
from multiprocessing import Queue
from typing import Optional
from functools import partial
from pyqtgraph.Qt import QtGui, QtWidgets

pg.setConfigOptions(antialias=True)

pg.setConfigOption('foreground', 'k')


class DataCollecter:

    def __init__(self, data_queue, labels=("Data",), maxlen=500, ):
        """
        A lightweight wrapper around a deque for storing and visualizing data.
        :param maxlen: Number of samples after which old values are discarded (oldest at index 0)
        """
        self.data_queue = data_queue
        self.data_deques = [deque(maxlen=maxlen) for label in labels]
        self.data_times = deque(maxlen=maxlen)
        self.labels = labels

        self.start_time = time.time()
        self.plot_app: Optional[ExoDataVisalizer] = None
        self.plot_widget: Optional[RealTimePlotWidget] = None
        self.timer: Optional[pg.QtCore.QTimer] = None

    def launch_plot(self):
        self.plot_app = ExoDataVisalizer(labels=self.labels, data_collecter=self)
        self.plot_widget = self.plot_app.plot_widget
        self.timer = pg.QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(10)  # Update the plot every 50 ms
        self.start_time = time.time()

        t = threading.Thread(target=lambda: self.plot_app.app.exec())
        t.daemon = True
        t.start()
        pass

    def update(self):
        data = None
        while not self.data_queue.empty():
            data = self.data_queue.get()
        if data is None:
            return
        self.data_times.append(time.time() - self.start_time)
        for data_item, data_deque in zip(data, self.data_deques):
            data_deque.append(data_item)
        if self.plot_widget is not None:
            self.plot_widget.update_plot(self.data_times, self.data_deques)

    def passive_update(self):
        if self.plot_widget is not None:
            self.plot_widget.update_plot(self.data_times, self.data_deques)


def generate_signal_data():
    """Generate a random value between -1 and 1."""
    return random.uniform(-1, 1)


class SignalGenerator(Thread):
    def __init__(self, data_queue, N):
        super().__init__()
        self.data_queue = data_queue
        self.N = N
        self.setDaemon(True)
        self.running = True


    def run(self):
        while self.running:
            new_data = (generate_signal_data(), generate_signal_data(), generate_signal_data())
            self.data_queue.put(new_data)
            time.sleep(0.01)  # Adjust this delay as needed

    def stop(self):
        self.running = False


class RealTimePlotWidget(PlotWidget):
    def __init__(self, dim=3, parent=None):
        super().__init__(parent)
        self.setYRange(-20, 20)  # Adjust the y-range as needed
        self.start_time = time.time()
        colors = ['#5975a4', '#cc8963', '#5f9e6e', '#b55d60']
        self.setBackground('#eaeaf2')

        # Adjust font and axis labels
        font = QtGui.QFont()
        font.setPointSize(40)  # Set the font size
        label_style = {'color': '#E1EAEF', 'font-size': '14pt'}
        self.setLabel('left', 'Signal Value', units='', **label_style)
        self.setLabel('bottom', 'Time', units='s', **label_style)
        self.getPlotItem().getAxis('left').setWidth(50)
        self.getPlotItem().getAxis('left').setStyle(tickLength=0)
        self.getPlotItem().getAxis('bottom').setHeight(50)
        self.getPlotItem().getAxis('bottom').setStyle(tickLength=0)
        self.active_plot = 0

        def setYRange(self, x_range):
            self.enableAutoRange(axis='y')
            self.setAutoVisible(y=True)

        self.sigXRangeChanged.connect(setYRange)

        self.curves = [self.plot(pen=pg.mkPen(colors[i], width=2)) for i in range(dim)]

    def update_plot(self, data_times, data_deques):
        plot_data = np.array(data_deques[self.active_plot]).T
        if plot_data.shape[0]<3:
            plot_data = np.vstack([plot_data, *([np.empty_like(plot_data)*np.nan]*(3-plot_data.shape[0]))])
        for curve, curve_data in zip(self.curves, plot_data):
            curve.setData(data_times, curve_data)  # Set the data for the curve item


class ExoDataVisalizer:
    def __init__(self, labels, data_collecter):
        self.app = QApplication(sys.argv)
        self.data_collecter: DataCollecter = data_collecter
        self.w = QtWidgets.QWidget()

        self.plot_widget = RealTimePlotWidget()

        self.w.setWindowTitle('PyQtGraph example')
        hbox = QtWidgets.QHBoxLayout(self.w)
        self.w.vbox = QtWidgets.QVBoxLayout()
        self.w.buttons = [QtWidgets.QPushButton(label) for label in labels]

        for i, button in enumerate(self.w.buttons):
            button.clicked.connect(partial(self.set_active_plot, i))
            self.w.vbox.addWidget(button)
        hbox.addLayout(self.w.vbox)

        hbox.addWidget(self.plot_widget)

        self.w.show()

    def set_active_plot(self, i):
        self.plot_widget.active_plot = i
        self.data_collecter.passive_update()


if __name__ == '__main__':
    data_queue = Queue()
    N = 100  # Number of data points to display

    signal_generator = SignalGenerator(data_queue, N)
    signal_generator.start()
    data_collecter = DataCollecter(data_queue, [str(n) for n in range(N)])
    data_collecter.launch_plot()
    data_collecter.plot_app.app.exec()
    signal_generator.join()
