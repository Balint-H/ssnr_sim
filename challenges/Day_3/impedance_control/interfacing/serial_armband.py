from typing import Optional

import serial
import struct
from collections import deque
import threading

class SerialArmband:
    def __init__(self, port, byte_count, bauderate=115200, frame_memory=5, checksum_func=None, checksum_length=0):
        self.port = port
        self.bauderate = bauderate
        self.byte_count = byte_count
        self.serial: Optional[serial.Serial] = None
        self.task = None
        self.frames = deque(maxlen=frame_memory)
        self.cur_frame = None
        self.checksum_func = checksum_func
        self.checksum_length = checksum_length
        self.thread = threading.Thread(target=self.update)
        self.thread.daemon = True
        self.buffer = b""
        self.stopped=False

    def read_data_frame(self):
        # Read bytes until byte_count + 1 (for the checksum byte) are received
        while len(self.buffer) < self.byte_count + self.checksum_length:
            if self.stopped or not self.serial.is_open:
                return
            incoming_bytes = self.serial.read(self.byte_count + self.checksum_length - len(self.buffer))
            self.buffer += incoming_bytes

        # Calculate the checksum over all the bytes in the data frame
        data = self.buffer
        self.buffer = b""
        if self.checksum_func is None or not self.checksum_func(data):
            # Checksum passed, unpack int16 values
            if self.checksum_length == 0:
                int_values = struct.unpack('<' + 'h' * (self.byte_count // 2), data)
                return int_values

            int_values = struct.unpack('<' +'h' * (self.byte_count // 2), data[:-self.checksum_length])
            return int_values

    def update(self):
        while self.serial.is_open and not self.stopped:

            int_values = self.read_data_frame()
            if int_values is not None:
                if self.cur_frame is not None:
                    self.frames.append(list(self.cur_frame))
                self.cur_frame = int_values

    def start(self):
        print("Connecting...")
        self.serial = serial.Serial(self.port, self.bauderate)
        self.serial.flushInput()
        self.serial.flushOutput()
        self.serial.write('s'.encode())
        self.thread.start()

    def stop(self):
        if self.stopped:
            return
        self.stopped = True
        self.serial.write('p'.encode())
        self.serial.close()

    def get_and_empty_data(self):
        current = list(self.cur_frame) if self.cur_frame is not None else None
        remaining = list(self.frames)
        self.frames.clear()
        self.cur_frame = None
        return current, remaining

    def __enter__(self):
        print("Entered armband")
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()

    def __del__(self):
        self.stop()


def main():
    import time
    import signal
    import numpy as np

    with SerialArmband("COM3", 960) as armband:
        signal.signal(signal.SIGINT, armband.stop)
        try:
            while True:
                current, past = armband.get_and_empty_data()
                if current is None:
                    continue
                current = np.array(current).reshape((12, -1))
                print(current[:, 0])
                print(len(past))
                time.sleep(1)
        except Exception as e:
            print(e)


if __name__ == '__main__':
    main()
