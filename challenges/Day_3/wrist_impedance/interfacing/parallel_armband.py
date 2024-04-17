from typing import Optional
import serial
import struct
from collections import deque
import threading
import multiprocessing as mp
from interfacing.serial_armband import SerialArmband
import time
import multiprocessing.connection


class ParallelSerialArmband(SerialArmband):
    def __init__(self, queue: mp.Queue, shutdown_conn: mp.connection.Connection,
                 **armband_kwargs):
        super().__init__(**armband_kwargs)
        self.queue = queue
        self.shutdown_conn = shutdown_conn

    def update(self):
        while self.serial.is_open and not self.stopped:
            if self.shutdown_conn.poll():
                self.stop()
                return
            int_values = self.read_data_frame()
            self.queue.put(int_values)


class ParallelSerialArmbandManager:
    def __init__(self, **armband_kwargs):
        self.armband_kwargs = armband_kwargs
        self.data_queue = mp.Queue()
        self.shutdown_receiver, self.shutdown_sender = mp.Pipe(duplex=False)
        self.process = None
    
    def __enter__(self):
        self.process = mp.Process(target=armband_process,
                                  kwargs={"queue": self.data_queue,
                                          "shutdown_connection": self.shutdown_receiver,
                                          "armband_kwargs": self.armband_kwargs},
                                  daemon=True)
        self.process.start()
        return self

    def get_data(self):
        list_out = []
        while not self.data_queue.empty():
            list_out.append(self.data_queue.get_nowait())
        return list_out  # Oldest first

    def __exit__(self, exc_type, exc_value, traceback):
        self.shutdown_sender.send("end")
        time.sleep(1)

    def __del__(self):
        self.shutdown_sender.send("end")
        time.sleep(1)

    def start(self):
        self.__enter__()


def armband_process(queue, shutdown_connection, armband_kwargs):
    print("Entered process!")
    with ParallelSerialArmband(queue, shutdown_connection, **armband_kwargs) as armband:
        try:
            armband.thread.join()
        except KeyboardInterrupt:
            armband.stop()
            time.sleep(1)
        

def main():
    with ParallelSerialArmbandManager(port="COM3", byte_count=960) as manager:
        try:
            old_time = time.time()
            while True:
                data = manager.get_data()
                if len(data) != 0:
                    print(data[0])
                    new_time = time.time()
                    print(new_time-old_time)
                    old_time = new_time
        except KeyboardInterrupt:
            print("Finish Gracefully")
            pass
    time.sleep(2)



if __name__ == '__main__':
    main()
