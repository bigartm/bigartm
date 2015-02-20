import os
import psutil
import subprocess
import time
import threading


def subprocess_memory(pid=None):
    if pid is None:
        pid = os.getpid()

    rss_memory = 0
    vms_memory = 0

    try:
        pp = psutil.Process(pid)
    except psutil.Error:
        return 0.0, 0.0

    descendants = list(pp.get_children(recursive=True))
    descendants = descendants + [pp]

    #calculate and sum up the memory of the subprocess and all its descendants
    for descendant in descendants:
        try:
            mem_info = descendant.get_memory_info()

            rss_memory += mem_info[0]
            vms_memory += mem_info[1]
        except psutil.Error:
            pass

    return float(rss_memory) / (2 ** 20), float(vms_memory) / (2 ** 20)


class ResourceTracker(object):

    def _get_current_seconds(self):
        _, _, _, _, elapsed_time = os.times()
        return elapsed_time

    def _get_current_mem(self):
        if self.pid > 0:
            mem_mb, _ = subprocess_memory(self.pid)
            return mem_mb
        else:
            return 0

    def __init__(self, pid=None, timeout=1):
        self.start_time = None
        self.tick_times = []
        self.tick_mem = []
        self.pid = pid if pid is not None else os.getpid()
        self.timeout = timeout
        self.tick_thread = None
        self.finished = None

    def __enter__(self):
        self.start_time = self._get_current_seconds()
        self.finished = False
        self.tick_thread = threading.Thread(target=ResourceTracker.tick_thread_func, args=(self,))
        self.tick_thread.start()
        return self

    def __exit__(self, type, value, traceback):
        self.tick()
        self.finished = True

    def set_pid(self, pid):
        self.pid = pid

    def tick_thread_func(self):
        while not self.finished:
            self.tick()
            time.sleep(self.timeout)

    def tick(self):
        time_from_start = self._get_current_seconds() - self.start_time
        self.tick_times.append(time_from_start)
        self.tick_mem.append(self._get_current_mem())
        self.elapsed_time = time_from_start

    def report(self):
        return {
            'total_time': self.elapsed_time,
            'max_memory_mb': max(self.tick_mem),
            'ticks': {
                'time': self.tick_times,
                'memory_mb': self.tick_mem,
            }
        }


def track_cmd_resource(cmd, timeout=1):
    with ResourceTracker(pid=-1, timeout=timeout) as tracker:
        p = subprocess.Popen(cmd)
        tracker.set_pid(p.pid)
        p.wait()
    return p.returncode, tracker

