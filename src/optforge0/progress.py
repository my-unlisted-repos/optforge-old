import time
from collections.abc import Iterable


class SimpleProgress:
    def __init__(self, iterable:Iterable, nprints = 1000, enable=True):
        self.sequence = list(iterable)
        self.len = len(self.sequence)
        self.nprints = nprints
        self.enable = enable

    def __iter__(self):
        start = time.time()
        for i, item in enumerate(self.sequence):
            yield item
            if self.enable and i % (max(1, self.len // self.nprints)) == 0 or i == self.len-1:
                time_passed = time.time() - start
                if i == 0: print(f"{i} / {self.len}", end = '          \r')
                else: 
                    ops_per_sec = i / max(1e-6, time_passed)
                    remaining = (self.len - i) / ops_per_sec
                    if i == self.len-1: i = i + 1
                    print(f"{i}/{self.len} | {time.time() - start:.2f}s/{remaining:.2f}s", end = '          \r')
        if self.enable: print()