# In the name of God
import re

import numpy as np


def ac_func(x):
    return -1 if x < .5 else 0 if -.5 <= x <= .5 else 1


def bipolar(x):
    return -1 if x < 0 else 1


def generate_data_set_from_file(file_path, target_character='A'):
    def read_file():
        with open(file_path) as f:
            lines = f.readlines()
            return ''.join(_ for _ in lines)

    map_pattern = re.compile('[\w:\s\t,]+[.#\s]+', re.I | re.M)
    maps = map_pattern.findall(read_file())

    data = list()
    target = list()

    title_pattern = re.compile('[\w\s,:]+', re.I | re.M)
    map_pattern = re.compile('[.#]+')

    for m in maps:
        m = m.strip()
        title = title_pattern.search(m).group(0).strip()
        target.append(np.array([1]) if target_character == title[~0] else np.array([-1]))
        map_ = map_pattern.findall(m)
        data.append(np.array(list(map(lambda x: 0 if x == '.' else 1, map_))))
    re.purge()
    return data, target
