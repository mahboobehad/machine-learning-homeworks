# In the name of God
import re

import numpy as np


def bipolar(x, theta):
    return -1 if x < theta else x if x == theta else 1


def binary(x):
    return -1 if x < 0 else 1


def generate_data_set_from_file(file_path, target_character='A', input_map=lambda x: 0 if x == '.' else 1):
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
        data.append(np.array(list(map(input_map, map_))))
    re.purge()
    return data, target


def geometric_series(a_0, q):
    while True:
        yield a_0
        a_0 *= q


def linear_neighbourhood(center, radius):
    return [center + r for r in range(-radius, radius + 1)]


def rectangular_neighbourhood(center, radius):
    return [center + i + j for i in range(-radius, radius + 1) for j in range(-radius, radius + 1)]


def data_to_map(data):
    units_repr = str()
    for i in range(data.shape[0]):
        if data[i] == 0:
            units_repr += '. '
        else:
            units_repr += '# '
        if (i + 1) % 7 == 0:
            units_repr += '\n'
    return units_repr.lstrip()


