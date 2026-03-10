import random

import numpy as np


def set_seed(seed):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
