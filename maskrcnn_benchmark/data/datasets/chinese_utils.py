import os
import numpy as np
def load_chars():
    cur_path = os.path.abspath(__file__)
    cur_dir  = os.path.dirname(cur_path)
    return np.load(os.path.join(cur_dir, 'chars.npy')).tolist()
