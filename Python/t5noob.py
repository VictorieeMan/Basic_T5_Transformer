"""Created: 2022-03-25
GitHub Repo: https://github.com/VictorieeMan/Basic_T5_Transformer.git
An attempt at creating an easy to mod and use T5 Transformer Model for the PyTorch framework.
"""

try:
    import os
    import json
    import pickle
    import warnings
    import datetime
    import numpy as np
    import matplotlib.pyplot as plt

    import torch

    from tqdm import tqdm # Progressbar
    # *https://www.pythonpool.com/python-tqdm/

except Exception as e:
    raise Exception("Couldn't load all modules. More info: {}".format(e))

print("Gooday ax shaft!")