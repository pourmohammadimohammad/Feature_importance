import itertools
from enum import Enum
import numpy as np
import pandas as pd
import socket

class DataUsed (Enum):
    INS = 1
    OOS = 2
    TOTAL = 3

class Estimators (Enum):
    MEAN = 1
    STD = 2
    PI = 3
    PI_2 = 4
    R_2 = 5
    MSE = 6
