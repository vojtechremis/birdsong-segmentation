import numpy as np
import sys
from pathlib import Path

# Adding parent to sys.path
sys.path += [str(Path().resolve().parent)]
from utils import Log

PRINT = Log.get_logger()


class Mask:
    def __init__(self, values, time_bins=None, frequency_bins=None):
        self.values = values
        self.time_bins = time_bins
        self.frequency_bins = frequency_bins

    def axis_scale(self):
        return self.time_bins is not None and self.frequency_bins is not None

    def is_numeric(self):
        return np.issubdtype(self.values.dtype, np.number)

    def is_binary(self):
        return np.isin(self.values, [0, 1]).all()

    def is_boolean(self):
        return self.values.dtype == np.bool_

    def to_segments(self):
        if all(len(np.unique(self.values[:, col])) == 1 for col in range(self.values.shape[1])):
            return self.values[0, :]
        else:
            PRINT.info('Transforming mask to segments require consistent rows (every column must contain only one unique value).')