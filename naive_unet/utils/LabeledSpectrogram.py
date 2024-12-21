import numpy as np
import sys
from pathlib import Path

sys.path += [str(Path().resolve().parent.parent)]

from utils import Log

PRINT = Log.get_logger()
class LabeledSpectrogram:
    def __init__(self, values, time_bins=None, frequency_bins=None, labels=None):
        self.values = values
        self.time_bins = time_bins
        self.frequency_bins = frequency_bins
        self.labels = labels

    def axis_scale(self):
        return self.time_bins is not None and self.frequency_bins is not None

    def save_as_raw_input(self, filepath):
        values_dict = {
                    'spectrum': self.values,
                    'time_bins': self.time_bins,
                    'frequency_bins': self.frequency_bins,
                    'labels': self.labels
                }

        try:
            np.savez_compressed(
                filepath,
                **values_dict
            )
        except Exception as e:
            PRINT.error(f'Spectrogram object cannot be saved due to following error: \n {e}.')