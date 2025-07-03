import numpy as np
import sys
from pathlib import Path
from torch import Tensor

sys.path += [str(Path().resolve().parent.parent), str(Path().resolve().parent)]

from utils import Log
from utils.Visualization import VisualizationHandler

PRINT = Log.get_logger()


class Spectrogram:
    values = None
    time_bins = None
    frequency_bins = None
    golden_mask_values = None

    def __init__(self, values, time_bins=None, frequency_bins=None, golden_mask_values=None):

        # Filepath expected
        if isinstance(values, str):
            self.load_raw_input(values)
        else:

            self.values = values
            self.time_bins = time_bins
            self.frequency_bins = frequency_bins
            self.golden_mask_values = golden_mask_values

        self.to_cpu()

        if len(self.values.shape) == 3:
            self.values = np.squeeze(self.values, axis=0)

        self.visualization_engine = VisualizationHandler()

    def __call__(self):
        return self

    def to_cpu(self):
        if isinstance(self.values, Tensor):
            self.values = self.values.cpu()

        if self.time_bins is not None and isinstance(self.time_bins, Tensor):
            self.time_bins = self.time_bins.cpu()

        if self.frequency_bins is not None and isinstance(self.frequency_bins, Tensor):
            self.frequency_bins = self.frequency_bins.cpu()

    def axis_scale(self):
        return self.time_bins is not None and self.frequency_bins is not None

    def save_as_raw_input(self, filepath):
        values_dict = {
            'spectrum': self.values,
            'time_bins': self.time_bins,
            'frequency_bins': self.frequency_bins,
        }

        try:
            np.savez_compressed(
                filepath,
                **values_dict
            )
        except Exception as e:
            PRINT.error(f'Spectrogram object cannot be saved due to following error: \n {e}.')

    def load_raw_input(self, filepath):
        try:
            data = np.load(filepath, allow_pickle=True)

            # Values extraction
            self.values = data['spectrum']
            self.time_bins = data['time_bins']
            self.frequency_bins = data['frequency_bins']

            # Closing the file
            data.close()

        except Exception as e:
            PRINT.error(f"Error while loading the spectrogram: {e}")

    def show(self, caption=None):
        return self.visualization_engine.plot_spectrogram(self.time_bins, self.frequency_bins, self.values, caption)
