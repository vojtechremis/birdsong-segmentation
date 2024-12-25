import os
import sys
from pathlib import Path

sys.path += [str(Path().resolve().parent.parent)]
from utils.Spectrogram import Spectrogram
from utils import Log

PRINT = Log.get_logger()

from dummy_network.data_handlers import SpectrogramDataset


class ClassificationSpectrogramDataset(SpectrogramDataset.SpectrogramDataset):
    """
    Class inherited from SpectrogramDataset <- Dataset. This class handles classification task, that means using six complexity
    grades (0 - 4) instead of number of segments or motifs.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx):
        # Get the ID of the current sample
        sample_id = self.sample_ids[idx]
        spectrum_path = os.path.join(self.spectrogram_dir, f'{sample_id}.npz')
        values = Spectrogram(spectrum_path).values

        # Get the corresponding label values
        label_entry = self.label_df[self.label_df['sample_id'] == sample_id].iloc[0]

        complexity = label_entry['complexity']
        segments = label_entry['segments']
        motifs = label_entry['motifs']

        peak_frequency = label_entry['peak_frequency']

        # Apply input transformations
        if self.transform:
            values = self.transform(values)

        return values, complexity