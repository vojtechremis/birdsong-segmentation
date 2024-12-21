# This scripts collects data from Excel with "a bit" annotated xeno-canto recordings
import os
from tqdm import tqdm
import shutil
import librosa
import sys
from pathlib import Path

sys.path += [str(Path().resolve().parent.parent)]

from utils import SegmentsHandler
from utils.Spectrogram import Spectrogram
from utils import Log

PRINT = Log.get_logger()

# Following steps are being done:
# 1. Extracting only samples with < 50 segments
# 2. Finding and gathering appropriate audio files
# 3. Calculating and saving Vanila spectrogram (letting option to calculate spectrograms with different
# parameters - saved in different folder)
# 4. Saving labels (#segments, #motives) 5. Saving samples information files
# (peak_frequency, subj_rec_quality, xeno_rec_quality)


# Importing modules
import pandas as pd


def calculate_spectrogram(audio_filepath):
    y, sr = librosa.load(audio_filepath, sr=22050)  # 'y' is amplitude, 'sr' is sampling rate

    thresh = None
    NFFT = 1024
    hop_length = 119
    transform_type = None

    time_bins, frequency_bins, spectrum = SegmentsHandler.calculate_spectrogram(y, sr, thresh=thresh, NFFT=NFFT,
                                                                                hop_length=hop_length,
                                                                                transform_type=transform_type)

    return Spectrogram(spectrum, time_bins, frequency_bins)


class CollectAnnotatedXenoData:
    def __init__(self, excel_filepath, annotations_filepath, source_audio_dir):
        excel_data_raw = pd.read_excel(excel_filepath)
        self.excel_data = excel_data_raw[excel_data_raw['segments'] < 50]

        # Setting directory names
        self.audio_dir = '/Volumes/T7/Datasets/XenoCanto/audio'
        self.labels_dir = '/Volumes/T7/Datasets/XenoCanto/labels'
        self.spectrograms_dir = '/Volumes/T7/Datasets/XenoCanto/spectrograms'

        # Creating directories
        for directory in [self.audio_dir, self.labels_dir, self.spectrograms_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)

        self.source_audio_dir = [os.path.join(source_audio_dir, subdir) for subdir in ['A', 'B', 'C', 'D']]

        self.suffix = 'mp3'

        self.annotations_filepath = annotations_filepath

    def collect(self, spectrogram, spectrogram_only):

        processed_samples = 0
        for index, row in tqdm(self.excel_data.iterrows(), 'Collecting data'):
            audio_path = None

            id = row['XenoCantoID']
            peak_freq = row['peak_frequency']
            segments = row['segments']
            motifs = row['element_types']
            subj_quality = row['rec_quality_subj']
            rec_quality = row['rec_quality_xeno']

            # Parsing integer ID
            if id.startswith("XC"):
                sample_name = id[2:]
            else:
                sample_name = id

            try:
                sample_name = int(sample_name)
            except Exception as e:
                PRINT.warning(f'Invalid XenoCanto ID: {sample_name}')

            # Finding audio
            try:
                audio_path = self.find_audio_by_id(id)
                if audio_path is not None:
                    target_file = os.path.join(self.audio_dir, f'{sample_name}.{self.suffix}')
                else:
                    continue

            except Exception as e:
                PRINT.error(f'Audio file not found: {e}')
                continue

            # If file allready exist, continue (solving no-space break)
            if os.path.exists(target_file) and spectrogram_only is False:
                continue

            # Calculating spectrogram
            # Saving spectrogram
            if spectrogram:
                try:
                    Spectrogram = calculate_spectrogram(audio_filepath=audio_path)
                except Exception as e:
                    PRINT.error(f"Calculation of spectrogram for '{sample_name}': {e}")
                    continue

                Spectrogram.save_as_raw_input(os.path.join(self.spectrograms_dir, str(sample_name)))

            if spectrogram_only is False:
                # Copying audio
                shutil.copy(audio_path, target_file)

                # Saving label
                self.save_label([sample_name, segments, motifs, peak_freq, subj_quality, rec_quality])

            processed_samples += 1

        print(f'Processed {processed_samples} samples.')


    def save_label(self, values):
        """
        Saving one row to annotation CSV file.
        """

        # Check if file exists
        file_exists = os.path.exists(self.annotations_filepath)

        # If file does not exist, add header
        columns = ['sample_id', 'segments', 'motifs', 'peak_frequency', 'rec_quality_subj', 'rec_quality_xeno']
        data_to_save = pd.DataFrame([values], columns=columns)
        data_to_save.to_csv(self.annotations_filepath, mode='a', header=not file_exists, index=False)

    def find_audio_by_id(self, sample_id):
        file_name = f'{sample_id}_'

        for parent_dir in self.source_audio_dir:
            for root, dirs, files in os.walk(parent_dir):
                for file in files:
                    if file.startswith(file_name):
                        found_file = os.path.join(root, file)
                        print(f'Matched: {file_name} --> {file}')
                        return found_file

        raise FileNotFoundError(f'File {file_name} not found!')


if __name__ == '__main__':
    CAX = CollectAnnotatedXenoData(
        excel_filepath='/Users/vojtechremis/Desktop/Projects/birdsong-segmentation/source_data/xeno-canto/song_raw.xlsx',
        annotations_filepath='/Volumes/T7/Datasets/XenoCanto/labels/labels.csv',
        source_audio_dir='/Users/vojtechremis/Desktop/Projects/birdsong-segmentation/source_data/xeno-canto/Song_files'
    )

    CAX.collect(spectrogram=True, spectrogram_only=False)
