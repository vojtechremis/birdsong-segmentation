import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import sys
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
from tqdm import tqdm
import torchvision.transforms as transforms
import random

# Adds parent directory to sys.path if not already included
parent_dir = Path(__file__).resolve().parent.parent
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))

from utils import Log
from utils import Visualization, Transformations

PRINT = Log.get_logger()

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

to_tensor = transforms.ToTensor()


class DefaultTransform(Transformations.Transformation):
    def input(self, image):
        image_resized = Transformations.resize_to_nearest_power_of_two(image, is_mask=False, debug=self.debug)
        image_normalized = Transformations.normalize_tensor(image_resized, self.mean, self.std)
        return image_normalized

    def input_inverse(self, image):
        img_transformed_inv = Transformations.denormalize_tensor(image, self.mean, self.std)
        return Transformations.resize_to_nearest_power_of_two_inverse(img_transformed_inv, img_transformed_inv,
                                                                      is_mask=False, debug=self.debug)

    def mask(self, mask):
        image_resized = Transformations.resize_to_nearest_power_of_two(mask, is_mask=True, debug=self.debug)
        return image_resized

    def mask_inverse(self, mask):
        return Transformations.resize_to_nearest_power_of_two_inverse(mask, is_mask=True,
                                                                      debug=self.debug)

    def inverse_input_mask(self, image, mask, freq_bins, time_bins, original_size, padding, debug=False):
        img_transformed_inv = Transformations.denormalize_tensor(image, self.mean, self.std)
        return Transformations.res_to_near_two_pow_full_inverse(img_transformed_inv, mask, freq_bins, time_bins,
                                                                original_size, padding,
                                                                debug=False)


class BirdUNetDataset(Dataset):
    def __init__(self, spectrogram_dir, mask_dir, bin_size=128, transform=None, debug=False,
                 transform_attrs=None):
        """
        :param image_dir: Cesta k adresáři obsahujícímu obrázky.
        :param annotation_dir: Cesta k adresáři obsahujícímu anotace (např. JSON, XML, atd.).
        :param transform: Možné transformace, které budou aplikovány na data.
        """

        self.mean = None
        self.std = None
        self.bin_size = bin_size
        self.file_ext = 'npz'
        self.debug = debug

        self.info = {}

        self.spectrogram_dir = spectrogram_dir
        self.mask_dir = mask_dir

        if spectrogram_dir is not None and mask_dir is not None:
            self.samples_ids_list = self.get_data_list(spectrogram_dir, mask_dir)

        # Transformations
        self.parse_transform_attrs(transform_attrs)

        if transform is None:
            self.transform = DefaultTransform(mean=self.mean, std=self.std)
        else:
            self.transform = transform

    def parse_transform_attrs(self, transform_attrs):
        if transform_attrs is not None:
            if 'mean' in transform_attrs.keys():
                self.mean = torch.tensor(transform_attrs['mean'])
            else:
                self.mean = None

            if 'std' in transform_attrs.keys():
                self.std = torch.tensor(transform_attrs['std'])
            else:
                self.std = None

        if self.mean is None or self.std is None:
            self.calculate_normalization()

    def get_segments_from_numpy_file(self, spec_path: str, mask_path: str, sample_segment_id_start=0, bin_size=88):
        """
        Load segments from numpy file including object with keys (image_id, spectrum, time_bins, frequency_bins).
        Loaded segments are in format of dictionary sample_segment_id's (incrementing integer starting from 'sample_segment_id_start') with value of
        dictionary with keys (image_id, segment_start, segment_end).
        This method does not return raw spectrogram and mask. If you want extract data from file in a different way, or they have different format,
        you can skip this method and use 'get_segments_with_valid_length()' method directly.
        :param bin_size:
        :param spec_path: Path to numpy file containing spectrogram data.
        :param mask_path: Path to mask file containing segmentation masks.
        :param sample_segment_id_start: Where should sample_segment_id start from.
        :return:
            Segments detail info.
        """

        spectrogram = np.load(spec_path)['spectrum']
        mask_ = np.load(mask_path)
        recording_id = mask_['image_id']
        mask = mask_['mask']

        return self.get_segments_with_valid_length(spectrogram, mask, recording_id,
                                                   sample_segment_id_start=sample_segment_id_start, bin_size=bin_size,
                                                   describe_dictionary=False)

    def get_segments_with_valid_length(self, spectrogram: np.array, mask: np.array, recording_id: int, bin_size,
                                       sample_segment_id_start=0, describe_dictionary=True, prob_to_drop_blank=0.8):
        """
        Extracts segments from spectrogram and mask.
        Extracted segments are in format of dictionary sample_segment_id's (incrementing integer starting from 'sample_segment_id_start') with value of
        either dictionary with keys (image_id, segment_start, segment_end) or just a tuple (image_id, segment_start, segments_end) - depending on 'describe_dictionary' parameter.
        This method does not return raw spectrogram and mask.

        :param spectrogram: Spectrogram to extract segments.
        :param mask: Mask to extract segments.
        :param recording_id: ID of recording file to extract segments from.
        :param bin_size: Which bin size does one segment have.
        :param sample_segment_id_start: Where should sample_segment_id start from.
        :param describe_dictionary: Whether add more keys to output dictionary, so it is more meaningful. If set to false, output is more memory-efficient.
        :return:
            Segments detail info.
        """
        segments_list = {}

        sample_segment_id = sample_segment_id_start
        num_bins = spectrogram.shape[1]
        for start in range(0, num_bins, self.bin_size):
            end = start + self.bin_size

            # If segment is blank
            if np.all(mask[:, start:end] == 0):

                # Skip with given probability
                if random.random() < prob_to_drop_blank:
                    self.info['removed_blank_segments'] += 1
                    continue
                else:
                    self.info['left_blank_segments'] += 1

            if describe_dictionary:
                segments_list[sample_segment_id] = {'original_file_id': recording_id, 'segment_start': start,
                                                    'segment_end': end}
            else:
                segments_list[sample_segment_id] = (int(recording_id), start, end)

            sample_segment_id += 1

        return segments_list

    def get_data_list(self, spectrogram_dir, mask_dir, limit=-1):
        # Get data - list of ids
        spectrogram_list = set()
        mask_list = set()

        for filename in os.listdir(spectrogram_dir):
            if filename.endswith(self.file_ext):
                spect_id = filename.split('.')[0]
                spectrogram_list.add(spect_id)

        for filename in os.listdir(mask_dir):
            if filename.endswith(self.file_ext):
                mask_id = filename.split('.')[0]
                mask_list.add(mask_id)

        id_pairs = list(spectrogram_list & mask_list)[0:limit]
        only_spect = spectrogram_list - mask_list
        only_mask = mask_list - spectrogram_list

        PRINT.info(
            f'Number of samples: {len(id_pairs)}.\t Spectrograms without mask: {len(only_spect)}.\t Masks without spectrogram: {len(only_mask)}')

        # Crop spectrograms and maks
        self.info['removed_blank_segments'] = 0
        self.info['left_blank_segments'] = 0

        sample_list = {}
        sample_id_last = -1
        for sample_id in tqdm(id_pairs, 'Cropping recordings spectrogram'):
            spec_path = os.path.join(spectrogram_dir, f"{sample_id}.{self.file_ext}")
            mask_path = os.path.join(mask_dir, f"{sample_id}.{self.file_ext}")

            segments_dict = self.get_segments_from_numpy_file(spec_path, mask_path,
                                                              sample_segment_id_start=sample_id_last + 1)

            # Checking conflict keys
            conflict_keys = segments_dict.keys() & sample_list.keys()
            if len(conflict_keys) != 0:
                PRINT.warning(
                    f'During recordings spectrogram splitting (id: {sample_id}) some conflict keys appeared: {conflict_keys}.')

            sample_list.update(segments_dict)

            sample_id_last = list(segments_dict.keys())[-1]

        # Return dict of {sample_segment_id: (sample_id, bin_start, bin_end)}
        PRINT.info(
            f'Dataset total size after cropping: {len(sample_list)}.\t Blank segments left: {self.info["left_blank_segments"]}. \t Blank segments removed: {self.info["removed_blank_segments"]}.')
        return sample_list

    def calculate_normalization(self):
        samples_attributes = self.samples_ids_list

        total_sum = 0.0
        total_sq_sum = 0.0
        total_count = 0

        for sample_id, sample_attrs in tqdm(samples_attributes.items(), 'Normalizing samples'):
            sample_filepath = f'{str(sample_attrs[0])}.{self.file_ext}'
            sample_bin_start = sample_attrs[1]
            sample_bin_end = sample_attrs[2]

            spectrum_path = os.path.join(self.spectrogram_dir, sample_filepath)
            spectrogram = np.load(spectrum_path)['spectrum']

            # Extract segment and pad if necessary
            spect_segment = spectrogram[:, sample_bin_start:sample_bin_end]

            # Pad if needed
            if spect_segment.shape[1] < self.bin_size:
                pad_width = self.bin_size - spect_segment.shape[1]
                spect_segment = np.pad(spect_segment, ((0, 0), (0, pad_width)), mode='constant')

            # Calculate sum, squared sum, and count incrementally
            total_sum += np.sum(spect_segment)
            total_sq_sum += np.sum(spect_segment ** 2)
            total_count += spect_segment.size

        # Compute mean and standard deviation
        self.mean = total_sum / total_count
        self.std = np.sqrt((total_sq_sum / total_count) - self.mean ** 2)

        PRINT.info(f"Mean: {self.mean},\t Std: {self.std}.")

    def post_process_mask(self, spectrum, frequency_bins):
        pass

    def __len__(self):
        """Returns number of samples in the dataset."""
        return len(self.samples_ids_list)

    def __getitem__(self, idx):
        """
        Loads a spectrum and corresponding mask.
        :param idx: Index of a sample.
        :return: Loaded sample and its corresponding mask.
        """

        # Loading spectrum and mask from a file
        sample_attributes = self.samples_ids_list[idx]
        sample_filepath = f'{str(sample_attributes[0])}.{self.file_ext}'
        sample_bin_start = sample_attributes[1]
        sample_bin_end = sample_attributes[2]

        spectrum_path = os.path.join(self.spectrogram_dir, sample_filepath)
        mask_path = os.path.join(self.mask_dir, sample_filepath)

        spectrogram_source = np.load(spectrum_path)
        mask_source = np.load(mask_path)

        # Converting to float32 due to MPS incompatibility of float64
        spectrogram = spectrogram_source['spectrum']
        time_bins = spectrogram_source['time_bins'].astype(np.float32)
        frequency_bins = spectrogram_source['frequency_bins'].astype(np.float32)
        image_id = spectrogram_source['image_id']

        mask = mask_source['mask']

        # Extract segment and pad if necessary
        spect_segment = spectrogram[:, sample_bin_start:sample_bin_end]
        mask_segment = mask[:, sample_bin_start:sample_bin_end]

        # Pad if needed
        if spect_segment.shape[1] < self.bin_size:
            pad_width = self.bin_size - spect_segment.shape[1]
            spect_segment = np.pad(spect_segment, ((0, 0), (0, pad_width)), mode='constant')
            mask_segment = np.pad(mask_segment, ((0, 0), (0, pad_width)), mode='constant')

        # Building one sample dictionary
        spectrum_segment_dict = {
            'spectrum_values': to_tensor(spect_segment),
            'time_bins': time_bins[sample_bin_start:sample_bin_end],
            'freq_bins': frequency_bins
        }

        mask_segment_dict = {
            'mask_values': to_tensor(mask_segment).long()
        }

        if self.debug:
            Visualization.spectrum_above_mask(
                spect_segment,
                mask_segment,
                spectrum_segment_dict['freq_bins'],
                spectrum_segment_dict['time_bins'],
                f'Idco-{image_id}',
                output_mode='display'
            )

        model_input = self.transform(spectrum_segment_dict)
        mask_golden = self.transform.mask(mask_segment_dict)

        plot_ = False
        if plot_:
            Visualization.spectrum_above_mask(model_input['spectrum_values'], mask_golden['mask_values'].squeeze(0),
                                              frequency_bins=model_input['freq_bins'],
                                              time_bins=model_input['time_bins'])

        return model_input, mask_golden


if __name__ == '__main__':
    Dataset = BirdUNetDataset(
        '/Users/vojtechremis/Desktop/Projects/birdsong-segmentation/naive_unet/data/spectrogram',
        '/Users/vojtechremis/Desktop/Projects/birdsong-segmentation/naive_unet/data/mask',
        debug=True,
        transform_attrs={'mean': -72.9381103515625, 'std': 13.615241050720215}
    )

    for i in range(1):
        Dataset.__getitem__(i)
