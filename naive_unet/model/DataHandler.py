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
import torch.nn.functional as F
from PIL import Image

# Adds parent directory to sys.path if not already included
parent_dir = Path(__file__).resolve().parent.parent
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))

from Inc import Log
from utils import visualization

PRINT = Log.get_logger()

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

to_tensor = transforms.ToTensor()

def nearest_power_of_two(x, pick_greater=False):
    """Finds the nearest power of two to x, either larger or smaller."""
    if x <= 0:
        return 1  # Handle non-positive inputs, returning 1 as the nearest power of two.

    # Find the previous power of two
    lower = 2 ** (x.bit_length() - 1)

    # Find the next power of two
    higher = 2 ** x.bit_length()

    if pick_greater:
        return higher

    # Compare distances to x
    if (x - lower) <= (higher - x):
        return lower
    else:
        return higher


def resize_to_nearest_power_of_two(image, process_mask=False, debug=False):
    """
    Resizes (or pads) the spectrum and mask so that the larger dimension is resized
    to the next power of two, and the smaller dimension is padded to match.

    :param debug:
    :param process_mask:
    :param image: Input spectrogram as a NumPy array.
    :param mask: Input mask as a NumPy array.
    :return: Resized spectrum and mask.
    """
    global to_tensor

    # If input variable is dictionary also containing time_bins, freq_bins
    time_bins, freq_bins = None, None
    dict_format = False

    if isinstance(image, dict) and all(
            value in ['spectrum_values', 'time_bins', 'freq_bins'] for value in image.keys()):
        image_dict = image.copy()
        time_bins = image['time_bins']
        freq_bins = image['freq_bins']
        image = image['spectrum_values']
        dict_format = True

    # Implemented for batch = 1
    b = 1
    c, height, width = image.shape

    if isinstance(image, np.ndarray):
        image = to_tensor(image)

    if debug:
        if process_mask:
            title_ = 'mask'
        else:
            title_ = 'spect'

        visualization.spectrum_above_mask(spectrum=image.squeeze(0), frequency_bins=freq_bins, time_bins=time_bins,
                                          sample_id=f'*before nearest power of 2 transform_{title_}*',
                                          output_mode='display')

    # Find the larger side and compute its next power of two
    if height >= width:
        new_height = nearest_power_of_two(height)
        # Scale the width proportionally
        new_width = int(new_height * width / height)
    else:
        new_width = nearest_power_of_two(width)
        # Scale the height proportionally
        new_height = int(new_width * height / width)

    if process_mask:
        image_resized = F.interpolate(image.unsqueeze(0).float(), size=(new_height, new_width), mode='nearest')
        image_resized = image_resized.long()
    else:
        image_resized = F.interpolate(image.unsqueeze(0), size=(new_height, new_width), mode='bilinear', align_corners=False)

    # Ensure smaller side is padded to match the aspect ratio
    padding = [0, 0]
    resized_width, resized_height = new_width, new_height

    if height >= width:
        new_width = nearest_power_of_two(new_width, pick_greater=True)
        padding[0] = new_width - resized_width
    else:
        new_height = nearest_power_of_two(new_height, pick_greater=True)
        padding[1] = new_height - resized_height

    spectrum = image_resized.squeeze(0)

    # Resize and pad spectrum
    padded_spectrum = torch.zeros((1, new_height, new_width), dtype=spectrum.dtype)
    padded_spectrum[:, :resized_height, :resized_width] = spectrum

    # Extracting important values for padding

    # Transform time_bins and freq_bins
    if dict_format:
        # Resize time bins
        original_time_range = time_bins[-1] - time_bins[0]
        time_bins = np.linspace(
            time_bins[0] - original_time_range * (new_width - spectrum.shape[2]) / (2 * spectrum.shape[2]),
            time_bins[-1] + original_time_range * (new_width - spectrum.shape[2]) / (2 * spectrum.shape[2]),
            new_width
        )

        # Resize frequency bins
        original_freq_range = freq_bins[-1] - freq_bins[0]
        freq_bins = np.linspace(
            freq_bins[0] - original_freq_range * (new_height - spectrum.shape[1]) / (2 * spectrum.shape[1]),
            freq_bins[-1] + original_freq_range * (new_height - spectrum.shape[1]) / (2 * spectrum.shape[1]),
            new_height
        )

    if debug:
        visualization.spectrum_above_mask(spectrum=padded_spectrum.squeeze(0),
                                          frequency_bins=freq_bins,
                                          time_bins=time_bins,
                                          sample_id=f'*after nearest power of 2 transform_{title_}*',
                                          output_mode='display')

    if dict_format:
        image_dict['spectrum_values'] = padded_spectrum
        image_dict['original_size'] = (width, height)
        image_dict['frequency_bins'] = freq_bins
        image_dict['time_bins'] = time_bins
        image_dict['padding'] = tuple(padding)

        resized = resize_to_nearest_power_of_two_inverse(image_dict, image_dict['spectrum_values'], debug=debug,
                                                         return_format='ndarray', process_mask=process_mask)

        return image_dict
    else:
        return padded_spectrum


def resize_to_nearest_power_of_two_inverse(image, mask, process_mask=False, debug=False, return_format='PIL'):
    # If input variable is dictionary also containing time_bins, freq_bins
    if not (isinstance(image,
                       dict) and 'spectrum_values' in image.keys() and 'time_bins' in image.keys() and 'freq_bins' in image.keys()):
        return None

    image_dict = image.copy()
    time_bins = image_dict['time_bins']
    freq_bins = image_dict['freq_bins']
    original_size = image_dict['original_size']
    padding = image_dict['padding']
    dict_format = True
    mask_ = mask.clone()


    if isinstance(mask, np.ndarray):
        mask = to_tensor(mask)

    # Removing padding
    right, bottom = padding

    b = 1
    c, height, width = mask.shape
    cropped_image = mask[:, :height - bottom, :width - right]

    # Resize mask to fit original image size
    if process_mask:
        resized_mask = F.interpolate(cropped_image.unsqueeze(0).float(), size=original_size[::-1], mode='nearest')
        resized_mask = resized_mask.long()
    else:
        resized_mask = F.interpolate(cropped_image.unsqueeze(0), size=original_size[::-1], mode='bilinear', align_corners=False)

    resized_mask = resized_mask.squeeze(0)

    # Adjust time_bins and freq_bins using original_size and padding
    original_width, original_height = original_size

    # Adjust time_bins based on the width scaling
    time_bins_resized = np.linspace(
        time_bins[0] - (time_bins[-1] - time_bins[0]) * (original_width - width) / (2 * width),
        time_bins[-1] + (time_bins[-1] - time_bins[0]) * (original_width - width) / (2 * width),
        original_width
    )

    # Adjust freq_bins based on the height scaling
    freq_bins_resized = np.linspace(
        freq_bins[0] - (freq_bins[-1] - freq_bins[0]) * (original_height - height) / (2 * height),
        freq_bins[-1] + (freq_bins[-1] - freq_bins[0]) * (original_height - height) / (2 * height),
        original_height
    )

    if debug:
        visualization.spectrum_above_mask(spectrum=resized_mask.squeeze(0),
                                          frequency_bins=freq_bins_resized,
                                          time_bins=time_bins_resized,
                                          sample_id=f'*resized*',
                                          output_mode='display')

    # Převod zpět na NumPy array, pokud je to potřeba
    if return_format == 'ndarray':
        return np.array(resized_mask)
    elif return_format == 'PIL':
        return resized_mask
    else:
        return resized_mask


class BirdUNetDataset(Dataset):
    def __init__(self, spectrogram_dir, mask_dir, bin_size=88, transform_input=None, transform_mask=None, debug=False,
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

        self.spectrogram_dir = spectrogram_dir
        self.mask_dir = mask_dir
        self.samples_ids_list = self.get_data_list(spectrogram_dir, mask_dir, limit=5)

        # Transformations
        self.parse_transform_attrs(transform_attrs)

        if transform_input is None:
            self.transform_input = transforms.Compose([
                transforms.Lambda(
                    lambda img: resize_to_nearest_power_of_two(img, process_mask=False, debug=self.debug)),
                transforms.Lambda(self.normalize_tensor)
            ])
        else:
            self.transform_input = transform_input

        if transform_mask is None:
            self.transform_mask = transforms.Compose([
                # transforms.ToPILImage(),
                transforms.Lambda(lambda img: resize_to_nearest_power_of_two(img, process_mask=True, debug=self.debug))
            ])
        else:
            self.transform_mask = transform_mask

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
                                       sample_segment_id_start=0, describe_dictionary=True):
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
        return sample_list

    def calculate_normalization(self):
        samples_attributes = self.samples_ids_list

        all_data = []
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

            all_data.append(spect_segment)

        all_data = np.concatenate([data.flatten() for data in all_data])

        self.mean = np.mean(all_data)
        self.std = np.std(all_data)

        PRINT.info(f"Mean: {self.mean},\t Std: {self.std}.")

    def normalize_tensor(self, spectrum):
        if isinstance(spectrum,
                      dict) and 'spectrum_values' in spectrum.keys() and 'time_bins' in spectrum.keys() and 'freq_bins' in spectrum.keys():
            spectrum_values = spectrum['spectrum_values']
            spectrum['spectrum_values'] = (spectrum_values - self.mean) / self.std
            return spectrum
        else:
            return (spectrum - self.mean) / self.std

    def denormalize(self, spectrum):
        if isinstance(spectrum,
                      dict) and 'spectrum_values' in spectrum.keys() and 'time_bins' in spectrum.keys() and 'freq_bins' in spectrum.keys():
            spectrum_values = spectrum['spectrum_values']
            spectrum['spectrum_values'] = (spectrum_values * self.std) + self.mean
            return spectrum
        else:
            return (spectrum * self.std) + self.mean

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

        spectrogram = spectrogram_source['spectrum']
        time_bins = spectrogram_source['time_bins']
        frequency_bins = spectrogram_source['frequency_bins']
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

        if self.debug:
            visualization.spectrum_above_mask(
                spect_segment,
                mask_segment,
                spectrum_segment_dict['freq_bins'],
                spectrum_segment_dict['time_bins'],
                f'Idco-{image_id}',
                output_mode='display'
            )

        mask_segment_pt = to_tensor(mask_segment).long()

        model_input = self.transform_input(spectrum_segment_dict)
        mask_segment_pt = self.transform_mask(mask_segment_pt)

        plot_ = False
        if plot_:
            visualization.spectrum_above_mask(model_input['spectrum_values'], mask_segment.squeeze(0),
                                              frequency_bins=model_input['freq_bins'],
                                              time_bins=model_input['time_bins'])

        return model_input, mask_segment_pt


if __name__ == '__main__':
    Dataset = BirdUNetDataset(
        '/Users/vojtechremis/Desktop/Projects/Birds/naive_unet/data/spectrogram',
        '/Users/vojtechremis/Desktop/Projects/Birds/naive_unet/data/mask',
        debug=True,
        transform_attrs={'mean': -72.9381103515625, 'std': 13.615241050720215}
    )

    for i in range(1):
        Dataset.__getitem__(i)
