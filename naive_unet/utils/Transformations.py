import sys
from pathlib import Path

# Adds parent directory to sys.path if not already included
parent_dir = Path(__file__).resolve().parent.parent
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))

from Inc import Log
from utils import Visualization
import torch.nn.functional as F
import numpy as np
import torch
from abc import ABC, abstractmethod


class Transformation:
    def __init__(self, mean=None, std=None, device='cpu', debug=False):
        self.mean = mean
        self.std = std
        self.device = device
        self.debug = debug

    def __call__(self, image_tensor, *args):
        return self.input(image_tensor)

    @abstractmethod
    def input(self, image_tensor, *args):
        return image_tensor

    @abstractmethod
    def input_inverse(self, image_tensor, *args):
        return image_tensor

    @abstractmethod
    def mask(self, mask_tensor, *args):
        return mask_tensor

    @abstractmethod
    def mask_inverse(self, mask_tensor, *args):
        return mask_tensor


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

        Visualization.spectrum_above_mask(spectrum=image.squeeze(0), frequency_bins=freq_bins, time_bins=time_bins,
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
        image_resized = F.interpolate(image.unsqueeze(0), size=(new_height, new_width), mode='bilinear',
                                      align_corners=False)

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
        Visualization.spectrum_above_mask(spectrum=padded_spectrum.squeeze(0),
                                          frequency_bins=freq_bins,
                                          time_bins=time_bins,
                                          sample_id=f'*after nearest power of 2 transform_{title_}*',
                                          output_mode='display')

    # Converting float64 to float32 due to incompatibility of MPS and float64
    if dict_format:
        image_dict['spectrum_values'] = padded_spectrum
        image_dict['original_size'] = (width, height)
        image_dict['freq_bins'] = freq_bins.astype(np.float32)
        image_dict['time_bins'] = time_bins.astype(np.float32)
        image_dict['padding'] = tuple(padding)

        # resized = resize_to_nearest_power_of_two_inverse(image_dict, image_dict['spectrum_values'], debug=debug,
        #                                                  return_format='ndarray', process_mask=process_mask)

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
        resized_mask = F.interpolate(cropped_image.unsqueeze(0), size=original_size[::-1], mode='bilinear',
                                     align_corners=False)

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
        Visualization.spectrum_above_mask(spectrum=resized_mask.squeeze(0),
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


def res_to_near_two_pow_full_inverse(image, mask, freq_bins, time_bins, original_size, padding, debug=False):
    # Convert freq_bins and time_bins to numpy
    freq_bins = freq_bins.cpu().numpy()
    time_bins = time_bins.cpu().numpy()

    if isinstance(mask, np.ndarray):
        mask = to_tensor(mask)

    # Removing padding
    right, bottom = padding

    b = 1
    c, height, width = mask.shape
    cropped_image = image[:, :height - bottom, :width - right]
    cropped_mask = mask[:, :height - bottom, :width - right]

    # Resize image
    resized_image = F.interpolate(cropped_image.unsqueeze(0), size=original_size[::-1], mode='bilinear',
                                  align_corners=False).squeeze(0)

    # Resize mask to fit original image size
    resized_mask = F.interpolate(cropped_mask.unsqueeze(0).float(), size=original_size[::-1], mode='nearest')
    resized_mask = resized_mask.long().squeeze(0)

    # Adjust time_bins and freq_bins using original_size and padding
    original_width, original_height = original_size

    original_width = original_width.cpu().numpy()

    # Adjust time_bins based on the width scaling
    time_bins_resized = np.linspace(
        time_bins[0] - (time_bins[-1] - time_bins[0]) * (original_width - width) / (2 * width),
        time_bins[-1] + (time_bins[-1] - time_bins[0]) * (original_width - width) / (2 * width),
        original_width
    )

    original_height = original_height.cpu().numpy()

    # Adjust freq_bins based on the height scaling
    freq_bins_resized = np.linspace(
        freq_bins[0] - (freq_bins[-1] - freq_bins[0]) * (original_height - height) / (2 * height),
        freq_bins[-1] + (freq_bins[-1] - freq_bins[0]) * (original_height - height) / (2 * height),
        original_height
    )

    if debug:
        Visualization.spectrum_above_mask(spectrum=resized_mask.squeeze(0),
                                          frequency_bins=freq_bins_resized,
                                          time_bins=time_bins_resized,
                                          sample_id=f'*resized*',
                                          output_mode='display')

    # Převod zpět na NumPy array, pokud je to potřeba
    return resized_image, resized_mask, freq_bins_resized, time_bins_resized


def normalize_tensor(spectrum, mean, std):
    if isinstance(spectrum,
                  dict) and 'spectrum_values' in spectrum.keys() and 'time_bins' in spectrum.keys() and 'freq_bins' in spectrum.keys():
        spectrum_values = spectrum['spectrum_values']
        spectrum['spectrum_values'] = (spectrum_values - mean) / std
        return spectrum
    else:
        return (spectrum - mean) / std


def denormalize_tensor(spectrum, mean, std):
    if isinstance(spectrum,
                  dict) and 'spectrum_values' in spectrum.keys() and 'time_bins' in spectrum.keys() and 'freq_bins' in spectrum.keys():
        # If spectrum is in complete dict format
        spectrum_values = spectrum['spectrum_values']

        device = spectrum_values.device
        mean = mean.to(device)
        std = std.to(device)
        spectrum['spectrum_values'] = (spectrum_values * std) + mean
        return spectrum

    else:
        # If spectrum is in raw (only tensor) format
        device = spectrum.device
        mean = mean.to(device)
        std = std.to(device)

        return (spectrum * std) + mean
