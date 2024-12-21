import sys
from pathlib import Path

# Adds parent directory to sys.path if not already included
parent_dir = Path(__file__).resolve().parent.parent
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))

from utils import Log
from utils import Visualization
import torch.nn.functional as F
import numpy as np
import torch
from abc import ABC, abstractmethod

PRINT = Log.get_logger()

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


def resize_to_nearest_power_of_two(input: dict, is_mask=False, debug=False):
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

    # Check if input dict has required keys
    valid_image_keys = ['spectrum_values', 'time_bins', 'freq_bins']
    valid_mask_keys = ['mask_values']

    resized_input = {}
    if is_mask:
        if not(all(value in valid_mask_keys for value in input.keys())):
            PRINT.error(f'Input dictionary has to have {valid_mask_keys} keys when mask provided as an input.')
            return None

        else:
            # Valid mask input
            input_image = input['mask_values']
            title_ = 'mask'

    if not is_mask:
        if not (
            all(value in valid_image_keys for value in
                input.keys()) or all(value in valid_mask_keys for value in
                input.keys())
        ):
            PRINT.error(f'Input dictionary has to have {valid_image_keys} keys when spectrum provided as an input.')
            return None
        else:
            # Valid spectrum input
            input_image = input['spectrum_values']
            title_ = 'spect'

            time_bins = input['time_bins']
            freq_bins = input['freq_bins']

    channels, height, width = input_image.shape


    if isinstance(input_image, np.ndarray):
        input_image = to_tensor(input_image)

    if debug and not is_mask:
        Visualization.spectrum_above_mask(spectrum=input_image.squeeze(0), frequency_bins=freq_bins, time_bins=time_bins,
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

    if is_mask:
        input_image_resized = F.interpolate(input_image.unsqueeze(0).float(), size=(new_height, new_width), mode='nearest')
        input_image_resized = input_image_resized.long()
    else:
        input_image_resized = F.interpolate(input_image.unsqueeze(0), size=(new_height, new_width), mode='bilinear',
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

    input_image_resized = input_image_resized.squeeze(0)

    # Resize and pad spectrum
    input_image_padded = torch.zeros((1, new_height, new_width), dtype=input_image_resized.dtype)
    input_image_padded[:, :resized_height, :resized_width] = input_image_resized

    # Transform time_bins and freq_bins for retrieval
    if not is_mask:
        # Resize time bins
        original_time_range = time_bins[-1] - time_bins[0]
        time_bins = np.linspace(
            time_bins[0] - original_time_range * (new_width - input_image_resized.shape[2]) / (2 * input_image_resized.shape[2]),
            time_bins[-1] + original_time_range * (new_width - input_image_resized.shape[2]) / (2 * input_image_resized.shape[2]),
            new_width
        )

        # Resize frequency bins
        original_freq_range = freq_bins[-1] - freq_bins[0]
        freq_bins = np.linspace(
            freq_bins[0] - original_freq_range * (new_height - input_image_resized.shape[1]) / (2 * input_image_resized.shape[1]),
            freq_bins[-1] + original_freq_range * (new_height - input_image_resized.shape[1]) / (2 * input_image_resized.shape[1]),
            new_height
        )

    if debug and not is_mask:
        Visualization.spectrum_above_mask(spectrum=input_image_padded.squeeze(0),
                                          frequency_bins=freq_bins,
                                          time_bins=time_bins,
                                          sample_id=f'*after nearest power of 2 transform_{title_}*',
                                          output_mode='display')

    # Converting float64 to float32 due to incompatibility of MPS and float64
    if is_mask:
        resized_input['mask_values'] = input_image_padded
        resized_input['original_size'] = (width, height)
        resized_input['padding'] = tuple(padding)
    else:
        resized_input['spectrum_values'] = input_image_padded
        resized_input['original_size'] = (width, height)
        resized_input['freq_bins'] = freq_bins.astype(np.float32)
        resized_input['time_bins'] = time_bins.astype(np.float32)
        resized_input['padding'] = tuple(padding)

    return resized_input


def resize_to_nearest_power_of_two_inverse(input: dict, is_mask=False, debug=False):

    # Check if input dict has required keys
    valid_image_keys = ['spectrum_values', 'time_bins', 'freq_bins', 'padding', 'original_size']
    valid_mask_keys = ['mask_values', 'padding', 'original_size']

    resized_input = {}
    if is_mask:
        if not(all(value in valid_mask_keys for value in input.keys())):
            PRINT.error(f'Input dictionary has to have {valid_mask_keys} keys when mask provided as an input.')
            return None

        else:
            # Valid mask input
            input_image = input['mask_values']

    if not is_mask:
        if not (
            all(value in valid_image_keys for value in
                input.keys()) or all(value in valid_mask_keys for value in
                input.keys())
        ):
            PRINT.error(f'Input dictionary has to have {valid_image_keys} keys when spectrum provided as an input.')
            return None
        else:
            # Valid spectrum input
            input_image = input['spectrum_values']

            time_bins = input['time_bins']
            freq_bins = input['freq_bins']

    # Original size and padding
    original_size = input['original_size']
    padding = input['padding']

    if isinstance(input_image, np.ndarray):
        input_image = to_tensor(input_image)

    # Removing padding
    right, bottom = padding

    channels, height, width = input_image.shape
    cropped_image = input_image[:, :height - bottom, :width - right]

    # Resize mask to fit original image size
    if is_mask:
        input_image_resized = F.interpolate(cropped_image.unsqueeze(0).float(), size=original_size[::-1], mode='nearest')
        input_image_resized = input_image_resized.long()
    else:
        input_image_resized = F.interpolate(cropped_image.unsqueeze(0), size=original_size[::-1], mode='bilinear',
                                     align_corners=False)

    input_image_resized = input_image_resized.squeeze(0)

    # Adjust time_bins and freq_bins using original_size and padding
    original_width, original_height = original_size

    # Adjust time_bins based on the width scaling
    if not is_mask:
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

    if debug and not is_mask:
        Visualization.spectrum_above_mask(spectrum=input_image_resized.squeeze(0),
                                          frequency_bins=freq_bins_resized,
                                          time_bins=time_bins_resized,
                                          sample_id=f'*resized*',
                                          output_mode='display')

    if is_mask:
        resized_input['mask_values'] = input_image_resized
    else:
        resized_input['spectrum_values'] = input_image_resized
        resized_input['time_bins'] = time_bins_resized
        resized_input['freq_bins'] = freq_bins_resized

    return resized_input


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
    if isinstance(spectrum, dict) and 'spectrum_values' in spectrum.keys():
        spectrum_values = spectrum['spectrum_values']
        spectrum['spectrum_values'] = (spectrum_values - mean) / std
        return spectrum

    elif isinstance(spectrum, dict) and 'mask_values' in spectrum.keys():
        mask_values = spectrum['mask_values']
        spectrum['mask_values'] = (mask_values - mean) / std
        return spectrum

    else:
        return (spectrum - mean) / std


def denormalize_tensor(spectrum, mean, std):
    if isinstance(spectrum, dict) and 'spectrum_values' in spectrum.keys():
        # If spectrum is in dict format
        spectrum_values = spectrum['spectrum_values']

        device = spectrum_values.device
        mean = mean.to(device)
        std = std.to(device)
        spectrum['spectrum_values'] = (spectrum_values * std) + mean
        return spectrum

    elif isinstance(spectrum, dict) and 'mask_values' in spectrum.keys():
        # If spectrum is in dict format
        mask_values = spectrum['mask_values']

        device = mask_values.device
        mean = mean.to(device)
        std = std.to(device)
        spectrum['mask_values'] = (mask_values * std) + mean
        return spectrum

    else:
        # If spectrum is in raw (only tensor) format
        device = spectrum.device
        mean = mean.to(device)
        std = std.to(device)

        return (spectrum * std) + mean