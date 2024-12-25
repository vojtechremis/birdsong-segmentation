import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import torch
import sys
from pathlib import Path

sys.path += [str(Path().resolve().parent.parent)]
from utils.Spectrogram import Spectrogram
from utils import Log

PRINT = Log.get_logger()
to_tensor = transforms.ToTensor()


class NormalizeInverse:
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)

    def __call__(self, tensor: torch.Tensor):
        return self.transform(tensor)

    def transform(self, tensor: torch.Tensor):
        return (tensor * self.std) + self.mean


def pad_spectrogram_with_noise(spectrogram, target_shape):
    """
    Pad spectrogram with noise extracted from low-intensity parts of the spectrogram.

    Parameters:
    - spectrogram (np.ndarray): Input spectrogram (2D numpy array).
    - target_shape (tuple): Desired output shape (rows, cols).

    Returns:
    - padded_spectrogram (np.ndarray): Spectrogram padded with noise to the target shape.
    """
    # Kontrola vstupního tvaru
    input_shape = spectrogram.shape
    if input_shape[0] > target_shape[0] or input_shape[1] > target_shape[1]:
        raise ValueError("Target shape must be larger than or equal to the input spectrogram shape.")

    # 1. Identifikace šumu (nízké intenzity)
    noise_mask = spectrogram < np.percentile(spectrogram, 10)  # Např. 10% nejnižší intenzity
    noise_values = spectrogram[noise_mask]

    if noise_values.size == 0:
        raise ValueError("No noise found in the spectrogram. Adjust the percentile threshold.")

    # 2. Generování šumu ze šumového profilu
    mean_noise = np.mean(noise_values)
    std_noise = np.std(noise_values)

    def generate_noise(shape):
        return np.random.normal(loc=mean_noise, scale=std_noise, size=shape)

    # 3. Vytvoření nového spectrogramu s paddingem
    padded_spectrogram = np.zeros(target_shape, dtype=np.float32)

    # Kopírování původního spectrogramu do horního levého rohu
    padded_spectrogram[:input_shape[0], :input_shape[1]] = spectrogram

    # Vyplnění paddingu šumem
    if input_shape[0] < target_shape[0]:  # Svislý padding
        vertical_noise = generate_noise((target_shape[0] - input_shape[0], input_shape[1]))
        padded_spectrogram[input_shape[0]:, :input_shape[1]] = vertical_noise

    if input_shape[1] < target_shape[1]:  # Horizontální padding
        horizontal_noise = generate_noise((target_shape[0], target_shape[1] - input_shape[1]))
        padded_spectrogram[:, input_shape[1]:] = horizontal_noise

    return padded_spectrogram


class RemoveNoise:
    def __init__(self, noise_threshold=1.5):
        self.noise_threshold = noise_threshold

    def __call__(self, spectrogram: torch.Tensor):
        return self.remove(spectrogram)

    def remove(self, spectrogram: torch.tensor):
        """
        Apply spectral gating to a precomputed STFT matrix.

        Parameters:
        - stft_matrix (np.ndarray): Magnitude spectrogram (STFT).
        - noise_threshold (float): Threshold factor to suppress noise (higher = more aggressive).

        Returns:
        - denoised_stft (np.ndarray): Denoised magnitude STFT.
        """
        # 1. Odhad šumového prahu jako průměrná energie spektra
        noise_profile = np.mean(np.abs(spectrogram), axis=1, keepdims=True)

        # 2. Aplikace prahu: Potlačení frekvencí pod šumovým prahem
        threshold = self.noise_threshold * noise_profile
        denoised_stft = np.where(np.abs(spectrogram) > threshold, spectrogram, 0)

        return denoised_stft


class SpectrogramDataset(Dataset):
    def __init__(self, spectrogram_dir, label_file, transform=None, transform_inverse=None, debug=False,
                 transform_attrs=None, time_binsize_limit=None):
        """
        Args:
            spectrogram_dir (str): Path to the directory containing spectrograms.
            label_file (str): Path to the CSV file containing labels.
            transform (callable, optional): Optional transform to be applied on a sample.
        """

        # Data attributes
        self.mean = None
        self.std = None
        self.time_binsize_limit = time_binsize_limit
        self.label_df = pd.read_csv(label_file, delimiter=',')

        # Setting directories
        self.spectrogram_dir = spectrogram_dir
        self.debug = debug

        self.spectrogram_suffix = 'npz'

        if self.time_binsize_limit is not None:
            self.label_df['time_binsize'] = self.label_df['sample_id'].apply(self.get_time_binsize)

            original_len = len(self.label_df)
            self.label_df.drop(self.label_df[self.label_df['time_binsize'] > self.time_binsize_limit].index,
                               inplace=True)
            PRINT.info(
                f'Time binsize limit ({self.time_binsize_limit}): {original_len} -> {len(self.label_df)} samples')

        self.sample_ids = self.label_df['sample_id'].values

        # Transformations
        self.transform = transform
        self.parse_transform_attrs(transform_attrs)

        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std)
            ])
        else:
            self.transform = transform

        if transform_inverse is None:
            self.transform_inverse = NormalizeInverse(mean=self.mean, std=self.std)
        else:
            self.transform_inverse = transform_inverse

    def get_time_binsize(self, sample_id):
        sample_filepath = os.path.join(self.spectrogram_dir, f'{sample_id}.{self.spectrogram_suffix}')
        time_bins = Spectrogram(sample_filepath).time_bins
        return time_bins.shape[0]

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

    def __len__(self):
        return len(self.sample_ids)

    def calculate_normalization(self):

        total_sum = 0.0
        total_sq_sum = 0.0
        total_count = 0

        for sample_id in tqdm(self.sample_ids, 'Normalizing samples'):
            sample_filepath = os.path.join(self.spectrogram_dir, f'{sample_id}.{self.spectrogram_suffix}')

            values = Spectrogram(sample_filepath).values

            # Calculate sum, squared sum, and count incrementally
            total_sum += np.sum(values)
            total_sq_sum += np.sum(values ** 2)
            total_count += values.size

        # Compute mean and standard deviation
        self.mean = total_sum / total_count
        self.std = np.sqrt((total_sq_sum / total_count) - self.mean ** 2)

        PRINT.info(f"Mean: {self.mean},\t Std: {self.std}.")

    def __getitem__(self, idx):
        # Get the ID of the current sample
        sample_id = self.sample_ids[idx]
        spectrum_path = os.path.join(self.spectrogram_dir, f'{sample_id}.npz')
        values = Spectrogram(spectrum_path).values

        # Get the corresponding label values
        label_entry = self.label_df[self.label_df['sample_id'] == sample_id].iloc[0]

        motifs = label_entry['motifs']
        segments = label_entry['segments']
        peak_frequency = label_entry['peak_frequency']

        # Apply input transformations
        if self.transform:
            values = self.transform(values)

        return values, segments, motifs


# # Create DataLoader for batching
# batch_size = 32
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

if __name__ == '__main__':
    Dataset = SpectrogramDataset(
        spectrogram_dir='/Users/vojtechremis/Desktop/Projects/birdsong-segmentation/dummy_network/data_handlers/spectrograms',
        label_file='/Users/vojtechremis/Desktop/Projects/birdsong-segmentation/dummy_network/data_handlers/labels/labels.csv',
        debug=True,
        time_binsize_limit=2000
        # transform_attrs={'mean': -72.9381103515625, 'std': 13.615241050720215}
    )

    for i in range(1):
        a = Dataset.__getitem__(i)

    print(1)

    spectr = Spectrogram(
        '/Users/vojtechremis/Desktop/Projects/birdsong-segmentation/dummy_network/data_handlers/spectrograms/1089.npz')
    spectr.show(caption='Spectrogram').show()

    spectr.values = pad_spectrogram_with_noise(spectr.values, (513, 5000))
    spectr.show(caption='Padded Spectrogram').show()

    spectr.values = np.squeeze(Dataset.transform(spectr.values), axis=0)
    spectr.show(caption='Normalized Spectrogram').show()

    rn = RemoveNoise()

    spectr.values = rn(torch.tensor(spectr.values))
    spectr.show(caption='RemovedNoise Spectrogram').show()

    # loop spectrums and show
