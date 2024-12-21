import sys
from pathlib import Path
import torch

# Přidání prarodiče aktuálního adresáře do sys.path
sys.path += [str(Path().resolve().parent.parent), str(Path().resolve().parent), f'{str(Path().resolve().parent)}/model']

import noisereduce as nr
import cv2
import numpy as np
from matplotlib import pyplot as plt

from utils import Log, SegmentsHandler, Visualization
from source_data.tweetynet.Inc.segments_handler import VisualizationHandler

sys.path.append('model')
from model.UNetHandler import UNet, load_config
from model.DataHandler import BirdUNetDataset
from utils.Mask import Mask
from utils.Spectrogram import Spectrogram

from scipy import ndimage
from skimage.morphology import remove_small_objects, binary_closing, disk

PRINT = Log.get_logger()

# Initialize Model
CONFIG = load_config('/Users/vojtechremis/Desktop/Projects/birdsong-segmentation/naive_unet/config.json')
MEAN = CONFIG['data']['augmentation'].get('normalization_mean')
STD = CONFIG['data']['augmentation'].get('normalization_std')


class InferenceProcess:
    def __init__(self, model, debug=False):
        self.model = model
        self.VH = VisualizationHandler()
        self.debug = debug

    def __call__(self, amplitude, sampling_rate):
        return self.pipeline(amplitude, sampling_rate)

    def pipeline(self, amplitude, sampling_rate):
        y_reduced_noise = self.reduce_noise(amplitude, sampling_rate)

        time_bins, frequency_bins, spectrum_reduced_noise = SegmentsHandler.calculate_spectrogram(y_reduced_noise,
                                                                                                  sampling_rate)
        spectrum = Spectrogram(spectrum_reduced_noise, time_bins, frequency_bins)

        if self.debug and spectrum.axis_scale():
            self.VH.plot_spectrogram(spectrum.time_bins, spectrum.frequency_bins, spectrum.values,
                                     title='After noise reducing')

        # Crop top
        cutoff_bin = self.crop_top(spectrum)

        if self.debug and spectrum.axis_scale():
            self.VH.plot_spectrogram(spectrum.time_bins, spectrum.frequency_bins[:cutoff_bin],
                                     spectrum.values[:cutoff_bin, :],
                                     title='Before inference')

        spectrum.values = spectrum.values[:cutoff_bin, :]
        spectrum.frequency_bins = spectrum.frequency_bins[:cutoff_bin]

        retrieved_binary_mask = self.inference(spectrum, 128)

        binary_mask_stripes = self.reconstruct_stripes_in_binary_mask(retrieved_binary_mask)
        # processed_binary_mask = self.connect_areas(retrieved_binary_mask)

        segments = binary_mask_stripes.to_segments()

        return segments

    def reduce_noise(self, amplitude_array, sr, noise_start=0, noise_duration=1):

        if self.debug:
            # Spektrogram před odstraněním šumu
            time_bins, frequency_bins, spectrum = SegmentsHandler.calculate_spectrogram(amplitude_array, sr)
            self.VH.plot_spectrogram(time_bins, frequency_bins, spectrum)

        # Výběr šumového profilu
        noise_sample_start = int(noise_start * sr)
        noise_sample_end = int((noise_start + noise_duration) * sr)
        noise_profile = amplitude_array[noise_sample_start:noise_sample_end]

        # Redukce šumu
        reduced_noise = nr.reduce_noise(y=amplitude_array, sr=sr, y_noise=noise_profile)
        return reduced_noise

    def crop_top(self, spectrum: Spectrogram):
        spectrogram = spectrum.values
        time_bins = spectrum.time_bins
        frequency_bins = spectrum.frequency_bins

        # Pokud má spektrogram float hodnoty, normalizujte ho na rozsah 0-255
        if spectrogram.max() > 1.0:
            spectrogram_normalized = (spectrogram - spectrogram.min()) / (
                    spectrogram.max() - spectrogram.min())
            spectrogram_normalized = (spectrogram_normalized * 255).astype(np.uint8)
        else:
            spectrogram_normalized = (spectrogram * 255).astype(np.uint8)

        # Aplikace Sobelova filtru
        sobel_x = cv2.Sobel(spectrogram_normalized, cv2.CV_64F, 1, 0, ksize=3)  # Detekce hran ve směru X
        sobel_y = cv2.Sobel(spectrogram_normalized, cv2.CV_64F, 0, 1, ksize=3)  # Detekce hran ve směru Y

        # Kombinace hran
        sobel_combined = cv2.magnitude(sobel_x, sobel_y)  # Vypočítej velikost gradientu

        # Normalizace výsledného obrázku
        sobel_combined = np.uint8(255 * sobel_combined / np.max(sobel_combined))

        edges_adaptive = cv2.adaptiveThreshold(sobel_combined, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11,
                                               2)

        if self.debug:
            self.VH.plot_spectrogram(time_bins, frequency_bins, edges_adaptive, title='Detected edges')

        # Spočítat počet hran v každém řádku
        edge_counts = np.sum(edges_adaptive < 250, axis=1)  # Počet nenulových pixelů pro každý řádek

        # Najít poslední řádek, kde počet hran je pod prahem
        threshold = int(edges_adaptive.shape[1] * 0.01)  # Prah pro hranice, 0.1% šířky obrázku
        cutoff_row = 0
        for i, count in enumerate(edge_counts[::-1]):
            if count > threshold:
                break  # Jakmile najdeme první řádek, který překračuje práh, zastavíme
            cutoff_row = i

        cutoff_bin = spectrogram_normalized.shape[0] - cutoff_row
        cutoff_values = frequency_bins[cutoff_bin]

        if self.debug:
            # Zobrazení spektrogramu s vyznačenou ořezávací čarou
            plt.imshow(spectrogram_normalized, aspect='auto', cmap='inferno', origin='lower',
                       extent=[time_bins.min(), time_bins.max(), frequency_bins.min(), frequency_bins.max()])
            plt.axhline(y=cutoff_values, color='r', linestyle='--')  # Přidání čáry pro ořezání
            plt.title('Spectrogram with Cutoff Line')
            plt.xlabel('Time')
            plt.ylabel('Frequency')
            plt.colorbar(label='Intensity')
            plt.show()

        print(f"cutoff_row: {cutoff_row}, {spectrogram_normalized.shape[0] - cutoff_row}, {cutoff_values}.")
        return cutoff_bin

    def inference(self, spectrum: Spectrogram, bin_size):
        if not spectrum.axis_scale():
            PRINT.info(f'The Spectrum object provided for inference must have both time_bins and frequency_bins defined.')
            return None

        segments = SegmentsHandler.crop_spectrogram(spectrum.values, spectrum.time_bins, crop_size=bin_size)

        outputs = []
        for segment in segments:

            # Inference by měla být pak provedena na objektu
            predicted_mask = self.model.inference(
                {
                    'spectrum_values': segment['spectrum'],
                    'time_bins': segment['time_bins'],
                    'freq_bins': spectrum.frequency_bins
                }
            )

            outputs.append(predicted_mask)

        output_full = torch.cat([segment['mask_values'] for segment in outputs], dim=2)

        original_time_lenght = spectrum.time_bins.shape[0]
        output_original_length = output_full[:, :, :original_time_lenght]

        # spectrum.values = output_original_length

        mask = Mask(values=output_original_length.numpy(), time_bins=spectrum.time_bins, frequency_bins=spectrum.frequency_bins)

        image_PIL = Visualization.spectrum_above_mask(spectrum=spectrum.values,
                                                      mask=mask.values.squeeze(0),
                                                      frequency_bins=spectrum.frequency_bins,
                                                      time_bins=spectrum.time_bins, output_mode='return')

        image_PIL.show()
        return mask

    def reconstruct_stripes_in_binary_mask(self, binary_mask: Mask, vertical_threshold: float = 0.5):
        binary_mask_ = binary_mask.values.squeeze(0)

        filtered_mask = remove_small_objects(binary_mask_, min_size=500)

        if self.debug and binary_mask.axis_scale():
            self.VH.plot_spectrogram(
                time_bins=binary_mask.time_bins,
                freq_bins=binary_mask.frequency_bins,
                spectrum=filtered_mask,
                title='Mask after removing small objects'
            )

        # Count ones in columns
        mask_height = filtered_mask.shape[0]
        counts = np.sum(filtered_mask == 1, axis=0)
        condition = (counts / mask_height) > vertical_threshold

        filtered_mask[:, condition] = 1
        filtered_mask[:, ~condition] = 0

        if self.debug and binary_mask.axis_scale():
            self.VH.plot_spectrogram(
                time_bins=binary_mask.time_bins,
                freq_bins=binary_mask.frequency_bins,
                spectrum=filtered_mask,
                title='Mask after stripes reconstruction'
            )

        return Mask(filtered_mask)

    # def connect_areas(self, binary_mask_dir):
    #
    #     binary_mask = binary_mask_dir['mask_values'].astype(bool)
    #
    #     filtered_mask = remove_small_objects(binary_mask, min_size=500)
    #
    #     self.VH.plot_spectrogram(
    #         time_bins=binary_mask_dir['time_bins'],
    #         freq_bins=binary_mask_dir['frequency_bins'],
    #         spectrum=filtered_mask.squeeze(axis=0),
    #         title='Mask after removing small objects'
    #     )
    #
    #     final_mask = ndimage.binary_closing(filtered_mask)
    #
    #     self.VH.plot_spectrogram(
    #         time_bins=binary_mask_dir['time_bins'],
    #         freq_bins=binary_mask_dir['frequency_bins'],
    #         spectrum=final_mask.squeeze(axis=0),
    #         title='Mask after connecting areas'
    #     )
    #
    #     return None


if __name__ == '__main__':
    input_file = '/Users/vojtechremis/Desktop/Projects/birdsong-segmentation/source_data/xeno-canto/Song_files/A/XC35220_Sarcops_calvus.mp3'
    time_bins, frequency_bins, spectrum, y, sr = SegmentsHandler.load_birdsong(input_file, return_sr=True, plot=False)

    # Setting up model
    dataset = BirdUNetDataset(None, None, transform_attrs={'mean': MEAN, 'std': STD}, debug=True)
    model = UNet(in_channels=1, num_classes=2, DataHandler=dataset)
    model.load_weights(
        '/Users/vojtechremis/Desktop/Projects/birdsong-segmentation/naive_unet/weights/model_weights_split.pth')

    # Run post process
    PP = InferenceProcess(model, debug=True)
    binary_mask = PP(y, sr)


    # # Načtení zpět
    # loaded = np.load("binary_mask.npz")
    # binary_mask = {key: loaded[key] for key in loaded}

    # output = PP.reconstruct_stripes_in_binary_mask(binary_mask=binary_mask, vertical_threshold=10)
