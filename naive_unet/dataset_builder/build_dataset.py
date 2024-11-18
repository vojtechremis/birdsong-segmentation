import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
import sys
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Adds parent directory to sys.path if not already included
parent_dir = Path(__file__).resolve().parent.parent
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))

import pathlib
from Datasource_tweetynet.Inc import segments_handler


def create_segmentation_mask(spectrogram, annotations, bin_size, bin_margin=0):
    # Creating mask with spectrogram shape
    mask = np.zeros(spectrogram.shape)

    # Transforming each annotation from time [s] –> time_bins
    for idx, ann in annotations.iterrows():
        motive_start, motive_end = ann['onset_s'], ann['offset_s']

        motive_start_idx = int(motive_start / bin_size)
        motive_end_idx = int(motive_end / bin_size)

        start_margin = motive_start_idx - bin_margin
        end_margin = motive_end_idx + bin_margin

        if motive_start_idx == 0:
            start_margin = motive_start_idx
        if motive_end_idx == len(spectrogram)-1:
            end_margin = motive_end_idx

        mask[:, start_margin:end_margin] = 1

    return mask


# /Users/vojtechremis/Desktop/Projects/Birds/naive_unet/data

# Preparation of segmentation target labels

if __name__ == '__main__':

    # Extracting data
    wav_data_folders = [
        '/Users/vojtechremis/Desktop/Projects/Birds/Datasource_tweetynet/llb3_data/llb3_songs',
        '/Users/vojtechremis/Desktop/Projects/Birds/Datasource_tweetynet/llb11_data/llb11_songs',
        '/Users/vojtechremis/Desktop/Projects/Birds/Datasource_tweetynet/llb16_data/llb16_songs'
    ]

    output_folder = '/Users/vojtechremis/Desktop/Projects/Birds/naive_unet/data'

    for subfolder in ['spectrogram', 'mask']:
        os.makedirs(os.path.join(output_folder, subfolder), exist_ok=True)

    for wav_data_folder in wav_data_folders:

        annotation_filename = str.split(Path(wav_data_folder).name, '_')[0]+'_annot.csv'
        annotations_df = pd.read_csv(os.path.join(Path(wav_data_folder).parent, annotation_filename))

        for filename in tqdm(os.listdir(wav_data_folder), f'Processing {wav_data_folder}'):
            file_path = os.path.join(wav_data_folder, filename)

            if file_path.endswith('.wav'):

                # Extracting spectrogram
                time_bins, frequency_bins, spectrum, y = segments_handler.load_birdsong(file_path, plot=False)

                # Extracting mask
                annotations_sample = annotations_df[annotations_df['audio_file'] == filename]

                if annotations_sample.empty:
                    print('For file {} no annotations'.format(filename))
                    continue

                bin_size = (time_bins[1] - time_bins[0])
                mask = create_segmentation_mask(spectrum, annotations_sample, bin_size, bin_margin=0)

                plot = False
                # if plot:
                #     # Plot spectrum and mask
                #     fig, ax = plt.subplots(2, 1, figsize=(12, 8))
                #
                #     # Displaying the spectrogram (spectrum)
                #     cax1 = ax[0].imshow(spectrum, aspect='auto', origin='lower', cmap='inferno')
                #     ax[0].set_title(f'Spectrogram of {filename}')
                #     ax[0].set_ylabel('Frequency (Hz)')
                #     ax[0].set_xlabel('Time (s)')
                #
                #     # Set x and y axis ticks
                #     n_time_ticks = 5
                #     n_freq_ticks = 5
                #
                #     # Set ticks for time axis
                #     time_tick_indices = np.linspace(0, len(time_bins) - 1, n_time_ticks, dtype=int)
                #     time_tick_labels = [f'{time_bins[i]:.2f}' for i in time_tick_indices]
                #     ax[0].set_xticks(time_tick_indices)
                #     ax[0].set_xticklabels(time_tick_labels)
                #
                #     # Set ticks for frequency axis
                #     freq_tick_indices = np.linspace(0, len(frequency_bins) - 1, n_freq_ticks, dtype=int)
                #     freq_tick_labels = [f'{frequency_bins[i]:.2f}' for i in freq_tick_indices]
                #     ax[0].set_yticks(freq_tick_indices)
                #     ax[0].set_yticklabels(freq_tick_labels)
                #
                #     # Displaying the mask
                #     cax2 = ax[1].imshow(mask, aspect='auto', origin='lower', cmap='Blues')
                #     ax[1].set_title(f'Segmentation Mask for {filename}')
                #     ax[1].set_ylabel('Frequency (Hz)')
                #     ax[1].set_xlabel('Time (s)')
                #
                #     # Set x and y axis ticks for the mask
                #     ax[1].set_xticks(time_tick_indices)
                #     ax[1].set_xticklabels(time_tick_labels)
                #     ax[1].set_yticks(freq_tick_indices)
                #     ax[1].set_yticklabels(freq_tick_labels)
                #
                #     # Adjust layout
                #     plt.tight_layout()
                #
                #     # Show plots
                #     plt.show()

                image_id_origin = filename.split('_')[1]
                image_group_id = filename.split('_')[0].split('llb')[1]
                image_id = str(image_group_id) + str(0) + str(image_id_origin) # Creating image_id with format llb11_00001_2018_05_03_16_34_20.wav -> 1101

                input = {
                    'image_id': image_id,
                    'spectrum': spectrum,
                    'time_bins': time_bins,
                    'frequency_bins': frequency_bins
                }

                target = {
                    'image_id': image_id,
                    'original_filename': filename,
                    'mask': mask
                }

                spectrogram_filepath = os.path.join(output_folder, 'spectrogram', image_id+'.npz')
                mask_filepath = os.path.join(output_folder, 'mask', image_id+'.npz')

                # saving
                np.savez_compressed(spectrogram_filepath, **input)
                np.savez_compressed(mask_filepath, **target)