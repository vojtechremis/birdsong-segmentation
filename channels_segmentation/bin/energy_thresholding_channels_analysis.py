# Segmentace pomocí:
# 1) Mixit separace signálu
# 2) vypočítání SNR (vybrání jednoho nebo K-Means)
# 3) Energetický thresholding, variance thresholdng

# Výstup: energetický thresholding je horší než variance thresholding

import os
import sys
import tensorflow.compat.v1 as tf
import numpy as np
from scipy.signal import spectrogram
import librosa
import noisereduce as nr
from sklearn.cluster import KMeans
from matplotlib.ticker import FuncFormatter
from matplotlib import pyplot as plt
import pandas as pd
import soundfile as sf

# Disable TensorFlow v2 behavior
tf.disable_v2_behavior()

MIXIT_MODEL_DIR = "/Users/vojtechremis/Desktop/Projects/birdsong-segmentation/background_segmentation/sound-separation/models/bird_mixit/output_sources4"
MIXIT_CHECKPOINT_PATH = "/Users/vojtechremis/Desktop/Projects/birdsong-segmentation/background_segmentation/sound-separation/models/bird_mixit/output_sources4/model.ckpt-3223090"
song_data = "/Users/vojtechremis/Desktop/Projects/birdsong-segmentation/source_data/xeno-canto/song_raw.xlsx"
labeled_data = pd.read_excel(song_data)
segment_comparison_file = "/Users/vojtechremis/Desktop/Projects/birdsong-segmentation/background_segmentation/energy_threshold_output/segment_comparison.txt"


def format_thousands(x, pos):
    """Formatter function to add commas as a thousand separators."""
    return f'{int(x):,}'.replace(',', ' ')


def plot_spectrogram(time_bins, freq_bins, spectrum, log=False, title=None, save_path=None):
    """
    Plot spectogram without annotations.
    """
    if log:
        spectrum = 10 * np.log10(spectrum + 1e-10)

    extent = [time_bins[0], time_bins[-1], freq_bins[0],
              freq_bins[-1]] if freq_bins is not None and time_bins is not None else None
    plt.figure(figsize=(12, 6))
    plt.imshow(spectrum, aspect='auto', origin='lower', extent=extent, cmap='viridis')
    plt.colorbar(label='Intensity (dB)')

    if title is not None:
        plt.title(title, y=1.10)
    else:
        plt.title('Spectrogram', y=1.05)

    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')

    # Format y-axis ticks with thousands separators
    plt.gca().yaxis.set_major_formatter(FuncFormatter(format_thousands))
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    # plt.show()

    return None


def apply_channels_separation(
        input_y,  # 2-channels input audio waveform
        model_dir=MIXIT_MODEL_DIR,
        checkpoint_path=MIXIT_CHECKPOINT_PATH,
        output_channels=4,
        input_tensor_name='input_audio/receiver_audio:0',
        output_tensor_name='denoised_waveforms:0',
):
    """
    Takes waveform and processes it with Google research MixIT model. Output is separated audio signals (list of waveforms).

    Args:
        model_dir (str): Path to the model directory containing inference.meta.
        input_y (np.ndarray): 2-channels input audio waveform.
        checkpoint_path (str): Optional path to a specific checkpoint.
        output_channels (int): Number of output channels to truncate to.
        input_tensor_name (str): Name of the input tensor in the model.
        output_tensor_name (str): Name of the output tensor in the model.

    Returns:
        np.ndarray: Separated audio signals, shape (samples, sources).
        int: Sample rate of the input audio.
    """

    # Load TF model for inference
    meta_graph_filename = os.path.join(model_dir, 'inference.meta')
    print(f"Importing meta graph: {meta_graph_filename}")

    with tf.Graph().as_default() as g:
        saver = tf.train.import_meta_graph(meta_graph_filename)
        meta_graph_def = g.as_graph_def()

    tf.reset_default_graph()
    input_wav = tf.cast(input_y, dtype=tf.float32)

    # Prepare the input tensor
    input_wav = tf.transpose(input_wav)
    input_wav = tf.expand_dims(input_wav, axis=0)

    # Import the graph and specify input/output tensors
    output_wav, = tf.import_graph_def(
        meta_graph_def,
        name='',
        input_map={input_tensor_name: input_wav},
        return_elements=[output_tensor_name]
    )

    # Post-process the output
    output_wav = tf.squeeze(output_wav, 0)  # shape: [sources, samples]
    output_wav = tf.transpose(output_wav)

    if output_channels > 0:
        output_wav = output_wav[:, :output_channels]

    # Restore from the checkpoint
    if not checkpoint_path:
        checkpoint_path = tf.train.latest_checkpoint(model_dir)
    print(f"Restoring from checkpoint: {checkpoint_path}")

    # Do inference
    with tf.Session() as sess:
        saver.restore(sess, checkpoint_path)
        separated_signals = sess.run(output_wav)

    return separated_signals


def calculate_snr(signal, noise_segment):
    """
    Calculate the Signal-to-Noise Ratio (SNR) using a noise reference segment.

    Parameters:
    - signal (np.ndarray): the audio waveform of the separated channel.
    - noise_segment (np.ndarray): the noise reference segment taken from the beginning of the original mix.

    Returns:
    - snr_db: float, the computed SNR in dB.
    """
    # Compute power of the signal
    signal_power = np.mean(signal ** 2)

    # Compute power of the noise
    noise_power = np.mean(noise_segment ** 2)

    # Avoid division by zero
    if noise_power == 0:
        noise_power = 1e-10

    # Compute SNR in decibels
    snr_db = 10 * np.log10(signal_power / noise_power)

    return snr_db


def VarianceThresholding(y, sr, window_size, hop_size, variance_threshold):
    window_samples = int(window_size * sr)
    hop_samples = int(hop_size * sr)
    variance_values = []
    times_variance = []

    for start in range(0, len(y) - window_samples, hop_samples):
        window = y[start:start + window_samples]
        var = np.var(window)
        variance_values.append(var)
        times_variance.append(start / sr)

    # Detekce segmentů pomocí variance thresholdingu
    segment_times_variance = []
    in_segment = False
    segment_start = 0

    for i, var in enumerate(variance_values):
        if var > variance_threshold:
            if not in_segment:
                segment_start = times_variance[i]
                in_segment = True
        else:
            if in_segment:
                segment_times_variance.append((segment_start, times_variance[i]))
                in_segment = False

    if in_segment:
        segment_times_variance.append((segment_start, times_variance[-1]))

    print("Variance thresholding segments:", segment_times_variance)

    return segment_times_variance


def process_audio_file(input_file, output_dir, segments_ground_true, energy_threshold=0.02):
    ## 1. Separate channels from the input audio
    y_raw, sr = librosa.load(input_file, sr=None)

    # Create details directory
    details_directory = os.path.join(output_dir, os.path.splitext(os.path.basename(input_file))[0])
    if not os.path.exists(details_directory):
        os.makedirs(details_directory)

    window_size = 1024
    overlap = 512
    frequency_bins, time_bins, spectrum_raw = spectrogram(y_raw, fs=sr, nperseg=window_size, noverlap=overlap)
    plot_spectrogram(time_bins, frequency_bins, spectrum_raw, log=True, title="Original audio spectrogram",
                     save_path=os.path.join(details_directory, "original_audio_spectrogram.png"))

    # Separate channels using MixIT model
    y = y_raw.reshape(-1, 1)

    if y.shape[1] == 1:
        padding = np.zeros((y.shape[0], 2 - y.shape[1]))
        y = np.hstack((y, padding))
    elif y.shape[1] > 2:
        print("Too many channels!")
        sys.exit(0)

    separated_channels = apply_channels_separation(y, output_channels=4)

    ## 2. Calculate SNR for each channel (signal-to-noise ratio) in order to determine foreground and background channels
    separated_spectrum_list = []
    snr_values = []

    for i in range(separated_channels.shape[1]):
        # Get mono signal back
        separated_signal_mono = separated_channels[:, i]

        frequency_bins, time_bins, spectrum = spectrogram(separated_signal_mono, fs=sr, nperseg=window_size,
                                                          noverlap=overlap)

        # Plot the spectrogram
        plot_spectrogram(time_bins, frequency_bins, spectrum, log=True, title=f"Channel num. {i + 1} - STFT Magnitude",
                         save_path=os.path.join(details_directory, f"channel_{i + 1}.png"))

        # Download audio file
        sf.write(os.path.join(details_directory, f"channel_{i + 1}.wav"), separated_signal_mono, sr)

        # Calculate noise in order to calculate SNR
        y_rn = nr.reduce_noise(y=separated_signal_mono, sr=sr)
        noise_estimate = separated_signal_mono - y_rn
        separated_spectrum_list.append((time_bins, frequency_bins, spectrum, separated_signal_mono, y_rn))

        # Calculate SNR
        snr_db_global = calculate_snr(separated_signal_mono, noise_estimate)
        snr_values.append((i, snr_db_global))  # (channel index, SNR value)

    ## 3. Use K-Means clustering for channel grouping (foreground and background channels)

    snr_values = sorted(snr_values, key=lambda x: x[1], reverse=True)

    print("SNR values (sorted):", [f"Channel num.:{i[0] + 1}, SNR value: {i[1]}" for i in snr_values])

    # Apply K-Means clustering
    # snr_values_only = np.array(snr_values)[:, 1].reshape(-1, 1)

    # kmeans = KMeans(n_clusters=2, random_state=42, n_init=10, init='k-means++')
    # labels = kmeans.fit_predict(snr_values_only)

    # foreground_cluster_label = np.argmax(kmeans.cluster_centers_)

    # # Channel groups
    # foreground_channels = [separated_spectrum_list[i] for i in range(len(separated_spectrum_list)) if
    #                        labels[i] == foreground_cluster_label]
    # background_channels = [separated_spectrum_list[i] for i in range(len(separated_spectrum_list)) if
    #                        labels[i] != foreground_cluster_label]
    #
    # foreground_channels_labels = [i for i in range(len(separated_spectrum_list)) if
    #                               labels[i] == foreground_cluster_label]
    # background_channels_labels = [i for i in range(len(separated_spectrum_list)) if
    #                                 labels[i] != foreground_cluster_label]

    # Foreground channel is the one with the highest SNR value
    snr_values = np.array(snr_values)
    foreground_channel_index = int(snr_values[np.argmax(snr_values[:, 1]), 0])
    foreground_channels_labels = [foreground_channel_index]
    background_channels_labels = [i for i in range(len(separated_spectrum_list)) if i != foreground_channel_index]

    print("Foreground channel index:", foreground_channel_index)

    foreground_channels = [separated_spectrum_list[foreground_channel_index]]

    print("Foreground channels numbers:", [i + 1 for i in foreground_channels_labels])
    print("Background channels numbers:", [i + 1 for i in background_channels_labels])

    # Save info to txt file
    with open(os.path.join(details_directory, "snr_values.txt"), "w") as f:
        f.write("SNR values (sorted):\n")
        for i in snr_values:
            f.write(f"Channel num.: {i[0] + 1}, SNR value: {i[1]}\n")
        f.write(f"\nForeground channels numbers: {[i + 1 for i in foreground_channels_labels]}")
        f.write(f"\nBackground channels numbers: {[i + 1 for i in background_channels_labels]}")

    # Plot the SNR values
    channels = np.array(snr_values)[:, 0].astype(int) + 1  # Channel index
    snr_values = np.array(snr_values)[:, 1]  # SNR values

    # Calculate relative SNR
    max_snr = np.max(snr_values)
    relative_snr = snr_values / max_snr if max_snr != 0 else snr_values

    plt.figure(figsize=(8, 5))
    plt.bar(channels, relative_snr, color='blue', alpha=0.7)
    plt.xlabel("Channels index")
    plt.ylabel("Relative SNR (normalized)")
    plt.title("Relative SNR (single channels)")
    plt.xticks(channels)
    plt.ylim(0, 1.1)
    plt.savefig(os.path.join(details_directory, "relative_snr.png"))

    ## 4. Perform segmentation using energy thresholding (foreground channels only)

    # Calculating energy threshold for original audio (for comparison)

    # Výpočet síly v určitém frekvenčním rozsahu (např. 1-5 kHz)
    low_freq = 1000  # Dolní hranice frekvence (Hz)
    high_freq = 10000  # Horní hranice frekvence (Hz)
    freq_indices = np.where((frequency_bins >= low_freq) & (frequency_bins <= high_freq))[0]
    y_raw_nr = nr.reduce_noise(y=y_raw, sr=sr)
    frequency_bins, time_bins, spectrum_raw_nr = spectrogram(y_raw_nr, fs=sr, nperseg=window_size, noverlap=overlap)

    energy_raw = np.sum(spectrum_raw_nr[freq_indices, :], axis=0)
    normalized_energy_raw = energy_raw / np.max(energy_raw)
    Sxx_log_raw = 10 * np.log10(spectrum_raw_nr + 1e-10)

    spectrum_nr_combined = 0
    y_nr_combined = 0
    for channel in foreground_channels:
        time_bins, frequency_bins, spectrum, y_raw, y_nr = channel[0], channel[1], channel[2], channel[3], channel[4]

        # Content below can be possibly replaced with different segmentation analysis
        # Variables to use:
        # - y_nr: waveform with reduced noise
        # - y_raw: original waveform
        # - spectrum: spectrogram of the raw channel (without noise removing)

        frequency_bins, time_bins, spectrum_nr = spectrogram(y_nr, fs=sr, nperseg=window_size, noverlap=overlap)

        spectrum_nr_combined += spectrum_nr
        y_nr_combined += y_nr

        # plot_spectrogram(time_bins, frequency_bins, spectrum_nr, log=True, title="Foreground channel (reduced noise)")

    # Energy peaks thresholding
    Sxx = spectrum_nr_combined  # spectrum_nr

    # Parametry segmentace
    low_freq = 1000  # Dolní hranice frekvence (Hz)
    high_freq = 10000  # Horní hranice frekvence (Hz)
    energy_threshold = 0.1  # Práh pro energy thresholding
    variance_threshold = 1e-5  # Práh pro variance thresholding
    window_size = 0.05  # Velikost okna pro variance (v sekundách)
    hop_size = 0.01  # Posun okna (v sekundách)

    # Výpočet energie ve vybraném frekvenčním rozsahu
    freq_indices = np.where((frequency_bins >= low_freq) & (frequency_bins <= high_freq))[0]
    energy = np.sum(Sxx[freq_indices, :], axis=0)
    normalized_energy = energy / np.max(energy)

    # Detekce segmentů pomocí energy thresholdingu
    segment_times_energy = []
    in_segment = False
    segment_start = 0

    for i, e in enumerate(normalized_energy):
        if e > energy_threshold:
            if not in_segment:
                segment_start = time_bins[i]
                in_segment = True
        else:
            if in_segment:
                segment_times_energy.append((segment_start, time_bins[i]))
                in_segment = False

    if in_segment:
        segment_times_energy.append((segment_start, time_bins[-1]))

    print("Energy thresholding segments:", segment_times_energy)

    # Výpočet variance v klouzavém okně
    segment_times_variance = VarianceThresholding(y_nr_combined, sr, window_size, hop_size, variance_threshold)

    # Výpočet variance v klouzavém okně (raw)
    segment_times_variance_raw = VarianceThresholding(y_raw_nr, sr, window_size, hop_size, variance_threshold)


    # Vizualizace výsledků
    plt.figure(figsize=(30, 10))

    # Horní graf – Spektrogram + Energy Thresholding
    plt.subplot(3, 1, 1)
    plt.pcolormesh(time_bins, frequency_bins, 10 * np.log10(Sxx + 1e-10), cmap='gray', shading='auto')
    plt.colorbar(label="Intensity [dB]")
    plt.title("Segmentace pomocí Energy Thresholding")
    plt.xlabel("Čas [s]")
    plt.ylabel("Frekvence [Hz]")

    # Zvýraznění detekovaných segmentů (Energy)
    for start, end in segment_times_energy:
        plt.axvspan(start, end, color='red', alpha=0.3,
                    label="Energy segment" if start == segment_times_energy[0][0] else "")

    # Spodní graf – Variance Thresholding
    plt.subplot(3, 1, 2)
    plt.pcolormesh(time_bins, frequency_bins, 10 * np.log10(Sxx + 1e-10), cmap='gray', shading='auto')
    plt.colorbar(label="Intensity [dB]")
    plt.plot(time_bins, normalized_energy * frequency_bins[-1], color='green', label="Energy (normalized)")
    plt.title("Segmentace pomocí Variance Thresholding")
    plt.xlabel("Čas [s]")
    plt.ylabel("Frekvence [Hz]")

    # Zvýraznění detekovaných segmentů (Variance)
    for start, end in segment_times_variance:
        plt.axvspan(start, end, color='blue', alpha=0.3,
                    label="Variance segment" if start == segment_times_variance[0][0] else "")

    plt.legend()
    plt.tight_layout()

    # Vykreslení dolního grafu (porovnání, jak by segmentace vypadala bez channel separation)
    plt.subplot(3, 1, 3)
    plt.pcolormesh(time_bins, frequency_bins, Sxx_log_raw, cmap='gray', shading='auto')
    plt.colorbar(label='Intensity [dB]')
    plt.title(f"Energy thresholding of original audio (for comparison)")
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.plot(time_bins, normalized_energy_raw * frequency_bins[-1], color='red', label="Energy (normalized)")

    # Zvýraznění detekovaných segmentů (Variance)
    for start, end in segment_times_variance_raw:
        plt.axvspan(start, end, color='green', alpha=0.3,
                    label="Variance segment" if start == segment_times_variance_raw[0][0] else "")

    plt.legend()

    # Uložení grafu do souboru
    audio_id = os.path.splitext(os.path.basename(input_file))[0]
    output_path = os.path.join(output_dir, audio_id + ".png")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.savefig(os.path.join(details_directory, "energy_thresholding.png"))

    # append to segment_comparison_file .txt
    with open(segment_comparison_file, "a") as f:
        f.write(
            f"ID: {audio_id}. \tEnergy thresholding segments: {len(segment_times_energy)}. \tVariance thresholding segments: {len(segment_times_variance)}.\t Ground true: {segments_ground_true}\n")

    with open(os.path.join(details_directory, "snr_values.txt"), "a") as f:
        f.write(
            f"\nID: {audio_id}. \tEnergy thresholding segments: {len(segment_times_energy)}. \tVariance thresholding segments: {len(segment_times_variance)}.\t Ground true: {segments_ground_true}\n")


    print(f"ID: {audio_id}. \tEnergy thresholding segments: {len(segment_times_energy)}. \tVariance thresholding segments: {len(segment_times_variance)}.\t Ground true: {segments_ground_true}\n")


if __name__ == "__main__":
    import random

    ## 1. Separate channels from the input audio
    # input_file = "/Users/vojtechremis/Downloads/Example_5_(amepip,_gcrfin,_whcspa)_mix.wav"
    # audio_folder = "/Users/vojtechremis/Desktop/Projects/birdsong-segmentation/naive_unet/postprocess/separated_audio"
    # output_dir = "/Users/vojtechremis/Desktop/Projects/birdsong-segmentation/background_segmentation/energy_threshold_output"
    #
    # process_audio_file(
    #     input_file,
    #     output_dir
    # )

    # Process files in the directory
    folder_path = "/Users/vojtechremis/Desktop/Projects/birdsong-segmentation/source_data/xeno-canto/Song_files/A"
    audio_folder = "/Users/vojtechremis/Desktop/Projects/birdsong-segmentation/naive_unet/postprocess/separated_audio"
    output_dir = "/Users/vojtechremis/Desktop/Projects/birdsong-segmentation/background_segmentation/energy_threshold_output"

    # Get names of the files from pd.dataframe
    labeled_data_filtered = labeled_data[(labeled_data["elements"] != 50) & (labeled_data["rec_quality_subj"] == "A")]

    # Pick random 150 samples
    num_files_to_process = min(150, len(labeled_data_filtered))
    random_rows = labeled_data_filtered.sample(n=num_files_to_process)

    selected_files = []
    for index, row in random_rows.iterrows():
        xeno_canto_id = row["XenoCantoID"]
        segments_gt = row["elements"]
        file_prefix = f"{xeno_canto_id}_"

        # try:
        file_name = next(f for f in os.listdir(folder_path) if f.startswith(file_prefix))

        file_path = os.path.join(folder_path, file_name)
        process_audio_file(file_path, output_dir, segments_gt)
        print(f"Processed: {file_name}")

        # except Exception as e:
        #     print("Could not process file:", file_prefix)
        #     continue
