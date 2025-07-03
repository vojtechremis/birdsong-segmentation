# Tento skript slouží k ověření hypotézy, že pokud segmentuji ptačí syllaby pomocí thresholdu (v našem případě korelační threshold),
# tak nějaký fixní threshold nebu dobrý pro všechny nahrávky, a tedy nelze najít obecný threshold.

#!/usr/bin/env python
import os
import sys
import tensorflow.compat.v1 as tf
import numpy as np
from scipy.signal import spectrogram
import librosa
import noisereduce as nr
import matplotlib.pyplot as plt

# Disable TensorFlow v2 behavior
tf.disable_v2_behavior()

# Cesty k modelu MixIT (upravte dle potřeby)
MIXIT_MODEL_DIR = "/Users/vojtechremis/Desktop/Projects/birdsong-segmentation/background_segmentation/sound-separation/models/bird_mixit/output_sources4"
MIXIT_CHECKPOINT_PATH = "/Users/vojtechremis/Desktop/Projects/birdsong-segmentation/background_segmentation/sound-separation/models/bird_mixit/output_sources4/model.ckpt-3223090"


def apply_channels_separation(
        input_y,
        model_dir=MIXIT_MODEL_DIR,
        checkpoint_path=MIXIT_CHECKPOINT_PATH,
        output_channels=4,
        input_tensor_name='input_audio/receiver_audio:0',
        output_tensor_name='denoised_waveforms:0',
):
    """
    Aplikuje MixIT model pro separaci kanálů.
    """
    meta_graph_filename = os.path.join(model_dir, 'inference.meta')
    print(f"Importing meta graph: {meta_graph_filename}")

    with tf.Graph().as_default() as g:
        saver = tf.train.import_meta_graph(meta_graph_filename)
        meta_graph_def = g.as_graph_def()
    tf.reset_default_graph()
    input_wav = tf.cast(input_y, dtype=tf.float32)
    input_wav = tf.transpose(input_wav)
    input_wav = tf.expand_dims(input_wav, axis=0)

    output_wav, = tf.import_graph_def(
        meta_graph_def,
        name='',
        input_map={input_tensor_name: input_wav},
        return_elements=[output_tensor_name]
    )

    output_wav = tf.squeeze(output_wav, 0)  # shape: [sources, samples]
    output_wav = tf.transpose(output_wav)

    if output_channels > 0:
        output_wav = output_wav[:, :output_channels]

    if not checkpoint_path:
        checkpoint_path = tf.train.latest_checkpoint(model_dir)
    print(f"Restoring from checkpoint: {checkpoint_path}")

    with tf.Session() as sess:
        saver.restore(sess, checkpoint_path)
        separated_signals = sess.run(output_wav)

    return separated_signals


def calculate_snr(signal, noise_segment):
    """
    Vypočítá SNR v dB.
    """
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise_segment ** 2)
    if noise_power == 0:
        noise_power = 1e-10
    snr_db = 10 * np.log10(signal_power / noise_power)
    return snr_db


def energy_segmentation_segments(spectrum, time_bins, frequency_bins, energy_threshold, low_freq=1000, high_freq=10000):
    """
    Aplikuje energy thresholding na daný spektrogram.

    Vrací seznam segmentů (začátek, konec) v sekundách.
    """
    freq_indices = np.where((frequency_bins >= low_freq) & (frequency_bins <= high_freq))[0]
    energy = np.sum(spectrum[freq_indices, :], axis=0)
    normalized_energy = energy / np.max(energy)

    segment_times = []
    in_segment = False
    segment_start = None
    for i, e in enumerate(normalized_energy):
        if e > energy_threshold:
            if not in_segment:
                segment_start = time_bins[i]
                in_segment = True
        else:
            if in_segment:
                segment_times.append((segment_start, time_bins[i]))
                in_segment = False
    if in_segment:
        segment_times.append((segment_start, time_bins[-1]))
    return segment_times


def get_foreground_spectrogram(input_file, nperseg=1024, noverlap=512):
    """
    Načte audio, provede separaci kanálů, vybere foreground kanál dle SNR a
    vrátí spektrogram získaný z redukovaného (noise reduced) signálu.
    """
    # Načtení audio souboru
    y_raw, sr = librosa.load(input_file, sr=None)

    # Připraví audio pro separaci (musí mít alespoň 2 kanály)
    y = y_raw.reshape(-1, 1)
    if y.shape[1] == 1:
        padding = np.zeros((y.shape[0], 2 - y.shape[1]))
        y = np.hstack((y, padding))
    elif y.shape[1] > 2:
        print("Too many channels!")
        sys.exit(0)

    # Separace kanálů pomocí MixIT
    separated_channels = apply_channels_separation(y, output_channels=4)

    # Výpočet SNR pro každý kanál a výběr foreground kanálu (nejvyšší SNR)
    snr_values = []
    separated_spectrum_list = []
    for i in range(separated_channels.shape[1]):
        separated_signal_mono = separated_channels[:, i]
        freq_bins, time_bins, spectrum = spectrogram(separated_signal_mono, fs=sr, nperseg=nperseg, noverlap=noverlap)
        y_rn = nr.reduce_noise(y=separated_signal_mono, sr=sr)
        noise_estimate = separated_signal_mono - y_rn
        snr_db = calculate_snr(separated_signal_mono, noise_estimate)
        snr_values.append((i, snr_db))
        separated_spectrum_list.append((time_bins, freq_bins, spectrum, separated_signal_mono, y_rn))

    snr_values = np.array(snr_values)
    foreground_channel_index = int(snr_values[np.argmax(snr_values[:, 1]), 0])
    foreground_data = separated_spectrum_list[foreground_channel_index]

    # Pro segmentaci použijeme redukovaný signál
    y_rn = foreground_data[4]
    time_bins, freq_bins, spectrum_nr = spectrogram(y_rn, fs=sr, nperseg=nperseg, noverlap=noverlap)
    return sr, time_bins, freq_bins, spectrum_nr, y_raw


def grid_search_energy_threshold(input_file, ground_truth_segments, threshold_values, nperseg=1024, noverlap=512,
                                 low_freq=1000, high_freq=10000):
    """
    Provede grid search přes hodnoty energy_threshold a pro každou hodnotu vypočítá
    počet detekovaných segmentů a absolutní chybu (|detekovaných - ground truth|).

    Vrací seznam výsledků: (energy_threshold, počet segmentů, chyba).
    """
    sr, time_bins, freq_bins, spectrum, y_raw = get_foreground_spectrogram(input_file, nperseg, noverlap)
    results = []
    for thresh in threshold_values:
        segments = energy_segmentation_segments(spectrum, time_bins, freq_bins, energy_threshold=thresh,
                                                low_freq=low_freq, high_freq=high_freq)
        num_segments = len(segments)
        error = abs(num_segments - ground_truth_segments)
        print(
            f"Threshold: {thresh:.3f}, Detected segments: {num_segments}, Ground truth: {ground_truth_segments}, Error: {error}")
        results.append((thresh, num_segments, error))
    return results


def main():
    # Upravte cesty a ground truth podle vašich dat
    input_file = "/path/to/your/audio/file.wav"  # Nahraďte skutečnou cestou k audio souboru
    ground_truth_segments = 10  # Nahraďte skutečným počtem segmentů z anotace

    # Definice rozsahu hodnot energy threshold k testování
    threshold_values = np.linspace(0.05, 0.2, num=16)  # např. 16 hodnot od 0.05 do 0.2

    results = grid_search_energy_threshold(input_file, ground_truth_segments, threshold_values)

    # Rozbalení výsledků a vykreslení grafu
    thresholds, detected_segments, errors = zip(*results)
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, errors, marker='o')
    plt.xlabel("Energy Threshold")
    plt.ylabel("Absolutní chyba (|detekovaných - ground truth|)")
    plt.title("Grid Search: Energy Threshold vs. Chyba segmentace")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()