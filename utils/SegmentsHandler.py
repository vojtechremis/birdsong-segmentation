from scipy.stats import mode
import pandas as pd
import numpy as np
import torch
import librosa
from matplotlib import pyplot as plt


import numpy as np
import librosa

def calculate_spectrogram(y, sr, thresh=None, NFFT=1024, hop_length=119, transform_type=None, use_mel=False, n_mels=128):
    """
    Vypočítá STFT nebo MEL spektrogram.

    Parametry:
    - y: Audio signál
    - sr: Vzorkovací frekvence
    - thresh: Prahová hodnota pro log transformaci
    - NFFT: Počet bodů pro FFT
    - hop_length: Posun okna mezi výpočty STFT
    - transform_type: Typ transformace ("log_spect", "log_spect_plus_one" nebo None)
    - use_mel: Použít MEL transformaci (True = výstup v MEL měřítku)
    - n_mels: Počet MEL frekvenčních pásem

    Návratové hodnoty:
    - time_bins: Časové osy binů
    - frequency_bins: Frekvenční osy binů (Hz nebo MEL)
    - spectrum: Výstupní spektrogram (STFT nebo MEL)
    """

    # Výpočet STFT
    fourier_transform = librosa.stft(y, n_fft=NFFT, hop_length=hop_length)
    spectrum = np.abs(fourier_transform)

    if use_mel:
        # Převod na MEL spektrogram
        spectrum = librosa.feature.melspectrogram(S=spectrum**2, sr=sr, n_mels=n_mels, fmax=sr/2)
        frequency_bins = librosa.mel_frequencies(n_mels=n_mels, fmin=0, fmax=sr/2)
    else:
        frequency_bins = np.linspace(0, sr / 2, int(1 + NFFT // 2))

    # Transformace spektrogramu
    if transform_type == "log_spect":
        spectrum /= spectrum.max()  # Normalizace
        spectrum = np.log10(spectrum)  # Logaritmická transformace
        if thresh:
            spectrum[spectrum < -thresh] = -thresh
    elif transform_type == "log_spect_plus_one":
        spectrum = np.log10(spectrum + 1)
        if thresh:
            spectrum[spectrum < thresh] = thresh
    else:
        # Převod na dB, pokud není zvolená žádná specifická transformace
        spectrum = librosa.amplitude_to_db(spectrum, ref=np.max)

    # Výpočet časové osy binů
    time_bins = librosa.frames_to_time(np.arange(spectrum.shape[1]), sr=sr, hop_length=hop_length)

    return time_bins, frequency_bins, spectrum


def load_birdsong(file_path, plot=True, transform_type=None, thresh=None, NFFT=1024, hop_length=119, return_sr=False):
    """
    Load songs from .mp3 and .wav files with optional log transformation.

    :param file_path: Path to the song file.
    :param plot: Boolean indicating whether amplitudes, spectrum, etc. should be plotted.
    :param transform_type: Logarithmic transformation type ('log_spect' or 'log_spect_plus_one').
    :param thresh: Threshold for minimum power in log spectrogram.
    :return: time_bins, frequency_bins, and transformed spectrum.
    """

    # Loading song file
    y, sr = librosa.load(file_path, sr=None)  # 'y' is amplitude, 'sr' is sampling rate

    time_bins, frequency_bins, spectrum = calculate_spectrogram(y, sr, thresh=thresh, NFFT=NFFT, hop_length=hop_length,
                                                                transform_type=transform_type)

    if plot:
        # Plot the waveform
        plt.figure(figsize=(10, 4))
        librosa.display.waveshow(y, sr=sr)
        plt.title('Waveform')
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude')
        plt.show()

        # Plot the spectrogram
        plt.figure(figsize=(10, 6))
        librosa.display.specshow(spectrum, sr=sr, hop_length=hop_length, x_axis='time', y_axis='linear')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Short-Term Fourier Transform (Spectrogram)')
        plt.xlabel('Time [s]')
        plt.ylabel('Frequency [Hz]')
        plt.tight_layout()
        plt.show()

    if return_sr:
        return time_bins, frequency_bins, spectrum, y, sr
    else:
        return time_bins, frequency_bins, spectrum, y


def crop_spectrogram(spectrum, time_bins, crop_size=128):
    """
    Rozdělí spektrogram na úseky o velikosti crop_size a doplní paddingem, pokud je to nutné.
    Vrátí seznam slovníků, každý obsahující oříznuté spektrum a odpovídající časové biny.

    Parameters:
    spectrum (np.ndarray): Spectogram with shape (frequency_bins, time_bins).
    time_bins (np.ndarray): An list of time bins corresponding to spectrum.
    crop_size (int): Maximum number of time_bins each segment may contain (size of time window). Default is 88.

    Returns:
    list: List of dictionaries, where each element of list contains:
        - 'spectrum': torch.Tensor with shape (1, frequency_bins, crop_size)
        - 'time_bins': np.ndarray with corresponding time bins corresponding to spectrum.
    """
    frequency_bins, original_time_bins = spectrum.shape
    cropped_segments = []

    # Looping through a spectrum
    for start_idx in range(0, original_time_bins, crop_size):
        end_idx = start_idx + crop_size

        # If cropped segment is shorter than crop_size, it will be padded with zeros
        if end_idx > original_time_bins:
            padding_size = end_idx - original_time_bins
            padded_spectrum = np.pad(spectrum[:, start_idx:original_time_bins],
                                     ((0, 0), (0, padding_size)), mode='constant')
            cropped_time_bins = time_bins[start_idx:original_time_bins]
        else:
            padded_spectrum = spectrum[:, start_idx:end_idx]
            cropped_time_bins = time_bins[start_idx:end_idx]

        # Convert to tensor (and adding batch dimension)
        crop_tensor = torch.tensor(padded_spectrum, dtype=torch.float32).unsqueeze(0)

        cropped_segments.append({
            'spectrum': crop_tensor,
            'time_bins': cropped_time_bins
        })

    return cropped_segments


def remove_short_segments(predictions, min_duration):
    """
    Removes short segments from predictions. Short segments are being replaced by 0.

    :param predictions: numpy array or torch tensor with predictions
    :param min_duration: minimal segment threshold
    :return: filtered_predictions
    """
    # Shape (time_bins, num_classes) is assumed
    filtered_predictions = np.zeros_like(predictions)
    current_segment = []
    start_index = None

    for i in range(predictions.shape[0]):
        # If bin is not a background
        if predictions[i] != 0:  # Note, that 'class 0' = 'background'
            if start_index is None:
                start_index = i
            current_segment.append(predictions[i])
        else:
            # If bin is a background
            if current_segment:
                if len(current_segment) >= min_duration:
                    filtered_predictions[start_index:i] = current_segment  # Uložení segmentu do outputu
                # Segment je krátký, nebude se ukládat (zůstane 0)
                current_segment = []
                start_index = None  # Reset počátečního indexu

    # Zpracování posledního segmentu
    if current_segment and len(current_segment) >= min_duration:
        filtered_predictions[start_index:] = current_segment  # Uložení segmentu na konec

    return filtered_predictions


def majority_vote_post_processing(predictions, time_bins):
    """
    Aplikuje většinové hlasování na segmenty predikcí a převádí segmenty zpět na časové intervaly.

    :param predictions: numpy array s predikcemi (časové bity), kde 0 znamená pozadí
    :param time_bins: numpy array s časovými biny (časové značky ve vteřinách)
    :return: processed_labels, segments_info
    """
    processed_labels = np.copy(predictions)
    segments_info = []  # Seznam pro ukládání informací o segmentech (onset, offset, label)

    current_segment = []
    start_index = None

    for i in range(len(predictions)):
        if predictions[i] != 0:  # Pokud predikce není pozadí (0)
            if start_index is None:
                start_index = i  # Uložení počátečního indexu segmentu
            current_segment.append(predictions[i])
        else:
            if current_segment:
                # Většinové hlasování pro aktuální segment
                most_common_label = mode(current_segment, keepdims=True).mode[0]
                processed_labels[start_index:i] = most_common_label  # Přepíše hodnoty segmentu na nejčastější label

                # Převod časového úseku segmentu na sekundy
                onset_time = time_bins[start_index]
                offset_time = time_bins[i - 1]  # Konec segmentu (poslední index segmentu)
                segments_info.append((onset_time, offset_time, most_common_label))

                # Reset segmentu
                current_segment = []
                start_index = None

    # Zpracování posledního segmentu, pokud existuje
    if current_segment:
        most_common_label = mode(current_segment, keepdims=True).mode[0]
        processed_labels[start_index:] = most_common_label
        onset_time = time_bins[start_index]
        offset_time = time_bins[-1]  # Konec segmentu (poslední časový bin)
        segments_info.append((onset_time, offset_time, most_common_label))

    return processed_labels, segments_info


def label_short_background_segments(labels, time_bins, min_duration_sec):
    label_to_assign = 999
    modified_labels = labels.copy()
    time_bin = time_bins[1] - time_bins[0]
    min_bins = int(min_duration_sec / time_bin)  # Minimální počet binů pro prahovou hodnotu
    start = None

    for i, label in enumerate(labels):
        if label == 0:  # Background segment
            if start is None:
                start = i  # Start nového background segmentu
        else:
            if start is not None:  # Konec background segmentu
                segment_length = i - start
                if segment_length < min_bins:  # Pokud je segment kratší než prahová hodnota
                    modified_labels[start:i] = label_to_assign  # Nastavení třídy 999 pro krátký segment
                start = None

    # Zpracování posledního background segmentu na konci
    if start is not None:
        segment_length = len(labels) - start
        if segment_length < min_bins:
            modified_labels[start:] = label_to_assign  # Nastavení třídy 999 pro krátký segment

    return modified_labels
