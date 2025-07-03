import sys
sys.path.append("..")
from ChannelsSeparations import process_wav
from utils import SegmentsHandler, Visualization
import os
import soundfile as sf
from matplotlib import pyplot as plt
import numpy as np
import noisereduce as nr
from sklearn.cluster import KMeans
from scipy.signal import spectrogram

input_file = "/Users/vojtechremis/Downloads/Example_5_(amepip,_gcrfin,_whcspa)_mix.wav"
audio_folder = "/Users/vojtechremis/Desktop/Projects/birdsong-segmentation/naive_unet/postprocess/separated_audio"

mixit_model_dir = "/Users/vojtechremis/Desktop/Projects/birdsong-segmentation/background_segmentation/sound-separation/models/bird_mixit/output_sources4"
mixit_checkpoint_path = "/Users/vojtechremis/Desktop/Projects/birdsong-segmentation/background_segmentation/sound-separation/models/bird_mixit/output_sources4/model.ckpt-3223090"

# num_sources = 4

time_bins, frequency_bins, spectrum, y, sr = SegmentsHandler.load_birdsong(input_file, return_sr=True,
                                                                           plot=False)
VH = Visualization.VisualizationHandler()
source_image = VH.plot_spectrogram(time_bins, frequency_bins, spectrum)
source_image.show()

reduced_noise = nr.reduce_noise(y=y, sr=sr)
noise_estimate = y - reduced_noise

# Uložení odšuměného zvuku
sf.write("birdsong_denoised.wav", reduced_noise, sr)

# Uložení extrahovaného šumu
sf.write("birdsong_noise_estimate.wav", noise_estimate, sr)

original_file = os.path.join(audio_folder, "original.wav")
sf.write(original_file, y, sr)

# Umělé vytvoření stereo signálu
y = y.reshape(-1, 1)
padding = np.zeros((y.shape[0], 2 - y.shape[1]))
y = np.hstack((y, padding))

separated_signals, sample_rate_ = process_wav(
    model_dir=mixit_model_dir,
    input_wav=y,
    sample_rate=sr,
    checkpoint_path=mixit_checkpoint_path,
    output_channels=4
)

print(f"Separated signals shape: {separated_signals.shape}")
print(f"Sample rate: {sample_rate_}")


def calculate_snr_global(signal, noise_segment):
    """
    Calculate the Signal-to-Noise Ratio (SNR) using a noise reference segment.

    Parameters:
    - signal: np.array, the audio waveform of the separated channel.
    - noise_segment: np.array, the noise reference segment taken from the beginning of the original mix.

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


# Extract a noise reference from the beginning of the original mix
# noise_duration = 0.1  # in seconds (adjust as needed)
# noise_samples = int(noise_duration * sr)  # Convert to samples
# noise_reference = y[:noise_samples]


snr_values_global = []
Spectrums = []
for i in range(separated_signals.shape[1]):
    # Zpětná transformace na mono signál
    separated_signal_mono = separated_signals[:, i]

    time_bins, frequency_bins, spectrum = SegmentsHandler.calculate_spectrogram(separated_signal_mono,
                                                                                sample_rate_)

    output_file = os.path.join(audio_folder, f"separated_signal_{i + 1}.wav")
    sf.write(output_file, separated_signal_mono, sample_rate_)

    plt.figure(figsize=(10, 4))
    plt.pcolormesh(time_bins, frequency_bins, np.abs(spectrum), shading='gouraud')
    plt.title(f"Source {i + 1} - STFT Magnitude")
    plt.ylabel("Frequency [Hz]")
    plt.xlabel("Time [s]")
    plt.colorbar(label="Magnitude")
    plt.show()

    reduced_noise = nr.reduce_noise(y=separated_signal_mono, sr=sample_rate_)
    noise_estimate = separated_signal_mono - reduced_noise

    Spectrums.append((time_bins, frequency_bins, spectrum, reduced_noise))


    snr_db_global = calculate_snr_global(separated_signal_mono, noise_estimate)
    snr_values_global.append((i, snr_db_global))

sorted_snr_values_global = sorted(snr_values_global, key=lambda x: x[1], reverse=True)
print("sorted_snr_values_global", sorted_snr_values_global)

# K-Means clustering
snr_array = np.array(sorted_snr_values_global)[:, 1].reshape(-1, 1)

# Extrahujeme hodnoty SNR
channels = np.array(sorted_snr_values_global)[:, 0].astype(int)  # Čísla kanálů
snr_values = np.array(sorted_snr_values_global)[:, 1]  # Hodnoty SNR

# Vypočítáme relativní SNR (vzhledem k největší hodnotě SNR)
max_snr = np.max(snr_values)
relative_snr = snr_values / max_snr if max_snr != 0 else snr_values  # Ochrana proti dělení nulou

# Vykreslíme graf
plt.figure(figsize=(8, 5))
plt.bar(channels, relative_snr, color='blue', alpha=0.7)

# Popisky grafu
plt.xlabel("Číslo kanálu")
plt.ylabel("Relativní SNR (normalizované)")
plt.title("Relativní SNR pro jednotlivé kanály")
plt.xticks(channels)  # Zajistíme, že osu x budou označovat čísla kanálů
plt.ylim(0, 1.1)  # Udržení osy y mezi 0 a mírně nad 1

# Zobrazíme graf
plt.show()

# Aplikujeme K-Means s dvěma clustery (popředí/pozadí)
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
labels = kmeans.fit_predict(snr_array)

# Určíme, který cluster má vyšší SNR (bude popředí)
foreground_cluster = np.argmax(kmeans.cluster_centers_)





# # Rozdělení kanálů
foreground_channels = [Spectrums[i] for i in range(len(Spectrums)) if labels[i] == foreground_cluster]
background_channels = [Spectrums[i] for i in range(len(Spectrums)) if labels[i] != foreground_cluster]

# # Výpis výsledků
print(f"Kanály patřící do popředí: {len(foreground_channels)}")

for channel in foreground_channels:
    time_bins, frequency_bins, spectrum, reduced_noise = channel[0], channel[1], channel[2], channel[3]

    channel_image = VH.plot_spectrogram(time_bins, frequency_bins, spectrum, title="Foregroud channel")
    channel_image.show()

    window_size = 1024
    overlap = 512

    frequency_bins, time_bins, channel_spectrum_rn = spectrogram(reduced_noise, fs=sample_rate_, nperseg=window_size, noverlap=overlap)
    channel_image_reduced_noise = VH.plot_spectrogram(time_bins, frequency_bins, channel_spectrum_rn, title="Foregroud channel - reduced noise")
    channel_image_reduced_noise.show()

    # Analýza pomocí energetických peaků
    Sxx = channel_spectrum_rn
    energy_threshold = 0.01

    # Výpočet síly v určitém frekvenčním rozsahu (např. 1-5 kHz)
    low_freq = 1000   # Dolní hranice frekvence (Hz)
    high_freq = 10000 # Horní hranice frekvence (Hz)
    freq_indices = np.where((frequency_bins >= low_freq) & (frequency_bins <= high_freq))[0]
    energy = np.sum(Sxx[freq_indices, :], axis=0)

    # Oprava problému s nulovými hodnotami v logaritmu
    Sxx_log = 10 * np.log10(Sxx + 1e-10)

    # Normalizace energie pro thresholding
    normalized_energy = energy / np.max(energy)

    print(f"time_bins shape: {time_bins.shape}")
    print(f"frequency_bins shape: {frequency_bins.shape}")
    print(f"Sxx_log shape: {Sxx_log.shape}")
    print(f"Sxx shape: {Sxx.shape}")

    # # Oprava dimenzí os (přidání posledního kroku)
    # if time_bins.shape[0] == Sxx_log.shape[1]:
    #     print(1)
    #     time_bins = np.append(time_bins, time_bins[-1] + (time_bins[-1] - time_bins[-2]))
    #
    # if frequency_bins.shape[0] == Sxx_log.shape[0]:
    #     print(1)
    #     frequency_bins = np.append(frequency_bins, frequency_bins[-1] + (frequency_bins[-1] - frequency_bins[-2]))
    #
    # print(f"time_bins shape: {time_bins.shape}")
    # print(f"frequency_bins shape: {frequency_bins.shape}")
    # print(f"Sxx_log shape: {Sxx_log.shape}")
    # print(f"Sxx shape: {Sxx.shape}")

    # Vykreslení horního grafu (spektrogram + energetické peaky)
    plt.figure(figsize=(20, 10))
    plt.subplot(2, 1, 1)
    plt.pcolormesh(time_bins, frequency_bins, Sxx_log, cmap='gray', shading='auto')
    plt.colorbar(label='Intenzita (dB)')
    plt.title(f"Spektrogram a intenzita zpěvu")
    plt.xlabel("Čas (s)")
    plt.ylabel("Frekvence (Hz)")
    plt.plot(time_bins, normalized_energy * frequency_bins[-1], color='red', label="Síla zpěvu (normalizovaná)")
    plt.legend()

    # Vykreslení dolního grafu (spektrogram s označenými segmenty)
    plt.subplot(2, 1, 2)
    plt.pcolormesh(time_bins, frequency_bins, Sxx_log, cmap='gray', shading='auto')
    plt.colorbar(label='Intenzita (dB)')
    plt.title(f"Spektrogram s označenými segmenty nad normalizovaným energetickým prahem {energy_threshold}")
    plt.xlabel("Čas (s)")
    plt.ylabel("Frekvence (Hz)")

    # Označení segmentů s energií nad threshold
    for i, t in enumerate(time_bins):
        if normalized_energy[i] > energy_threshold:
            plt.axvspan(t, t + (time_bins[1] - time_bins[0]), color='red', alpha=0.05)

    plt.show()

    # # Uložení grafu do souboru
    # output_path = os.path.join(output_dir, os.path.splitext(os.path.basename(file_path))[0] + ".png")
    # plt.tight_layout()
    # plt.savefig(output_path)
    # plt.close()
