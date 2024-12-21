import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import os
import noisereduce as nr

# Funkce pro zpracování jednoho souboru
def process_audio_file(file_path, output_dir, noise_red=False, energy_threshold=0.01):
    # Načtení zvuku pomocí librosa
    data, sample_rate = librosa.load(file_path, sr=None, mono=True)

    if noise_red:
        data = reduce_noise(data, sample_rate)

    # Parametry okénkové Fourierovy transformace
    window_size = 1024  # Velikost okna (počet vzorků)
    overlap = 512       # Překryv mezi okny
    frequencies, times, Sxx = spectrogram(data, fs=sample_rate, nperseg=window_size, noverlap=overlap)

    # Výpočet síly v určitém frekvenčním rozsahu (např. 1-5 kHz)
    low_freq = 1000   # Dolní hranice frekvence (Hz)
    high_freq = 10000 # Horní hranice frekvence (Hz)
    freq_indices = np.where((frequencies >= low_freq) & (frequencies <= high_freq))[0]
    energy = np.sum(Sxx[freq_indices, :], axis=0)

    # Oprava problému s nulovými hodnotami v logaritmu
    Sxx_log = 10 * np.log10(Sxx + 1e-10)

    # Normalizace energie pro thresholding
    normalized_energy = energy / np.max(energy)

    # Vykreslení horního grafu (spektrogram + energetické peaky)
    plt.figure(figsize=(20, 10))
    plt.subplot(2, 1, 1)
    plt.pcolormesh(times, frequencies, Sxx_log, cmap='gray', shading='auto')
    plt.colorbar(label='Intenzita (dB)')
    plt.title(f"Spektrogram a intenzita zpěvu - {os.path.basename(file_path)}")
    plt.xlabel("Čas (s)")
    plt.ylabel("Frekvence (Hz)")
    plt.plot(times, normalized_energy * frequencies[-1], color='red', label="Síla zpěvu (normalizovaná)")
    plt.legend()

    # Vykreslení dolního grafu (spektrogram s označenými segmenty)
    plt.subplot(2, 1, 2)
    plt.pcolormesh(times, frequencies, Sxx_log, cmap='gray', shading='auto')
    plt.colorbar(label='Intenzita (dB)')
    plt.title(f"Spektrogram s označenými segmenty nad normalizovaným energetickým prahem {energy_threshold}")
    plt.xlabel("Čas (s)")
    plt.ylabel("Frekvence (Hz)")

    # Označení segmentů s energií nad threshold
    for i, t in enumerate(times):
        if normalized_energy[i] > energy_threshold:
            plt.axvspan(t, t + (times[1] - times[0]), color='red', alpha=0.05)

    # Uložení grafu do souboru
    output_path = os.path.join(output_dir, os.path.splitext(os.path.basename(file_path))[0] + ".png")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"Zpracováno: {file_path} -> {output_path}")



# Hlavní funkce pro zpracování všech souborů v adresáři
def process_audio_directory(directory, output_directory):
    for file_name in os.listdir(directory):
        if file_name.lower().endswith((".mp3", ".wav", ".flac")):  # Podporované formáty
            file_path = os.path.join(directory, file_name)
            process_audio_file(file_path, output_directory)


def reduce_noise(amplitude_array, sr, noise_start=0, noise_duration=1):

    # Výběr šumového profilu
    noise_sample_start = int(noise_start * sr)
    noise_sample_end = int((noise_start + noise_duration) * sr)
    noise_profile = amplitude_array[noise_sample_start:noise_sample_end]

    # Redukce šumu
    reduced_noise = nr.reduce_noise(y=amplitude_array, sr=sr, y_noise=noise_profile)
    return reduced_noise





# Zadejte cestu k adresáři se zvukovými soubory
audio_directory = "/Users/vojtechremis/Desktop/Xeno-Canto_A/audiofiles"  # Nahraďte cestou ke svému adresáři
output_directory = "/Users/vojtechremis/Desktop/Xeno-Canto_A"
process_audio_directory(audio_directory, output_directory)
#
# file_path = '/Users/vojtechremis/Desktop/Xeno-Canto_A/audiofiles/XC1691_Elaenia_obscura.mp3'
# process_audio_file(file_path, '/Users/vojtechremis/Desktop/Xeno-Canto_A', noise_red=True)

