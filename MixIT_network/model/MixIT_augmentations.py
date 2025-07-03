import numpy as np
import librosa
from scipy.signal import butter, filtfilt
import random

class MixITAugmentations:
    """Augmentace pro MixIT trénink"""
    
    def __init__(self, sample_rate=16000):
        """
        Args:
            sample_rate (int): Vzorkovací frekvence
        """
        self.sample_rate = sample_rate
    
    def time_crop(self, audio, target_length):
        """
        Náhodný výřez z audio signálu
        
        Args:
            audio (np.ndarray): Vstupní audio signál
            target_length (int): Požadovaná délka výřezu ve vzorcích
            
        Returns:
            np.ndarray: Oříznutý audio signál
        """
        # Pokud je signál kratší nebo stejně dlouhý jako cílová délka, opakujeme ho
        if len(audio) <= target_length:
            num_repeats = int(np.ceil(target_length / max(1, len(audio))))
            audio = np.tile(audio, num_repeats)
            # Pro jistotu oříznutí na požadovanou délku
            return audio[:target_length]
        
        # Náhodný start
        max_start = len(audio) - target_length
        if max_start <= 0:  # Další kontrola pro jistotu
            return audio[:target_length]
            
        start = np.random.randint(0, max_start)
        return audio[start:start + target_length]
    
    def random_energy_gain(self, audio, min_peak=0.05, max_peak=0.75):
        """
        Náhodné zesílení/zeslabení signálu na cílový peak
        
        Args:
            audio (np.ndarray): Vstupní audio signál
            min_peak (float): Minimální cílový peak
            max_peak (float): Maximální cílový peak
            
        Returns:
            np.ndarray: Upravený audio signál
        """
        # Výpočet aktuálního peaku
        current_peak = np.max(np.abs(audio)) + 1e-8  # Prevence dělení nulou
        
        # Náhodný cílový peak
        target_peak = np.random.uniform(min_peak, max_peak)
        
        # Výpočet gain faktoru
        gain_factor = target_peak / current_peak
        
        return audio * gain_factor
    
    def add_background_noise(self, audio, noise_audio, snr_db=20, lowpass_freq=1000):
        """
        Přidání pozadí a šumu do signálu
        
        Args:
            audio (np.ndarray): Vstupní audio signál
            noise_audio (np.ndarray): Audio signál pro šum
            snr_db (float): Signál/šum poměr v dB
            lowpass_freq (float): Frekvence low-pass filtru
            
        Returns:
            np.ndarray: Signál s přidaným šumem
        """
        # Oříznutí šumu na stejnou délku
        if len(noise_audio) > len(audio):
            start = np.random.randint(0, len(noise_audio) - len(audio))
            noise_audio = noise_audio[start:start + len(audio)]
        else:
            noise_audio = np.tile(noise_audio, int(np.ceil(len(audio) / max(1, len(noise_audio)))))
            noise_audio = noise_audio[:len(audio)]
        
        # Aplikace low-pass filtru na šum
        nyquist = self.sample_rate / 2
        normal_cutoff = lowpass_freq / nyquist
        b, a = butter(4, normal_cutoff, btype='low')
        noise_audio = filtfilt(b, a, noise_audio)
        
        # Výpočet výkonu signálu a šumu
        signal_power = np.mean(audio ** 2) + 1e-8  # Prevence dělení nulou
        noise_power = np.mean(noise_audio ** 2) + 1e-8  # Prevence dělení nulou
        
        # Výpočet gain faktoru pro dosažení požadovaného SNR
        gain_factor = np.sqrt(signal_power / (noise_power * 10 ** (snr_db / 10)))
        
        # Přidání šumu
        return audio + noise_audio * gain_factor
    
    def apply_augmentations(self, audio, noise_sample=None, target_length=None, snr_db=20, lowpass_freq=1000):
        """
        Aplikace všech augmentací na signál
        
        Args:
            audio (np.ndarray): Vstupní audio signál
            noise_sample (np.ndarray, optional): Audio signál pro šum
            target_length (int, optional): Požadovaná délka výřezu
            snr_db (float): Signál/šum poměr v dB
            lowpass_freq (float): Frekvence low-pass filtru
            
        Returns:
            np.ndarray: Augmentovaný audio signál
        """
        # Pokud není zadána cílová délka, použijeme aktuální délku
        if target_length is None:
            target_length = len(audio)
            
        # Time crop
        audio = self.time_crop(audio, target_length)
        
        # Přidání šumu, pokud je k dispozici
        if noise_sample is not None:
            audio = self.add_background_noise(audio, noise_sample, snr_db, lowpass_freq)

        # Random energy gain
        audio = self.random_energy_gain(audio)
        
        return audio 