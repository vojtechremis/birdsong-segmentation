"""
This file contains the MixITDataset class, which is used to load the MixIT dataset.
"""

import os
import torch
from torch.utils.data import Dataset
import numpy as np
import librosa
import random
import json
from tqdm import tqdm

import os
import sys
sys.path.append(os.path.abspath("../../")) 
# from .MixITAugmentations import MixITAugmentations
from utils import Log

PRINT = Log.get_logger()

class MixITDataset(Dataset):
    """
    Dataset class for MixIT training supports two generation of MoMs modes:
    - 'dynamic': x1 and x2 (x = x1 + x2) are randomly selected at each call of __getitem__.
    - 'pregenerate': x1 and x2 are pregenerated by calling generate_groups() mehod. This can be done at the
        initialization of the class or at any other time. Suitable for validation/test sets which require consistency.
        Generation of groups isn't done automatically and must be called explicitly.
    """
    def __init__(self, data_dir, sample_rate=16000, segment_length_sec=6, 
                 augment_prob=0.5, num_sources_for_mix=4, num_samples=1000, 
                 generation_mode='dynamic', groups_file_path=None):
        """
        Args:
            data_dir (str): Path to the directory with data (.wav files).
            sample_rate (int): Target sample rate.
            segment_length_sec (float): Length of audio segment in seconds.
            augment_prob (float): Probability of applying augmentations to the source.
            num_sources_for_mix (int): Number of sources used to create one mixture x = x1 + x2.
                                     Must be a positive even number.
            num_samples (int): Number of samples (mixtures) in one "epoch" of the dataset.
            generation_mode (str): Generation mode ('pregenerate' or 'dynamic').
            groups_file_path (str): Path to a JSON file containing pre-defined groups of source file paths.
        """
        self.data_dir = data_dir
        self.sample_rate = sample_rate
        self.segment_length_sec = segment_length_sec
        self.segment_length = int(sample_rate * segment_length_sec) 
        self.augment_prob = augment_prob

        self.num_sources_for_mix = num_sources_for_mix
        self.num_samples = num_samples
        self.generation_mode = generation_mode
        self.groups_file_path = groups_file_path
        
        # Validation of parameters
        if num_sources_for_mix <= 0 or num_sources_for_mix % 2 != 0:
             raise ValueError("Parameter 'num_sources_for_mix' must be a positive even number for MixIT.")
        
        if generation_mode not in ['pregenerate', 'dynamic']:
            raise ValueError("Parameter 'generation_mode' must be 'pregenerate' or 'dynamic'.")

        # Initialize augmentations
        self.augmentations = None# MixITAugmentations(sample_rate)
        
        # Loading list of files
        self.audio_files = self._load_audio_folder(data_dir)
       
        
        # Initialize pregenerated groups list if generation mode is 'pregenerate'
        if self.generation_mode == 'pregenerate':
            self.mixit_groups = []

    def _load_audio_folder(self, dir_path):
        """Loads audio file, crops/repeats to the desired length and checks SR."""

        PRINT.info(f"Scanning for .wav files in {dir_path}...")

        audio_files = []

        for root, _, files in os.walk(dir_path):
            for file in files:
                if file.lower().endswith('.wav'):
                    audio_files.append(os.path.join(root, file))
        
        if len(audio_files) < self.num_sources_for_mix:
            raise ValueError(f"Not enough audio files ({len(audio_files)}) found in {dir_path} "
                            f"to create mixes with {self.num_sources_for_mix} sources.")
            
        PRINT.info(f"Loaded {len(audio_files)} audio file paths. Check carefuly if there is "
                   f"enough audio_files for creating {self.num_samples} MoMs!")
        
        return audio_files
        
    def generate_groups(self):
        """Generates a list of source file paths for 'pregenerate' mode."""
        if self.generation_mode != 'pregenerate':
            PRINT.warning("Method generate_groups called but dataset is in 'dynamic' mode. No action taken.")
            return

        PRINT.info(f"Generating {self.num_samples} source groups...")
        self._create_mixit_groups()
        PRINT.info(f"Finished pregenerating {len(self.mixit_groups)} groups.")

    def load_groups_from_file(self, file_path: str):
        """Loads pre-defined groups of source file paths from a JSON file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Groups JSON file not found: {file_path}")
        
        try:
            with open(file_path, 'r') as f:
                loaded_groups = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding JSON from {file_path}: {e}")

        if not isinstance(loaded_groups, list):
            raise ValueError(f"Groups JSON file {file_path} must contain a JSON list.")

        self.mixit_groups = []
        for i, group in enumerate(loaded_groups):
            if not isinstance(group, list):
                raise ValueError(f"Group at index {i} in {file_path} is not a list. Expected list of file paths.")
            if len(group) != self.num_sources_for_mix:
                raise ValueError(f"Group at index {i} in {file_path} has {len(group)} files, "
                                 f"but expected {self.num_sources_for_mix} based on num_sources_for_mix.")
            
            # Validation of file paths (since they come from JSON file)
            for_all_files_exist_in_group = True
            for source_file_path in group:
                if not os.path.exists(source_file_path):
                    PRINT.warning(f"File path in group {i} does not exist: {source_file_path}. Skipping group.")
                    for_all_files_exist_in_group = False
                    break
            
            if for_all_files_exist_in_group:
                 self.mixit_groups.append(group)
            else:
                 PRINT.warning(f"Group {i} from file {file_path} was skipped due to missing files.")

        PRINT.info(f"Loaded {len(self.mixit_groups)} groups from {file_path}.")

        if not self.mixit_groups:
            raise ValueError(f"No valid groups loaded from {file_path}. Please check file content and paths.")

        self.num_samples = len(self.mixit_groups)
        PRINT.info(f"Successfully loaded {self.num_samples} groups from {file_path}.")

    def _create_mixit_groups(self):
        """Internal method for creating a list of source file paths."""
        self.mixit_groups = []
        num_files = len(self.audio_files)
        
        if num_files < self.num_sources_for_mix:
            raise ValueError(f"Cannot create groups of {self.num_sources_for_mix} because only {num_files} files are available.")

        actual_samples_to_generate = self.num_samples
        PRINT.info(f"Generating {actual_samples_to_generate} groups of {self.num_sources_for_mix} sources...")
        
        for _ in tqdm(range(actual_samples_to_generate), desc="Generating MixIT Groups"):
            indices = random.sample(range(num_files), self.num_sources_for_mix)
            group = [self.audio_files[i] for i in indices]
            self.mixit_groups.append(group) # pole polí o velikosti 4 (pro klasický případ MixIT)
        
        if len(self.mixit_groups) != self.num_samples:
            raise ValueError(f"Generated {len(self.mixit_groups)} groups instead of {self.num_samples}.")

    def _load_audio(self, file_path):
        """Loads audio file, crops/repeats to the desired length and checks SR."""
        try:
             audio, sr = librosa.load(file_path, sr=self.sample_rate, mono=True, res_type='kaiser_fast')
        except Exception as e:
             raise ValueError(f"Error loading audio file {file_path}: {e}")

        if audio is None or len(audio) == 0:
             raise ValueError(f"Warning: Loaded audio is empty for {file_path}.")

        target_len = self.segment_length 
        current_len = len(audio)

        # If the audio is already the correct length, return it
        if current_len == target_len:
             return audio
        
        # If the audio is shorter than the target length, repeat it
        elif current_len < target_len:
            num_repeats = int(np.ceil(target_len / current_len))
            audio = np.tile(audio, num_repeats)
            return audio[:target_len]
        
        # If the audio is longer than the target length, randomly crop it
        else:
            start = random.randint(0, current_len - target_len)
            return audio[start:start + target_len]

    def _get_random_noise(self):
        """Získá náhodný audio soubor pro šum a načte ho."""
        for _ in range(5): # Try to load several times
            noise_file = random.choice(self.audio_files)
            noise_audio = self._load_audio(noise_file)
            if noise_audio is not None:
                return noise_audio
        PRINT.warning("Could not load a valid noise sample after multiple attempts. Returning silence.")
        return np.zeros(self.segment_length) 

    def _normalize(self, audio_array):
        """Normalizes audio data (array) to max absolute value 1."""
        max_abs = np.max(np.abs(audio_array))
        if max_abs > 1e-8: # Added for stability
             return audio_array / max_abs
        return audio_array

    def __len__(self):
        """Returns the defined number of samples per epoch."""
        if self.generation_mode == 'pregenerate' and self.mixit_groups:
            return len(self.mixit_groups)
        else:
            return self.num_samples

    def __getitem__(self, idx):
        """
        Returns one MixIT sample 'MoM' (Mixture of Mixtures - "x = x1 + x2"), i. e. x, x1, x2.
        MixIT sample is selected either from pregenerated groups or by random selection (based on generation_mode class configuration).
        """
        
        # Getting source_files
        source_files = []
        if self.generation_mode == 'pregenerate':
            if not self.mixit_groups: # Ensure that pregenerated groups exist
                 raise ValueError(f"Pregenerated groups are empty. Cannot get item {idx}.")
            
            actual_group_count = self.__len__()

            if idx >= actual_group_count: # Ensure index is valid for the actual number of pregenerated groups
                 raise ValueError(f"Warning: Index {idx} out of bounds for pregenerated groups ({actual_group_count}). Wrapping around.")
            
            source_files = self.mixit_groups[idx]
        
        elif self.generation_mode == 'dynamic':
            # Random selection of sources directly here
            indices = random.sample(range(len(self.audio_files)), self.num_sources_for_mix)
            source_files = [self.audio_files[i] for i in indices]
    
        else:
             raise RuntimeError(f"Invalid generation mode: {self.generation_mode}")

        # Loading audio data for selected files
        sources_to_mix = []
        for file_path in source_files:
            audio = self._load_audio(file_path)
            if audio is None:
                raise ValueError(f"Failed to load source {file_path} for item {idx}. Skipping sample.")
            
            sources_to_mix.append(audio)

        # You may apply augmentations here ...
        

        # --- Creating Mixture of Mixtures ---
        
        # Separating sources into two groups
        num_sources_per_mix = self.num_sources_for_mix // 2
        group1_sources = sources_to_mix[:num_sources_per_mix]
        group2_sources = sources_to_mix[num_sources_per_mix:]

        mix1 = np.zeros(self.segment_length, dtype=np.float32)
        for src in group1_sources:
            mix1 += src
            
        mix2 = np.zeros(self.segment_length, dtype=np.float32)
        for src in group2_sources:
            mix2 += src
            
        # Input for model: two mixed mixtures
        # Normalizing input mixtures separately!!!

        normalized_mix1 = self._normalize(mix1)
        normalized_mix2 = self._normalize(mix2)

        normalized_input_mix = np.stack([normalized_mix1, normalized_mix2], axis=0)
        
        # Converting to PyTorch tensors
        input_mix_tensor = torch.from_numpy(normalized_input_mix)  # Shape: [2, T]
        
        # Returning a dictionary that will be processed in DataLoader
        return input_mix_tensor

# Příklad použití (mimo třídu)
if __name__ == '__main__':
    # Nastavení cest a parametrů pro testování
    # POUŽIJTE CESTU K VAŠEMU TRÉNINKOVÉMU ADRESÁŘI
    user_train_data_dir = '/Users/vojtechremis/Desktop/Projects/birdsong-segmentation/MixIT_network/data/train'
    
    # Adresář pro dočasné testovací soubory (např. JSON se skupinami)
    # Můžete si ho vytvořit ve vašem projektu, nebo použít /tmp/
    temp_test_dir = './temp_test_data_handler' # Vytvoří se v aktuálním adresáři, kde spouštíte skript
    os.makedirs(temp_test_dir, exist_ok=True)
    
    test_groups_json_path = os.path.join(temp_test_dir, 'test_mixit_groups.json')

    # Testovací parametry
    sr_test = 16000 # Mělo by odpovídat vašim datům, pokud možno
    len_sec_test = 3  # Délka segmentu v sekundách pro test
    num_src_for_mix_test = 4 # Počet zdrojů pro jednu MixIT směs
    num_samples_test = 5   # Počet vzorků pro testovací datasety

    PRINT.info(f"Použitý adresář s trénovacími daty: {user_train_data_dir}")
    PRINT.info(f"Dočasný adresář pro testy: {temp_test_dir}")

    # --- Příprava testovacího JSON souboru se skupinami --- 
    # Zkusíme načíst prvních pár souborů z vašeho adresáře pro vytvoření JSONu
    # Tato část je jen pro vytvoření smysluplného test_mixit_groups.json
    # Pokud tato část selže (např. málo souborů), test načítání z JSON se přeskočí.
    
    # Načteme prvních pár souborů z user_train_data_dir pro testovací JSON
    # Toto je zjednodušené, v praxi byste měli svůj validační JSON připravený
    try:
        potential_files_for_json = []
        for root, _, files in os.walk(user_train_data_dir):
            for file in files:
                if file.lower().endswith('.wav'):
                    potential_files_for_json.append(os.path.join(root, file))
            if len(potential_files_for_json) >= num_src_for_mix_test * 2: # Potřebujeme alespoň 2 skupiny
                break 
        
        groups_for_json_creation = []
        if len(potential_files_for_json) >= num_src_for_mix_test:
            groups_for_json_creation.append(potential_files_for_json[:num_src_for_mix_test])
        if len(potential_files_for_json) >= num_src_for_mix_test * 2:
            groups_for_json_creation.append(potential_files_for_json[num_src_for_mix_test : num_src_for_mix_test * 2])

        if groups_for_json_creation and len(groups_for_json_creation[0]) == num_src_for_mix_test:
            with open(test_groups_json_path, 'w') as f_json:
                json.dump(groups_for_json_creation, f_json)
            PRINT.info(f"Testovací JSON soubor '{test_groups_json_path}' vytvořen s {len(groups_for_json_creation)} skupinami.")
            json_file_created_successfully = True
        else:
            PRINT.warning(f"Nepodařilo se vytvořit testovací JSON: nedostatek souborů v {user_train_data_dir} (potřeba {num_src_for_mix_test*2}).")
            json_file_created_successfully = False
    except Exception as e:
        PRINT.error(f"Chyba při vytváření testovacího JSON souboru: {e}")
        json_file_created_successfully = False

    # --- Test 1: Dynamický režim --- 
    PRINT.info("\n--- Test 1: 'dynamic' mode ---")
    try:
        dynamic_dataset = MixITDataset(
            data_dir=user_train_data_dir, 
            sample_rate=sr_test,
            segment_length_sec=len_sec_test,
            num_sources_for_mix=num_src_for_mix_test, 
            num_samples=num_samples_test, 
            generation_mode='dynamic'
        )
        PRINT.info(f"Dynamic dataset - Počet vzorků: {len(dynamic_dataset)}")
        if len(dynamic_dataset) > 0:
            for i in range(min(2, len(dynamic_dataset))): # Otestujeme první 2 vzorky
                sample = dynamic_dataset[i]
                if sample:
                    PRINT.info(f"  Vzorek {i} (dynamic):")
                    PRINT.info(f"    Mixture_of_mixtures tvar: {sample['mixture_of_mixtures'].shape}, dtype: {sample['mixture_of_mixtures'].dtype}")
                    PRINT.info(f"   Mixtures_sources tvar: {sample['mixtures_sources'].shape}, dtype: {sample['mixtures_sources'].dtype}")
                else:
                    PRINT.warning(f"  Vzorek {i} (dynamic) je None (možná chyba načítání souboru).")
    except Exception as e:
        PRINT.error(f"CHYBA při vytváření/použití dynamického datasetu: {e}", exc_info=True)

    # --- Test 2: Předgenerovaný režim s načtením skupin z JSON souboru --- 
    if json_file_created_successfully:
        PRINT.info("\n--- Test 2: 'pregenerate' mode with groups_file_path ---")
        try:
            pregen_file_dataset = MixITDataset(
                data_dir=user_train_data_dir, # data_dir může být stále potřeba pro _get_random_noise, pokud JSON neobsahuje absolutní cesty
                sample_rate=sr_test, 
                segment_length_sec=len_sec_test,
                num_sources_for_mix=num_src_for_mix_test, 
                num_samples=num_samples_test, # Bude přepsáno počtem skupin v souboru
                generation_mode='pregenerate',
                groups_file_path=test_groups_json_path
            )
            PRINT.info(f"Pregenerated (from file) dataset - Počet vzorků: {len(pregen_file_dataset)}")
            pregen_file_dataset.generate_groups()
            if len(pregen_file_dataset) > 0:
                sample = pregen_file_dataset[0]
                if sample:
                    PRINT.info(f"  Vzorek 0 (pregen from file):")
                    PRINT.info(f"    Mixture_of_mixtures tvar: {sample['mixture_of_mixtures'].shape}, dtype: {sample['mixture_of_mixtures'].dtype}")
                    PRINT.info(f"    Mixtures_sources tvar: {sample['mixtures_sources'].shape}, dtype: {sample['mixtures_sources'].dtype}")
                else:
                    PRINT.warning("  Vzorek 0 (pregen from file) je None.")
        except Exception as e:
            PRINT.error(f"CHYBA při vytváření/použití předgenerovaného datasetu ze souboru: {e}", exc_info=True)
    else:
        PRINT.warning("\n--- Test 2: 'pregenerate' mode with groups_file_path SKIPPED (JSON file not created) ---")

    # --- Test 3: Předgenerovaný režim s náhodným generováním skupin --- 
    PRINT.info("\n--- Test 3: 'pregenerate' mode with random generation ---")
    try:
        pregen_random_dataset = MixITDataset(
            data_dir=user_train_data_dir, 
            sample_rate=sr_test, 
            segment_length_sec=len_sec_test,
            num_sources_for_mix=num_src_for_mix_test, 
            num_samples=num_samples_test, # Požadujeme N náhodně generovaných skupin
            generation_mode='pregenerate',
            groups_file_path=None # Bez souboru, vynutí náhodné generování
        )
        pregen_random_dataset.generate_groups()
        PRINT.info(f"Pregenerated (random) dataset - Počet vzorků: {len(pregen_random_dataset)}")
        if len(pregen_random_dataset) > 0:
            sample = pregen_random_dataset[0]
            if sample:
                PRINT.info(f"  Vzorek 0 (pregen random):")
                PRINT.info(f"    Mixture_of_mixtures tvar: {sample['mixture_of_mixtures'].shape}, dtype: {sample['mixture_of_mixtures'].dtype}")
                PRINT.info(f"    Mixtures_sources tvar: {sample['mixtures_sources'].shape}, dtype: {sample['mixtures_sources'].dtype}")
            else:
                PRINT.warning("  Vzorek 0 (pregen random) je None.")
            
            # Test explicitní regenerace
            PRINT.info("  Testuji explicitní regeneraci skupin...")
            pregen_random_dataset.generate_groups() # Zavoláme znovu náhodné generování
            PRINT.info(f"  Dataset po regeneraci - Počet vzorků: {len(pregen_random_dataset)}")
            if len(pregen_random_dataset) > 0:
                sample = pregen_random_dataset[0]
                if sample:
                    PRINT.info(f"    Vzorek 0 (po explicitní regeneraci):")
                    PRINT.info(f"      Mixture_of_mixtures tvar: {sample['mixture_of_mixtures'].shape}, dtype: {sample['mixture_of_mixtures'].dtype}")
                    PRINT.info(f"      Mixtures_sources tvar: {sample['mixtures_sources'].shape}, dtype: {sample['mixtures_sources'].dtype}")
                else:
                    PRINT.warning("    Vzorek 0 (po explicitní regeneraci) je None.")
    except Exception as e:
        PRINT.error(f"CHYBA při vytváření/použití předgenerovaného náhodného datasetu: {e}", exc_info=True)
    
    # --- Úklid testovacích souborů ---
    try:
        if json_file_created_successfully and os.path.exists(test_groups_json_path):
            os.remove(test_groups_json_path)
            PRINT.info(f"Dočasný JSON soubor '{test_groups_json_path}' smazán.")
        if os.path.exists(temp_test_dir):
            # Zkusíme smazat adresář, jen pokud je prázdný
            if not os.listdir(temp_test_dir):
                 os.rmdir(temp_test_dir)
                 PRINT.info(f"Dočasný adresář '{temp_test_dir}' smazán.")
            else:
                 PRINT.info(f"Dočasný adresář '{temp_test_dir}' nebyl prázdný, nebyl smazán.")
    except Exception as e:
        PRINT.warning(f"Chyba při úklidu testovacích souborů: {e}")

    PRINT.info("\nTestování MixITDataset dokončeno.")
    PRINT.info("POZNÁMKA: Pokud __getitem__ vrátí None (např. kvůli chybě načítání souboru), Váš DataLoader bude potřebovat vlastní `collate_fn` pro jejich odfiltrování.")
    PRINT.info("Příklad collate_fn: `def collate_fn(batch): batch = [b for b in batch if b is not None]; return torch.utils.data.dataloader.default_collate(batch) if batch else None`")