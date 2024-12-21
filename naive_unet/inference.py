import sys
from pathlib import Path

# Adds parent directory to sys.path if not already included
parent_dir = Path().resolve().parent.parent
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))

sys.path.append('model')

import os
import random

import torch
from model.UNetHandler import UNet, load_config
from model.DataHandler import BirdUNetDataset
from utils import Log

PRINT = Log.get_logger()


# sample_filepath = '/Users/vojtechremis/Desktop/Projects/birdsong-segmentation/source_data/tweetynet/llb3_data/llb3_songs/llb3_0014_2018_04_23_15_18_14.wav'

def inference(sample_filepath):
    CONFIG = load_config('config.json')
    MEAN = CONFIG['data']['augmentation'].get('normalization_mean')
    STD = CONFIG['data']['augmentation'].get('normalization_std')

    dataset = BirdUNetDataset(None, None,
                              transform_attrs={'mean': MEAN, 'std': STD}, debug=True)

    model = UNet(in_channels=1, num_classes=2, DataHandler=dataset)

    model.load_weights('model_weights.pth')

    outputs = model.inference_by_filepath(sample_filepath, plot_predictions=False)


if __name__ == '__main__':

    # Cesta ke složce
    folder_path = '/Users/vojtechremis/Desktop/Projects/birdsong-segmentation/source_data/tweetynet/llb11_data/llb11_songs'

    # Získání seznamu všech souborů ve složce
    all_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if
                 os.path.isfile(os.path.join(folder_path, file))]

    # Zajištění, že je ve složce dostatek souborů
    if len(all_files) < 25:
        print(f"Ve složce je pouze {len(all_files)} souborů. Nelze vybrat 50.")
    else:
        # Výběr 50 náhodných souborů
        random_files = random.sample(all_files, 25)

        # Výpis náhodně vybraných souborů
        print("Vybrané soubory:")
        for file_path in random_files:
            inference(file_path)
