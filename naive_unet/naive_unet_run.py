import sys
from pathlib import Path
import random

# Adds parent directory to sys.path if not already included
parent_dir = Path(__file__).resolve().parent.parent
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))

import torch
from torch.utils.data import DataLoader, random_split, Subset

sys.path.append('model')
from model.UNetHandler import UNet, load_config
from model.DataHandler import BirdUNetDataset
from pytorch_lightning import Trainer
from Inc import Log

PRINT = Log.get_logger()

if __name__ == '__main__':
    CONFIG = load_config('config.json')

    LEARNING_RATE = CONFIG['training'].get('learning_rate')
    BATCH_SIZE = CONFIG['training'].get('batch_size')
    NUM_EPOCHS = CONFIG['training'].get('epochs')
    MAX_DATASET_SIZE = CONFIG['training'].get('maximum_dataset_size')

    WORKERS = CONFIG['setup'].get('number_of_workers')
    DEBUG = CONFIG['setup']['debug']

    SPECTROGRAM_DIR = CONFIG['data'].get('spectrogram_dir')
    MASK_DIR = CONFIG['data'].get('mask_dir')
    MEAN = CONFIG['data']['augmentation'].get('normalization_mean')
    STD = CONFIG['data']['augmentation'].get('normalization_std')
    TRAIN_VAL_TEST_SPLIT = CONFIG['data'].get('train_val_test_split')

    CHANNELS = CONFIG['model'].get('input_channels')
    CLASSES = CONFIG['model'].get('number_of_classes')
    WEIGHTS_DIR = CONFIG['model'].get('weights_to_be_saved_path')

    PROJECT_NAME = CONFIG['logging'].get('project_name')
    MODEL_NAME = CONFIG['logging'].get('model_name')
    WAND_KEY = CONFIG['logging'].get('wand_api_key_filepath')

    generator = torch.Generator()
    generator.manual_seed(42)

    dataset = BirdUNetDataset(SPECTROGRAM_DIR, MASK_DIR,
                              transform_attrs={'mean': MEAN, 'std': STD}, debug=DEBUG)

    # Using only limited number of samples for training
    total_samples = len(dataset)
    num_samples = min(MAX_DATASET_SIZE, total_samples)
    random_indices = random.sample(range(total_samples), num_samples)

    # Create subset from these indices
    sampled_dataset = Subset(dataset, random_indices)

    train_dataset, val_dataset, test_dataset = random_split(sampled_dataset, TRAIN_VAL_TEST_SPLIT, generator=generator)

    test_file = 'test_samples.txt'
    with open(test_file, 'w') as file:
        for val in test_dataset.dataset.dataset.samples_ids_list.items():
            file.write(str(val[1]) + '\n')

    PRINT.info(
        f'Dataset sizes: \t Train = {len(train_dataset)}\t Val = {len(val_dataset)}\t Test = {len(test_dataset)}.')

    if torch.cuda.is_available():
        device = torch.device('cuda')
        num_cores = torch.cuda.device_count()
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        num_cores = torch.mps.device_count()
    else:
        device = torch.device('cpu')
        num_cores = torch.cpu.device_count()

    PRINT.info(f'Using device: {device}.')

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=WORKERS, shuffle=True, persistent_workers=True)
    val_dataloader = DataLoader(val_dataset, num_workers=WORKERS, batch_size=BATCH_SIZE, persistent_workers=True)

    log_params = {
        'project_name': PROJECT_NAME,
        'model_name': MODEL_NAME,
        'key_file_path': WAND_KEY
    }
    model = UNet(in_channels=CHANNELS, num_classes=CLASSES, DataHandler=dataset, log_mode='online', log_params=log_params).to(device)

    trainer = Trainer(max_epochs=NUM_EPOCHS, callbacks=model.callbacks, devices=num_cores, accelerator=str(device))
    trainer.fit(model, train_dataloader, val_dataloader)

    # Save the trained model
    model.save(WEIGHTS_DIR)