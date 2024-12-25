import random

import sys
from pathlib import Path

sys.path += [str(Path().resolve().parent.parent.parent)]

import torch
from torch.utils.data import DataLoader, random_split, Subset

import os
from dummy_network.data_handlers.SpectrogramDataset import SpectrogramDataset
from ResNet_parts import pad_to_largest
from ResNetHandler import LittleResNetLSTM, ResNetHandler
from pytorch_lightning import Trainer
from utils import Log
from utils.utils import load_config


PRINT = Log.get_logger()

def collate_fn(batch):
    batch_imgs = pad_to_largest([sample[0] for sample in batch])
    segments = torch.Tensor([sample[1] for sample in batch])
    motifs = torch.Tensor([sample[2] for sample in batch])

    return batch_imgs, (segments, motifs)


if __name__ == '__main__':
    # Print system info
    TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
    CUDA_VERSION = torch.__version__.split("+")[-1]
    PRINT.info(f'torch: {TORCH_VERSION}; cuda: {CUDA_VERSION}')

    CONFIG = load_config('../LittleResNet/LittleResNet_config.json')

    LEARNING_RATE = CONFIG['training'].get('learning_rate')
    BATCH_SIZE = CONFIG['training'].get('batch_size')
    NUM_EPOCHS = CONFIG['training'].get('epochs')
    MAX_DATASET_SIZE = CONFIG['training'].get('maximum_dataset_size')
    TIME_LIMIT = CONFIG['training'].get('time_binsize_limit')
    DROPOUT = CONFIG['training'].get('dropout')

    WORKERS = CONFIG['setup'].get('number_of_workers')
    DEBUG = CONFIG['setup']['debug']

    SPECTROGRAM_DIR = CONFIG['data'].get('spectrogram_dir')
    LABELS_FILE = CONFIG['data'].get('labels_file')

    MEAN = CONFIG['data']['augmentation'].get('normalization_mean')
    STD = CONFIG['data']['augmentation'].get('normalization_std')
    TRAIN_VAL_SPLIT = CONFIG['data'].get('train_val_split')
    TEST_IDS_PATH = CONFIG['data'].get('test_ids_to_be_saved_path')

    WEIGHTS_DIR = CONFIG['model'].get('weights_to_be_saved_path')

    PROJECT_NAME = CONFIG['logging'].get('project_name')
    MODEL_NAME = CONFIG['logging'].get('model_name')
    WAND_KEY = CONFIG['logging'].get('wand_api_key_filepath')

    generator = torch.Generator()
    generator.manual_seed(42)

    dataset = SpectrogramDataset(SPECTROGRAM_DIR, LABELS_FILE, transform_attrs={'mean': MEAN, 'std': STD},
                                 debug=DEBUG, time_binsize_limit=TIME_LIMIT)

    # Using only limited number of samples for training
    total_samples = len(dataset)
    num_samples = min(MAX_DATASET_SIZE, total_samples)
    random_indices = random.sample(range(total_samples), num_samples)

    # Create subset from these indices
    sampled_dataset = Subset(dataset, random_indices)

    train_dataset, val_dataset = random_split(sampled_dataset, TRAIN_VAL_SPLIT, generator=generator)

    # with open(TEST_IDS_PATH, 'w') as file:
    #     for idx in test_dataset.indices:
    #         file.write(str(test_dataset.dataset.dataset.sample_ids[idx]) + '\n')

    # PRINT.info(
    #     f'Dataset sizes: \t Train = {len(train_dataset)}\t Val = {len(val_dataset)}\t Test = {len(test_dataset)}.')

    # Selecting device
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

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=WORKERS, collate_fn=collate_fn,
                                  shuffle=False,
                                  persistent_workers=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=WORKERS,
                                persistent_workers=True)

    log_params = {
        'project_name': PROJECT_NAME,
        'model_name': MODEL_NAME,
        'key_file_path': WAND_KEY
    }
    model = ResNetHandler(model=LittleResNetLSTM(input_channels=1, dropout_prob=DROPOUT), dataset=SpectrogramDataset, log_mode='wandb', log_params=log_params).to(device)

    trainer = Trainer(max_epochs=NUM_EPOCHS, callbacks=model.callbacks, precision='16-mixed', accumulate_grad_batches=8, devices=num_cores, accelerator=str(device))
    trainer.fit(model, train_dataloader, val_dataloader)