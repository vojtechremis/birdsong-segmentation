import sys

import torch
from torch.utils.data import DataLoader, random_split

sys.path.append('model')
from model.UNetHandler import UNet
from model.DataHandler import BirdUNetDataset
from pytorch_lightning import Trainer
from torch.utils.data import default_collate




if __name__ == '__main__':
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 16
    NUM_EPOCHS = 1
    SPECTROGRAM_DIR = '/Users/vojtechremis/Desktop/Projects/Birds/naive_unet/data/spectrogram/'
    MASK_DIR = '/Users/vojtechremis/Desktop/Projects/Birds/naive_unet/data/mask/'

    dataset = BirdUNetDataset(SPECTROGRAM_DIR, MASK_DIR, transform_attrs={'mean': [-75.14263153076172], 'std': [11.666242599487305]})
    generator = torch.Generator()
    generator.manual_seed(42)
    train_dataset, val_dataset = random_split(dataset, [0.8, 0.2], generator=generator)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    model = UNet(in_channels=1, num_classes=2).to(device)

    num_gpus = torch.cuda.device_count()
    num_cpus = torch.cpu.device_count()
    trainer = Trainer(max_epochs=NUM_EPOCHS, callbacks=model.callbacks, devices=num_cpus, accelerator=str(device))
    trainer.fit(model, train_dataloader, val_dataloader)