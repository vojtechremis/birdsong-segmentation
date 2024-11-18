# Definition of paths
import sys
from pathlib import Path

PARENT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PARENT_DIR))

import torch
import torch.nn as nn
import pytorch_lightning as pl
import wandb
from pytorch_lightning import Trainer
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from matplotlib import pyplot as plt
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torchmetrics.classification import Accuracy, Precision, Recall, F1Score, JaccardIndex
from utils.visualization import spectrum_above_mask
import UNet_parts
import os
import numpy as np

from Inc import Log

PRINT = Log.get_logger()

# Print system info
TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
PRINT.info(f'torch: {TORCH_VERSION}; cuda: {CUDA_VERSION}')


class WandbLogger:
    def __init__(self, project_name, name_of_model, api_key=None, key_file_path='_private/wandb_key.txt'):

        # Prepare logging
        if api_key is not None:
            os.environ["WANDB_API_KEY"] = api_key
        else:
            try:
                with open(key_file_path, 'r') as key_file:
                    os.environ["WANDB_API_KEY"] = key_file.read().strip()
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"WANDB API key file '{key_file_path}' not found. Please ensure it exists or pass it directly as "
                    f"'api_key' parameter.")

        try:
            self.wandb_instance = wandb.init(project=project_name, name=name_of_model)
        except wandb.errors.CommError as e:
            raise ConnectionError("Failed to initialize WANDB session.") from e
        except Exception as e:
            raise RuntimeError(f"An unexpected error occurred during WANDB initialization: {e}") from e

    def log_metrics(self, key, value):
        self.wandb_instance.log({key: value})

    def log_image(self, image: Image, image_id='NoID Image', caption='Segmentation visualization'):
        self.wandb_instance.log({
            f"validation_image_{image_id}": wandb.Image(image, caption=caption)
        })


class UNetArchitecture(nn.Module):
    def __init__(self, in_channels, num_of_classes, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Steps down
        self.step_down_1 = UNet_parts.DownSample(in_channels, 64)
        self.step_down_2 = UNet_parts.DownSample(64, 128)
        self.step_down_3 = UNet_parts.DownSample(128, 256)
        self.step_down_4 = UNet_parts.DownSample(256, 512)

        self.bottleneck = UNet_parts.DoubleConv(512, 1024)

        self.step_up_1 = UNet_parts.UpSample(1024, 512)
        self.step_up_2 = UNet_parts.UpSample(512, 256)
        self.step_up_3 = UNet_parts.UpSample(256, 128)
        self.step_up_4 = UNet_parts.UpSample(128, 64)

        self.output = nn.Conv2d(64, num_of_classes, kernel_size=1)

    def forward(self, x):
        # Going down
        enc_1, enc_pool_1 = self.step_down_1(x)
        enc_2, enc_pool_2 = self.step_down_2(enc_pool_1)
        enc_3, enc_pool_3 = self.step_down_3(enc_pool_2)
        enc_4, enc_pool_4 = self.step_down_4(enc_pool_3)

        # Bottleneck
        bott = self.bottleneck(enc_pool_4)

        # Going up
        dec_1 = self.step_up_1(bott, enc_4)
        dec_2 = self.step_up_2(dec_1, enc_3)
        dec_3 = self.step_up_3(dec_2, enc_2)
        dec_4 = self.step_up_4(dec_3, enc_1)
        output = self.output(dec_4)

        soft_max = F.softmax(output, dim=1)

        return soft_max


class UNet(pl.LightningModule):
    def __init__(self, in_channels, num_classes, callbacks='default', log_mode='all'):
        super().__init__()
        self.num_classes = num_classes
        self.loss_fn = None
        self.model = UNetArchitecture(in_channels, num_classes)
        self.callbacks = []
        self.validation_losses = []

        self.log_mode = log_mode
        self.wand_logger = None
        try:
            self.wand_logger = WandbLogger(project_name='Birds Naive UNet', name_of_model='Testing')
        except Exception as e:
            PRINT.error(f'During WANDB engine initialization following error occurred: {e}')

        # Setting metrics
        self.accuracy = Accuracy(num_classes=num_classes, average='macro', task='binary')
        self.precision = Precision(num_classes=num_classes, average='macro', task='binary')
        self.recall = Recall(num_classes=num_classes, average='macro', task='binary')
        self.f1_score = F1Score(num_classes=num_classes, average='macro', task='binary')
        self.iou = JaccardIndex(num_classes=num_classes, average='macro', task='binary')

        if callbacks == 'default':
            self.checkpoint_saver_init()
            self.early_stopping_init()

        self.set_loss()
        self.configure_optimizers()

    def log_scalar(self, key, value, prog_bar=True):
        if self.log_mode == 'all':
            self.log(key, value, prog_bar=True)

            if self.wand_logger is not None:
                self.wand_logger.log_metrics(key, value)

        if self.log_mode == 'local_only':
            self.log(key, value, prog_bar=value)

    def log_image(self, image: Image, image_id='NoID Image', caption='Segmentation visualization'):
        if self.log_mode == 'all':
            if self.wand_logger is not None:
                self.wand_logger.log_image(image, image_id, caption)
                plt.imshow(image)
                plt.axis('off')  # Hide axes for cleaner display
                plt.show()

        if self.log_mode == 'local_only':
            plt.imshow(image)
            plt.axis('off')  # Hide axes for cleaner display
            plt.show()

    def forward(self, x):
        return self.model(x)

    def set_loss(self, loss_fn=None):
        if loss_fn is None:
            self.loss_fn = nn.CrossEntropyLoss()  # nn.BCEWithLogitsLoss()
        else:
            self.loss_fn = loss_fn

    # def model_probabilities(self, images):
    #     masks_pred = self.model(images)
    #     probs = F.softmax(masks_pred, dim=1)
    #     return probs

    def training_step(self, batch, batch_idx):
        images, masks = batch
        spectrum_values = images['spectrum_values']
        masks_chw = masks.squeeze(1)
        masks_pred = self.model(spectrum_values)
        loss = self.loss_fn(masks_pred, masks_chw)
        self.log_scalar('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):  # Is being called for each validation batch
        images, masks = batch

        spectrum_values = images['spectrum_values']
        masks_pred = self.model(spectrum_values)

        masks_squeezed = masks.squeeze(1)  # (batch, H, W)
        masks_one_hot = F.one_hot(masks_squeezed, num_classes=self.num_classes)  # (batch, H, W, num_classes)
        masks_one_hot = masks_one_hot.permute(0, 3, 1, 2).float()

        loss = self.loss_fn(masks_pred, masks_squeezed)

        self.validation_losses.append(loss.item())
        self.log_scalar('val_loss_batch', loss)

        # Log loss and metrics
        self.accuracy.update(masks_pred, masks_one_hot)
        self.precision.update(masks_pred, masks_one_hot)
        self.recall.update(masks_pred, masks_one_hot)
        self.f1_score.update(masks_pred, masks_one_hot)
        self.iou.update(masks_pred, masks_one_hot)

        return loss

    def on_validation_epoch_end(self):  # Is being called after whole validation process
        # Logging metrics
        accuracy_epoch = self.accuracy.compute()
        precision_epoch = self.precision.compute()
        recall_epoch = self.recall.compute()
        f1_epoch = self.f1_score.compute()
        iou_epoch = self.iou.compute()

        self.log_scalar('val_loss', np.mean(self.validation_losses))
        self.log_scalar('val_accuracy', accuracy_epoch, prog_bar=True)
        self.log_scalar('val_precision', precision_epoch, prog_bar=True)
        self.log_scalar('val_recall', recall_epoch, prog_bar=True)
        self.log_scalar('val_f1_score', f1_epoch, prog_bar=True)
        self.log_scalar('val_iou', iou_epoch, prog_bar=True)

        # Resetting metrics
        self.accuracy.reset()
        self.precision.reset()
        self.recall.reset()
        self.f1_score.reset()
        self.iou.reset()

    def configure_optimizers(self, optimizer=None):
        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

    def checkpoint_saver_init(self):
        checkpoint_callback = ModelCheckpoint(
            dirpath='checkpoints/',
            filename='epoch-{epoch:02d}',
            monitor='val_loss',
            save_top_k=1,
            every_n_epochs=1,
            verbose=True
        )

        self.callbacks.append(checkpoint_callback)

    def early_stopping_init(self):
        early_stopping_callback = EarlyStopping(
            monitor='val_loss',
            patience=3,
            mode='min',
            verbose=True
        )

        self.callbacks.append(early_stopping_callback)
