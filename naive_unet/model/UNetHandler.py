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
from utils.Visualization import spectrum_above_mask
from utils import Transformations
import UNet_parts
import os
import numpy as np
import random
import json

from utils import Log, SegmentsHandler

PRINT = Log.get_logger()


def load_config(config_file_path):
    with open(config_file_path, 'r') as file:
        config = json.load(file)
    return config


class WandbLogger:
    def __init__(self, project_name, name_of_model, api_key=None,
                 key_file_path=None):

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
    def __init__(self, in_channels, num_classes, DataHandler, callbacks='default', debug=False, log_mode='all',
                 log_params={}):
        super().__init__()
        self.num_classes = num_classes
        self.loss_fn = None
        self.model = UNetArchitecture(in_channels, num_classes)
        self.callbacks = []
        self.validation_losses = []

        self.DataHandler = DataHandler

        self.bin_size = 128

        self.debug = debug
        self.log_mode = log_mode
        self.wand_logger = None
        try:
            project_name = log_params.get('project_name', 'UNet_project')
            name_of_model = log_params.get('model_name', 'UNet_model')
            key_file_path = log_params.get('key_file_path', None)
            self.wand_logger = WandbLogger(project_name=project_name, name_of_model=name_of_model,
                                           key_file_path=key_file_path)
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
        if self.log_mode in ['all', 'wandb']:
            if self.wand_logger is not None:
                self.wand_logger.log_metrics(key, value)

        self.log(key, value, prog_bar=prog_bar, sync_dist=True)

    def log_image(self, image: Image, image_id='NoID Image', caption='Segmentation visualization'):
        if self.log_mode in ['all', 'wandb']:
            if self.wand_logger is not None:
                self.wand_logger.log_image(image, image_id, caption)

        if self.log_mode in ['all', 'local']:
            plt.imshow(image)
            plt.axis('off')  # Hide axes for cleaner display
            plt.show()

    def forward(self, x):
        return self.model(x)

    def set_loss(self, loss_fn=None):
        if loss_fn is None:
            self.loss_fn = nn.CrossEntropyLoss()
        else:
            self.loss_fn = loss_fn

    def training_step(self, batch, batch_idx):
        images, masks = batch
        spectrum_values = images['spectrum_values']
        mask_values = masks['mask_values']

        masks_chw = mask_values.squeeze(1)
        masks_pred = self.model(spectrum_values)
        loss = self.loss_fn(masks_pred, masks_chw)
        self.log_scalar('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):  # Is being called for each validation batch
        images, masks = batch
        spectrum_values = images['spectrum_values']
        mask_values = masks['mask_values']

        masks_pred = self.model(spectrum_values)

        batch_size = spectrum_values.shape[0]

        masks_squeezed = mask_values.squeeze(1)  # (batch, H, W)
        masks_one_hot = F.one_hot(masks_squeezed, num_classes=self.num_classes)  # (batch, H, W, num_classes)
        masks_one_hot = masks_one_hot.permute(0, 3, 1, 2).float()

        loss = self.loss_fn(masks_pred, masks_squeezed)

        self.validation_losses.append(loss.item())

        # Log loss and metrics
        self.accuracy.update(masks_pred, masks_one_hot)
        self.precision.update(masks_pred, masks_one_hot)
        self.recall.update(masks_pred, masks_one_hot)
        self.f1_score.update(masks_pred, masks_one_hot)
        self.iou.update(masks_pred, masks_one_hot)

        # Log validation image
        num_log_images = 4
        random_indices = random.sample(range(batch_size), min(num_log_images, batch_size))

        for idx in random_indices:
            spectrum_values = images['spectrum_values'][idx]  # shape (chan, height, width)
            mask_values = torch.argmax(masks_pred[idx], 0).unsqueeze(
                0)  # format 0 if background, 1 if object -> shape (chan, height, width)

            image_transformed_inverse, mask_transformed_inverse, freq_bins, time_bins = \
                self.DataHandler.transform.inverse_input_mask(
                    spectrum_values,
                    mask_values,
                    images['freq_bins'][idx],
                    images['time_bins'][idx],
                    (images['original_size'][0][idx], images['original_size'][1][idx]),
                    (images['padding'][0][idx], images['padding'][1][idx])
                )

            image_PIL = spectrum_above_mask(spectrum=image_transformed_inverse.cpu().squeeze(0),
                                            mask=mask_transformed_inverse.cpu().squeeze(0), frequency_bins=freq_bins,
                                            time_bins=time_bins, sample_id=f'Val_{idx}', output_mode='return')
            self.log_image(image_PIL, f'Val_{idx}')

        return loss

    def on_validation_epoch_end(self):  # Is being called after whole validation process
        # Logging metrics
        accuracy_epoch = self.accuracy.compute()
        precision_epoch = self.precision.compute()
        recall_epoch = self.recall.compute()
        f1_epoch = self.f1_score.compute()
        iou_epoch = self.iou.compute()

        self.log_scalar('val_loss', np.mean(self.validation_losses), prog_bar=True)
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
            min_delta=10e-3,
            patience=3,
            mode='min',
            verbose=True
        )

        self.callbacks.append(early_stopping_callback)

    def save(self, model_path):
        torch.save(self.state_dict(), model_path)
        PRINT.info(f"Model saved to {model_path}.")

    def load_weights(self, model_weights_path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_dict = torch.load(model_weights_path, map_location=device, weights_only=True)

        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace("model.", "") if key.startswith("model.") else key
            new_state_dict[new_key] = value

        if self.debug:
            PRINT.info(f'Model weights loaded from {model_weights_path} using device: {device}.')

        self.model.load_state_dict(new_state_dict)

    def inference_by_filepath(self, file_path, bin_size, plot_predictions=False):
        time_bins, frequency_bins, spectrum, y = SegmentsHandler.load_birdsong(file_path)
        segments = SegmentsHandler.crop_spectrogram(spectrum, time_bins, crop_size=bin_size)

        outputs = []
        for segment in segments:
            predicted_mask = self.inference(
                {
                    'spectrum_values': segment['spectrum'],
                    'time_bins': segment['time_bins'],
                    'freq_bins': frequency_bins
                }
            )

            outputs.append(predicted_mask)

            if plot_predictions:
                image_PIL = spectrum_above_mask(spectrum=segment['spectrum'].cpu().squeeze(0),
                                                mask=predicted_mask['mask_values'].cpu().squeeze(0),
                                                frequency_bins=frequency_bins,
                                                time_bins=segment['time_bins'], output_mode='return')

                image_PIL.show()

        output_full = torch.cat([segment['mask_values'] for segment in outputs], dim=2)

        original_time_lenght = time_bins.shape[0]
        output_original_length = output_full[:, :, :original_time_lenght]

        image_PIL = spectrum_above_mask(spectrum=torch.tensor(spectrum),
                                        mask=output_original_length.squeeze(0),
                                        frequency_bins=frequency_bins,
                                        time_bins=time_bins, output_mode='return')

        image_PIL.show()
        # image_PIL.save('/Users/vojtechremis/Desktop/Projects/birdsong-segmentation/naive_unet/_work/' +
        #                file_path.split('/')[-1].split('.')[0] + '.jpg')

        return time_bins, frequency_bins, spectrum, outputs

    def inference(self, x):
        # Setting evaluation mode
        self.eval()

        x_preprocessed = self.DataHandler.transform(x)

        # Disabling gradients
        with torch.no_grad():
            y_pred = self.model(x_preprocessed['spectrum_values'].unsqueeze(0))

        y_pred_argmax = torch.argmax(y_pred, 1)

        # Creating suitable mask format for mask_inverse() method

        padding = x_preprocessed['padding']
        original_size = x_preprocessed['original_size']
        mask_output = self.DataHandler.transform.mask_inverse(
            {
                'mask_values': y_pred_argmax,
                'padding': padding,
                'original_size': original_size
            }
        )

        return mask_output
