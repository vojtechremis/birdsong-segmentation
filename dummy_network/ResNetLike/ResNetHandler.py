import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
import random
from PIL import Image
from matplotlib import pyplot as plt
import sys
from pathlib import Path

sys.path += [str(Path().resolve().parent.parent)]

from ResNet_parts import SegmentsLoss, CustomAccuracy, CustomMetric, ResidualBlock
from utils.Spectrogram import Spectrogram
from utils.WandbLogger import WandbLogger
from utils import Log

PRINT = Log.get_logger()


class ResNet18Architecture(nn.Module):
    def __init__(self, dropout_prob=0, *args, **kwargs):
        # Předtrénovaný ResNet-18
        super().__init__(*args, **kwargs)
        self.resnet = models.resnet18(pretrained=True)

        # Replacing fully connected layer with Global Average Pooling
        # Note: Transforming CNN output of shape (batch, channels, height, width) -> (batch, channels)

        # Adjusting resnet to accept 1-channel input
        self.resnet.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=self.resnet.conv1.out_channels,
            kernel_size=self.resnet.conv1.kernel_size,
            stride=self.resnet.conv1.stride,
            padding=self.resnet.conv1.padding,
            bias=self.resnet.conv1.bias
        )

        # Removing fc layer
        self.resnet.fc = nn.Identity()

        self.final_droupout = nn.Dropout(p=dropout_prob)

        # Initializing extra parts
        # self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.regression_head = nn.Linear(512, 2)

    def forward(self, x):
        x = self.resnet(x)
        # x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.final_droupout(x)
        x = self.regression_head(x)
        return x


class ResNet50Architecture(nn.Module):
    def __init__(self, dropout_prob=0, *args, **kwargs):
        # Předtrénovaný ResNet-50
        super().__init__(*args, **kwargs)
        self.resnet = models.resnet50(pretrained=True)

        # Replacing fully connected layer with Global Average Pooling
        # Note: Transforming CNN output of shape (batch, channels, height, width) -> (batch, channels)

        # Adjusting resnet to accept 1-channel input
        self.resnet.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=self.resnet.conv1.out_channels,
            kernel_size=self.resnet.conv1.kernel_size,
            stride=self.resnet.conv1.stride,
            padding=self.resnet.conv1.padding,
            bias=self.resnet.conv1.bias
        )

        # Removing fc layer
        self.resnet.fc = nn.Identity()

        self.final_droupout = nn.Dropout(p=dropout_prob)

        # Initializing extra parts
        # self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.regression_head = nn.Linear(2048, 2)

    def forward(self, x):
        x = self.resnet(x)
        # x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.final_droupout(x)
        x = self.regression_head(x)
        return x


class LittleResNet(nn.Module):
    def __init__(self, input_channels=1, dropout_prob=0.3):
        super().__init__()
        self.dropout_prob = dropout_prob

        # Input layer
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.dropout1 = nn.Dropout2d(p=dropout_prob)

        # Residual layers
        self.layer1 = self._make_layer(ResidualBlock, 32, 32, stride=1)  # Stage 1
        self.layer2 = self._make_layer(ResidualBlock, 32, 64, stride=2)  # Stage 2
        self.layer3 = self._make_layer(ResidualBlock, 64, 128, stride=2)  # Stage 3
        self.layer4 = self._make_layer(ResidualBlock, 128, 256, stride=2)  # Stage 4

        # Dropout before regression head
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout2 = nn.Dropout(p=dropout_prob)
        self.regression_head = nn.Linear(256, 2)

    def _make_layer(self, block, in_channels, out_channels, stride):
        layers = [block(in_channels, out_channels, stride)]
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout1(out)  # Dropout after input layer
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = self.dropout2(out)  # Dropout before flattening
        out = torch.flatten(out, 1)
        out = self.regression_head(out)
        return out


class LSTMHead(nn.Module):
    def __init__(self, input_channels, lstm_hidden_size, lstm_num_layers, output_dim, dropout_prob=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_channels,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout_prob if lstm_num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(lstm_hidden_size, output_dim)

    def forward(self, x):
        # LSTM forward
        lstm_out, _ = self.lstm(x)  # (batch_size, seq_len, lstm_hidden_size)
        out = lstm_out[:, -1, :]  # Use the last time step (batch_size, lstm_hidden_size)
        out = self.fc(out)  # (batch_size, output_dim)
        return out


class LittleResNetLSTM(nn.Module):
    def __init__(self, input_channels=1, lstm_hidden_size=128, lstm_num_layers=1, dropout_prob=0.3):
        super().__init__()
        self.dropout_prob = dropout_prob

        # Input layer
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout2d(p=dropout_prob)

        # Residual layers
        self.layer1 = self._make_layer(ResidualBlock, 32, 32, stride=1)  # Stage 1
        self.layer2 = self._make_layer(ResidualBlock, 32, 64, stride=2)  # Stage 2
        self.layer3 = self._make_layer(ResidualBlock, 64, 128, stride=2)  # Stage 3
        self.layer4 = self._make_layer(ResidualBlock, 128, 128, stride=2)  # Stage 4

        # Dropout before regression head

        self.dropout2 = nn.Dropout(p=dropout_prob)

        # Pooling over height to reduce dimensions
        self.global_pool = nn.AdaptiveAvgPool2d((1, None))  # Pool over height

        # Two LSTM heads
        self.lstm_head1 = LSTMHead(
            input_channels=128, lstm_hidden_size=lstm_hidden_size, lstm_num_layers=lstm_num_layers, output_dim=1
        )
        self.lstm_head2 = LSTMHead(
            input_channels=128, lstm_hidden_size=lstm_hidden_size, lstm_num_layers=lstm_num_layers, output_dim=1
        )

        self.regression_head1 = nn.Linear(lstm_hidden_size, 1)

    def _make_layer(self, block, in_channels, out_channels, stride):
        layers = [block(in_channels, out_channels, stride)]
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.pool1(out)

        out = self.dropout1(out)  # Dropout after input layer
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)


        # Global Pooling over height
        out = self.global_pool(out)  # (batch, channels, 1, width)
        out = out.squeeze(2)  # Remove the height dimension -> (batch, channels, width)
        out = out.permute(0, 2, 1).contiguous() # (batch, channels, width)  -> (batch, seq_len, features)
        # Pass through LSTM heads
        out1 = self.lstm_head1(out)  # First LSTM head
        out2 = self.lstm_head2(out)  # Second LSTM head

        # Concatenate outputs
        combined_out = torch.cat((out1, out2), dim=1)  # (batch_size, 2)

        return combined_out

class ResNetHandler(pl.LightningModule):
    def __init__(self, model, dataset=None, callbacks='default', debug=False, log_mode='all', log_params=None):
        super().__init__()

        self.model = model
        self.loss_fn = None

        self.debug = debug
        self.log_mode = log_mode
        self.wand_logger = None
        try:
            project_name = log_params.get('project_name', 'dummy_network')
            name_of_model = log_params.get('model_name', 'ResNet18')
            key_file_path = log_params.get('key_file_path', None)
            self.wand_logger = WandbLogger(project_name=project_name, name_of_model=name_of_model,
                                           key_file_path=key_file_path)
        except Exception as e:
            PRINT.error(f'During WANDB engine initialization following error occurred: {e}')

        self.callbacks = []
        self.validation_losses = []

        if callbacks == 'default':
            self.checkpoint_saver_init()
            # self.early_stopping_init()

        self.set_loss()
        self.configure_optimizers()

        # Setting metrics
        self.accuracy_segment = CustomAccuracy().to(self.device)
        self.metric_segment = CustomMetric().to(self.device)

        self.accuracy_motif = CustomAccuracy().to(self.device)
        self.metric_motif = CustomMetric().to(self.device)

        # Setting training metrics

        self.train_accuracy_segment = CustomAccuracy().to(self.device)
        self.train_metric_segment = CustomMetric().to(self.device)

        self.train_accuracy_motif = CustomAccuracy().to(self.device)
        self.train_metric_motif = CustomMetric().to(self.device)

    def forward(self, x):
        return self.model(x)

    def log_scalar(self, key, value, prog_bar=True):
        if not self.trainer.global_rank == 0:
            return None

        if self.log_mode in ['all', 'wandb']:
            if self.wand_logger is not None:
                self.wand_logger.log_metrics(key, value)

        self.log(key, value, prog_bar=prog_bar, sync_dist=True)

    def log_image(self, image: Image, image_id='NoID Image', caption='Batch visualization'):

        if self.log_mode in ['all', 'wandb']:
            if self.wand_logger is not None:
                self.wand_logger.log_image(image, image_id, caption)

        if self.log_mode in ['all', 'local']:
            plt.imshow(image)
            plt.axis('off')  # Hide axes for cleaner display
            plt.show()

    def training_step(self, batch, batch_idx):

        images, y_true = batch
        y_hat = self.model(images)

        loss = self.loss_fn(y_hat, y_true)

        self.train_losses.append(loss.item())

        # Extracting segments and motifs - each y_* has shape (batch, 2)
        y_hat_seg = torch.round(y_hat[:, 0])
        y_hat_mot = torch.round(y_hat[:, 1])

        y_true_seg = y_true[0]
        y_true_mot = y_true[1]

        # Append to Segment metrics
        self.train_accuracy_segment(y_hat_seg, y_true_seg)
        self.train_metric_segment(y_hat_seg, y_true_seg)

        # Append to Motif metrics
        self.train_accuracy_motif(y_hat_mot, y_true_mot)
        self.train_metric_motif(y_hat_mot, y_true_mot)

        return loss

    def on_train_epoch_end(self):  # on_train_epoch_end() is called after validation

        # Log loss
        self.log_scalar('train_loss', np.mean(self.train_losses), prog_bar=True)

        # Calculate
        accuracy_segment = self.train_accuracy_segment.compute()
        metric_segment = self.train_metric_segment.compute()

        # Log
        self.log_scalar('train_acc_seg', accuracy_segment, prog_bar=True)
        self.log_scalar('train_prec_seg', metric_segment['precision'], prog_bar=True)
        self.log_scalar('train_rec_seg', metric_segment['recall'], prog_bar=True)
        self.log_scalar('train_f1_seg', metric_segment['f1_score'], prog_bar=True)

        # Reset
        self.train_accuracy_segment.reset()
        self.train_metric_segment.reset()

        # Motif metrics
        #

        # Calculate
        accuracy_motif = self.train_accuracy_motif.compute()
        metric_motif = self.train_metric_motif.compute()

        # Log
        self.log_scalar('train_acc_mot', accuracy_motif, prog_bar=True)
        self.log_scalar('train_prec_mot', metric_motif['precision'], prog_bar=True)
        self.log_scalar('train_rec_mot', metric_motif['recall'], prog_bar=True)
        self.log_scalar('train_f1_mot', metric_motif['f1_score'], prog_bar=True)

        # Reset
        self.train_accuracy_motif.reset()
        self.train_metric_motif.reset()

    def validation_step(self, batch, batch_idx):  # Is being called for each validation batch

        images, y_true = batch

        y_hat = self.model(images)
        loss = self.loss_fn(y_hat, y_true)

        self.validation_losses.append(loss.item())

        # Extracting segments and motifs - each y_* has shape (batch, 2)
        y_hat_seg = torch.round(y_hat[:, 0])
        y_hat_mot = torch.round(y_hat[:, 1])

        y_true_seg = y_true[0]
        y_true_mot = y_true[1]

        # Append to Segment metrics
        self.accuracy_segment(y_hat_seg, y_true_seg)
        self.metric_segment(y_hat_seg, y_true_seg)

        # Append to Motif metrics
        self.accuracy_motif(y_hat_mot, y_true_mot)
        self.metric_motif(y_hat_mot, y_true_mot)

        # # Log validation image
        # num_log_images = 2
        # batch_size = images.shape[0]
        # random_indices = random.sample(range(batch_size), min(num_log_images, batch_size))
        #
        # for idx in random_indices:
        #     Spectrum = Spectrogram(images[idx, :])
        #     Spectrum.to_cpu()
        #
        #     caption = f'Pred: {y_hat_seg[idx]}s, {y_hat_mot[idx]}m.    True: {int(y_true_seg[idx])}s, {int(y_true_mot[idx])}m'
        #     image = Spectrum.show(caption=caption)
        #     self.log_image(image, f'Val_{idx}')

        return loss

    def on_validation_epoch_end(self):  # Is being called after whole validation process

        # Log loss
        self.log_scalar('val_loss', np.mean(self.validation_losses), prog_bar=True)

        # Segment metrics
        #

        # Calculate
        accuracy_segment = self.accuracy_segment.compute()
        metric_segment = self.metric_segment.compute()

        # Log
        self.log_scalar('val_acc_seg', accuracy_segment, prog_bar=True)
        self.log_scalar('val_prec_seg', metric_segment['precision'], prog_bar=True)
        self.log_scalar('val_rec_seg', metric_segment['recall'], prog_bar=True)
        self.log_scalar('val_f1_seg', metric_segment['f1_score'], prog_bar=True)

        # Reset
        self.accuracy_segment.reset()
        self.metric_segment.reset()

        # Motif metrics
        #

        # Calculate
        accuracy_motif = self.accuracy_motif.compute()
        metric_motif = self.metric_motif.compute()

        # Log
        self.log_scalar('val_acc_mot', accuracy_motif, prog_bar=True)
        self.log_scalar('val_prec_mot', metric_motif['precision'], prog_bar=True)
        self.log_scalar('val_rec_mot', metric_motif['recall'], prog_bar=True)
        self.log_scalar('val_f1_mot', metric_motif['f1_score'], prog_bar=True)

        # Reset
        self.accuracy_motif.reset()
        self.metric_motif.reset()

    def set_loss(self):
        self.loss_fn = SegmentsLoss()

    def configure_optimizers(self, optimizer=None):
        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

    def checkpoint_saver_init(self):
        checkpoint_callback = ModelCheckpoint(
            dirpath='checkpoints/',
            filename='epoch-{epoch:02d}',
            monitor='val_loss',
            save_last=True,
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
