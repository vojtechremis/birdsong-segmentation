import torch
import torch.nn as nn
import torchmetrics
import torch.nn.functional as F
import numpy as np
import sys
from pathlib import Path

sys.path += [str(Path().resolve().parent.parent)]

from dummy_network.ResNetLike.ResNet_parts import SegmentsLoss, ResidualBlock
from dummy_network.ResNetLike.ResNetHandler import LSTMHead
from utils import NetworkHandler
from utils import Log

PRINT = Log.get_logger()


class ClassificationLittleResNet(nn.Module):
    def __init__(self, input_channels, num_classes, dropout_prob=0.3):
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
        self.classification_head = nn.Linear(256, num_classes)

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
        out = self.classification_head(out)
        return out


class xxxLittleResNetLSTM(nn.Module):
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
        out = out.permute(0, 2, 1).contiguous()  # (batch, channels, width)  -> (batch, seq_len, features)
        # Pass through LSTM heads
        out1 = self.lstm_head1(out)  # First LSTM head
        out2 = self.lstm_head2(out)  # Second LSTM head

        # Concatenate outputs
        combined_out = torch.cat((out1, out2), dim=1)  # (batch_size, 2)

        return combined_out


class ClassificationResNetHandler(NetworkHandler.NetworkHandler):
    def __init__(self, model, num_classes, learning_rate=0.001, callbacks='default', debug=False, log_mode='all', log_params=None):
        super().__init__(model=model, learning_rate=learning_rate, callbacks=callbacks, debug=debug, log_mode=log_mode, log_params=log_params)

        # Setting loss and optimizer
        self.set_loss()
        self.configure_optimizers()

        # Setting metrics
        self.accuracy_metric = torchmetrics.Accuracy(num_classes=num_classes, task='multiclass', average='macro')
        self.precision_metric = torchmetrics.Precision(num_classes=num_classes, task='multiclass', average='macro')
        self.recall_metric = torchmetrics.Recall(num_classes=num_classes, task='multiclass', average='macro')

    def training_step(self, batch, batch_idx):

        images, y_true = batch
        logits = self.model(images)
        loss = self.loss_fn(logits, y_true)
        self.train_losses.append(loss.item())

        return loss

    def validation_step(self, batch, batch_idx):  # Is being called for each validation batch

        images, true_labels = batch

        logits = self(images)
        loss = self.loss_fn(logits, true_labels)
        self.validation_losses.append(loss.item())

        # Compute predictions
        preds = torch.argmax(logits, dim=1)

        # Update metrics
        self.accuracy_metric.update(preds, true_labels)
        self.precision_metric.update(preds, true_labels)
        self.recall_metric.update(preds, true_labels)

        return loss

    def on_validation_epoch_end(self):  # Is being called after whole validation process

        # Log loss
        self.log_scalar('val_loss', np.mean(self.validation_losses), prog_bar=True)

        # Segment metrics
        #

        # Calculate
        accuracy = self.accuracy_metric.compute()
        precision = self.precision_metric.compute()
        recall = self.recall_metric.compute()

        # Log
        self.log_scalar('accuracy_val', accuracy, prog_bar=True)
        self.log_scalar('precision_val', precision, prog_bar=True)
        self.log_scalar('recall_val', recall, prog_bar=True)

        # Reset
        self.accuracy_metric.reset()
        self.precision_metric.reset()
        self.recall_metric.reset()

    def set_loss(self):
        class_weights = torch.tensor([7.93100000e+03, 2.64366667e+02, 6.89652174e+01, 1.26289809e+01, 1.10814587e+00], dtype=torch.float32).to(self.device)
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    def configure_optimizers(self, optimizer=None):
        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer