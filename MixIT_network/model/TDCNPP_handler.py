import sys
import os
sys.path.append("/Users/vojtechremis/Desktop/Projects/birdsong-segmentation")
sys.path.append(os.path.abspath("../../"))

from utils import Log

# Definition of paths
from pathlib import Path

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np

from utils.WandbLogger import WandbLogger
from TDCNPP_architecture import TDCNPPArchitecture
from TDCNPP_parts import MixIT_Loss
from MixIT_metrics import SiSNR_Metric



PRINT = Log.get_logger()


class TDCNPP(pl.LightningModule):
    def __init__(self, model_configuration, training_configuration, save_dir, callbacks='default', debug=False, log_mode='all',
                 log_params={}):
        super().__init__()
        self.model_configuration = model_configuration
        self.training_configuration = training_configuration

        self.model = TDCNPPArchitecture(self.model_configuration)

        self.callbacks = []
        self.validation_losses = []

        # Parameters
        self.num_epochs = self.training_configuration.get('num_epochs')

        # Administration
        self.save_dir = save_dir
        self.debug = debug
        self.log_mode = log_mode
        self.wand_logger = None
        try:
            project_name = log_params.get('project_name', 'Unknown_MixIT_project')
            name_of_model = log_params.get('model_name', 'MixIT_model')
            key_file_path = log_params.get('key_file_path', None)
            self.wand_logger = WandbLogger(project_name=project_name, name_of_model=name_of_model,
                                           key_file_path=key_file_path)
        except Exception as e:
            PRINT.error(f'During WANDB engine initialization following error occurred: {e}')

        # Setting metrics
        self.SiSNR = SiSNR_Metric()
        self.SiSNRi = SiSNR_Metric()

        if callbacks == 'default':
            self.checkpoint_saver_init()
            self.early_stopping_init()

        self.set_loss()
        self.configure_optimizer(self.training_configuration.get('optimizer_configuration'))

    def log_scalar(self, key, value, prog_bar=True):
        if self.log_mode in ['all', 'wandb']:
            if self.wand_logger is not None:
                self.wand_logger.log_metrics(key, value)

        self.log(key, value, prog_bar=prog_bar, sync_dist=True)

    def forward(self, x):
        return self.model(x)

    def set_loss(self, loss_fn=None):
        if loss_fn is None:
            self.loss_fn = MixIT_Loss()
        else:
            self.loss_fn = loss_fn

    def training_step(self, batch, batch_idx):
        target_sources = batch
        model_input = torch.sum(target_sources, dim=1, keepdim=True)

        estimated_sources = self.forward(model_input)
        loss = self.loss_fn(estimated_sources, target_sources)

        self.log_scalar('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):  # Is being called for each validation batch
        target_sources = batch
        model_input = torch.sum(target_sources, dim=1, keepdim=True)

        estimated_sources = self.forward(model_input)

        validation_loss = self.loss_fn(estimated_sources, target_sources)

        self.validation_losses.append(validation_loss.item())

        # Log loss and metrics
        if estimated_sources.shape[1] == target_sources.shape[1]:
            self.SiSNR.update(preds=estimated_sources, target=target_sources)
            self.SiSNRi.update(preds=estimated_sources, target=target_sources, mixture_input=model_input)
        else:
            PRINT.warning(f"Počet odhadovaných zdrojů ({estimated_sources.shape[1]}) se neshoduje s cílovými zdroji pro evaluaci ({target_sources.shape[1]}). Metriky se nepočítají.")

        return validation_loss

    def on_validation_epoch_end(self):  # Is being called after whole validation process
        # Computing metrics
        SiSNR_epoch = self.SiSNR.compute()
        SiSNRi_epoch = self.SiSNRi.compute()

        # Logging metrics
        self.log_scalar('val_loss', np.mean(self.validation_losses), prog_bar=True)
        self.log_scalar('SiSNR', SiSNR_epoch, prog_bar=True)
        self.log_scalar('SiSNRi', SiSNRi_epoch, prog_bar=True)

        # Resetting metrics
        self.SiSNR.reset()
        self.SiSNRi.reset()
    
    def configure_optimizer(self, optimizer_config):

        optimizer_name = optimizer_config.get('optimizer')
        learning_rate = optimizer_config.get('learning_rate')

        # Initialization of optimizer
        if optimizer_name == 'Adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        # Scheduler
        scheduler = None
        if optimizer_config.get("scheduler") is not None and optimizer_config.get("warmup_epochs") is not None:
            scheduler_name = optimizer_config.get("scheduler")
            warmup_epochs = optimizer_config.get("warmup_epochs")

            if scheduler_name == "cosine":
                # Cosine Annealing s warmupem
                def lr_lambda(current_epoch):
                    if current_epoch < warmup_epochs:
                        return float(current_epoch + 1) / float(warmup_epochs)
                    else:
                        cosine_progress = (current_epoch - warmup_epochs) / max(1, self.num_epochs - warmup_epochs)
                        return 0.5 * (1 + torch.cos(torch.tensor(cosine_progress * np.pi)))

                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

            else:
                raise ValueError(f"Unsupported scheduler: {scheduler_name}.")

        return optimizer, scheduler

    def checkpoint_saver_init(self):
        checkpoint_callback = ModelCheckpoint(
            dirpath=f'{self.save_dir}/checkpoints/',
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
        pass

    def inference(self, x):
        # Setting evaluation mode
        pass

if __name__ == '__main__':
    config = {
        "input_mixtures": 2,
        "target_sources": 4,
        "encoder_channels": 512,
        "encoder_kernel_size": 16,
        "TDCN_Separation_Block": {
            "initial_bottleneck_channels": 128,
            "skip_connections_channels": 128,
            "number_of_repeats": 3,
            "number_of_blocks_per_repeat": 8,
            "add_skip_residual_connections": False,
            "OneDConv_Block": {
                "internal_channels": 512,
                "kernel_size": 3,
                "scale_init": 0.9
            }
        },
        "optimizer_configuration": {
            "optimizer_name": "Adam",
            "learning_rate": 10e-3
        },
        "window_size_ms": 2,
        "sample_rate": 16000
    }

    model = TDCNPPArchitecture(model_config = config)

    # model = TDCNPPArchitecture_nnmodule(config = config)
    from torchinfo import summary
    batch_size = 24
    summary(model, input_size=(batch_size, 1, 16000*6), depth=10)  # např. (1, 16000) pro audio

    # dummy_input = torch.randn(1, 1, 16)
    # torch.onnx.export(model, dummy_input, "model.onnx")




    # from torchviz import make_dot

    # x = torch.randn(1, 1, 16)  # např. vstupní waveform
    # y = model(x)

    # make_dot(y, params=dict(model.named_parameters())).render("model", format="png")


    # from torch.utils.tensorboard import SummaryWriter

    # writer = SummaryWriter()
    # x = torch.randn(1, 1, 16)  # vstupní vzorek
    # writer.add_graph(model, x)
    # writer.close()