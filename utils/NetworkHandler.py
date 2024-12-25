import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from PIL import Image
from matplotlib import pyplot as plt
import sys
from pathlib import Path

sys.path += [str(Path().resolve().parent.parent)]

from utils.WandbLogger import WandbLogger
from utils import Log

PRINT = Log.get_logger()


class NetworkHandler(pl.LightningModule):
    def __init__(self, model, learning_rate, callbacks='default', debug=False, log_mode='all', log_params=None):
        super().__init__()

        self.loss_fn = None

        self.model = model

        self.wand_logger = None
        self.debug = debug
        self.log_mode = log_mode

        self.wandb_init(log_params)

        self.callbacks = []
        self.validation_losses = []
        self.train_losses = []

        if callbacks == 'default':
            self.checkpoint_saver_init()

        self.learning_rate = learning_rate

        # Setting metrics
        #

    def wandb_init(self, log_params):
        try:
            project_name = log_params.get('project_name', 'dummy_network')
            name_of_model = log_params.get('model_name', 'ResNet')
            key_file_path = log_params.get('key_file_path', None)
            self.wand_logger = WandbLogger(project_name=project_name, name_of_model=name_of_model,
                                           key_file_path=key_file_path)
        except Exception as e:
            PRINT.error(f'During WANDB engine initialization following error occurred: {e}')

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
