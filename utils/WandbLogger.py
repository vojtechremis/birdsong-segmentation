import wandb
from PIL import Image, ImageDraw, ImageFont
import os


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
