import sys
from pathlib import Path
import json
import torch
from torch.utils.data import DataLoader, random_split
import os
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from .TDCNPP_handler import TDCNHandler
from .MixIT_dataHandler import MixITDataset

def load_config(config_path):
    """Načte konfigurační soubor"""
    with open(config_path, 'r') as f:
        return json.load(f)

def load_wandb_key(key_path):
    """Načte Weights & Biases API klíč ze souboru"""
    try:
        with open(key_path, 'r') as f:
            return f.read().strip()
    except Exception as e:
        print(f"Chyba při načítání W&B klíče: {e}")
        return None

def main():
    # Nastavení cesty pro konfiguraci
    # Získání absolutní cesty ke skriptu a k adresáři
    script_path = Path(__file__).resolve()
    script_dir = script_path.parent
    project_root = script_dir.parent.parent  # Kořen projektu
    config_path = script_dir / 'config.json'
    wandb_key_path = project_root / '_private' / 'wandb_key.txt'
    
    # Načtení konfigurace
    config = load_config(config_path)
    
    # Načtení W&B API klíče
    wandb_key = load_wandb_key(wandb_key_path)
    if wandb_key:
        print("Weights & Biases API klíč úspěšně načten")
        os.environ["WANDB_API_KEY"] = wandb_key
    else:
        print("Weights & Biases API klíč nenalezen, bude vyžadována manuální autentizace.")
    
    # Parametry modelu
    model_config = config['model']
    training_config = config['training']
    data_config = config['data']
    logging_config = config['logging']
    
    # Vytvoření adresářů pro ukládání, pokud neexistují
    log_dir = Path(logging_config['log_dir'])
    save_dir = Path(logging_config['save_dir'])
    log_dir.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Vytvoření datasetu
    dataset = MixITDataset(
        data_dir=data_config['train_dir'],
        sample_rate=data_config['sample_rate'],
        segment_length=data_config['segment_length'],
        augment_prob=0
    )
    
    # Rozdělení datasetu
    generator = torch.Generator()
    generator.manual_seed(42)
    
    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        dataset, 
        [train_size, val_size],
        generator=generator
    )
    
    print(f'Dataset sizes: Train = {len(train_dataset)}, Val = {len(val_dataset)}')
    
    # Vytvoření dataloaderů
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=training_config['batch_size'],
        shuffle=True,
        num_workers=data_config['number_of_workers'],
        persistent_workers=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=training_config['batch_size'],
        shuffle=False,
        num_workers=data_config['number_of_workers'],
        persistent_workers=True
    )
    
    # Creating model
    model = TDCNHandler(model_config, training_config, save_dir)
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=str(save_dir),
            filename='tdcnpp-{epoch:02d}-{val_loss:.2f}',
            save_top_k=3,
            monitor='val_loss',
            mode='min'
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            mode='min'
        )
    ]
    
    # Logger
    logger = WandbLogger(
        project=logging_config.get('project_name', 'tdcnpp-mixit'),
        name=logging_config.get('run_name', 'tdcnpp-mixit-test'),
        offline=wandb_key is None  # Offline mód, pokud není k dispozici klíč
    )
    
    # Device selection
    if torch.cuda.is_available():
        device = 'gpu'
        accelerator = 'cuda'
    elif torch.backends.mps.is_available():  # Pro Apple Silicon
        device = 'mps'
        accelerator = 'mps'
    else:
        device = 'cpu'
        accelerator = 'cpu'
    
    print(f'Using device: {device}')
    
    # Trainer
    trainer = Trainer(
        max_epochs=training_config['num_epochs'],
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=logging_config['log_interval'],
        accelerator=accelerator,
        devices=1
    )
    
    # Training
    trainer.fit(model, train_dataloader, val_dataloader)
    
    # Saving model
    model_save_path = save_dir / 'tdcnpp_final.pt'
    torch.save(model.state_dict(), str(model_save_path))
    print(f'Model saved to {model_save_path}')

if __name__ == '__main__':
    main() 