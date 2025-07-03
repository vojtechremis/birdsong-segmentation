import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
from TDCNPP_parts import TDCNPP_Separation_Block


class TDCNPPArchitecture(pl.LightningModule):
    """TDCN++ model with MixIT training architecture."""

    def __init__(self, model_config):
        super().__init__()
        self.save_hyperparameters()
        self.config = model_config
        
        self.input_mixtures = self.config.get('input_mixtures', 2)
        
        # Počet zdrojů, které model interně odhaduje (M)
        self.num_estimated_sources = self.config.get('target_sources', 4) # M
        
        # Detection branch
        self.use_detection_branch = self.config.get('use_detection_branch', False)
        self.detection_branch_pooling = self.config.get('detection_branch_pooling', 'avg')

        # Network architecture
        # ----------------------------

        # Encoder / [B, 1, T_audio] -> [B, C_encoder, T_latent]
        # Vstupem jsou 2 směsi (mix1, mix2) stackované jako kanály
        C_encoder = self.config.get('encoder_channels')
        kernel_size = self.config.get('window_size_ms') * self.config.get('sample_rate') // 1000
        self.encoder = nn.Conv1d(
            in_channels=1, 
            out_channels=C_encoder,
            kernel_size=kernel_size,
            stride=kernel_size // 2,
            bias=False
        )
        
        # Separation block / [B, C_encoder, T_latent] -> [B, C_encoder, M, T_latent] (masky)
        TDCN_Separation_Block_config = self.config.get('TDCN_Separation_Block')
        self.separation_block = TDCNPP_Separation_Block(
            in_channels=C_encoder, # Vstupní kanály z enkodéru
            target_sources=self.num_estimated_sources, # Počet interně odhadovaných zdrojů M
            internal_latent_channels=TDCN_Separation_Block_config.get('initial_bottleneck_channels'),
            number_of_repeats=TDCN_Separation_Block_config.get('number_of_repeats'),
            number_of_blocks=TDCN_Separation_Block_config.get('number_of_blocks_per_repeat'),
            add_skip_residual_connections=TDCN_Separation_Block_config.get('add_skip_residual_connections'),
            skip_connections_channels=TDCN_Separation_Block_config.get('skip_connections_channels'),
            OneDConv_Block_channels=TDCN_Separation_Block_config.get('OneDConv_Block').get('internal_channels'),
            OneDConv_Block_kernel_size=TDCN_Separation_Block_config.get('OneDConv_Block').get('kernel_size'),
            OneDConv_Block_scale_init=TDCN_Separation_Block_config.get('OneDConv_Block').get('scale_init'),
            OneDConv_Block_index_offset=0,
            use_detection_branch=self.use_detection_branch,
            detection_branch_pooling=self.detection_branch_pooling
        )

        # Decoder / [B*M, C_encoder, T_latent] -> [B, 1, T_audio_reconstructed]
        # Dekodér zpracovává každý z M odhadovaných zdrojů samostatně.
        self.decoder = nn.ConvTranspose1d(
            in_channels=C_encoder,
            out_channels=1, # Chceme mono výstup pro každý separovaný zdroj
            kernel_size=self.config.get('encoder_kernel_size'),
            stride=self.config.get('encoder_kernel_size') // 2,
            bias=False
        )

    def forward(self, x_input_mixtures):
        # x_input_mixtures má tvar [B, 2, T_audio] (mix1, mix2)

        # Vytvoření vstupního audia z mixů
        x_input_mixtures = torch.sum(x_input_mixtures, dim=1, keepdim=True) # shape: [B, 1, T_audio] 
        
        # 1. Enkódování
        # [B, 1, T_audio] -> [B, C_encoder, T_latent]
        x_latent = self.encoder(x_input_mixtures)
        
        batch_size, c_encoder, t_latent = x_latent.shape

        # 2. Generování masek
        # [B, C_encoder, T_latent] -> [B, C_encoder, M, T_latent]

        if self.use_detection_branch:
            masks, detection_logits = self.separation_block(x_latent)
        else:
            masks = self.separation_block(x_latent) # M = self.num_estimated_sources

        # 3. Aplikace masek
        # Chceme vynásobit každý z M masek s x_latent.
        # x_latent:       [B, C_encoder, T_latent]
        # masks:          [B, C_encoder, M, T_latent]
        # Rozšíříme x_latent pro broadcasting:
        x_latent_expanded = x_latent.unsqueeze(2)  # Tvar: [B, C_encoder, 1, T_latent]
        
        # Násobení (element-wise):
        # [B, C_encoder, 1, T_latent] * [B, C_encoder, M, T_latent] -> [B, C_encoder, M, T_latent]
        masked_latent_sources = x_latent_expanded * masks
        
        # 4. Dekódování každého zdroje
        # Přeuspořádání pro dekodér: chceme [B*M, C_encoder, T_latent]
        # Aktuální tvar: [B, C_encoder, M, T_latent]
        masked_latent_sources = masked_latent_sources.permute(0, 2, 1, 3) # Tvar: [B, M, C_encoder, T_latent]
        
        # Zploštění pro batch processing v dekodéru
        # Tvar: [B*M, C_encoder, T_latent]
        masked_latent_sources_flat = masked_latent_sources.reshape(
            batch_size * self.num_estimated_sources, 
            c_encoder, 
            t_latent
        )
        
        # Dekódování:
        # [B*M, C_encoder, T_latent] -> [B*M, 1, T_audio_reconstructed]
        decoded_sources_flat = self.decoder(masked_latent_sources_flat)
        
        # Odstranění kanálové dimenze (protože out_channels=1)
        # Tvar: [B*M, T_audio_reconstructed]
        decoded_sources_squeezed = decoded_sources_flat.squeeze(1) 
        
        # Přeuspořádání zpět na oddělené zdroje pro každý prvek v batchi
        # Tvar: [B, M, T_audio_reconstructed]
        estimated_sources = decoded_sources_squeezed.view(
            batch_size, 
            self.num_estimated_sources, 
            -1 # Délka se odvodí automaticky
        )

        if self.use_detection_branch:
            scaling_factors = detection_logits.unsqueeze(-1)
            scaled_sources = estimated_sources * scaling_factors
            return scaled_sources
        
        else:
            return estimated_sources


if __name__ == "__main__":
    from torchinfo import summary

    config = {
        "input_mixtures": 2,
        "target_sources": 4,
        "encoder_channels": 512,
        "window_size_ms": 2,
        "TDCN_Separation_Block": {
            "initial_bottleneck_channels": 128,
            "skip_connections_channels": 128,
            "number_of_repeats": 3,
            "number_of_blocks_per_repeat": 8,
            "add_skip_residual_connections": True,
            "OneDConv_Block": {
                "internal_channels": 512,
                "kernel_size": 3,
                "scale_init": 0.9
            }
        },
        "optimizer_configuration": {
            "optimizer_name": "Adam",
            "learning_rate": 0.001
        },
        "sample_rate": 16000
        }

    TDCN_Separation_Block_config = config.get('TDCN_Separation_Block')
    model = TDCNPP_Separation_Block(
            in_channels=config.get('encoder_channels'), # Vstupní kanály z enkodéru
            target_sources=config.get('target_sources'), # Počet interně odhadovaných zdrojů M
            internal_latent_channels=TDCN_Separation_Block_config.get('initial_bottleneck_channels'),
            number_of_repeats=TDCN_Separation_Block_config.get('number_of_repeats'),
            number_of_blocks=TDCN_Separation_Block_config.get('number_of_blocks_per_repeat'),
            add_skip_residual_connections=False,
            skip_connections_channels=TDCN_Separation_Block_config.get('skip_connections_channels'),
            OneDConv_Block_channels=TDCN_Separation_Block_config.get('OneDConv_Block').get('internal_channels'),
            OneDConv_Block_kernel_size=TDCN_Separation_Block_config.get('OneDConv_Block').get('kernel_size'),
            OneDConv_Block_scale_init=TDCN_Separation_Block_config.get('OneDConv_Block').get('scale_init'),
            OneDConv_Block_index_offset=0
        )
    
    summary(model, input_size=(4, config.get('encoder_channels'), 5999))