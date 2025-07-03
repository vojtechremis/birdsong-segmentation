import torch
import torch.nn as nn
import itertools

class FeatureWiseLayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, elementwise_affine=True, time_dim=2):
        """
        Feature-wise LayerNorm: normalizes each channel independently over the time dimension.
        Expects input shape [B, C, T] if time_dim=2, [B, T, C] if time_dim=1.

        Args:
            num_features (int): Number of features in the input tensor (C).
            eps (float, optional): A small constant which avoids division by zero.
            affine (bool, optional): If True, use learnable parameters.
        """
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.time_dim = time_dim

        if self.time_dim not in [1, 2]:
            raise ValueError(f"Parameter 'time_dim' must be 1 or 2, got {self.time_dim}!")

        if self.elementwise_affine:
            self.beta_vector = nn.Parameter(torch.zeros(num_features)) # shape: [C]
            self.gamma_vector = nn.Parameter(torch.ones(num_features)) # shape: [C]

    def forward(self, x):
        mean = x.mean(dim=self.time_dim, keepdim=True)
        std = x.std(dim=self.time_dim, keepdim=True, unbiased=False)
        x_norm = (x - mean) / (std + self.eps)

        if self.elementwise_affine:
            if self.time_dim == 2:  # x is [B, C, T]
                shape = (1, -1, 1)  # broadcast to [B, C, T]
            else:  # time_dim == 1, x is [B, T, C]
                shape = (1, 1, -1)  # broadcast to [B, T, C]

            gamma = self.gamma_vector.view(*shape)
            beta = self.beta_vector.view(*shape)
            x_norm = gamma * x_norm + beta

        return x_norm


class OneDConv_Block(nn.Module):
    """
    1DConv_Block designed in Conv-TasNet paper and modified in UNIVERSAL SOUND SEPARATION paper.
    This block contains dilated convolutions, prelu, layernorm and skip connection.
    """
    def __init__(self, in_channels, skip_channels, dilation, kernel_size, internal_channels, scale: bool = True, scale_init: float = 0.9): # scale_init = \alpha^L
        super().__init__()
        self.conv1x1_1 = nn.Conv1d(in_channels, internal_channels, 1, bias=True)
        self.prelu1 = nn.PReLU()
        self.fw_layernorm1 = FeatureWiseLayerNorm(internal_channels) # Feature-wise LayerNorm
        self.d_conv = nn.Conv1d(internal_channels, internal_channels, kernel_size, dilation=dilation, padding=(kernel_size-1) * dilation // 2, groups=internal_channels)
        self.prelu2 = nn.PReLU()
        self.fw_layernorm2 = FeatureWiseLayerNorm(internal_channels) # Feature-wise LayerNorm
        self.conv1x1_skip = nn.Conv1d(internal_channels, skip_channels, 1, bias=True)
        self.conv1x1_output = nn.Conv1d(internal_channels, in_channels, 1, bias=True)

        self.scale = scale
        if self.scale:
            self.scale1 = nn.Parameter(torch.ones(1) * scale_init)
            self.scale2 = nn.Parameter(torch.ones(1) * scale_init)

    def forward(self, x):
        x = self.conv1x1_1(x)
        if self.scale:
            x  = x * self.scale1

        x = self.prelu1(x)
        x = self.fw_layernorm1(x)
        x = self.d_conv(x)
        x = self.prelu2(x)
        x = self.fw_layernorm2(x)
        output = self.conv1x1_output(x)

        if self.scale:
            output = output * self.scale2

        skip_connection_out = self.conv1x1_skip(x)

        return output, skip_connection_out


def compute_dilation_base2(block_number):
    return 2**block_number


class TDCNPP_Repeat_Block(nn.Module):
    """
    TDCN++ Repeat Block designed in UNIVERSAL SOUND SEPARATION paper.
    This block contains set of 1DConv_Blocks.
    """
    def __init__(self, in_channels, number_of_blocks, skip_channels, dilation_indexer = compute_dilation_base2, kernel_size = 4, internal_latent_channels = 128, scale_init = 0.9, OneDConv_Block_index_offset = 0):
        super().__init__()

        self.in_channels = in_channels
        self.number_of_blocks = number_of_blocks
        self.skip_channels = skip_channels
        self.dilation_indexer = dilation_indexer
        self.kernel_size = kernel_size

        self.dilations = [dilation_indexer(i) for i in range(number_of_blocks)]

        # Create #number_of_blocks blocks
        self.blocks = nn.ModuleList([
            OneDConv_Block(
                in_channels, skip_channels,
                dilation, kernel_size,
                internal_latent_channels,
                scale=True,
                scale_init=scale_init**(i + OneDConv_Block_index_offset)
            ) for i, dilation in enumerate(self.dilations)
        ])

    def forward(self, x):
        skip_connections = []
        for block in self.blocks:
            x, x_skip = block(x)
            skip_connections.append(x_skip)

        # Sum all skip connections
        skip_connections_sum = sum(skip_connections)

        return x, skip_connections_sum
    
    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"blocks={len(self.blocks)}, "
            f"dilations={self.dilations}, "
            f"kernel_size={self.kernel_size})"
        )


class TDCNPP_Separation_Block(nn.Module):
    """
    TDCN++ block designed in UNIVERSAL SOUND SEPARATION paper.
    This block mainly contains set of 1DConv_Blocks.

    Pozn.: V paperu USS jsou přidány skip connections mezi repeat blocks.
    "We add longer-range skip-residual connections from earlier
    repeat inputs to later repeat inputs after passing them through dense layers. This presumably helps with gradient flow from layer
    to layer during training."
    """
    def __init__(
            self,
            in_channels, # Počet kanálů na vstupu do Separation_Block (z enkodéru, např. C_encoder)
            target_sources, # Počet zdrojů M, které má separation blok interně odhadnout (např. 4)
            internal_latent_channels,
            number_of_repeats,
            number_of_blocks,
            add_skip_residual_connections,
            skip_connections_channels, # Počet kanálů pro skip connections sčítané z TDCNPP_Repeat_Block
            OneDConv_Block_channels,
            OneDConv_Block_kernel_size,
            OneDConv_Block_scale_init,
            OneDConv_Block_index_offset: int = 0,
            use_detection_branch: bool = False,
            detection_branch_pooling: str = 'avg'
        ):
        super().__init__()

        self.in_channels = in_channels # Ukládáme pro použití v out_channels conv1x1_2 (C_encoder)
        self.target_sources = target_sources # Ukládáme pro použití v out_channels conv1x1_2 (M)
        self.skip_connections_channels = skip_connections_channels # Ukládáme pro in_channels conv1x1_2

        self.number_of_repeats = number_of_repeats
        self.number_of_blocks = number_of_blocks
        self.use_detection_branch = use_detection_branch
        self.detection_branch_pooling = detection_branch_pooling

        # Pre-layer
        # self.fw_layernorm1 = FeatureWiseLayerNorm(in_channels)
        self.relu = nn.ReLU()
        self.conv1x1_1 = nn.Conv1d(in_channels, internal_latent_channels, 1, bias=True)

        
        # Create #number_of_repeats repeat blocks
        self.from_block = [] 
        self.to_block = []
        if add_skip_residual_connections:
            for k in range(number_of_repeats):
                for j in range(k):
                    self.from_block.append(j)
                    self.to_block.append(k)
        
        self.repeat_blocks = nn.ModuleList([
            TDCNPP_Repeat_Block(
                internal_latent_channels, # Vstup do RepeatBlock
                number_of_blocks,
                skip_connections_channels, # Výstup skip_connection_sum z RepeatBlock
                kernel_size=OneDConv_Block_kernel_size,
                internal_latent_channels=OneDConv_Block_channels, # interní kanály v OneDConv_Block
                scale_init=OneDConv_Block_scale_init,
                OneDConv_Block_index_offset=OneDConv_Block_index_offset + i * number_of_blocks
            )
            for i in range(number_of_repeats) # Opraveno i místo _
        ])

        self.skip_connection_dense = nn.ModuleList([
            nn.Conv1d(in_channels=internal_latent_channels, out_channels=internal_latent_channels, kernel_size=1, bias=True)
            for _ in range(len(self.from_block))
        ])

        # Post-layer
        self.prelu = nn.PReLU()
        # self.in_channels zde odkazuje na původní vstupní kanály do Separation bloku (C_encoder)
        # self.target_sources je M (počet odhadovaných zdrojů)
        # self.skip_connections_channels jsou kanály ze sumy skipů z Repeat bloků
        self.conv1x1_2 = nn.Conv1d(
            in_channels=self.skip_connections_channels,
            out_channels=self.in_channels * self.target_sources, # Např. C_encoder * M
            kernel_size=1,
            bias=True
        )
        self.sigmoid = nn.Sigmoid()

        if self.use_detection_branch:
            if detection_branch_pooling not in ['avg', 'max']:
                raise ValueError(f"Invalid pooling type: {detection_branch_pooling}. Must be 'avg' or 'max'.")
            
            self.detection_conv1x1 = nn.Conv1d(self.in_channels, 1, kernel_size=1, bias=True) # Shape [B, C_Encoder, M] -> [B, 1, M]
            self.detection_softmax = nn.Softmax(dim=1) 

    def detection_branch(self, x):
        print(f"x shape: {x.shape}")
        if self.detection_branch_pooling == 'avg':  # Shape [B, C_encoder, M, T_latent] -> [B, C_Encoder, M]
                detection_x = x.mean(dim=-1, keepdim=False)
        elif self.detection_branch_pooling  == 'max':
                detection_x = x.max(dim=-1, keepdim=False)[0]
        else:
            raise ValueError(f"Invalid pooling type: {self.detection_branch_pooling}. Must be 'avg' or 'max'.")
        

        print(f"detection_x shape: {detection_x.shape}")
        
        detection_x = self.detection_conv1x1(detection_x) # Shape [B, C_Encoder, M] -> [B, 1, M]
        detection_x = detection_x.squeeze(dim=1) # Shape [B, 1, M] -> [B, M]
        detection_logits = self.detection_softmax(detection_x)

        return detection_logits

    def forward(self, x):
        batch_size, c_encoder, time_steps = x.shape # c_encoder je self.in_channels konstruktoru

        # Pre-layer
        # x_processed = self.fw_layernorm1(x)
        x_processed = self.relu(x)
        x_processed = self.conv1x1_1(x_processed) # Nyní tvar [B, internal_latent_channels, T_latent]

        # Main layer - Repeat blocks
        repeat_inputs = []
        
        # Inicializace pro sčítání skip connections z TDCNPP_Repeat_Block
        # Musí mít self.skip_connections_channels
        sum_of_all_skip_connections = torch.zeros(batch_size, self.skip_connections_channels, time_steps, device=x_processed.device)
        
        current_x = x_processed # Pracovní proměnná pro hlavní větev
        
        for block_idx, repeat_block in enumerate(self.repeat_blocks):
            repeat_inputs.append(current_x) # Uložíme vstup do aktuálního repeat bloku

            # Získáme indexy 'from' pro skip connections směřující do tohoto bloku
            # (V kódu je tato logika, jen ji zde pro přehlednost nezobrazuji celou)
            from_repeat_idxs = [i for i in range(len(self.to_block)) if self.to_block[i] == block_idx]

            if from_repeat_idxs:
                residual_sum = sum([
                    self.skip_connection_dense[i](repeat_inputs[self.from_block[i]])
                    for i in from_repeat_idxs
                ])
                current_x = current_x + residual_sum # Aplikujeme skip na vstup do repeat_block
            
            # Výstup z repeat_block: x_main_branch má internal_latent_channels, skip_sum má skip_connections_channels
            x_main_branch, skip_sum_from_repeat = repeat_block(current_x)
            
            sum_of_all_skip_connections = sum_of_all_skip_connections + skip_sum_from_repeat # Sčítáme skipy
            
            current_x = x_main_branch # Výstup hlavní větve se stane vstupem pro další iteraci (nebo finálním x_main_branch)
            
        # Post-layer: Použijeme sum_of_all_skip_connections
        masks_input = self.prelu(sum_of_all_skip_connections)
        masks_raw = self.conv1x1_2(masks_input) # conv1x1_2 má in_channels=self.skip_connections_channels

        # Očekávaný tvar masks_raw: [B, C_encoder * M, T_latent]
        # self.in_channels je C_encoder, self.target_sources je M
        output_masks = masks_raw.view(batch_size, self.in_channels, self.target_sources, time_steps)

        print(f"output_masks shape: {output_masks.shape}")
        print(batch_size, self.in_channels, self.target_sources, time_steps)

        # Detection layer --optional
        if self.use_detection_branch:
            detection_logits = self.detection_branch(output_masks)

            # if self.detection_branch_pooling == 'avg':  # Shape [B, C_encoder, M, T_latent] -> [B, C_Encoder, M]
            #     detection_x = output_masks.mean(dim=-1, keepdim=False)
            # elif self.detection_branch_pooling  == 'max':
            #     detection_x = output_masks.max(dim=-1, keepdim=False)[0]
            # else:
            #     raise ValueError(f"Invalid pooling type: {self.detection_branch_pooling}. Must be 'avg' or 'max'.")
            

            # print(f"detection_x shape: {detection_x.shape}")
            
            # detection_x = self.detection_conv1x1(detection_x) # Shape [B, C_Encoder, M] -> [B, 1, M]
            # detection_x = detection_x.squeeze(dim=1) # Shape [B, 1, M] -> [B, M]
            # detection_logits = self.detection_softmax(detection_x)

        output_masks = self.sigmoid(output_masks) # Tvar [B, C_encoder, M, T_latent]

        if self.use_detection_branch:
            return output_masks, detection_logits
        else:
            return output_masks
    
    def __repr__(self):
        return (
            f"{self.__class__.__name__}(repeats={self.number_of_repeats}, "
            f"blocks_per_repeat={self.number_of_blocks}, "
            f"skip_conns={len(self.from_block)})"
            f"use_detection_layer={self.use_detection_branch})"
        )
    

def SNR_Loss(input_signal: torch.Tensor, estimated_signal: torch.Tensor, tau: float) -> torch.Tensor:
    """
    Signal-to-noise ratio loss for single signal (1D tensor)
    """
    input_energy = torch.sum(input_signal ** 2)
    noise_energy = torch.sum((input_signal - estimated_signal) ** 2)
    return -10 * torch.log10(input_energy / (noise_energy + tau * input_energy))

# MixIT loss
class MixIT_Loss(nn.Module):
    def __init__(self, loss_fn=SNR_Loss, tau: float = 1e-6):
        """
        loss_fn: funkce, která bere (pred, target) a vrací ztrátu tvaru [B]
        """
        super().__init__()
        self.loss_fn = loss_fn
        self.tau = tau

    def forward(self, estimated_sources: torch.Tensor, input_mixtures: torch.Tensor) -> torch.Tensor:
        """
        estimated_sources: [B, M, T]  — M input sources
        input_mixtures:    [B, 2, T]  — mixtures x1 and x2

        Return value: [B] — loss for each sample in batch
        """
        B, M, T = estimated_sources.shape
        _, J, _ = input_mixtures.shape
        assert J == 2, f"Second dimension of 'input_mixtures' must be equal to {2}. Shape of 'input_mixtures': {input_mixtures.shape}."

        # Generate all possible assignment matrices
        A_system = self.matrix_system_A(M)

        losses = torch.full((B,), float('inf'), device=estimated_sources.device)

        for batch_idx in range(B):

            s_est = estimated_sources[batch_idx]
            input_mixture = input_mixtures[batch_idx] # [2, T]

            for A in A_system:
                A = A.to(dtype=s_est.dtype, device=s_est.device)
                loss_ = torch.zeros((1), device=estimated_sources.device)
                for i in range(2):
                    loss_ += self.loss_fn(input_mixture[i], (A@s_est)[i], self.tau)

                losses[batch_idx] = torch.minimum(losses[batch_idx], loss_)

        return losses.mean()

    def matrix_system_A(self, M: int, allow_empty_groups=True):
        """
        Returns list of all binary matrices A ∈ {0,1}^{2xM}, where each column has one one.
        """
        A_system = []
        for assign in itertools.product([0, 1], repeat=M):  # 2^M possible assignments
            if allow_empty_groups and (sum(assign) == 0 or sum(assign) == M):
                continue
            A = torch.zeros(2, M, dtype=torch.int)
            for i, val in enumerate(assign):
                A[val, i] = 1
            A_system.append(A)

        return A_system