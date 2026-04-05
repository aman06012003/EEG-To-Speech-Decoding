import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
from functools import partial

from s4_block.s4_model import S4Model
from models import SubjectDiscriminator

try:
    from peft import LoraConfig, get_peft_model
except ImportError:
    LoraConfig, get_peft_model = None, None

# Import LaBraM modeling (assuming LaBraM is in the same directory or PYTHONPATH)
# Based on debug_labram.py, we need to ensure the path is correct
LABRAM_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'LaBraM')
if LABRAM_PATH not in sys.path:
    sys.path.append(LABRAM_PATH)

try:
    from LaBraM.modeling_finetune import NeuralTransformer
except ImportError:
    # Fallback for different directory structures
    try:
        from modeling_finetune import NeuralTransformer
    except ImportError:
        NeuralTransformer = None

class block_conv(nn.Module):
    def __init__(self, in_channel: int=512, out_channel: int=512, 
                 kernel_size: int=3, stride: int=1, padding: int=1, output_padding: int=1,
                 norm=nn.GroupNorm, dropout: float=0.3, residual: bool=False,
                 act_layer = nn.GELU, conv: str='conv'):
        '''A single convolution block with normaliation and dropout'''
        super().__init__()

        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.padding = padding
        self.output_padding = output_padding
        self.stride = stride
        self.dropout = dropout

        if residual is not False:
            raise NotImplementedError

        if conv == 'conv':
            self.conv_layer = nn.Conv1d(in_channels=self.in_channel,
                                        out_channels=self.out_channel,
                                        kernel_size=self.kernel_size,
                                        stride=self.stride, padding=self.padding)
        else:
            self.conv_layer = nn.ConvTranspose1d(in_channels=self.in_channel,
                                        out_channels=self.out_channel,
                                        kernel_size=self.kernel_size,
                                        stride=self.stride, padding=self.padding,
                                        output_padding=self.output_padding)
        
        self.norm_layer = norm(1, out_channel)

        self.dropout_layer = nn.Dropout1d(self.dropout)

        if act_layer is not None:
            self.act_layer = act_layer()
        else:
            self.act_layer = None

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.dropout_layer(x)
        x = self.norm_layer(x)

        if self.act_layer is not None:
            x = self.act_layer(x)
        return x

class block_last_decoder(block_conv):
    def __init__(self, in_channel: int = 512, out_channel: int = 512, kernel_size: int = 3, stride: int = 1, padding: int = 1, output_padding: int = 1, norm=nn.GroupNorm, dropout: float = 0.3, residual: bool = False, act_layer=nn.GELU, conv: str = 'conv'):
        super().__init__(in_channel, out_channel, kernel_size, stride, padding, output_padding, norm, dropout, residual, act_layer, conv)
    
    def forward(self, x):
        x = self.conv_layer(x)
        return x
    
class conv_encoder_tueg(nn.Module):
    def __init__(self, n_layers: int=4, input_channels: int=19, embedding_size: int=512, 
                 kernel_size: int=3, stride: int=2, padding: int=1, output_padding: int=1,
                 norm=nn.GroupNorm, dropout: float=0.3, residual: bool=False,
                 act_layer = nn.GELU):
        super().__init__()

        self.n_layers = n_layers

        self.block_layers = nn.ModuleList()

        self.block_layers.append(block_conv(input_channels, embedding_size,
                                            kernel_size, stride, padding, output_padding,
                                            norm, dropout, residual, act_layer))
        
        for i in range(n_layers - 1):
            self.block_layers.append(block_conv(embedding_size, embedding_size,
                                            kernel_size, stride, padding, output_padding,
                                            norm, dropout, residual, act_layer))

        
    def forward(self, x):
        for i, block_layer in enumerate(self.block_layers):
            x = block_layer(x)
        return x
    
class conv_decoder_tueg(nn.Module):
    def __init__(self, n_layers: int=4, input_channels: int=19, embedding_size: int=512, 
                 kernel_size: int=3, stride: int=2, padding: int=1, output_padding: int=1,
                 norm=nn.GroupNorm, dropout: float=0.3, residual: bool=False,
                 act_layer = nn.GELU, is_last_layer=False):
        super().__init__()

        self.n_layers = n_layers

        self.block_layers = nn.ModuleList()
        
        for i in range(n_layers - 1 * is_last_layer):
            self.block_layers.append(block_conv(embedding_size, embedding_size,
                                            kernel_size, stride, padding, output_padding,
                                            norm, dropout, residual, act_layer, 'deconv'))
        
        if is_last_layer:
            self.block_layers.append(block_last_decoder(embedding_size, input_channels,
                                            kernel_size, stride, padding, output_padding,
                                            norm, dropout, residual, None, 'deconv'))
        
    def forward(self, x):
        for i, block_layer in enumerate(self.block_layers):
            x = block_layer(x)
        return x



class EEGModule(nn.Module):
    def __init__(self, n_layers_cnn: int=6,
                 use_s4=False,
                n_layers_s4: int=8,
                 device: str='cuda', embedding_size: int=512,
                 is_mask: bool=True, in_channels: int=19,
                 num_subjects: int=25,
                 use_labram: bool=False,
                 labram_checkpoint: str=None,
                 use_dora: bool=False,
                 lora_rank: int=8,
                 lora_alpha: int=16):
        super().__init__()

        self.n_layers_cnn = n_layers_cnn
        self.use_s4 = use_s4
        self.n_layers_s4 = n_layers_s4
        self.device = device
        self.is_mask = is_mask
        self.use_labram = use_labram
        self.in_channels = in_channels
        self.embedding_size = embedding_size
        self.use_dora = use_dora
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha

        if self.use_labram:
            print(f"Initializing EEGModule with LaBraM backbone...")
            assert NeuralTransformer is not None, "NeuralTransformer could not be imported from LaBraM. Check paths."
            
            # LaBraM base config
            self.labram_backbone = NeuralTransformer(
                patch_size=200,
                embed_dim=200,
                depth=12,
                num_heads=10,
                mlp_ratio=4,
                qkv_bias=False,
                qk_norm=partial(nn.LayerNorm, eps=1e-6),
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                init_values=0.1,
                num_classes=0
            )
            
            if labram_checkpoint and os.path.exists(labram_checkpoint):
                self._load_labram_checkpoint(labram_checkpoint)
            elif labram_checkpoint:
                print(f"WARNING: LaBraM checkpoint not found at {labram_checkpoint}")

            if self.use_dora:
                if get_peft_model is None:
                    print("WARNING: peft not installed. Proceeding with full fine-tuning.")
                else:
                    print(f"Applying DoRA to LaBraM (rank={self.lora_rank}, alpha={self.lora_alpha})...")
                    peft_config = LoraConfig(
                        r=self.lora_rank,
                        lora_alpha=self.lora_alpha,
                        target_modules=["qkv", "fc1", "fc2"], # Standard for NeuralTransformer
                        lora_dropout=0.05,
                        bias="none",
                        task_type=None,
                        use_dora=True
                    )
                    self.labram_backbone = get_peft_model(self.labram_backbone, peft_config)
                    self.labram_backbone.print_trainable_parameters()

            # Projection from LaBraM (200) to system embedding_size (e.g. 192 or 512)
            self.proj_labram = nn.Linear(200, embedding_size)
            
            # Fixed channel mapping for N400 BioSemi (A1-D32 etc -> 1-128)
            # We map CLS to 0, and then up to 128 channels sequentially
            self.register_buffer('input_chans', torch.arange(min(in_channels, 128) + 1))
            
        # Shared EEG Decoder for reconstruction loss
        self.deconv_encoder = conv_decoder_tueg(n_layers_cnn, stride=1, padding=3, kernel_size=4, output_padding=0, embedding_size=embedding_size, input_channels=in_channels, is_last_layer=True)
        self.deconv_encoder2 = conv_decoder_tueg(1, stride=3, padding=2, kernel_size=4, output_padding=2, embedding_size=embedding_size, input_channels=embedding_size, is_last_layer=False)

        if self.use_labram:
            # Already initialized labram_backbone and proj_labram
            pass
        else:
            self.conv_encoder = conv_encoder_tueg(n_layers_cnn, stride=1, padding=3, kernel_size=4, embedding_size=embedding_size, input_channels=in_channels)
            self.conv_encoder2 = conv_encoder_tueg(1, stride=3, padding=2, kernel_size=4, embedding_size=embedding_size, input_channels=embedding_size)

            self.s4_model = S4Model(d_input=embedding_size,
            d_output=embedding_size,
            d_model=embedding_size,
            n_layers=n_layers_s4,
            dropout=0.3,
            prenorm=False)

        self.num_subjects = num_subjects
        self.subject_discriminator = SubjectDiscriminator(embedding_size, embedding_size, num_subjects)

    def _load_labram_checkpoint(self, path):
        print(f"Loading LaBraM weights from {path}")
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        state_dict = checkpoint["model"]
        new_state_dict = {}
        for k, v in state_dict.items():
            # Strip 'student.' prefix if present
            k_clean = k[len("student."):] if k.startswith("student.") else k

            # Map 'norm' to 'fc_norm' (transformer's final normalization layer)
            if k_clean == "norm.weight":
                new_state_dict["fc_norm.weight"] = v
            elif k_clean == "norm.bias":
                new_state_dict["fc_norm.bias"] = v
            # Skip pre-training specific heads used in Masked EEG Modeling
            elif k_clean in ("logit_scale", "lm_head.weight", "lm_head.bias",
                       "projection_head.0.weight", "projection_head.0.bias",
                       "mask_token"):
                continue
            else:
                new_state_dict[k_clean] = v
        msg = self.labram_backbone.load_state_dict(new_state_dict, strict=False)
        print(f"LaBraM loaded with msg: {msg}")

    def forward(self, x, alpha=1.0):
        if type(x) is list:
            x = torch.cat((x[0], x[1]), dim=1)
        x = x.to(self.device)
        
        batch_size, n_channels, n_time = x.shape
        mask = None

        if self.use_labram:
            PATCH_SIZE = 200
            # Truncate channels if they exceed LaBraM's 128 pos_embed limit
            x_in = x[:, :128, :] if n_channels > 128 else x
            curr_channels = x_in.shape[1]
            
            # Trim time dimension to multiple of patch_size
            n_patches = n_time // PATCH_SIZE
            if n_patches == 0:
                x_in = F.pad(x_in, (0, PATCH_SIZE - n_time))
                n_patches = 1
            else:
                x_in = x_in[:, :, :n_patches * PATCH_SIZE]
            
            # Reshape for LaBraM: [B, Ch, A, T]
            x_labram = x_in.view(batch_size, curr_channels, n_patches, PATCH_SIZE)
            input_chans = self.input_chans[:curr_channels + 1].tolist()
            
            # Encoder
            features = self.labram_backbone.forward_features(
                x_labram, 
                input_chans=input_chans, 
                return_patch_tokens=True
            )
            features = self.proj_labram(features) # [B, N*A, Hidden]
            features = features.transpose(1, 2)   # [B, Hidden, N_tokens]
            
            # Temporal Resampling to match Speech Decoder (approx 86Hz)
            target_seq_len = int(n_time * (22050 / 256) / 256)
            if features.shape[2] != target_seq_len and target_seq_len > 0:
                mid_output = F.interpolate(features, size=target_seq_len, mode='linear', align_corners=False)
            else:
                mid_output = features
        else:
            if self.is_mask:
                # Placeholder for mask logic as called in EEGModule_copy
                # For now keeping it as identity as no mask method was provided
                masked_input = x
            else: 
                masked_input = x

            conv_output = self.conv_encoder(masked_input.clone())
            conv_output = self.conv_encoder2(conv_output)

            if self.use_s4:
                mid_output = self.s4_model(conv_output.transpose(-1,-2).clone())
                mid_output = mid_output.transpose(1,2)
            else:
                mid_output = conv_output

        # --- Shared EEG Reconstruction Decoder (style of EEGModule_copy) ---
        decoder_out = self.deconv_encoder2(mid_output.clone())
        eeg_decoder_out = self.deconv_encoder(decoder_out)
        
        # Slicing for temporal alignment
        eeg_decoder_out = eeg_decoder_out[:, :, :n_time]

        # Subject Discriminator logic for Feature Disentanglement
        # Input to discriminator is the subject-agnostic latent (mid_output)
        # pooled over time
        mid_output_pooled = mid_output.mean(dim=2)
        sid_logits = self.subject_discriminator(mid_output_pooled, alpha)

        return x, mask, mid_output, eeg_decoder_out, sid_logits