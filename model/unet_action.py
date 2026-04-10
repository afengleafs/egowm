"""Action-conditioned SVD UNet:仅以 (per-frame) 动作序列作为额外条件。"""
from typing import Optional, Tuple, Union

import torch
from diffusers.configuration_utils import register_to_config
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.unets.unet_spatio_temporal_condition import (
    UNetSpatioTemporalConditionModel,
)
from einops import rearrange

from .common import UNetSpatioTemporalConditionOutput, spatiotemporal_res_forward2


class DebugActionUnetFwise2(UNetSpatioTemporalConditionModel):
    """SVD UNet + action embedding。

    - 新增 action_proj / add_action_embedding 把动作映射到 time embedding 空间
    - 用 spatiotemporal_res_forward2 替换所有 resnet 的 forward,使 resblock
      能接收 [time token + F 个 per-frame action token] 的拼接 embedding
    """

    @register_to_config
    def __init__(
        self,
        sample_size: Optional[int] = None,
        in_channels: int = 8,
        out_channels: int = 4,
        down_block_types: Tuple[str] = (
            "CrossAttnDownBlockSpatioTemporal",
            "CrossAttnDownBlockSpatioTemporal",
            "CrossAttnDownBlockSpatioTemporal",
            "DownBlockSpatioTemporal",
        ),
        up_block_types: Tuple[str] = (
            "UpBlockSpatioTemporal",
            "CrossAttnUpBlockSpatioTemporal",
            "CrossAttnUpBlockSpatioTemporal",
            "CrossAttnUpBlockSpatioTemporal",
        ),
        block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
        addition_time_embed_dim: int = 256,
        projection_class_embeddings_input_dim: int = 768,
        layers_per_block: Union[int, Tuple[int]] = 2,
        cross_attention_dim: Union[int, Tuple[int]] = 1024,
        transformer_layers_per_block: Union[int, Tuple[int], Tuple[Tuple]] = 1,
        num_attention_heads: Union[int, Tuple[int]] = (5, 10, 20, 20),
        num_frames: int = 25,
    ):
        super().__init__(
            sample_size,
            in_channels,
            out_channels,
            down_block_types,
            up_block_types,
            block_out_channels,
            addition_time_embed_dim,
            projection_class_embeddings_input_dim,
            layers_per_block,
            cross_attention_dim,
            transformer_layers_per_block,
            num_attention_heads,
            num_frames,
        )

        time_embed_dim = block_out_channels[0] * 4
        # 32 sinusoidal channels × 3 action dims = 96 input features
        self.action_proj = Timesteps(32, True, downscale_freq_shift=0)
        self.add_action_embedding = TimestepEmbedding(96, time_embed_dim)

        self._patch_resnet_forwards()

    def _patch_resnet_forwards(self):
        for ii in range(len(self.down_blocks)):
            for jj in range(len(self.down_blocks[ii].resnets)):
                blk = self.down_blocks[ii].resnets[jj]
                if hasattr(blk, "temporal_res_block"):
                    blk.forward = spatiotemporal_res_forward2(blk)

        for ii in range(len(self.up_blocks)):
            for jj in range(len(self.up_blocks[ii].resnets)):
                blk = self.up_blocks[ii].resnets[jj]
                if hasattr(blk, "temporal_res_block"):
                    blk.forward = spatiotemporal_res_forward2(blk)

        for jj in range(len(self.mid_block.resnets)):
            blk = self.mid_block.resnets[jj]
            if hasattr(blk, "temporal_res_block"):
                blk.forward = spatiotemporal_res_forward2(blk)

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        added_time_ids: torch.Tensor,
        actions: torch.Tensor,
        return_dict: bool = True,
    ) -> Union[UNetSpatioTemporalConditionOutput, Tuple]:
        default_overall_up_factor = 2 ** self.num_upsamplers
        forward_upsample_size = any(
            s % default_overall_up_factor != 0 for s in sample.shape[-2:]
        )
        upsample_size = None

        # ---- 1. time embedding ----
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        batch_size, num_frames = sample.shape[:2]
        timesteps = timesteps.expand(batch_size)

        t_emb = self.time_proj(timesteps).to(dtype=sample.dtype)
        emb = self.time_embedding(t_emb)

        time_embeds = self.add_time_proj(added_time_ids.flatten())
        time_embeds = time_embeds.reshape((batch_size, -1)).to(emb.dtype)
        emb = emb + self.add_embedding(time_embeds)

        # ---- 2. action embedding (per-frame, concat 策略) ----
        b, nf, a = actions.shape
        actions_flat = rearrange(actions, "b f a -> (b f a)")
        action_embeds = self.action_proj(actions_flat)
        action_embeds = rearrange(action_embeds, "(b f a) c -> b f (a c)", b=b, f=nf)
        action_embeds = action_embeds.to(emb.dtype)
        aug_action_emb = self.add_action_embedding(action_embeds)  # [B, F, time_embed_dim]

        # ---- 3. 拼装 all_emb = [time token, action tokens] ----
        sample = sample.flatten(0, 1)
        emb = emb.unsqueeze(1)
        all_emb = torch.cat([emb, aug_action_emb], dim=1)  # [B, 1+F, D]

        encoder_hidden_states = encoder_hidden_states.repeat_interleave(num_frames, dim=0)

        # ---- 4. UNet ----
        sample = self.conv_in(sample)
        image_only_indicator = torch.zeros(
            batch_size, num_frames, dtype=sample.dtype, device=sample.device
        )

        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if getattr(downsample_block, "has_cross_attention", False):
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=all_emb,
                    encoder_hidden_states=encoder_hidden_states,
                    image_only_indicator=image_only_indicator,
                )
            else:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=all_emb,
                    image_only_indicator=image_only_indicator,
                )
            down_block_res_samples += res_samples

        sample = self.mid_block(
            hidden_states=sample,
            temb=all_emb,
            encoder_hidden_states=encoder_hidden_states,
            image_only_indicator=image_only_indicator,
        )

        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets):]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if getattr(upsample_block, "has_cross_attention", False):
                sample = upsample_block(
                    hidden_states=sample,
                    temb=all_emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    upsample_size=upsample_size,
                    image_only_indicator=image_only_indicator,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=all_emb,
                    res_hidden_states_tuple=res_samples,
                    upsample_size=upsample_size,
                    image_only_indicator=image_only_indicator,
                )

        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        sample = sample.reshape(batch_size, num_frames, *sample.shape[1:])
        return sample
