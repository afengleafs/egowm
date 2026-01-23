import inspect
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import PIL.Image
import torch
import torch.nn as nn
from diffusers import StableVideoDiffusionPipeline
from diffusers.configuration_utils import register_to_config
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.unets.unet_spatio_temporal_condition import UNetSpatioTemporalConditionModel
from diffusers.utils.outputs import BaseOutput
from diffusers.utils.torch_utils import randn_tensor
from einops import rearrange


@dataclass
class UNetSpatioTemporalConditionOutput(BaseOutput):
    """
    The output of [`UNetSpatioTemporalConditionModel`].

    Args:
        sample (`torch.Tensor` of shape `(batch_size, num_frames, num_channels, height, width)`):
            The hidden states output conditioned on `encoder_hidden_states` input. Output of last layer of model.
    """

    sample: torch.Tensor = None



def spatiotemporal_res_forward2(self):
    def dbg_st_forward2(
        hidden_states: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        image_only_indicator: Optional[torch.Tensor] = None,
    ):
        num_frames = image_only_indicator.shape[-1]
        b = temb.shape[0]
        time_emb = temb[:, 0, :]
        action_emb = temb[:, 1:, :]

        time_emb = time_emb.unsqueeze(1).repeat(1, num_frames, 1)
        timeaction_emb = time_emb + action_emb
        timeaction_emb = timeaction_emb.reshape(b * num_frames, -1)

        hidden_states = self.spatial_res_block(hidden_states, timeaction_emb)

        batch_frames, channels, height, width = hidden_states.shape
        batch_size = batch_frames // num_frames

        hidden_states_mix = (
            hidden_states[None, :]
            .reshape(batch_size, num_frames, channels, height, width)
            .permute(0, 2, 1, 3, 4)
        )
        hidden_states = (
            hidden_states[None, :]
            .reshape(batch_size, num_frames, channels, height, width)
            .permute(0, 2, 1, 3, 4)
        )

        if temb is not None:
            timeaction_emb = timeaction_emb.reshape(batch_size, num_frames, -1)

        hidden_states = self.temporal_res_block(hidden_states, timeaction_emb)
        hidden_states = self.time_mixer(
            x_spatial=hidden_states_mix,
            x_temporal=hidden_states,
            image_only_indicator=image_only_indicator,
        )

        hidden_states = hidden_states.permute(0, 2, 1, 3, 4).reshape(batch_frames, channels, height, width)
        return hidden_states

    return dbg_st_forward2



class DebugActionUnetFwise2(UNetSpatioTemporalConditionModel):
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
        self.action_proj = Timesteps(32, True, downscale_freq_shift=0)
        self.add_action_embedding = TimestepEmbedding(96, time_embed_dim)

        for ii in range(len(self.down_blocks)):
            for jj in range(len(self.down_blocks[ii].resnets)):
                if hasattr(self.down_blocks[ii].resnets[jj], "temporal_res_block"):
                    self.down_blocks[ii].resnets[jj].forward = spatiotemporal_res_forward2(
                        self.down_blocks[ii].resnets[jj]
                    )

        for ii in range(len(self.up_blocks)):
            for jj in range(len(self.up_blocks[ii].resnets)):
                if hasattr(self.up_blocks[ii].resnets[jj], "temporal_res_block"):
                    self.up_blocks[ii].resnets[jj].forward = spatiotemporal_res_forward2(
                        self.up_blocks[ii].resnets[jj]
                    )

        for jj in range(len(self.mid_block.resnets)):
            if hasattr(self.mid_block.resnets[jj], "temporal_res_block"):
                self.mid_block.resnets[jj].forward = spatiotemporal_res_forward2(self.mid_block.resnets[jj])

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        added_time_ids: torch.Tensor,
        actions: torch.Tensor,
        return_dict: bool = True,
    ) -> Union[UNetSpatioTemporalConditionOutput, Tuple]:
        default_overall_up_factor = 2**self.num_upsamplers
        forward_upsample_size = False
        upsample_size = None

        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            forward_upsample_size = True

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

        t_emb = self.time_proj(timesteps)
        t_emb = t_emb.to(dtype=sample.dtype)

        emb = self.time_embedding(t_emb)

        time_embeds = self.add_time_proj(added_time_ids.flatten())
        time_embeds = time_embeds.reshape((batch_size, -1))
        time_embeds = time_embeds.to(emb.dtype)
        aug_emb = self.add_embedding(time_embeds)
        emb = emb + aug_emb

        b, nf, a = actions.shape
        actions_flat = rearrange(actions, "b f a -> (b f a)")
        action_embeds = self.action_proj(actions_flat)
        action_embeds = rearrange(action_embeds, "(b f a) c -> b f (a c)", b=b, f=nf)
        action_embeds = action_embeds.to(emb.dtype)
        aug_action_emb = self.add_action_embedding(action_embeds)

        sample = sample.flatten(0, 1)
        emb = emb.unsqueeze(1)
        all_emb = torch.cat([emb, aug_action_emb], dim=1)

        encoder_hidden_states = encoder_hidden_states.repeat_interleave(num_frames, dim=0)

        sample = self.conv_in(sample)

        image_only_indicator = torch.zeros(batch_size, num_frames, dtype=sample.dtype, device=sample.device)

        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
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

            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
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


class ActionSVDFwise(nn.Module):
    def __init__(self, unet, img_enc, vae, feat_ex, scheduler):
        super().__init__()

        self.unet = unet
        self.vae = vae
        self.image_encoder = img_enc
        self.feature_extractor = feat_ex
        self.scheduler = scheduler

    def set_untrained_vae_imgenc(self):
        for param in self.vae.parameters():
            param.requires_grad = False
        for param in self.image_encoder.parameters():
            param.requires_grad = False

    def forward(
        self,
        latent_model_input,
        image_embeddings,
        added_time_ids,
        actions,
        t,
        ):
        sample = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=image_embeddings,
                added_time_ids=added_time_ids,
                actions=actions,
                return_dict=False,
                )
        
        return sample
    

class SVD_action_wrapper(nn.Module):
    def __init__(self, vdiff):
        super().__init__()

        self.backbone = vdiff
        self.backbone.set_untrained_vae_imgenc()

    def forward(self, latent_model_input, init_frame_embeddings, add_time_ids, actions, timesteps):
        pred_latents = self.backbone(
            latent_model_input,
            init_frame_embeddings,
            add_time_ids, 
            actions,
            timesteps,
            )

        return pred_latents


@dataclass
class StableVideoDiffusionPipelineOutput(BaseOutput):
    r"""
    Output class for Stable Video Diffusion pipeline.

    Args:
        frames (`[List[List[PIL.Image.Image]]`, `np.ndarray`, `torch.Tensor`]):
            List of denoised PIL images of length `batch_size` or numpy array or torch tensor of shape `(batch_size,
            num_frames, height, width, num_channels)`.
    """

    frames: Union[List[List[PIL.Image.Image]], np.ndarray, torch.Tensor]


def _append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
    return x[(...,) + (None,) * dims_to_append]


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class DebugSVDActionPipeline(StableVideoDiffusionPipeline):
    @torch.no_grad()
    def __call__(
        self,
        image: Union[PIL.Image.Image, List[PIL.Image.Image], torch.Tensor],
        height: int = 576,
        width: int = 1024,
        num_frames: Optional[int] = None,
        num_inference_steps: int = 25,
        sigmas: Optional[List[float]] = None,
        min_guidance_scale: float = 1.0,
        max_guidance_scale: float = 3.0,
        fps: int = 7,
        motion_bucket_id: int = 127,
        noise_aug_strength: float = 0.02,
        actions: torch.Tensor = None, 
        decode_chunk_size: Optional[int] = None,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        return_dict: bool = True,
    ):
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        num_frames = num_frames if num_frames is not None else self.unet.config.num_frames
        decode_chunk_size = decode_chunk_size if decode_chunk_size is not None else num_frames

        self.check_inputs(image, height, width)

        if isinstance(image, PIL.Image.Image):
            batch_size = 1
        elif isinstance(image, list):
            batch_size = len(image)
        else:
            batch_size = image.shape[0]
        device = self._execution_device
        self._guidance_scale = max_guidance_scale

        image_embeddings = self._encode_image(image, device, num_videos_per_prompt, self.do_classifier_free_guidance)

        fps = fps - 1

        image = self.video_processor.preprocess(image, height=height, width=width).to(device)
        noise = randn_tensor(image.shape, generator=generator, device=device, dtype=image.dtype)
        image = image + noise_aug_strength * noise

        needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast
        if needs_upcasting:
            self.vae.to(dtype=torch.float32)

        image_latents = self._encode_vae_image(
            image,
            device=device,
            num_videos_per_prompt=num_videos_per_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
        )

        image_latents = image_latents.to(image_embeddings.dtype)

        if needs_upcasting:
            self.vae.to(dtype=torch.float16)

        image_latents = image_latents.unsqueeze(1).repeat(1, num_frames, 1, 1, 1)

        added_time_ids = self._get_add_time_ids(
            fps,
            motion_bucket_id,
            noise_aug_strength,
            image_embeddings.dtype,
            batch_size,
            num_videos_per_prompt,
            self.do_classifier_free_guidance,
        )
        added_time_ids = added_time_ids.to(device)

        actions = actions.to(device)
        actions = actions.repeat(batch_size * num_videos_per_prompt, 1, 1)

        if self.do_classifier_free_guidance:
            actions = torch.cat([actions, actions])

        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, None, sigmas)

        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_frames,
            num_channels_latents,
            height,
            width,
            image_embeddings.dtype,
            device,
            generator,
            latents,
        )

        guidance_scale = torch.linspace(min_guidance_scale, max_guidance_scale, num_frames).unsqueeze(0)
        guidance_scale = guidance_scale.to(device, latents.dtype)
        guidance_scale = guidance_scale.repeat(batch_size * num_videos_per_prompt, 1)
        guidance_scale = _append_dims(guidance_scale, latents.ndim)

        self._guidance_scale = guidance_scale

        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                latent_model_input = torch.cat([latent_model_input, image_latents], dim=2)

                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=image_embeddings,
                    added_time_ids=added_time_ids,
                    actions=actions,
                    return_dict=False,
                )

                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)

                latents = self.scheduler.step(noise_pred, t, latents).prev_sample

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)

                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()

        if not output_type == "latent":
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
            frames = self.decode_latents(latents, num_frames, decode_chunk_size)
            frames = self.video_processor.postprocess_video(video=frames, output_type=output_type)
        else:
            frames = latents

        self.maybe_free_model_hooks()

        if not return_dict:
            return frames

        return StableVideoDiffusionPipelineOutput(frames=frames)





class DebugActionUnetFwise2state(UNetSpatioTemporalConditionModel) :

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
        self.action_proj = Timesteps(32, True, downscale_freq_shift=0) #64 for 4 frame concat and 32 for 8 frame
        self.add_action_embedding = TimestepEmbedding(800, time_embed_dim) #768 = 64 x 25 (action space) x 4 (num frames)


        self.state_proj = Timesteps(32, True, downscale_freq_shift=0) #64 for 4 frame concat and 32 for 8 frame
        self.add_state_embedding = TimestepEmbedding(800, time_embed_dim) #768 = 64 x 25 (action space) x 4 (num frames)

        for ii in range(len(self.down_blocks)) :

            print("2Down<><>===<><>Len", len(self.down_blocks[ii].resnets))

            for jj in range(len(self.down_blocks[ii].resnets)) :


                if hasattr(self.down_blocks[ii].resnets[jj], 'temporal_res_block'):

                    print("DOWN<><>===<><>", ii, jj)

                    self.down_blocks[ii].resnets[jj].forward = spatiotemporal_res_forward2(self.down_blocks[ii].resnets[jj])
                    


        for ii in range(len(self.up_blocks)) :

            print("2UP<><>===<><>Len", len(self.up_blocks[ii].resnets))

            for jj in range(len(self.up_blocks[ii].resnets)) :

                if hasattr(self.up_blocks[ii].resnets[jj], 'temporal_res_block'):

                    print("UP<><>===<><>", ii, jj)

                    self.up_blocks[ii].resnets[jj].forward = spatiotemporal_res_forward2(self.up_blocks[ii].resnets[jj])
                    

        print("2MID<><>===<><>Len", len(self.mid_block.resnets))
        for jj in range(len(self.mid_block.resnets)) :

            if hasattr(self.mid_block.resnets[jj], 'temporal_res_block'):

                print("MID<><>===<><>", jj)

                self.mid_block.resnets[jj].forward = spatiotemporal_res_forward2(self.mid_block.resnets[jj])
                


 





    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        added_time_ids: torch.Tensor,
        actions : torch.Tensor,
        init_state : torch.Tensor,
        return_dict: bool = True,
    ) -> Union[UNetSpatioTemporalConditionOutput, Tuple]:
        r"""
        The [`UNetSpatioTemporalConditionModel`] forward method.

        Args:
            sample (`torch.Tensor`):
                The noisy input tensor with the following shape `(batch, num_frames, channel, height, width)`.
            timestep (`torch.Tensor` or `float` or `int`): The number of timesteps to denoise an input.
            encoder_hidden_states (`torch.Tensor`):
                The encoder hidden states with shape `(batch, sequence_length, cross_attention_dim)`.
            added_time_ids: (`torch.Tensor`):
                The additional time ids with shape `(batch, num_additional_ids)`. These are encoded with sinusoidal
                embeddings and added to the time embeddings.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unet_slatio_temporal.UNetSpatioTemporalConditionOutput`] instead
                of a plain tuple.
        Returns:
            [`~models.unet_slatio_temporal.UNetSpatioTemporalConditionOutput`] or `tuple`:
                If `return_dict` is True, an [`~models.unet_slatio_temporal.UNetSpatioTemporalConditionOutput`] is
                returned, otherwise a `tuple` is returned where the first element is the sample tensor.
        """
        # By default samples have to be AT least a multiple of the overall upsampling factor.
        # The overall upsampling factor is equal to 2 ** (# num of upsampling layears).
        # However, the upsampling interpolation output size can be forced to fit any upsampling size
        # on the fly if necessary.
        default_overall_up_factor = 2**self.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            #logger.info("Forward upsample size to force interpolation output size.")
            forward_upsample_size = True

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        batch_size, num_frames = sample.shape[:2]
        timesteps = timesteps.expand(batch_size)

        t_emb = self.time_proj(timesteps)
        print("------ time_embeds t_emb", t_emb.shape)

        # `Timesteps` does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=sample.dtype)

        emb = self.time_embedding(t_emb)
        print("------ emb", emb.shape)

        time_embeds = self.add_time_proj(added_time_ids.flatten())
        print("------ time_embeds added_ids", time_embeds.shape)
        time_embeds = time_embeds.reshape((batch_size, -1))
        time_embeds = time_embeds.to(emb.dtype)
        aug_emb = self.add_embedding(time_embeds)
        print("------ aug_emb", aug_emb.shape)
        emb = emb + aug_emb




        ########condition with actions
        
        #pooling logic
        '''b,nf,a =  actions.shape
        actions_flat = rearrange(actions, 'b f a -> (b f a)')
        action_embeds = self.action_proj(actions_flat)
        print("------ action_embeds added_ids", action_embeds.shape)
        action_embeds = action_embeds.reshape((b, nf, -1)).flatten(0,1)
        action_embeds = action_embeds.to(emb.dtype)
        aug_action_emb = self.add_action_embedding(action_embeds)
        aug_action_emb = rearrange(aug_action_emb, '(b f) d -> b f d', b=b, f=nf)
        aug_action_emb = aug_action_emb.sum(1)
        print("------ aug_action_emb", aug_action_emb.shape, aug_action_emb.dtype)
        emb = emb + aug_action_emb'''


        #concat logic
        b,nf,a =  actions.shape
        actions_flat = rearrange(actions, 'b f a -> (b f a)')
        action_embeds = self.action_proj(actions_flat)
        print("------ action_embeds added_ids", action_embeds.shape)
        action_embeds = rearrange(action_embeds, '(b f a) c -> b f (a c)', b=b, f=nf)
        print("------ action_embeds added_ids flatten", action_embeds.shape)  # b, f, d
        action_embeds = action_embeds.to(emb.dtype)
        aug_action_emb = self.add_action_embedding(action_embeds)
        print("------ aug_action_emb", aug_action_emb.shape, aug_action_emb.dtype)


        b,a =  init_state.shape
        state_flat = rearrange(init_state, 'b a -> (b a)')
        state_embeds = self.state_proj(state_flat)
        print("------ state_embeds added_ids", state_embeds.shape)
        state_embeds = rearrange(state_embeds, '(b a) c -> b (a c)', b=b)
        print("------ state_embeds added_ids flatten", state_embeds.shape)  # b, d
        state_embeds = state_embeds.to(emb.dtype)
        aug_state_emb = self.add_state_embedding(state_embeds)
        print("------ aug_state_emb", aug_state_emb.shape, aug_state_emb.dtype)
        emb = emb + aug_state_emb

        
        
        # Flatten the batch and frames dimensions
        # sample: [batch, frames, channels, height, width] -> [batch * frames, channels, height, width]
        sample = sample.flatten(0, 1)
        # Repeat the embeddings num_video_frames times
        # emb: [batch, channels] -> [batch * frames, channels]
        #emb = emb.repeat_interleave(num_frames, dim=0)
        emb = emb.unsqueeze(1) #[batch, channels]
        all_emb = torch.cat([emb,aug_action_emb],dim=1)

        # encoder_hidden_states: [batch, 1, channels] -> [batch * frames, 1, channels]
        encoder_hidden_states = encoder_hidden_states.repeat_interleave(num_frames, dim=0)

        # 2. pre-process
        sample = self.conv_in(sample)

        image_only_indicator = torch.zeros(batch_size, num_frames, dtype=sample.dtype, device=sample.device)

        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
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

        # 4. mid
        sample = self.mid_block(
            hidden_states=sample,
            temb=all_emb,
            encoder_hidden_states=encoder_hidden_states,
            image_only_indicator=image_only_indicator,
        )

        # 5. up
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
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

        # 6. post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        # 7. Reshape back to original shape
        sample = sample.reshape(batch_size, num_frames, *sample.shape[1:])

        #if not return_dict:
        #    return (sample,)

        return sample


class ActionSVDFwisestate(nn.Module):

    def __init__(self, unet, img_enc, vae, feat_ex, scheduler):# unet, vae, scheduler):
        super().__init__()
        

        self.unet = unet
        self.vae = vae
        self.image_encoder = img_enc
        self.feature_extractor = feat_ex
        self.scheduler = scheduler
        
        #self.set_untrained_all()
        #self.set_untrained_vae_textenc()
        

        
    

    def set_untrained_vae_imgenc(self):
        
        for param in self.vae.parameters():
            param.requires_grad = False
        for param in self.image_encoder.parameters():
            param.requires_grad = False

    


    def forward(self,
        latent_model_input,
        image_embeddings,
        added_time_ids,
        actions,
        init_state,
        t,
        ):
       

        sample = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=image_embeddings,
                added_time_ids=added_time_ids,
                actions=actions,
                init_state=init_state,
                return_dict=False,
                )
        
        
        return sample









class SVD_actionstate_wrapper(nn.Module) :

    def __init__(self, vdiff) :
        super().__init__()


        self.backbone = vdiff
        self.backbone.set_untrained_vae_imgenc()



    def forward(self, latent_model_input, init_frame_embeddings, add_time_ids, actions, init_state, timesteps) :


        pred_latents = self.backbone(
            latent_model_input,
            init_frame_embeddings,
            add_time_ids, 
            actions,
            init_state,
            timesteps,
            )


        return pred_latents


def build_actionfwise2_svd_model(sd_id="pretrained/stable-video-diffusion-img2vid-xt"):
    unet = DebugActionUnetFwise2.from_pretrained(sd_id, subfolder="unet", low_cpu_mem_usage=False)
    pipe = DebugSVDActionPipeline.from_pretrained(sd_id)

    vdiff = ActionSVDFwise(unet, pipe.image_encoder, pipe.vae, pipe.feature_extractor, pipe.scheduler)
    return vdiff






def build_actionfwise2state_svd_model(sd_id='/mnt/fsx/Anurag/SVD/im2vid/stable-video-diffusion-img2vid-xt'):

    print(sd_id)
    unet = DebugActionUnetFwise2state.from_pretrained(sd_id, subfolder="unet", low_cpu_mem_usage=False)
    print(".....")
    pipe = DebugSVDActionPipeline.from_pretrained(sd_id)
    

    vdiff = ActionSVDFwisestate(unet, pipe.image_encoder, pipe.vae, pipe.feature_extractor, pipe.scheduler)
    return vdiff

class DebugSVDActionStateConstcfgPipeline(StableVideoDiffusionPipeline):

    @torch.no_grad()
    def __call__(
        self,
        image: Union[PIL.Image.Image, List[PIL.Image.Image], torch.Tensor],
        height: int = 576,
        width: int = 1024,
        num_frames: Optional[int] = None,
        num_inference_steps: int = 25,
        sigmas: Optional[List[float]] = None,
        min_guidance_scale: float = 1.0,
        max_guidance_scale: float = 3.0,
        fps: int = 7,
        motion_bucket_id: int = 127,
        noise_aug_strength: float = 0.02,
        actions: torch.Tensor = None, 
        init_state: torch.Tensor = None, 
        decode_chunk_size: Optional[int] = None,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        return_dict: bool = True,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            image (`PIL.Image.Image` or `List[PIL.Image.Image]` or `torch.Tensor`):
                Image(s) to guide image generation. If you provide a tensor, the expected value range is between `[0,
                1]`.
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_frames (`int`, *optional*):
                The number of video frames to generate. Defaults to `self.unet.config.num_frames` (14 for
                `stable-video-diffusion-img2vid` and to 25 for `stable-video-diffusion-img2vid-xt`).
            num_inference_steps (`int`, *optional*, defaults to 25):
                The number of denoising steps. More denoising steps usually lead to a higher quality video at the
                expense of slower inference. This parameter is modulated by `strength`.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            min_guidance_scale (`float`, *optional*, defaults to 1.0):
                The minimum guidance scale. Used for the classifier free guidance with first frame.
            max_guidance_scale (`float`, *optional*, defaults to 3.0):
                The maximum guidance scale. Used for the classifier free guidance with last frame.
            fps (`int`, *optional*, defaults to 7):
                Frames per second. The rate at which the generated images shall be exported to a video after
                generation. Note that Stable Diffusion Video's UNet was micro-conditioned on fps-1 during training.
            motion_bucket_id (`int`, *optional*, defaults to 127):
                Used for conditioning the amount of motion for the generation. The higher the number the more motion
                will be in the video.
            noise_aug_strength (`float`, *optional*, defaults to 0.02):
                The amount of noise added to the init image, the higher it is the less the video will look like the
                init image. Increase it for more motion.
            decode_chunk_size (`int`, *optional*):
                The number of frames to decode at a time. Higher chunk size leads to better temporal consistency at the
                expense of more memory usage. By default, the decoder decodes all frames at once for maximal quality.
                For lower memory usage, reduce `decode_chunk_size`.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of videos to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for video
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `pil`, `np` or `pt`.
            callback_on_step_end (`Callable`, *optional*):
                A function that is called at the end of each denoising step during inference. The function is called
                with the following arguments:
                    `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`.
                `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableVideoDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableVideoDiffusionPipelineOutput`] is
                returned, otherwise a `tuple` of (`List[List[PIL.Image.Image]]` or `np.ndarray` or `torch.Tensor`) is
                returned.
        """
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        

        num_frames = num_frames if num_frames is not None else self.unet.config.num_frames
        decode_chunk_size = decode_chunk_size if decode_chunk_size is not None else num_frames


        print("----", height, width, num_frames)

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(image, height, width)

        # 2. Define call parameters
        if isinstance(image, PIL.Image.Image):
            batch_size = 1
        elif isinstance(image, list):
            batch_size = len(image)
        else:
            batch_size = image.shape[0]
        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        self._guidance_scale = max_guidance_scale

        # 3. Encode input image
        image_embeddings = self._encode_image(image, device, num_videos_per_prompt, self.do_classifier_free_guidance)
        print("---- im emb", image_embeddings.shape)

        # NOTE: Stable Video Diffusion was conditioned on fps - 1, which is why it is reduced here.
        # See: https://github.com/Stability-AI/generative-models/blob/ed0997173f98eaf8f4edf7ba5fe8f15c6b877fd3/scripts/sampling/simple_video_sample.py#L188
        fps = fps - 1

        # 4. Encode input image using VAE
        image = self.video_processor.preprocess(image, height=height, width=width).to(device)
        noise = randn_tensor(image.shape, generator=generator, device=device, dtype=image.dtype)
        image = image + noise_aug_strength * noise

        needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast
        if needs_upcasting:
            self.vae.to(dtype=torch.float32)

        image_latents = self._encode_vae_image(
            image,
            device=device,
            num_videos_per_prompt=num_videos_per_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
        )

        print("---- im latents", image_latents.shape)
        image_latents = image_latents.to(image_embeddings.dtype)

        # cast back to fp16 if needed
        if needs_upcasting:
            self.vae.to(dtype=torch.float16)

        # Repeat the image latents for each frame so we can concatenate them with the noise
        # image_latents [batch, channels, height, width] ->[batch, num_frames, channels, height, width]
        image_latents = image_latents.unsqueeze(1).repeat(1, num_frames, 1, 1, 1)

        # 5. Get Added Time IDs
        added_time_ids = self._get_add_time_ids(
            fps,
            motion_bucket_id,
            noise_aug_strength,
            image_embeddings.dtype,
            batch_size,
            num_videos_per_prompt,
            self.do_classifier_free_guidance,
        )
        added_time_ids = added_time_ids.to(device)
        print("----- added_time_ids", added_time_ids.shape)

        actions = actions.to(device)
        actions = actions.repeat(batch_size * num_videos_per_prompt, 1, 1)
        print("----- actions", actions.shape)

        init_state = init_state.to(device)
        init_state = init_state.repeat(batch_size * num_videos_per_prompt, 1)
        print("----- init_state", init_state.shape)

        if self.do_classifier_free_guidance:
            actions = torch.cat([actions, actions])
            print("cfg----- actions", actions.shape)

            init_state = torch.cat([init_state, init_state])
            print("cfg----- init_state", init_state.shape)

        # 6. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, None, sigmas)

        # 7. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_frames,
            num_channels_latents,
            height,
            width,
            image_embeddings.dtype,
            device,
            generator,
            latents,
        )
        print("-------- latents", latents.shape)

        # 8. Prepare guidance scale
        cfg_scale = 0.5*(min_guidance_scale + max_guidance_scale)
        guidance_scale = torch.ones(1,num_frames)*cfg_scale#torch.linspace(min_guidance_scale, max_guidance_scale, num_frames).unsqueeze(0)
        guidance_scale = guidance_scale.to(device, latents.dtype)
        guidance_scale = guidance_scale.repeat(batch_size * num_videos_per_prompt, 1)
        guidance_scale = _append_dims(guidance_scale, latents.ndim)

        self._guidance_scale = guidance_scale
        print("----- guidance_scale", guidance_scale.shape)

        # 9. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # Concatenate image_latents over channels dimension
                latent_model_input = torch.cat([latent_model_input, image_latents], dim=2)

                print("---- latent_model_input", latent_model_input.shape)

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=image_embeddings,
                    added_time_ids=added_time_ids,
                    actions=actions,
                    init_state=init_state,
                    return_dict=False,
                )

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                    print("----noise pred cond", noise_pred_cond.shape)
                    print("----noise pred uncond", noise_pred_uncond.shape)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        if not output_type == "latent":
            # cast back to fp16 if needed
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
            print("pre decode ------- latents", latents.shape, latents.max(), latents.min())
            frames = self.decode_latents(latents, num_frames, decode_chunk_size)
            print("post decode ------- frames", frames.shape, frames.max(), frames.min())
            frames = self.video_processor.postprocess_video(video=frames, output_type=output_type)
            print("post process ------- frames", np.array(frames).shape, np.array(frames).max(), np.array(frames).min())
        else:
            frames = latents

        self.maybe_free_model_hooks()

        if not return_dict:
            return frames

        return StableVideoDiffusionPipelineOutput(frames=frames)




