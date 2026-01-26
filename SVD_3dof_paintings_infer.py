
import argparse
import datetime
import json
import random

from datetime import datetime
import numpy as np
import os
import opts

import torch

from diffusers.utils.outputs import BaseOutput
from diffusers import StableVideoDiffusionPipeline


from diffusers.utils.torch_utils import randn_tensor
from models.svd_wrapper import build_actionfwise2_svd_model, SVD_action_wrapper
from diffusers.utils import load_image, export_to_video
from tqdm import tqdm
from typing import Callable, Dict, List, Optional, Union, Tuple
import h5py
import io
import imageio
from PIL import Image
from dataclasses import dataclass
import inspect
import pickle
import matplotlib.pyplot as plt


import PIL.Image








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
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
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





class DebugSVDActionConstcfgPipeline(StableVideoDiffusionPipeline):

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

        if self.do_classifier_free_guidance:
            actions = torch.cat([actions, actions])
            print("cfg----- actions", actions.shape)

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
        #with self.progress_bar(total=num_inference_steps) as progress_bar:
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

                #if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                #    progress_bar.update()

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








METRIC_WAYPOINT_SPACING_SCAND = 0.38
METRIC_WAYPOINT_SPACING_TARTAN = 0.72


def normalize_xy_data(data, stats):
    # nomalize to [0,1]
    print(type(data), data)
    print(type(stats['min']), stats['min'])
    print(type(stats['max']), stats['max'])

    ndata = (data - stats['min']) / (stats['max'] - stats['min'])
    # normalize to [-1, 1]
    ndata = ndata * 2 - 1
    return ndata


def yaw_rotmat(yaw: float) -> np.ndarray:
    return np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ],
    )


def to_local_coords(
    positions: np.ndarray, curr_pos: np.ndarray, curr_yaw: float
) -> np.ndarray:
    """
    Convert positions to local coordinates

    Args:
        positions (np.ndarray): positions to convert
        curr_pos (np.ndarray): current position
        curr_yaw (float): current yaw
    Returns:
        np.ndarray: positions in local coordinates
    """
    rotmat = yaw_rotmat(curr_yaw)
    if positions.shape[-1] == 2:
        rotmat = rotmat[:2, :2]
    elif positions.shape[-1] == 3:
        pass
    else:
        raise ValueError

    return (positions - curr_pos).dot(rotmat)




def angle_difference(theta1, theta2):
    delta_theta = theta2 - theta1    
    delta_theta = delta_theta - 2 * np.pi * np.floor((delta_theta + np.pi) / (2 * np.pi))    
    return delta_theta


def rand_log_normal(shape, loc=0., scale=1., device='cpu', dtype=torch.float32):
    """Draws samples from an lognormal distribution."""
    u = torch.rand(shape, dtype=dtype, device=device) * (1 - 2e-7) + 1e-7
    return torch.distributions.Normal(loc, scale).icdf(u).exp()



def _compute_padding(kernel_size):
    """Compute padding tuple."""
    # 4 or 6 ints:  (padding_left, padding_right,padding_top,padding_bottom)
    # https://pytorch.org/docs/stable/nn.html#torch.nn.functional.pad
    if len(kernel_size) < 2:
        raise AssertionError(kernel_size)
    computed = [k - 1 for k in kernel_size]

    # for even kernels we need to do asymmetric padding :(
    out_padding = 2 * len(kernel_size) * [0]

    for i in range(len(kernel_size)):
        computed_tmp = computed[-(i + 1)]

        pad_front = computed_tmp // 2
        pad_rear = computed_tmp - pad_front

        out_padding[2 * i + 0] = pad_front
        out_padding[2 * i + 1] = pad_rear

    return out_padding


def _filter2d(input, kernel):
    # prepare kernel
    b, c, h, w = input.shape
    tmp_kernel = kernel[:, None, ...].to(device=input.device, dtype=input.dtype)

    tmp_kernel = tmp_kernel.expand(-1, c, -1, -1)

    height, width = tmp_kernel.shape[-2:]

    padding_shape: List[int] = _compute_padding([height, width])
    input = torch.nn.functional.pad(input, padding_shape, mode="reflect")

    # kernel and input tensor reshape to align element-wise or batch-wise params
    tmp_kernel = tmp_kernel.reshape(-1, 1, height, width)
    input = input.view(-1, tmp_kernel.size(0), input.size(-2), input.size(-1))

    # convolve the tensor with the kernel.
    output = torch.nn.functional.conv2d(input, tmp_kernel, groups=tmp_kernel.size(0), padding=0, stride=1)

    out = output.view(b, c, h, w)
    return out


def _gaussian(window_size: int, sigma):
    if isinstance(sigma, float):
        sigma = torch.tensor([[sigma]])

    batch_size = sigma.shape[0]

    x = (torch.arange(window_size, device=sigma.device, dtype=sigma.dtype) - window_size // 2).expand(batch_size, -1)

    if window_size % 2 == 0:
        x = x + 0.5

    gauss = torch.exp(-x.pow(2.0) / (2 * sigma.pow(2.0)))

    return gauss / gauss.sum(-1, keepdim=True)


def _gaussian_blur2d(input, kernel_size, sigma):
    if isinstance(sigma, tuple):
        sigma = torch.tensor([sigma], dtype=input.dtype)
    else:
        sigma = sigma.to(dtype=input.dtype)

    ky, kx = int(kernel_size[0]), int(kernel_size[1])
    bs = sigma.shape[0]
    kernel_x = _gaussian(kx, sigma[:, 1].view(bs, 1))
    kernel_y = _gaussian(ky, sigma[:, 0].view(bs, 1))
    out_x = _filter2d(input, kernel_x[..., None, :])
    out = _filter2d(out_x, kernel_y[..., None])

    return out







def _resize_with_antialiasing(input, size, interpolation="bicubic", align_corners=True):
    h, w = input.shape[-2:]
    factors = (h / size[0], w / size[1])

    # First, we have to determine sigma
    # Taken from skimage: https://github.com/scikit-image/scikit-image/blob/v0.19.2/skimage/transform/_warps.py#L171
    sigmas = (
        max((factors[0] - 1.0) / 2.0, 0.001),
        max((factors[1] - 1.0) / 2.0, 0.001),
    )

    # Now kernel size. Good results are for 3 sigma, but that is kind of slow. Pillow uses 1 sigma
    # https://github.com/python-pillow/Pillow/blob/master/src/libImaging/Resample.c#L206
    # But they do it in the 2 passes, which gives better results. Let's try 2 sigmas for now
    ks = int(max(2.0 * 2 * sigmas[0], 3)), int(max(2.0 * 2 * sigmas[1], 3))

    # Make sure it is odd
    if (ks[0] % 2) == 0:
        ks = ks[0] + 1, ks[1]

    if (ks[1] % 2) == 0:
        ks = ks[0], ks[1] + 1

    input = _gaussian_blur2d(input, ks, sigmas)

    output = torch.nn.functional.interpolate(input, size=size, mode=interpolation, align_corners=align_corners)
    return output


def create_trajectory_frames(positions, yaw_angles, target_height, line_thickness=6):
    if len(positions) == 0:
        return []

    positions = np.array(positions)
    yaw_angles = np.array(yaw_angles)

    pos_centered = positions - positions[0]
    x_range = pos_centered[:, 0].max() - pos_centered[:, 0].min()
    y_range = pos_centered[:, 1].max() - pos_centered[:, 1].min()
    max_range = max(max(x_range, y_range) * 1.4, 0.5)

    traveled_color = "#00ffff"
    frames = []

    for current_idx in range(len(positions)):
        fig_size = target_height / 80
        fig, ax = plt.subplots(figsize=(fig_size, fig_size), dpi=80)
        ax.set_facecolor("#1a1a2e")
        fig.patch.set_facecolor("#1a1a2e")

        current_pos = positions[current_idx]
        current_yaw = yaw_angles[current_idx]

        ego_positions = positions - current_pos
        rotation_angle = np.pi / 2 - current_yaw
        cos_r, sin_r = np.cos(rotation_angle), np.sin(rotation_angle)

        ego_rotated = np.column_stack(
            [
                ego_positions[:, 0] * cos_r - ego_positions[:, 1] * sin_r,
                ego_positions[:, 0] * sin_r + ego_positions[:, 1] * cos_r,
            ]
        )

        plot_x = ego_rotated[:, 0]
        plot_y = ego_rotated[:, 1]

        for i in range(current_idx):
            ax.plot(
                [plot_x[i], plot_x[i + 1]],
                [plot_y[i], plot_y[i + 1]],
                color=traveled_color,
                linewidth=line_thickness,
                alpha=0.9,
            )

        if current_idx > 0:
            ax.scatter(
                plot_x[:current_idx],
                plot_y[:current_idx],
                c="#ff8c00",
                s=60,
                zorder=6,
                edgecolors="white",
                linewidths=0.5,
                alpha=0.8,
            )

        if current_idx > 0:
            ax.scatter(
                plot_x[0],
                plot_y[0],
                c="#00ff88",
                s=150,
                marker="o",
                zorder=10,
                edgecolors="white",
                linewidths=2.5,
            )

        if current_idx < len(positions) - 1:
            for i in range(current_idx, len(positions) - 1):
                ax.plot(
                    [plot_x[i], plot_x[i + 1]],
                    [plot_y[i], plot_y[i + 1]],
                    color="white",
                    linewidth=line_thickness,
                    alpha=0.5,
                )

        tri_height = max_range * 0.045
        tri_width = max_range * 0.08
        triangle = plt.Polygon(
            [
                (0, tri_height),
                (-tri_width / 2, -tri_height * 0.4),
                (tri_width / 2, -tri_height * 0.4),
            ],
            closed=True,
            facecolor="#ff8c00",
            edgecolor="white",
            linewidth=2.5,
            zorder=20,
            alpha=1.0,
        )
        ax.add_patch(triangle)

        half_range = max_range / 2
        ax.set_xlim(-half_range, half_range)
        ax.set_ylim(-half_range, half_range)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")

        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.set_aspect("equal")
        plt.tight_layout(pad=0)

        fig.canvas.draw()
        img_array = np.array(fig.canvas.buffer_rgba())[:, :, :3]
        frames.append(Image.fromarray(img_array))
        plt.close(fig)

    return frames


def create_traj_only_mp4(trajectory_frames, output_path, fps=4, output_width=512, output_height=512):
    if not trajectory_frames:
        return False

    frames_to_write = []
    for frame in trajectory_frames:
        if frame.mode != "RGB":
            frame = frame.convert("RGB")
        resized = frame.resize((output_width, output_height), Image.LANCZOS)
        frames_to_write.append(np.array(resized))

    imageio.mimwrite(output_path, frames_to_write, fps=fps)
    return True



def main(args) :

    args.gpu = args.device
    print("device : ", args.gpu)

    seed = args.seed #+ utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


    backbone = build_actionfwise2_svd_model(args.pretrained_path)#VDiffFeatExtractor()
    model = SVD_action_wrapper(backbone)

    model.to(args.gpu)


    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.backbone.unet.load_state_dict(checkpoint['unet'])
        resume_epoch = checkpoint['epoch']
        print("checkpoint loaded : ", resume_epoch)
    else:
        print("no checkpoint---------")
        resume_epoch = -999



    action_stats = {
                    'min': np.array([-2.5, -4]),
                    'max': np.array([5, 4]) 
            }

    ########## VISUALIZE SAMPLES

    print("---vis---")
    pipe = DebugSVDActionConstcfgPipeline.from_pretrained(
        args.pretrained_path,
        unet = model.backbone.unet,
        vae = model.backbone.vae,
        image_encoder = model.backbone.image_encoder,
        torch_dtype=torch.float16, 
        variant="fp16")
    
    if args.dataset_root is None:
        dataset_root = "data/scand" if args.dataset == "scand" else "data/tartan"
    else:
        dataset_root = args.dataset_root

    split_dataset = "scand" if args.dataset == "scand" else "tartan_drive"
    test_anno = os.path.join(args.split_root, split_dataset, "test", "traj_names.txt")

    with open(test_anno, "r") as f:
        file_lines = f.read()
        test_traj_names = file_lines.split("\n")
    if "" in test_traj_names:
        test_traj_names.remove("")

    #random.shuffle(test_traj_names)
    val_set = test_traj_names[0:50] #randomly selected to validate

    args.iteration = resume_epoch



    o_dir = os.path.join(args.val_vis, f"iter_{args.iteration}")
    if not os.path.exists(o_dir) :
        os.makedirs(o_dir)


    stride = 16//args.num_frames#64//args.num_frames



    
    painting_dir = "data/paintings"
    painting_idxs = sorted(
        [
            int(fname.replace("painting_", "").replace(".png", ""))
            for fname in os.listdir(painting_dir)
            if fname.startswith("painting_") and fname.endswith(".png")
        ]
    )

    ###some randomly selected trajectories from the test set
    if args.dataset == "tartan":
        val_set = [
            "20210829_heightmaps_3_20210829_37_0",
            "20210829_heightmaps_1_20210829_2_0",
            "20210902_heightmaps_3_20210902_91_1",
            "20210903_heightmaps_5_20210903_143_0",
        ]
    else:
        val_set = [
            "random_mdps_B_Spot_JCL_JCL_Mon_Nov_15_108_4",
            "random_mdps_A_Spot_Library_Fountain_Tue_Nov_9_35_2",
            "random_mdps_B_Spot_RLM_OsCafe_Mon_Nov_15_117_2",
        ]


    for traj in tqdm(val_set) :

        with open(os.path.join(dataset_root, traj, "traj_data.pkl"), "rb") as f:
            traj_data = pickle.load(f)
        for k,v in traj_data.items():
            traj_data[k] = v.astype('float')

        

        position_data = traj_data["position"]
        yaw_data = traj_data["yaw"]

        ep_length = len(np.array(yaw_data))

        print("=======ep_length", ep_length)


        clip_starts = [c for c in list(range(0,ep_length,stride*args.num_frames)) if c+stride*args.num_frames < ep_length]
        clip_starts_2 = clip_starts[0:4]

        if len(clip_starts_2) == 0 :
            print("empty ::::", traj, ep_length)

        for clip_start in clip_starts_2 :

            clip_end = clip_start + stride*args.num_frames

            sample_indx = list(range(clip_start, clip_end+1))

            metric_spacing = (
                METRIC_WAYPOINT_SPACING_SCAND
                if args.dataset == "scand"
                else METRIC_WAYPOINT_SPACING_TARTAN
            )
            positions = []
            yaw_angles = []
            for t in sample_indx:
                x_y = position_data[t]
                yaw = yaw_data[t]
                positions.append([x_y[0] / metric_spacing, x_y[1] / metric_spacing])
                yaw_angles.append(float(yaw))

            traj_frames = create_trajectory_frames(positions, yaw_angles, target_height=512, line_thickness=16)


            actionsx_y = []
            actionsyaw = []


            init_indx = sample_indx[0]
            init_x_y = position_data[init_indx]
            init_yaw = yaw_data[init_indx]
            init_frame = Image.open(os.path.join(dataset_root, traj, f"{init_indx}.jpg")).convert('RGB')



            s_dir = os.path.join(args.val_vis, f"iter_{args.iteration}", traj, f"{clip_start}_{clip_end}")
            if not os.path.exists(s_dir) :
                os.makedirs(s_dir)

            if args.traj_only:
                if traj_frames:
                    create_traj_only_mp4(traj_frames, os.path.join(s_dir, "traj.mp4"), fps=4)
                continue

            if traj_frames:
                create_traj_only_mp4(traj_frames, os.path.join(s_dir, "traj.mp4"), fps=4)

            init_frame.save(os.path.join(s_dir, f"f_{clip_start}.png"), format='PNG')


            gt_root = os.path.join(args.val_vis, f"iter_{args.iteration}", traj, f"{clip_start}_{clip_end}", "GT")
            gt_dir = os.path.join(gt_root, "frames")
            if not os.path.exists(gt_dir):
                os.makedirs(gt_dir)


            

            t_idx=0
            gt_frames = []
            for t in sample_indx[1:] :
                t_idx+=1

                img = Image.open(os.path.join(dataset_root, traj, f"{t}.jpg")).convert('RGB')
                img = img.resize((512, 512))
                img.save(os.path.join(gt_dir, f"f_{t_idx}.png"), format='PNG')
                gt_frames.append(np.array(img))


                x_y = position_data[t]
                yaw = yaw_data[t]

                print("x y yaw ======>>>", x_y, yaw)

                action_x_y = to_local_coords(x_y, init_x_y, init_yaw) / metric_spacing
                action_x_y = normalize_xy_data(action_x_y, action_stats)
                actionsx_y.append(action_x_y)
                
                action_yaw = angle_difference(init_yaw, yaw)
                actionsyaw.append(action_yaw)


                print("actions x y yaw ======>>>", action_x_y, action_yaw)

                init_x_y = x_y
                init_yaw = yaw

            actionsx_y = torch.tensor(actionsx_y)
            actionsyaw = torch.tensor(actionsyaw)
            
            actions = torch.stack([actionsx_y[:,0],actionsx_y[:,1],actionsyaw]).permute(1,0)

            print("actions all :", actions.shape)

            #image = load_image(init_frame)
            #image = image.resize((512, 512))


            for pidx in painting_idxs :

                init_painting = Image.open(os.path.join(painting_dir, f"painting_{pidx}.png")).convert("RGB")

                pred_root = os.path.join(
                    args.val_vis, f"iter_{args.iteration}", traj, f"{clip_start}_{clip_end}", f"pred_{pidx}"
                )
                pred_dir = os.path.join(pred_root, "frames")
                if not os.path.exists(pred_dir):
                    os.makedirs(pred_dir)


                init_painting.save(os.path.join(args.val_vis, f"iter_{args.iteration}", traj, f"{clip_start}_{clip_end}", f"pred_{pidx}", f"painting_{pidx}.png"), format='PNG')


                image = load_image(init_painting)
                image = image.resize((512, 512))



                with torch.no_grad() :

                    output_frames = []
                    for i in range(stride) :

                        actions_i = actions[args.num_frames*i : args.num_frames*(i+1)]
                        print("actions i : ", args.num_frames*i,  args.num_frames*(i+1), actions_i.shape)

                        frame_chunk = pipe(
                                image, 
                                decode_chunk_size=8, 
                                generator=torch.manual_seed(args.seed), 
                                motion_bucket_id=180, 
                                noise_aug_strength=0.1,
                                actions=actions_i,
                                fps=7,
                                height=512,
                                width=512,
                                num_frames=args.num_frames,
                                ).frames[0]

                        
                            
                        image = frame_chunk[-1]
                        output_frames += frame_chunk


                    tt=0
                    
                    for f in output_frames :
                        tt=tt+1

                        print(np.array(f).shape, type(f))
                        
                        f.save(os.path.join(pred_dir, f"f_{tt}.png"), format='PNG')

                    if gt_frames:
                        imageio.mimwrite(os.path.join(gt_root, "gt.mp4"), gt_frames, fps=4)
                    if output_frames:
                        pred_frames = [np.array(f) for f in output_frames]
                        imageio.mimwrite(os.path.join(pred_root, "pred.mp4"), pred_frames, fps=4)

                            



    

def set_path(args):
    now = datetime.now()
    dt_string = now.strftime("%Y_%m_%d_%H_%M")
    args.launch_timestamp = dt_string
    name_prefix = f"{args.name_prefix}"
    exp_path = os.path.join(args.out_dir, name_prefix)
    log_path = os.path.join(exp_path, 'log')
    
    val_vis = os.path.join(exp_path, 'val_vis')
    if not os.path.exists(log_path): 
        os.makedirs(log_path)
    
    if not os.path.exists(val_vis): 
        os.makedirs(val_vis)
    
    with open(f'{log_path}/running_command.txt', 'a') as f:
        json.dump({'command_time_stamp':dt_string, **args.__dict__}, f, indent=2)
        f.write('\n')

    return log_path, val_vis








if __name__ == "__main__" :

    parser = argparse.ArgumentParser('RVOSNet training and evaluation script', parents=[opts.get_args_parser()])
    parser.add_argument(
        "--pretrained_path",
        default="pretrained/stable-video-diffusion-img2vid-xt",
        help="Path to the base SVD pretrained weights directory.",
    )
    parser.add_argument(
        "--out_dir",
        default="output",
        help="Root output directory for logs, models, and visualizations.",
    )
    parser.add_argument(
        "--dataset",
        default="scand",
        choices=["scand", "tartan"],
        help="Dataset source to use for trajectories and images.",
    )
    parser.add_argument(
        "--dataset_root",
        default=None,
        help="Optional override for dataset root. If unset, defaults by dataset.",
    )
    parser.add_argument(
        "--split_root",
        default="data/splits",
        help="Root for dataset split files.",
    )
    parser.add_argument(
        "--traj_only",
        action="store_true",
        help="Save only trajectory mp4s for each clip.",
    )
    args = parser.parse_args()
    import json
    print(json.dumps(args.__dict__, indent = 4))
    args.log_path, args.val_vis = set_path(args)

    main(args)

    