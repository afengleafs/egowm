
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



from decord import VideoReader, cpu
from models.svd_wrapper import build_actionfwise2state_svd_model, SVD_actionstate_wrapper, DebugSVDActionStateConstcfgPipeline
from diffusers.utils import load_image
from tqdm import tqdm
from typing import Callable, Dict, List, Optional, Union, Tuple
import h5py
import io
import imageio
from PIL import Image
from dataclasses import dataclass
import inspect
import pickle


import PIL.Image

def normalize_action_data(data, stats):
    # nomalize to [0,1]
    #print(type(data), data)
    #print(type(stats['min']), stats['min'])
    #print(type(stats['max']), stats['max'])

    ndata = (data - stats['min']) / (stats['max'] - stats['min'])
    # normalize to [-1, 1]
    ndata = ndata * 2 - 1
    return ndata











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



def main(args) :

    args.gpu = args.device
    print("device : ", args.gpu)

    seed = args.seed #+ utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


    backbone = build_actionfwise2state_svd_model(args.pretrained_path)#VDiffFeatExtractor()
    model = SVD_actionstate_wrapper(backbone)

    model.to(args.gpu)


    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.backbone.unet.load_state_dict(checkpoint['unet'])
        resume_epoch = checkpoint['epoch']
        print("checkpoint loaded : ", resume_epoch)
    else:
        print("no checkpoint---------")
        resume_epoch = -999

    args.iteration = resume_epoch



    action_stats = {
                    'max' : np.array([0.6775, 0.1794, 0.6841, 1.9395, 0.0638, 0.4112, 2.1809, 1.2214, 0.9985,
                        1.7021, 1.3078, 0.9962, 1.5630, 2.0908, 1.2251, 0.8223, 1.8282, 1.2627,
                        1.4591, 1.5652, 0.7138, 1.0000, 1.0000, 2.7938, 2.0000]),

                    'min' : np.array([-0.6337, -0.1194, -1.3144, -1.0353, -0.0811, -0.6756, -1.7197, -1.3794,
                            -0.9571, -1.8178, -1.3653, -0.9234, -1.1744, -1.8419, -1.3763, -1.1573,
                            -1.6800, -1.2870, -1.1016, -1.3727, -0.7889, -1.0000, -1.0000, -1.9542,
                            -2.8913])
                }

    action_stats_6 = {
        'max' : np.array([0.6772, 0.2021, 0.8041, 1.9454, 0.1002, 0.5036, 2.1714, 1.2248, 0.9923,
                    1.7115, 1.3096, 0.9908, 1.5655, 2.0927, 1.2239, 0.8698, 1.8549, 1.2650,
                    1.4743, 1.5530, 0.7287, 1.0000, 1.0000, 2.9645, 3.0000]),

        'min' : np.array([-0.6365, -0.1772, -1.3225, -1.1737, -0.1114, -0.6768, -1.7728, -1.3745,
                -1.0611, -1.8209, -1.3658, -0.9281, -1.3181, -1.9257, -1.3698, -1.1672,
                -1.6832, -1.2960, -1.1032, -1.3914, -0.7891, -1.0000, -1.0000, -2.2354,
                -2.8913])
    }

    state_stats = {

        'max' : np.array([ 7.7411e-01,  5.1655e-01, -1.3295e-03,  2.5882e+00,  3.0082e-01,
                1.8440e-01,  1.0049e+00,  1.8385e+00,  5.3803e-01, -5.0709e-02,
                8.0730e-01,  8.7204e-01,  5.2016e-01,  1.0011e+00,  9.2817e-02,
                1.4485e+00, -1.4477e-02,  1.2063e+00,  8.8060e-01,  1.5753e+00,
                5.2621e-01,  1.0000e+00,  1.0000e+00,  1.5000e+00,  1.5000e+00]),

        'min' : np.array([-0.9295, -0.5172, -1.5726,  0.0175, -0.3063, -1.0203, -2.0873, -0.0955,
                -1.4701, -2.2058, -1.2059, -0.8234, -1.5123, -2.0198, -1.9361, -0.5423,
                -2.2147, -0.8241, -0.8492, -0.5450, -0.3191,  0.0000,  0.0000, -1.5000,
                -1.5000])
    }

    ########## VISUALIZE SAMPLES

    print("---vis---")
    pipe = DebugSVDActionStateConstcfgPipeline.from_pretrained(
        args.pretrained_path, 
        unet = model.backbone.unet,
        vae = model.backbone.vae,
        image_encoder = model.backbone.image_encoder,
        torch_dtype=torch.float16, 
        variant="fp16")
    
    
    anno_path = args.anno_path
    val_set = os.listdir(anno_path)

    #random.shuffle(test_traj_names)
    #val_set = ["sample_178.json", "sample_244.json", "sample_13.json", "sample_28.json"] #randomly selected to validate.  "sample_83.json", "sample_89.json"
    #val_set = ['sample_26626.json', 'sample_23872.json', 'sample_4567.json', 'sample_8761.json', 'sample_7870.json', 'sample_10131.json', 'sample_11103.json'] #nav only
    print(val_set)


    nav_cnt = 0 
    bimanual_cnt = 0
    left_cnt = 0 
    right_cnt = 0 






    o_dir = os.path.join(args.val_vis, f"iter_{args.iteration}")
    if not os.path.exists(o_dir) :
        os.makedirs(o_dir)



    for traj in tqdm(val_set) :



        
        with open(os.path.join(anno_path, traj), "r") as f :

            traj_dict = json.load(f)

        
        
        task = traj_dict['task']


        rank = traj_dict['rank']

        #print("<<<<<<<<<<<<<<<<==", traj, task, nav_cnt, left_cnt, right_cnt, bimanual_cnt)


        if task != 'nav' :
            continue

        

        

        
        states_seg = np.array(traj_dict['states'])
        print("states_seg : ", states_seg.shape, states_seg.dtype)

        f_start = traj_dict['f_start']
        f_end = traj_dict['f_end']
        seg_idx = traj_dict['seg_idx']


        ep_length = f_end - f_start + 1
        print("=======ep_length", ep_length, f_start, f_end)



        clip_starts = [c for c in list(range(f_start,f_end+1,50)) if c+2*(args.num_frames+1)*6 < (f_end)]
        print("---",clip_starts)

        
        for clip_start in clip_starts :

            

            clip_end = clip_start + (2*(args.num_frames)+1)*6

            print(">>>>>>>", clip_start, clip_end)




            seg_start = clip_start - f_start
            seg_end = clip_end - f_start
            states = states_seg[seg_start : seg_end : 6]
            print("clip states : ", states.shape, states.dtype, seg_start, seg_end)
            actions = states[1:] - states[:-1]
            #print("actions : ", actions)

            actions = torch.from_numpy(normalize_action_data(actions, action_stats_6))


            video_path = os.path.join(args.val_root, f"video_{0}.mp4")
            

            frame_ids = list(range(clip_start, clip_end, 6))
            print("=<><><>=", frame_ids)
            vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)
            print(0, "==",len(vr))
            assert (np.array(frame_ids) < len(vr)).all()
            assert (np.array(frame_ids) >= 0).all()
            vr.seek(0)
            frames = vr.get_batch(frame_ids).asnumpy()
            frames = frames.astype(np.uint8)
            #frames = torch.from_numpy(frames).permute(0, 3, 1, 2)  # (l, c, h, w)
            print("loaded frames : ", frames.shape, frames.max(), frames.min(), frames.dtype)


            frames = [Image.fromarray(frame) for frame in frames]
            init_frame = frames[0]
            future_frames = frames[1:]

            traj_name = traj.split('.')[0]


            s_dir = os.path.join(args.val_vis, f"iter_{args.iteration}", f'{traj_name}_{0}_{seg_idx}_{task}', f"{clip_start}_{clip_end}")
            if not os.path.exists(s_dir) :
                os.makedirs(s_dir)

            init_frame.save(os.path.join(s_dir, f"f_{clip_start}.png"), format='PNG')


            gt_dir = os.path.join(args.val_vis, f"iter_{args.iteration}", f'{traj_name}_{0}_{seg_idx}_{task}', f"{clip_start}_{clip_end}", "GT")
            if not os.path.exists(gt_dir) :
                os.makedirs(gt_dir)


            pred_dir = os.path.join(args.val_vis, f"iter_{args.iteration}", f'{traj_name}_{0}_{seg_idx}_{task}', f"{clip_start}_{clip_end}", "pred")
            if not os.path.exists(pred_dir) :
                os.makedirs(pred_dir)

            t_idx=clip_start
            for img in future_frames :
                t_idx+=6
                img.save(os.path.join(gt_dir, f"f_{t_idx}.png"), format='PNG')

            gt_mp4_path = os.path.join(s_dir, "gt.mp4")
            gt_video_frames = [init_frame] + future_frames
            imageio.mimwrite(
                gt_mp4_path,
                [np.array(frame) for frame in gt_video_frames],
                fps=7,
            )


                

            image = load_image(init_frame)
            image = image.resize((512, 512))

            print("actions all :", actions.shape)


            

            

            with torch.no_grad() :

                output_frames = []
                for i in range(2) :

                    actions_i = actions[args.num_frames*i : args.num_frames*(i+1)]
                    print("actions i : ", args.num_frames*i,  args.num_frames*(i+1), actions_i.shape)

                    init_state_i = torch.from_numpy(normalize_action_data(states[args.num_frames*i], state_stats))
                    print("initial_state i : ", args.num_frames*i, init_state_i.shape)

                    frame_chunk = pipe(
                            image, 
                            decode_chunk_size=8, 
                            generator=torch.manual_seed(args.seed), 
                            motion_bucket_id=180, 
                            noise_aug_strength=0.1,
                            actions=actions_i,
                            init_state=init_state_i,
                            fps =7,
                            height=512,
                            width=512,
                            num_frames=args.num_frames,
                            ).frames[0]

                    
                        
                    image = frame_chunk[-1]
                    output_frames += frame_chunk


                tt=clip_start
                
                for f in output_frames :
                    tt=tt+6

                    print(np.array(f).shape, type(f))
                    
                    f.save(os.path.join(pred_dir, f"f_{tt}.png"), format='PNG')

                pred_mp4_path = os.path.join(s_dir, "pred.mp4")
                pred_video_frames = [init_frame] + output_frames
                imageio.mimwrite(
                    pred_mp4_path,
                    [np.array(frame) for frame in pred_video_frames],
                    fps=7,
                )

                        



    

def set_path(args):
    now = datetime.now()
    dt_string = now.strftime("%Y_%m_%d_%H_%M")
    args.launch_timestamp = dt_string
    name_prefix = f"{args.name_prefix}"
    exp_path = os.path.join(args.out_dir, name_prefix)
    log_path = os.path.join(exp_path, 'log')
    model_path = os.path.join(exp_path, 'model')
    train_vis = os.path.join(exp_path, 'train_vis')
    val_vis = os.path.join(exp_path, 'val_vis')
    if not os.path.exists(log_path): 
        os.makedirs(log_path)
    if not os.path.exists(model_path): 
        os.makedirs(model_path)
    if not os.path.exists(train_vis): 
        os.makedirs(train_vis)
    if not os.path.exists(val_vis): 
        os.makedirs(val_vis)
    
    with open(f'{log_path}/running_command.txt', 'a') as f:
        json.dump({'command_time_stamp':dt_string, **args.__dict__}, f, indent=2)
        f.write('\n')

    return log_path, model_path, train_vis, val_vis








if __name__ == "__main__" :

    parser = argparse.ArgumentParser('RVOSNet training and evaluation script', parents=[opts.get_args_parser()])
    parser.add_argument(
        "--out_dir",
        default="output",
        help="Root directory for logs/models/visualizations.",
    )
    parser.add_argument(
        "--anno_path",
        default="/data/EVE1x/meta/val",
        help="Path to validation annotations directory.",
    )
    parser.add_argument(
        "--val_root",
        default="/data/EVE1x/val_v2.0_raw",
        help="Root directory containing validation videos.",
    )
    parser.add_argument(
        "--pretrained_path",
        default="pretrained/stable-video-diffusion-img2vid-xt",
        help="Path to the pretrained Stable Video Diffusion model.",
    )
    args = parser.parse_args()
    import json
    print(json.dumps(args.__dict__, indent = 4))
    args.log_path, args.model_path, args.train_vis, args.val_vis = set_path(args)

    main(args)

    