"""共享工具:输出 dataclass、resnet forward patch、timestep/维度辅助函数。"""
import inspect
from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import PIL.Image
import torch
from diffusers.utils.outputs import BaseOutput
# BaseOutput 来自 diffusers，它是官方常用的输出基类，支持类似“既能按属性取值，也能按字典风格取值”的输出对象。


@dataclass
class UNetSpatioTemporalConditionOutput(BaseOutput):
    """UNetSpatioTemporalConditionModel 的输出包装。
    视频张量
    sample: (batch, num_frames, channels, height, width)
    """
    sample: torch.Tensor = None


@dataclass
class StableVideoDiffusionPipelineOutput(BaseOutput):
    """SVD pipeline 的输出包装。
    最终生成的视频结果List[List[PIL.Image.Image]] 往往表示“batch 内每个样本对应一个帧列表”

    """
    frames: Union[List[List[PIL.Image.Image]], np.ndarray, torch.Tensor]




def _append_dims(x, target_dims):
    """在 x 后面补 1 维直到 ndim == target_dims。"""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    return x[(...,) + (None,) * dims_to_append]

#  spatiotemporal_res_forward2 是整个项目最核心的修改：
#  这样空间分支和时间分支都能感知到"第 i 帧该执行什么动作"，实现了动作条件控制视频生成
# t:     999   850   700   500   300   100    0
# σ:    14.6   9.2   5.8   2.9   1.1   0.3   0.0 

def spatiotemporal_res_forward2(self):
    """替换 SpatioTemporalResBlock.forward 的闭包。

    作用:把 temb 里 [time | per-frame action] 拆开——time 广播到所有帧,
    再叠加每帧的 action embedding,使 spatial/temporal resblock 能在每帧
    感知到不同的 action 条件。

    让 spatial_res_block 和 temporal_res_block 都能“看到”每一帧对应的条件。

    第 0 个 token 是 time_emb，表示整段视频的时间条件；
    剩下 F 个 token 是每一帧的 action_emb，表示每帧不同的动作条件。
    """

    def dbg_st_forward2(
        hidden_states: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        image_only_indicator: Optional[torch.Tensor] = None,
    ):
        # - 从 image_only_indicator 的最后一维拿到帧数 F
        # - 从 temb 的第 0 维拿到 batch 大小 B

        num_frames = image_only_indicator.shape[-1]
        b = temb.shape[0]
        # temb: [B, 1+F, D] —— 第 0 个 token 是 time emb,其余 F 个是 per-frame action emb
        time_emb = temb[:, 0, :]
        action_emb = temb[:, 1:, :]

        time_emb = time_emb.unsqueeze(1).repeat(1, num_frames, 1)
        # 每帧条件 = 全局时间条件 + 该帧动作条件
        timeaction_emb = time_emb + action_emb

        # 对接空间分支，一般空间 block 处理的是“把帧当成 batch 展开”的输入格式
        timeaction_emb = timeaction_emb.reshape(b * num_frames, -1)

        # 空间分支
        # 把每帧对应的条件传进空间残差块
        # 这样每帧的 2D 特征提取都带有自己的条件信息，而不是整段视频共用一个条件
        # 否则空间分支可能只能看到统一 temb，无法区分“第 3 帧”和“第 8 帧”该执行什么动作
        hidden_states = self.spatial_res_block(hidden_states, timeaction_emb)

        batch_frames, channels, height, width = hidden_states.shape
        batch_size = batch_frames // num_frames

        hidden_states_mix = (
            hidden_states[None, :]
            .reshape(batch_size, num_frames, channels, height, width)
            .permute(0, 2, 1, 3, 4)
        )

        # hidden_states_mix 用来保留“空间分支输出”
        # hidden_states 用来继续喂给“时间分支”
        # 后面 time_mixer 要把空间结果和时间结果融合，所以要保留两份

        hidden_states = (
            hidden_states[None, :]
            .reshape(batch_size, num_frames, channels, height, width)
            .permute(0, 2, 1, 3, 4)
        )

        if temb is not None:
            timeaction_emb = timeaction_emb.reshape(batch_size, num_frames, -1)

        # 时间分支
        hidden_states = self.temporal_res_block(hidden_states, timeaction_emb)
        hidden_states = self.time_mixer(
            x_spatial=hidden_states_mix,
            x_temporal=hidden_states,
            image_only_indicator=image_only_indicator,
        )

        hidden_states = hidden_states.permute(0, 2, 1, 3, 4).reshape(
            batch_frames, channels, height, width
        )

        # 如何理解时间分支和空间分支的作用？
        return hidden_states

    return dbg_st_forward2


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """diffusers 常见的 timesteps / sigmas 二选一 helper。"""
    if timesteps is not None and sigmas is not None:
        raise ValueError(
            "Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values"
        )
    if timesteps is not None:
        accepts = "timesteps" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accepts:
            raise ValueError(
                f"Scheduler {scheduler.__class__} does not support custom timestep schedules."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accepts = "sigmas" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accepts:
            raise ValueError(
                f"Scheduler {scheduler.__class__} does not support custom sigmas schedules."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps
