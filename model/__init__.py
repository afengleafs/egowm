"""Decomposed reproduction of models/svd_wrapper.py.

模块划分:
    common      — 共享输出类、spatiotemporal_res_forward2 patch、工具函数
    unet_action — DebugActionUnetFwise2   (仅 action 条件的 SVD UNet)
    unet_action_state — DebugActionUnetFwise2state (action + init_state 条件)
    wrapper     — ActionSVDFwise / ActionSVDFwisestate 及外层 nn.Module 包装
    pipeline    — 对应 diffusers StableVideoDiffusionPipeline 的两个变体
    builder     — 工厂函数,从预训练 SVD 权重构建模型
"""

from .common import (
    UNetSpatioTemporalConditionOutput,
    StableVideoDiffusionPipelineOutput,
    spatiotemporal_res_forward2,
    retrieve_timesteps,
    _append_dims,
)
from .unet_action import DebugActionUnetFwise2
from .unet_action_state import DebugActionUnetFwise2state
from .wrapper import (
    ActionSVDFwise,
    SVD_action_wrapper,
    ActionSVDFwisestate,
    SVD_actionstate_wrapper,
)
from .pipeline import DebugSVDActionPipeline, DebugSVDActionStateConstcfgPipeline
from .builder import build_actionfwise2_svd_model, build_actionfwise2state_svd_model

__all__ = [
    "UNetSpatioTemporalConditionOutput",
    "StableVideoDiffusionPipelineOutput",
    "spatiotemporal_res_forward2",
    "retrieve_timesteps",
    "DebugActionUnetFwise2",
    "DebugActionUnetFwise2state",
    "ActionSVDFwise",
    "SVD_action_wrapper",
    "ActionSVDFwisestate",
    "SVD_actionstate_wrapper",
    "DebugSVDActionPipeline",
    "DebugSVDActionStateConstcfgPipeline",
    "build_actionfwise2_svd_model",
    "build_actionfwise2state_svd_model",
]
