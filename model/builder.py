"""工厂函数:从预训练 SVD 权重构建 action / action+state 模型。"""
from .pipeline import DebugSVDActionPipeline
from .unet_action import DebugActionUnetFwise2
from .unet_action_state import DebugActionUnetFwise2state
from .wrapper import ActionSVDFwise, ActionSVDFwisestate


def build_actionfwise2_svd_model(
    sd_id: str = "pretrained/stable-video-diffusion-img2vid-xt",
):
    """构建 action-only 的 SVD world model(3-DoF 场景)。"""
    unet = DebugActionUnetFwise2.from_pretrained(
        sd_id, subfolder="unet", low_cpu_mem_usage=False
    )
    pipe = DebugSVDActionPipeline.from_pretrained(sd_id)
    return ActionSVDFwise(
        unet, pipe.image_encoder, pipe.vae, pipe.feature_extractor, pipe.scheduler
    )


def build_actionfwise2state_svd_model(
    sd_id: str = "pretrained/stable-video-diffusion-img2vid-xt",
):
    """构建 action+init_state 的 SVD world model(25-DoF 导航场景)。"""
    unet = DebugActionUnetFwise2state.from_pretrained(
        sd_id, subfolder="unet", low_cpu_mem_usage=False
    )
    pipe = DebugSVDActionPipeline.from_pretrained(sd_id)
    return ActionSVDFwisestate(
        unet, pipe.image_encoder, pipe.vae, pipe.feature_extractor, pipe.scheduler
    )
