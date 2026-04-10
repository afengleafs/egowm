"""把 UNet + VAE + image encoder + scheduler 组合成可训练模块的外层封装。"""
import torch.nn as nn


class ActionSVDFwise(nn.Module):
    """Action-only 版本:冻结 VAE 和 image encoder,只训练 UNet。"""

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
        return self.unet(
            latent_model_input,
            t,
            encoder_hidden_states=image_embeddings,
            added_time_ids=added_time_ids,
            actions=actions,
            return_dict=False,
        )


class SVD_action_wrapper(nn.Module):
    """最外层训练包装:调用时触发冻结设置。"""

    def __init__(self, vdiff):
        super().__init__()
        self.backbone = vdiff
        self.backbone.set_untrained_vae_imgenc()

    def forward(
        self, latent_model_input, init_frame_embeddings, add_time_ids, actions, timesteps
    ):
        return self.backbone(
            latent_model_input,
            init_frame_embeddings,
            add_time_ids,
            actions,
            timesteps,
        )


class ActionSVDFwisestate(nn.Module):
    """Action+state 版本。"""

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
        init_state,
        t,
    ):
        return self.unet(
            latent_model_input,
            t,
            encoder_hidden_states=image_embeddings,
            added_time_ids=added_time_ids,
            actions=actions,
            init_state=init_state,
            return_dict=False,
        )


class SVD_actionstate_wrapper(nn.Module):
    def __init__(self, vdiff):
        super().__init__()
        self.backbone = vdiff
        self.backbone.set_untrained_vae_imgenc()

    def forward(
        self,
        latent_model_input,
        init_frame_embeddings,
        add_time_ids,
        actions,
        init_state,
        timesteps,
    ):
        return self.backbone(
            latent_model_input,
            init_frame_embeddings,
            add_time_ids,
            actions,
            init_state,
            timesteps,
        )
