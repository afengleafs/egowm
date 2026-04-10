# Model Code Layout

For easier reproduction and debugging, the monolithic implementation in `models/svd_wrapper.py` has also been decomposed into a modular `model/` package.

- `model/common.py`: shared outputs and helper utilities
- `model/unet_action.py`: action-conditioned SVD UNet
- `model/unet_action_state.py`: action + initial-state conditioned SVD UNet
- `model/wrapper.py`: trainable outer wrappers around UNet, VAE, image encoder, and scheduler
- `model/pipeline.py`: inference pipelines corresponding to the custom SVD variants
- `model/builder.py`: factory functions for constructing models from pretrained SVD weights

The existing training and inference scripts still import from `models/svd_wrapper.py`. The `model/` directory is provided as a structurally equivalent, easier-to-read reproduction for follow-up development and verification.
