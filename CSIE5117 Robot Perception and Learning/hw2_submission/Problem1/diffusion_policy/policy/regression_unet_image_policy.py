from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.diffusion.positional_embedding import SinusoidalPosEmb


class RegressionUnetImagePolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            obs_encoder: MultiImageObsEncoder,
            horizon, 
            n_action_steps, 
            n_obs_steps,
            obs_as_global_cond=True,
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
            use_sinusoidal_emb=False,
            # parameters passed to step
            **kwargs):
        super().__init__()

        # parse shapes
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_feature_dim = obs_encoder.output_shape()[0]

        # Initialize embeddings
        self.use_sinusoidal_emb = use_sinusoidal_emb
        if self.use_sinusoidal_emb:
            self.input_emb = SinusoidalPosEmb(action_dim)
        else:
            self.input_emb = nn.Embedding(horizon, action_dim)  # learnable queries

        # Create ConditionalUnet1D model
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            global_cond_dim = obs_feature_dim * n_obs_steps

        self.model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale
        )

        self.obs_encoder = obs_encoder
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs

    # ========= inference  ============
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert 'past_action' not in obs_dict
        nobs = self.normalizer.normalize(obs_dict)
        B = next(iter(nobs.values())).shape[0]
        T = self.horizon
        Da = self.action_dim

        # handle different ways of passing observation
        global_cond = None
        if self.obs_as_global_cond:
            this_nobs = dict_apply(nobs, lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            global_cond = nobs_features.reshape(B, -1)

        # Generate input embeddings
        if self.use_sinusoidal_emb:
            action_ts = torch.linspace(-1, 1, T, device=self.device)
            inputs = self.input_emb(action_ts).unsqueeze(0).expand(B, -1, -1)
        else:
            inputs = self.input_emb.weight.unsqueeze(0).expand(B, -1, -1)

        # TODO: Predict the action (set timestep to 0)
        # action_pred = ...
        action_pred = self.model(
            inputs,
            timestep=0,
            global_cond=global_cond
        )
        action = self.normalizer['action'].unnormalize(action_pred)

        result = {
            'action': action,
            'action_pred': action_pred
        }
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        B = nactions.shape[0]

        # handle different ways of passing observation
        global_cond = None
        if self.obs_as_global_cond:
            this_nobs = dict_apply(nobs, lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            global_cond = nobs_features.reshape(B, -1)

        # Generate input embeddings
        if self.use_sinusoidal_emb:
            action_ts = torch.linspace(-1, 1, self.horizon, device=self.device)
            inputs = self.input_emb(action_ts).unsqueeze(0).expand(B, -1, -1)
        else:
            inputs = self.input_emb.weight.unsqueeze(0).expand(B, -1, -1)

        # TODO: Set timestep to 0 and forward pass through the model
        # pred = ...
        pred = self.model(
            inputs,
            timestep=0,
            global_cond=global_cond
        )

        # TODO: Compute loss directly against the ground truth trajectory
        # loss = ...
        loss = F.mse_loss(pred, nactions)

        return loss
