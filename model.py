from typing import Any

import numpy as np
from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn
from torch.distributions import MultivariateNormal
import torch
from pytorch_lightning import LightningModule
from torch import nn, optim, Tensor
from torch.nn import functional as tf


class Encoder(LightningModule):

    def __init__(self, init_size: int = 256, latent_size: int = 22, lr: float = 1e-5, weight_decay: float = 0.0,
                 scheduler_gamma: float = .7, betas: tuple[float, float] = (.9, .99), *args: Any, **kwargs: Any):
        super().__init__()
        self.init_size = init_size
        self.latent_size = latent_size
        self.output_size = 6
        self.automatic_optimization = False
        self.lr = lr
        self.weight_decay = weight_decay
        self.scheduler_gamma = scheduler_gamma
        self.betas = betas

        self.encoder_games = nn.Sequential(
            nn.Linear(self.init_size, self.latent_size * 2),
            nn.SiLU(),
            nn.Linear(self.latent_size * 2, self.latent_size * 2),
            nn.SiLU(),
            nn.Linear(self.latent_size * 2, self.latent_size),
        )

        self.apply_opp = nn.Sequential(
            nn.Linear(self.latent_size * 2, self.latent_size),
            nn.SiLU(),
            nn.Linear(self.latent_size, self.output_size),
        )

        dnn_to_bnn(self.apply_opp, bnn_prior_parameters = {
              "prior_mu": 0.0,
              "prior_sigma": 1.0,
              "posterior_mu_init": 0.0,
              "posterior_rho_init": -4.0,
              "type": "Reparameterization",  # Flipout or Reparameterization
              "moped_enable": False,
        })

    def encode(self, x):
        return self.encoder_games(x)

    def forward(self, x, y):
        x = self.encode(x)
        y = self.encode(y)
        x = self.apply_opp(torch.cat([x, y], dim=-1))
        return x

    def loss_function(self, y, y_pred):
        return tf.mse_loss(y, y_pred)

    def on_fit_start(self) -> None:
        if self.trainer.is_global_zero and self.logger:
            self.logger.log_graph(self, self.example_input_array)

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        train_loss = self.train_val_get(batch, batch_idx)
        opt.zero_grad()
        self.manual_backward(train_loss, retain_graph=True)
        opt.step()

    def validation_step(self, batch, batch_idx):
        self.train_val_get(batch, batch_idx, 'val')

    def on_train_epoch_end(self) -> None:
        sch = self.lr_schedulers()

        # If the selected scheduler is a ReduceLROnPlateau scheduler.
        if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
            sch.step(self.trainer.callback_metrics["val_loss"])
        else:
            sch.step()

    def on_validation_epoch_end(self) -> None:
        self.log('lr', self.lr_schedulers().get_last_lr()[0], prog_bar=True, rank_zero_only=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),
                                      lr=self.lr,
                                      weight_decay=self.weight_decay,
                                      betas=self.betas,
                                      eps=1e-7)
        if self.scheduler_gamma is None:
            return optimizer
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.scheduler_gamma)
        '''scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, cooldown=self.params['step_size'],
                                                         factor=self.params['scheduler_gamma'], threshold=1e-5)'''

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def train_val_get(self, batch, batch_idx, kind='train'):
        team, opp, targets = batch

        results = self.forward(team, opp)
        train_loss = self.loss_function(results, targets)

        self.log_dict({f'{kind}_loss': train_loss}, on_epoch=True,
                      prog_bar=True, rank_zero_only=True)
        return train_loss


class Predictor(LightningModule):

    def __init__(self, init_size: int = 256, latent_size: int = 22, lr: float = 1e-5, weight_decay: float = 0.0,
                 scheduler_gamma: float = .7, betas: tuple[float, float] = (.9, .99), num_samples: int = 10,
                 *args: Any, **kwargs: Any):
        super().__init__()
        self.init_size = init_size
        self.latent_size = latent_size
        self.automatic_optimization = False
        self.lr = lr
        self.weight_decay = weight_decay
        self.scheduler_gamma = scheduler_gamma
        self.betas = betas
        self.num_samples = num_samples

        self.encoding_mu = nn.Sequential(
            nn.Linear(self.init_size, self.init_size),
            nn.GELU(),
            nn.Linear(self.init_size, self.init_size),
            nn.GELU(),
            nn.Dropout(.5),
            nn.Linear(self.init_size, self.latent_size),
        )

        self.encoding_var = nn.Sequential(
            nn.Linear(self.init_size, self.init_size),
            nn.GELU(),
            nn.Linear(self.init_size, self.init_size),
            nn.GELU(),
            nn.Dropout(.5),
            nn.Linear(self.init_size, self.latent_size),
            nn.Softplus(),
        )

        self.combine = nn.Sequential(
            nn.Conv1d(2, 1, 1, 1, 0),
            nn.GELU(),
        )

        self.predict_layer = nn.Sequential(
            nn.Linear(latent_size, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, y):

        # Draw samples for each of the teams
        x_hat = self.team_prep(x)
        y_hat = self.team_prep(y)

        xy = self.combine(torch.cat([x_hat.unsqueeze(1), y_hat.unsqueeze(1)], dim=1))

        xy = self.predict_layer(xy.squeeze(1))

        return xy

    def loss_function(self, y, y_pred):
        # return tf.mse_loss(torch.repeat_interleave(y, 100, dim=0), y_pred)
        return tf.binary_cross_entropy(y_pred, y)
        # return tf.mse_loss(y, y_pred)

    def on_fit_start(self) -> None:
        if self.trainer.is_global_zero and self.logger:
            self.logger.log_graph(self, self.example_input_array)

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        train_loss = self.train_val_get(batch, batch_idx)
        opt.zero_grad()
        self.manual_backward(train_loss, retain_graph=True)
        opt.step()

    def validation_step(self, batch, batch_idx):
        self.train_val_get(batch, batch_idx, 'val')

    def on_train_epoch_end(self) -> None:
        sch = self.lr_schedulers()

        # If the selected scheduler is a ReduceLROnPlateau scheduler.
        if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
            sch.step(self.trainer.callback_metrics["val_loss"])
        else:
            sch.step()

    def on_validation_epoch_end(self) -> None:
        self.log('lr', self.lr_schedulers().get_last_lr()[0], prog_bar=True, rank_zero_only=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),
                                      lr=self.lr,
                                      weight_decay=self.weight_decay,
                                      betas=self.betas,
                                      eps=1e-7)
        if self.scheduler_gamma is None:
            return optimizer
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.scheduler_gamma)
        '''scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, cooldown=self.params['step_size'],
                                                         factor=self.params['scheduler_gamma'], threshold=1e-5)'''

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def train_val_get(self, batch, batch_idx, kind='train'):
        tx, ty, label = batch

        results = self.forward(tx, ty)
        train_loss = self.loss_function(label, results)
        acc = (label.bool() & (results > .5)).sum() / label.shape[0]

        self.log_dict({f'{kind}_loss': train_loss, f'{kind}_acc': acc}, on_epoch=True,
                      prog_bar=True, rank_zero_only=True)
        return train_loss


class EncodePDF(LightningModule):

    def __init__(self, inp_sz: int = 256, latent: int = 22, depth: int = 3):
        super().__init__()
        self.linear_layer = nn.Sequential(
            nn.Linear(inp_sz, inp_sz),
            nn.GELU(),
            nn.Linear(inp_sz, latent),
            nn.GELU(),
        )
        self.conv_layer = nn.Sequential(
            nn.Conv1d(1, depth, 1, 1, 0),
            nn.GELU(),
            nn.Conv1d(depth, depth, 1, 1, 0),
        )

        self.mu_layer = nn.Sequential(
            nn.Linear(latent, latent),
            nn.GELU(),
            nn.Linear(latent, latent),
        )

        self.latent = latent

    def forward(self, x):
        x = self.linear_layer(x)
        mu = self.mu_layer(x)
        x = self.conv_layer(x.view(-1, 1, self.latent))

        return mu, torch.bmm(x.mT, x)


