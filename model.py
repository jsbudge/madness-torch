from typing import Any
import numpy as np
from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn
import torch
from pytorch_lightning import LightningModule
from torch import nn, Tensor
from torch.nn import functional as tf


def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


class Encoder(LightningModule):

    def __init__(self, init_size: int = 256, latent_size: int = 22, lr: float = 1e-5, weight_decay: float = 0.0,
                 scheduler_gamma: float = .7, betas: tuple[float, float] = (.9, .99), *args: Any, **kwargs: Any):
        super().__init__()
        self.init_size = init_size
        self.latent_size = latent_size
        self.output_size = 1
        self.automatic_optimization = False
        self.lr = lr
        self.weight_decay = weight_decay
        self.scheduler_gamma = scheduler_gamma
        self.betas = betas

        self.encoder = nn.Sequential(
            nn.Linear(self.init_size, self.latent_size),
            nn.SiLU(),
            nn.Linear(self.latent_size, self.latent_size),
            nn.SiLU(),
            nn.Linear(self.latent_size, self.latent_size),
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_size, self.latent_size),
            nn.SiLU(),
            nn.Linear(self.latent_size, self.init_size),
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        x = self.encode(x)
        return self.decode(x)

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
        team, team_target = batch

        results = self.forward(team)
        train_loss = self.loss_function(results, team_target)

        self.log_dict({f'{kind}_loss': train_loss}, on_epoch=True,
                      prog_bar=True, rank_zero_only=True)
        return train_loss


class Predictor(LightningModule):

    def __init__(self, init_size: int = 70, latent_size: int = 100, lr: float = 1e-5, weight_decay: float = 0.0,
                 encoded_sz: int = 10, sigma: float = 10., scheduler_gamma: float = .7, betas: tuple[float, float] = (.9, .99), *args: Any, **kwargs: Any):
        super().__init__()
        self.init_size = init_size
        self.latent_size = latent_size
        self.output_size = 1
        self.automatic_optimization = False
        self.lr = lr
        self.weight_decay = weight_decay
        self.scheduler_gamma = scheduler_gamma
        self.betas = betas
        self.encoded_size = self.init_size * 2 * encoded_sz
        self.e_sz = encoded_sz
        self.sigma = sigma

        self.apply_opp = nn.Sequential(
            nn.Linear(self.encoded_size * 2, self.latent_size),
            nn.SiLU(),
            nn.Linear(self.latent_size, self.output_size),
            nn.Sigmoid()
        )

        dnn_to_bnn(self.apply_opp, bnn_prior_parameters={
            "prior_mu": 0.0,
            "prior_sigma": 1.0,
            "posterior_mu_init": 0.0,
            "posterior_rho_init": -4.0,
            "type": "Reparameterization",  # Flipout or Reparameterization
            "moped_enable": False,
        })

    def forward(self, x, y):
        ex = positional_encoding(x, self.sigma, self.e_sz)
        ey = positional_encoding(y, self.sigma, self.e_sz)
        x = self.apply_opp(torch.cat([ex, ey], dim=-1))
        return x

    def loss_function(self, y, y_pred):
        return tf.binary_cross_entropy(y, y_pred)

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


class GameSequencePredictor(LightningModule):

    def __init__(self, init_size: int = 256, extra_param_size: int = 256, latent_size: int = 22, in_channels: int = 5,
                 lr: float = 1e-5, weight_decay: float = 0.0, scheduler_gamma: float = .7,
                 betas: tuple[float, float] = (.9, .99), *args, **kwargs):
        super().__init__()
        self.init_size = init_size
        self.latent_size = latent_size
        self.output_size = 1
        self.automatic_optimization = False
        self.lr = lr
        self.weight_decay = weight_decay
        self.scheduler_gamma = scheduler_gamma
        self.betas = betas
        self.encode0 = nn.Sequential(
            nn.Linear(self.init_size, latent_size),
            nn.SELU(),
            nn.Dropout(.5),
            nn.Conv1d(in_channels, 1, 1, 1, 0),
            nn.SELU(),
            nn.Dropout(.5),
        )

        self.encode0.apply(init_weights)

        self.extra_layer = nn.Sequential(
            nn.Linear(extra_param_size, latent_size),
            nn.SELU(),
            nn.Dropout(.5),
        )

        self.extra_layer.apply(init_weights)

        self.classifier = nn.Sequential(
            nn.Linear(latent_size * 4, 1),
            nn.Sigmoid()
        )

        self.classifier.apply(init_weights)

        dnn_to_bnn(self.classifier, bnn_prior_parameters={
            "prior_mu": 0.0,
            "prior_sigma": .33,
            "posterior_mu_init": 0.0,
            "posterior_rho_init": -4.0,
            "type": "Reparameterization",  # Flipout or Reparameterization
            "moped_enable": False,
        })

        self.latent = latent_size

    def forward(self, x, y, tav, oav):
        x = self.encode(x)
        x = torch.cat([x.squeeze(1), self.extra_layer(tav)], dim=-1)
        y = self.encode(y)
        y = torch.cat([y.squeeze(1), self.extra_layer(oav)], dim=-1)
        x = self.classifier(torch.cat([x, y], dim=-1))
        return x.squeeze(1)

    def encode(self, x):
        # positional_encoding(x, self.sigma, self.e_sz)
        x = self.encode0(x)
        return x

    def loss_function(self, y_pred, y):
        return tf.binary_cross_entropy(y_pred, y)

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
        team, opp, t_av, o_av, targets = batch

        results = self.forward(team, opp, t_av, o_av)
        train_loss = self.loss_function(results, targets)

        self.log_dict({f'{kind}_loss': train_loss}, on_epoch=True,
                      prog_bar=True, rank_zero_only=True)
        return train_loss


class TournamentPredictor(LightningModule):

    def __init__(self, init_size: int = 256, latent_size: int = 22, in_channels: int = 5, lr: float = 1e-5, weight_decay: float = 0.0,
                 encoded_sz: int = 10, sigma: float = 10., scheduler_gamma: float = .7, betas: tuple[float, float] = (.9, .99), *args, **kwargs):
        super().__init__()
        self.init_size = init_size
        self.latent_size = latent_size
        self.output_size = 1
        self.automatic_optimization = False
        self.lr = lr
        self.weight_decay = weight_decay
        self.scheduler_gamma = scheduler_gamma
        self.betas = betas
        self.encoded_size = self.init_size * 2 * encoded_sz
        self.e_sz = encoded_sz
        self.sigma = sigma
        self.encoding_layer = nn.Sequential(
            nn.Linear(self.encoded_size, latent_size),
            nn.SiLU(),
            nn.Dropout(.25),
            nn.Conv1d(in_channels, 1, 3, 1, 1),
            nn.SiLU(),
            nn.Dropout(.25),
            nn.Linear(latent_size, latent_size),
        )

        self.classifier = nn.Sequential(
            nn.Linear(latent_size * 2 + 3, latent_size),
            nn.SiLU(),
            nn.Linear(latent_size, 1),
            nn.Sigmoid()
        )

        self.latent = latent_size

    def forward(self, x, y, loc):
        x = self.encode(x)
        y = self.encode(y)
        x = self.classifier(torch.cat([x.squeeze(1), y.squeeze(1), loc], dim=-1))
        return x.squeeze(1)

    def encode(self, x):
        x = positional_encoding(x, self.sigma, self.e_sz)
        return self.encoding_layer(x)

    def loss_function(self, y, y_pred):
        return tf.binary_cross_entropy(y, y_pred)

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
        team, opp, loc, targets = batch

        results = self.forward(team, opp, loc)
        train_loss = self.loss_function(results, targets)

        self.log_dict({f'{kind}_loss': train_loss}, on_epoch=True,
                      prog_bar=True, rank_zero_only=True)
        return train_loss


def positional_encoding(
        v: Tensor,
        sigma: float,
        m: int) -> Tensor:
    r"""Computes :math:`\gamma(\mathbf{v}) = (\dots, \cos{2 \pi \sigma^{(j/m)} \mathbf{v}} , \sin{2 \pi \sigma^{(j/m)} \mathbf{v}}, \dots)`
        where :math:`j \in \{0, \dots, m-1\}`

    Args:
        v (Tensor): input tensor of shape :math:`(N, *, \text{input_size})`
        sigma (float): constant chosen based upon the domain of :attr:`v`
        m (int): [description]

    Returns:
        Tensor: mapped tensor of shape :math:`(N, *, 2 \cdot m \cdot \text{input_size})`

    See :class:`~rff.layers.PositionalEncoding` for more details.
    """
    j = torch.arange(m, device=v.device)
    coeffs = 2 * np.pi * sigma ** (j / m)
    vp = coeffs * torch.unsqueeze(v, -1)
    vp_cat = torch.cat((torch.cos(vp), torch.sin(vp)), dim=-1)
    return vp_cat.flatten(-2, -1)
