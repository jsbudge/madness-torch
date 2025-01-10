from typing import Any

import numpy as np
import torch
from pytorch_lightning import LightningModule
from torch import nn, optim, Tensor
from torch.nn import functional as tf


class Encoder(LightningModule):

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.init_size = 256
        self.latent_size = 22
        self.automatic_optimization = False

        self.encoder = nn.Sequential(
            nn.Linear(self.init_size, self.latent_size),
            nn.GELU(),
            nn.Linear(self.latent_size, self.latent_size),
            nn.GELU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_size, self.latent_size),
            nn.GELU(),
            nn.Linear(self.latent_size, self.init_size),
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x_hat):
        return self.decoder(x_hat)

    def forward(self, x):
        return self.decode(self.encode(x))

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

    def on_after_backward(self) -> None:
        if self.trainer.is_global_zero and self.global_step % 100 == 0 and self.logger:
            for name, params in self.named_parameters():
                self.logger.experiment.add_histogram(name, params, self.global_step)

    def on_train_epoch_end(self) -> None:
        sch = self.lr_schedulers()

    def on_validation_epoch_end(self) -> None:
        sch = self.lr_schedulers()

        # If the selected scheduler is a ReduceLROnPlateau scheduler.
        if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
            sch.step(self.trainer.callback_metrics["val_loss"])
            self.log('LR', sch.get_last_lr()[0], rank_zero_only=True)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(),
                                lr=self.params['LR'],
                                weight_decay=self.params['weight_decay'],
                                betas=self.params['betas'],
                                eps=1e-7)
        optims = [optimizer]
        if self.params['scheduler_gamma'] is None:
            return optims
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optims[0], cooldown=self.params['step_size'],
                                                         factor=self.params['scheduler_gamma'], threshold=1e-5)
        scheds = [scheduler]

        return optims, scheds

    def train_val_get(self, batch, batch_idx, kind='train'):
        img, img = batch

        results = self.forward(img)
        train_loss = self.loss_function(results, img)

        self.log_dict({f'{kind}_loss': train_loss}, on_epoch=True,
                      prog_bar=True, rank_zero_only=True)
        return train_loss


class Predictor(LightningModule):

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.init_size = 22

        self.team_prep = nn.Sequential(
            nn.Linear(self.init_size, 256),
            nn.GELU(),
        )

        self.combine = nn.Sequential(
            nn.Conv1d(2, 1, 1, 1, 0),
            nn.GELU(),
        )

        self.predict_layer = nn.Sequential(
            nn.Linear(256, 12),
            nn.GELU(),
            nn.Linear(12, 2),
            nn.Softmax(-1),
        )

    def forward(self, x, y):
        x = self.team_prep(x)
        y = self.team_prep(y)
        xy = self.combine(torch.cat([x.unsqueeze(1), y.unsqueeze(1)], dim=1))
        return self.predict_layer(xy.squeeze(1))