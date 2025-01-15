import torch
from pytorch_lightning import Trainer, loggers, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, StochasticWeightAveraging, ModelCheckpoint
from dataloader import EncoderDataModule
from model import Encoder, Predictor
from sklearn.decomposition import KernelPCA
import numpy as np
from tqdm import tqdm
import itertools
import yaml


if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    gpu_num = 1
    device = f'cuda:{gpu_num}' if torch.cuda.is_available() else 'cpu'
    seed_everything(np.random.randint(1, 2048), workers=True)

    with open('./run_params.yaml', 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    # seed_everything(43, workers=True)

    data = EncoderDataModule(**config['dataloader'])
    data.setup()

    # Get the model, experiment, logger set up
    model = Encoder(**config['model'])
    logger = loggers.TensorBoardLogger(config['training']['log_dir'], name=config['model']['model_name'])
    expected_lr = max((config['model']['lr'] * target_config.scheduler_gamma ** (target_config.max_epochs *
                                                                target_config.swa_start)), 1e-9)
    trainer = Trainer(logger=logger, max_epochs=target_config.max_epochs, default_root_dir=target_config.weights_path,
                      log_every_n_steps=target_config.log_epoch, devices=[gpu_num], callbacks=
                      [EarlyStopping(monitor='train_loss', patience=target_config.patience,
                                     check_finite=True),
                       StochasticWeightAveraging(swa_lrs=expected_lr,
                                                 swa_epoch_start=target_config.swa_start),
                       ModelCheckpoint(monitor='train_loss')])

    print("======= Training =======")
    try:
        if target_config.warm_start:
            trainer.fit(model, ckpt_path=f'{target_config.weights_path}/{target_config.model_name}.ckpt',
                        datamodule=data)
        else:
            trainer.fit(model, datamodule=data)
    except KeyboardInterrupt:
        if trainer.is_global_zero:
            print('Training interrupted.')
        else:
            print('adios!')
            exit(0)
    if target_config.save_model:
        trainer.save_checkpoint(f'{target_config.weights_path}/{target_config.model_name}.ckpt')