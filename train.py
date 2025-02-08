import torch
from pytorch_lightning import Trainer, loggers, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, StochasticWeightAveraging, ModelCheckpoint
from dataloader import EncoderDataModule, PredictorDataModule
from model import Encoder, Predictor
from sklearn.decomposition import KernelPCA
import numpy as np
from tqdm import tqdm
import itertools
import yaml


if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    gpu_num = 0
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
    config['model']['init_size'] = data.train_dataset.data_len
    model = Encoder(**config['model'])
    logger = loggers.TensorBoardLogger(config['training']['log_dir'], name=config['model']['name'])
    expected_lr = max((config['model']['lr'] * config['model']['scheduler_gamma'] ** (config['training']['max_epochs'] *
                                                                config['training']['swa_start'])), 1e-9)
    trainer = Trainer(logger=logger, max_epochs=config['training']['max_epochs'],
                      default_root_dir=config['training']['weights_path'],
                      log_every_n_steps=config['training']['log_epoch'], devices=[gpu_num], callbacks=
                      [EarlyStopping(monitor='train_loss', patience=config['training']['patience'],
                                     check_finite=True),
                       StochasticWeightAveraging(swa_lrs=expected_lr,
                                                 swa_epoch_start=config['training']['swa_start']),
                       ModelCheckpoint(monitor='train_loss')])

    print("======= Training =======")
    try:
        if config['training']['warm_start']:
            trainer.fit(model, ckpt_path=f'{config['training']['weights_path']}/{config['model']['name']}.ckpt',
                        datamodule=data)
        else:
            trainer.fit(model, datamodule=data)
    except KeyboardInterrupt:
        if trainer.is_global_zero:
            print('Training interrupted.')
        else:
            print('adios!')
            exit(0)
    if config['training']['save_model']:
        trainer.save_checkpoint(f'{config['training']['weights_path']}/{config['model']['name']}.ckpt')