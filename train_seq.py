from glob import glob
from pathlib import Path
import pandas as pd
import torch
from pytorch_lightning import Trainer, loggers, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, StochasticWeightAveraging, ModelCheckpoint
from dataloader import GameDataModule
from model import GameSequencePredictor
import numpy as np
from tqdm import tqdm
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

    # First, run the encoding to try and reduce the dimension of the data
    data = GameDataModule(**config['dataloader'])
    data.setup()

    # Get the model, experiment, logger set up
    config['seq_predictor']['init_size'] = data.train_dataset.data_len
    mdl_name = f"{config['seq_predictor']['name']}"
    model = GameSequencePredictor(**config['seq_predictor'])
    logger = loggers.TensorBoardLogger(config['seq_predictor']['training']['log_dir'], name=mdl_name)
    expected_lr = max((config['seq_predictor']['lr'] * config['seq_predictor']['scheduler_gamma'] ** (config['seq_predictor']['training']['max_epochs'] *
                                                                config['seq_predictor']['training']['swa_start'])), 1e-9)
    trainer = Trainer(logger=logger, max_epochs=config['seq_predictor']['training']['max_epochs'],
                      default_root_dir=config['seq_predictor']['training']['weights_path'],
                      log_every_n_steps=config['seq_predictor']['training']['log_epoch'],
                      num_sanity_val_steps=0, detect_anomaly=False, callbacks=
                      [EarlyStopping(monitor='train_loss', patience=config['seq_predictor']['training']['patience'],
                                     check_finite=True),
                       StochasticWeightAveraging(swa_lrs=expected_lr,
                                                 swa_epoch_start=config['seq_predictor']['training']['swa_start']),
                       ModelCheckpoint(monitor='train_loss')])

    print("======= Training =======")
    try:
        if config['seq_predictor']['training']['warm_start']:
            trainer.fit(model, ckpt_path=f"{config['seq_predictor']['training']['weights_path']}/{mdl_name}.ckpt",
                        datamodule=data)
        else:
            trainer.fit(model, datamodule=data)
    except KeyboardInterrupt:
        if trainer.is_global_zero:
            print('Training interrupted.')
        else:
            print('adios!')
            exit(0)
    if config['seq_predictor']['training']['save_model']:
        trainer.save_checkpoint(f"{config['seq_predictor']['training']['weights_path']}/{mdl_name}.ckpt")

    start = 2004
    end = 2024
    datapath = config['dataloader']['datapath']
    results = pd.DataFrame()
    for season in tqdm(range(start, end)):
        dp = f'{datapath}/t{season}'
        if Path(f'{datapath}/{season}').exists():
            files = glob(f'{dp}/*.pt')
            if len(files) > 0:
                ch_d = [torch.load(g) for g in files]
                t_data = torch.cat([c[0].unsqueeze(0) for c in ch_d], dim=0)
                o_data = torch.cat([c[1].unsqueeze(0) for c in ch_d], dim=0)
                locs = torch.cat([c[2].unsqueeze(0) for c in ch_d], dim=0)
                targets = np.array([c[3] for c in ch_d])
                predictions = model(t_data, o_data, locs).detach().numpy()

                file_data = [Path(c).stem for c in files]
                gid = [int(c.split('_')[0]) for c in file_data]
                tid = [int(c.split('_')[1]) for c in file_data]
                oid = [int(c.split('_')[2]) for c in file_data]
                seas = [season for _ in file_data]
                results = pd.concat((results,
                                     pd.DataFrame(data=np.stack([gid, seas, tid, oid, predictions, targets]).T,
                                                  columns=['gid', 'season', 'tid', 'oid', 'Res', 'truth'])))
    results = results.set_index(['gid', 'season', 'tid', 'oid'])
    corrects = sum(np.round(results['Res']) - results['truth'] == 0) / results.shape[0]
    # config.season
    print(f'{corrects} correct.')



