from glob import glob
from pathlib import Path
import pandas as pd
import torch
from pytorch_lightning import Trainer, loggers, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, StochasticWeightAveraging, ModelCheckpoint

from bracket import generateBracket, applyResultsToBracket, scoreBracket
from dataloader import GameDataModule
from load_data import getPossMatches
from model import GameSequencePredictor
import numpy as np
from tqdm import tqdm
import yaml

from prep_data import loadFramesForTorch, formatForTorch

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
    config['seq_predictor']['extra_size'] = data.train_dataset.extra_len
    mdl_name = f"{config['seq_predictor']['name']}"
    model = GameSequencePredictor(**config['seq_predictor'])
    logger = loggers.TensorBoardLogger(config['seq_predictor']['training']['log_dir'], name=mdl_name)
    trainer = Trainer(logger=logger, max_epochs=config['seq_predictor']['training']['max_epochs'],
                      default_root_dir=config['seq_predictor']['training']['weights_path'],
                      log_every_n_steps=config['seq_predictor']['training']['log_epoch'],
                      num_sanity_val_steps=0, detect_anomaly=False, callbacks=
                      [EarlyStopping(monitor='val_loss', patience=config['seq_predictor']['training']['patience'],
                                     check_finite=True),
                       ModelCheckpoint(monitor='val_loss')])

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

    model.eval()

    datapath = config['dataloader']['datapath']
    season = config['dataloader']['season']
    results = pd.DataFrame()
    dp = f'{datapath}/t{season}'
    if Path(f'{datapath}/t{season}').exists():
        files = glob(f'{dp}/*.pt')
        if len(files) > 0:
            ch_d = [torch.load(g) for g in files]
            t_data = torch.cat([c[0].unsqueeze(0) for c in ch_d], dim=0)
            o_data = torch.cat([c[1].unsqueeze(0) for c in ch_d], dim=0)
            tav_data = torch.cat([c[2].unsqueeze(0) for c in ch_d], dim=0)
            oav_data = torch.cat([c[3].unsqueeze(0) for c in ch_d], dim=0)
            targets = np.array([c[4] for c in ch_d])
            predictions = model(t_data, o_data, tav_data, oav_data).detach().numpy()

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


    season = 2023
    adf, avodf = loadFramesForTorch(datapath)
    extra_df, extra0_df = getPossMatches(avodf, season=season, datapath=datapath, gender='M')
    poss_results = pd.DataFrame(index=extra_df.index, columns=['Res'])
    for i in tqdm(range(0, extra_df.shape[0], 128)):
        block = extra_df.iloc[i:i + 128]
        torch_data = []
        for idx in block.index:
            torch_data.append(formatForTorch(adf, extra_df, extra0_df, season, idx, config['seq_predictor']['in_channels'], .5))
        t_data = torch.cat([c[0].unsqueeze(0) for c in torch_data], dim=0)
        o_data = torch.cat([c[1].unsqueeze(0) for c in torch_data], dim=0)
        tav_data = torch.cat([c[2].unsqueeze(0) for c in torch_data], dim=0)
        oav_data = torch.cat([c[3].unsqueeze(0) for c in torch_data], dim=0)
        predictions = 1 - model(t_data, o_data, tav_data, oav_data).detach().numpy()
        poss_results.loc[block.index, 'Res'] = predictions
    truth_br = generateBracket(season, True, datapath=datapath, gender='M')
    test = generateBracket(season, True, datapath=datapath, gender='M')
    res = 0
    if season < 2025:
        for r in range(100):
            test = applyResultsToBracket(test, poss_results, select_random=True, random_limit=.05)
            '''with open(f'./outputs/run_{r}.txt', 'w') as f:
                f.write(str(test))'''
            res += scoreBracket(test, truth_br) / 100
    else:
        test = applyResultsToBracket(test, poss_results, select_random=True, random_limit=.15)



