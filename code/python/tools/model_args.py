import pytorch_lightning as pl
import argparse
from argparse import ArgumentParser
from pytorch_lightning import Trainer

# Prepare Argparse Namespace with respect to the model, dataloader and experiment
def get_args(model: pl.LightningModule, dataloader: pl.LightningDataModule,
             jupyter_flag=None) -> argparse.Namespace:

    parser = ArgumentParser(add_help=False)
    # add model specific args
    parser = model.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    parser = dataloader.add_argparse_args(parser)
    parser.add_argument('--mode', type=str, default='training')
    parser.add_argument('--month', type=int, default=1)
    parser.add_argument('--num_trees', type=int, default=100)
    parser.add_argument('--continue', type=bool, default=False)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--nonlinear', type=str, default='siren')
    parser.add_argument('--seed', type=int, default=1742)

    parser.add_argument('--w0_initial', type=float, default=15.)
    parser.add_argument('--w0', type=float, default=4.)

    parser.add_argument(
        '--data_path_train',
        type=str,
        default='../../data/training/SWIFT-AI_train_month_01.nc')#SWIFT-AI_train_month_01.nc
    parser.add_argument(
        '--data_path_test',
        type=str,
        default='../../data/testing/SWIFT-AI_test_month_01.nc')#SWIFT-AI_test_month_01.nc
    parser.add_argument(
        '--data_path_meanstddev',
        type=str, 
        default='../../data/mean_stddev_training_data.nc')
    
    parser.add_argument("--features_in", type=str, default='none')
    parser.add_argument("--features_out", type=str, default='none')
    
    parser.add_argument(
            '--project_name',
            type=str,
            default='none')
    parser.add_argument(
            '--run_name',
            type=str
    )
    parser.add_argument(
            '--group_name',
            type=str,
            default='none'),
    parser.add_argument(
            '--job_type',
            type=str,
            default='training'),  

    if jupyter_flag:
        hparams = parser.parse_args([])
    else:
        hparams = parser.parse_args()

    # setup general
    hparams.conda_env = "None"
    hparams.notification_email = "helge.mohn@awi.de"
    hparams.seed = 1742
    hparams.default_root_dir = "wandb"
    # setup LightningModule
    # setup LightningDataLoader
    #hparams.num_traindata = 10000#10e3
    #hparams.num_validata = 100
    # setup Trainer
    hparams.num_nodes = 1
    #hparams.accelerator = 'ddp'
    hparams.num_workers = 8
    #hparams.gpus = 1
    hparams.auto_select_gpus = True
    #hparams.show_progress_bar = True
    hparams.log_every_n_steps = 100
    #hparams.flush_logs_every_n_steps = 200
    hparams.val_check_interval = 0.01
    return hparams