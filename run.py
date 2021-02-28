from argparse import ArgumentParser, Namespace
import hashlib
import json
from models import *
import numpy as np
import os
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TestTubeLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import torch.backends.cudnn as cudnn
import yaml


def get_experiment_hash(hparams):
    """Generates a unique hash-value depending on the provided experimental parameters."""
    experiment_parameters = Namespace(**vars(hparams))

    return hashlib.md5(json.dumps(vars(experiment_parameters), sort_keys=True).encode('utf-8')).hexdigest()


if __name__ == '__main__':
    """This is the main function to run experiments. The most convenient way to use it, is via calling
        
        $ python run.py --config /path/to/config-file --data_root /path/to/data-folder
        
    See the description of additional parameters below.
    """
    parser = ArgumentParser(description='Generic runner for sound event localization models.')

    parser.add_argument('--config', '-c', dest='filename', metavar='FILE', help='Path to config file')
    parser.add_argument('--data_root', default='./data', help='Path to database')
    parser.add_argument('--tmp_dir', default='./tmp', help='Path to store temporary files')
    parser.add_argument('--log_dir', default='./experiments', help='Path to store models and experiment logs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of parallel subprocesses')

    args = parser.parse_args()

    with open(args.filename, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    cudnn.deterministic = True
    cudnn.benchmark = False

    hparams = Namespace(**config)

    experiment_hash = get_experiment_hash(hparams)
    experiment_dir = os.path.join(args.log_dir, experiment_hash)

    results_dir = os.path.join(experiment_dir, 'results')

    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    hparams.data_root = args.data_root
    hparams.tmp_dir = args.tmp_dir
    hparams.logging_dir = args.log_dir
    hparams.batch_size = args.batch_size
    hparams.num_workers = args.num_workers
    hparams.results_dir = results_dir

    for dataset in config['dataset']:
        dataset_path = os.path.join(args.data_root, dataset)

        for cv_fold_idx in range(1, 4):
            torch.manual_seed(config['manual_seed'])
            np.random.seed(config['manual_seed'])

            model = models[config['name']](dataset_path, cv_fold_idx, hparams)

            model_name = '_'.join([config['name'], dataset, 'fold' + str(cv_fold_idx)])

            print('Model: ' + model_name + ', Num. parameters: ' + str(sum(p.numel() for p in model.parameters() if p.requires_grad)))

            logger = TestTubeLogger(save_dir=os.path.join(experiment_dir, 'logs'),
                                    name=model_name,
                                    debug=False,
                                    create_git_tag=False,
                                    version=0)
            checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(experiment_dir, 'checkpoints'),
                                                  filename=config['name'] + '_' + str(cv_fold_idx) + '.ckpt',
                                                  monitor='val_loss',
                                                  mode='min')

            trainer_args = {
                'max_epochs': config['max_epochs'],
                'gradient_clip_val': config['gradient_clip_val'],
                'logger': logger,
                'checkpoint_callback': checkpoint_callback,
                'progress_bar_refresh_rate': 5,
            }

            if torch.cuda.is_available():
                device = 'cuda'
                trainer = Trainer(gpus=1, **trainer_args)
            else:
                device = 'cpu'
                trainer = Trainer(**trainer_args)

            if not os.path.isfile(os.path.join(results_dir, model_name + '.json')):
                trainer.fit(model.to(device))
                trainer.test(model.to(device))
