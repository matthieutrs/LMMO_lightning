import os
import pytorch_lightning as pl
from lightning_denoiser import Denoiser
from data_module import DataModule
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from argparse import ArgumentParser

if __name__ == '__main__':

    class KnownNamespace(object):
        pass

    known_namespace = KnownNamespace()

    # PROGRAM args
    parser = ArgumentParser()
    parser.add_argument('--model_name', type=str, default='GS_DRUNet')
    parser.add_argument('--optimizer_lr', type=float, default=1e-4)
    parser.add_argument('--loss_name', type=str, default='l2')
    parser.add_argument('--save_images', dest='save_images', action='store_true')
    parser.add_argument('--log_folder', type=str, default='/scratch/space1/dc153/logs_experiments/logs_LMMO/')
    parser.add_argument('--pretrained_checkpoint', default=None)  # 'ckpts/GS_DRUNet_new.ckpt'
    parser.add_argument('--act_mode', type=str, default='s')  # Nonlinearities
    parser.add_argument('--min_sigma_train', type=int, default=180)
    parser.add_argument('--max_sigma_train', type=int, default=180)
    parser.add_argument('--nc_in', type=int, default=3)
    parser.add_argument('--nc_out', type=int, default=3)
    parser.add_argument('--sigma_list_test', type=int, nargs='+', default=[180])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--filt_shape', type=int, default=3)  # ONLY USEFUL FOR QNN
    parser.add_argument('--num_filt', type=int, default=64)  # ONLY USEFUL FOR QNN
    parser.set_defaults(save_images=True)  # Used to be False

    parser.parse_known_args(namespace=known_namespace)
    if 'DRUNet' in known_namespace.model_name:
        parser.add_argument('--batch_size_train', type=int, default=16)
        parser.add_argument('--train_patch_size', type=int, default=128)
        parser.add_argument('--gradient_clip_val', type=float, default=1e-2)
        parser.add_argument('--scheduler_milestones', type=int, nargs='+', default=[365, 365*2, 365*3, 365*4, 365*5, 365*6, 365*7, 365*8])
        parser.add_argument('--scheduler_gamma', type=float, default=0.5)
    elif 'DNCNN' in known_namespace.model_name:
        parser.add_argument('--batch_size_train', type=int, default=128)
        parser.add_argument('--train_patch_size', type=int, default=64)
        parser.add_argument('--gradient_clip_val', type=float, default=1e-2)
        parser.add_argument('--scheduler_milestones', type=int, nargs='+', default=[365, 365*2, 365*3, 365*4, 365*5, 365*6, 365*7, 365*8])
        parser.add_argument('--scheduler_gamma', type=float, default=0.5)
    if 'BF' in known_namespace.model_name:
        parser.set_defaults(bias=False)
    else:
        parser.set_defaults(bias=True)

    # MODEL args
    parser = Denoiser.add_model_specific_args(parser)
    # DATA args
    parser = DataModule.add_data_specific_args(parser)
    # OPTIM args
    parser = Denoiser.add_optim_specific_args(parser)

    hparams = parser.parse_args()

    model_name = hparams.model_name + '_' + hparams.act_mode

    pl.seed_everything(hparams.seed)  # Used to use native random package, but same seed between epochs

    str_filts = ''  # To add a manual identifier if needed

    if hparams.pretrained_checkpoint is None:
        id_str = model_name + '_nch_' + str(hparams.nc_in) + '_seed_' + str(hparams.seed) + str_filts + '_ljr_' \
                 + str(hparams.jacobian_loss_weight) + '_jt_'+hparams.jacobian_compute_type \
                 + '_nit_' + str(hparams.power_method_nb_step) + '_loss_' + str(hparams.loss_name)\
                 + '_lr_' + str(hparams.optimizer_lr) + '_sigma_' + str(hparams.min_sigma_train)\
                 + '-' + str(hparams.max_sigma_train) + '_single'
    else:
        id_str = model_name + '_PRE_nch_' + str(hparams.nc_in) + str_filts + '_ljr_' \
                     + str(hparams.jacobian_loss_weight) + '_jt_' + hparams.jacobian_compute_type \
                     + '_nit_' + str(hparams.power_method_nb_step) + '_loss_' + str(hparams.loss_name) \
                     + '_lr_' + str(hparams.optimizer_lr) + '_sigma_' + str(hparams.min_sigma_train)\
                     + '-' + str(hparams.max_sigma_train) + '_single'


    # Logging info
    if not os.path.exists(hparams.log_folder):
        os.mkdir(hparams.log_folder)
    log_path = hparams.log_folder + '/' + hparams.model_name
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    tb_logger = pl_loggers.TensorBoardLogger(log_path, version=id_str)

    # Saving some training samples for inspection
    if not os.path.exists('images/'+model_name):
        if not os.path.exists('images/'):
            os.mkdir('images/')
        os.mkdir('images/'+model_name)

    model = Denoiser(hparams)
    dm = DataModule(hparams)

    early_stop_callback = EarlyStopping(
        monitor='val/avg_val_loss',
        min_delta=0.00,
        patience=hparams.early_stopping_patiente,
        verbose=True,
        mode='min'
    )
    from pytorch_lightning.callbacks import LearningRateMonitor
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    max_epochs = 3000  # DRUNet training

    trainer = pl.Trainer.from_argparse_args(hparams, logger=tb_logger, gpus=-1,
                                            val_check_interval=hparams.val_check_interval,
                                            gradient_clip_val=hparams.gradient_clip_val,
                                            gradient_clip_algorithm="value",
                                            max_epochs=max_epochs, precision=32,
                                            callbacks=[lr_monitor], strategy='ddp', log_every_n_steps=1)

    trainer.fit(model, dm, ckpt_path=hparams.pretrained_checkpoint)



