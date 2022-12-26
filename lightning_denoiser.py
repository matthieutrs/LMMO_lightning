import os
import torch
from torch import nn
import pytorch_lightning as pl
from torch.optim import Adam
from torch.optim import lr_scheduler
import random
import torchmetrics
try:  # version issues
    from torchmetrics import PSNR
except:
    from torchmetrics import PeakSignalNoiseRatio as PSNR
from argparse import ArgumentParser
import cv2
import torchvision
from test_utils import test_mode
from models.network_unet import UNetRes
from models.DNCNN import DnCNN, weights_init_kaiming


class DenoisingModel(pl.LightningModule):
    '''
    Standard Denoiser model
    '''
    def __init__(self, model_name, pretrained, pretrained_checkpoint, act_mode, DRUNet_nb, bias, nc_in=3, nc_out=3):
        super().__init__()
        self.model_name = model_name
        if 'DRUNet' in self.model_name:
            self.model = UNetRes(in_nc=nc_in+1, out_nc=nc_out, nc=[64, 128, 256, 512], nb=DRUNet_nb, act_mode=act_mode,
                                 downsample_mode='strideconv', upsample_mode='convtranspose')
        elif 'DNCNN' in self.model_name:
            if 'BF' in self.model_name:
                bias = False
            self.model = DnCNN(nc_in, nc_out, 20, act_mode, bias=bias)   # Modified depth to 20 because variable noise level
            self.model.apply(weights_init_kaiming)
        self.model.to(self.device)
        if pretrained:
            checkpoint = torch.load(pretrained_checkpoint, map_location=self.device)
            state_dict = checkpoint['state_dict']
            new_state_dict = {}
            for key, val in state_dict.items():
                new_state_dict[key[6:]] = val
            self.model.load_state_dict(new_state_dict, strict=False)

    def forward(self, x, sigma):
        if 'DRUNet' in self.model_name:
            noise_level_map = torch.FloatTensor(x.size(0), 1, x.size(2), x.size(3)).fill_(sigma).to(self.device)
            x = torch.cat((x, noise_level_map), 1)
        out = self.model(x)
        return out


def normalize_min_max(A):
    '''
    Required for Gradient Step Denoiser
    '''
    AA = A.clone()
    AA = AA.view(A.size(0), -1)
    AA -= AA.min(1, keepdim=True)[0]
    AA /= AA.max(1, keepdim=True)[0]
    AA = AA.view(A.size())
    return AA


class Denoiser(pl.LightningModule):
    '''
    Gradient Step Denoiser
    '''

    def __init__(self, hparams):
        super().__init__()

        self.save_hyperparameters(hparams)
        self.denoiser = DenoisingModel(self.hparams.model_name, self.hparams.pretrained_student,
                                       self.hparams.pretrained_checkpoint, self.hparams.act_mode,
                                       self.hparams.DRUNet_nb, self.hparams.bias,
                                       nc_in=self.hparams.nc_in, nc_out=self.hparams.nc_out,
                                       filt_shape=self.hparams.filt_shape, num_filt=self.hparams.num_filt)
        if 'GS_' in self.hparams.model_name:
            self.hparams.grad_matching = True
        self.train_PSNR = PSNR(data_range=1.0)
        self.val_PSNR = PSNR(data_range=1.0)
        self.train_teacher_PSNR = PSNR(data_range=1.0)

    def forward(self, x, sigma):
        '''
        Denoising (either Gradient Step Denoiser or regular denoising)
        :param x:  torch.tensor input image
        :param sigma: Denoiser level (std)
        :return: Denoised image x_hat, Dg(x) gradient of the regularizer g at x
        '''
        if 'DNCNN' in self.denoiser.model_name or 'QNN' in self.denoiser.model_name:
            x_hat = self.denoiser.forward(x, sigma)
        elif 'DRUNet' in self.denoiser.model_name:
            x_hat = self.denoiser.forward(x, sigma)
        Dg = x - x_hat
        return x_hat, Dg

    def lossfn(self, x, y):  # L2 or L1 loss
        if self.hparams.loss_name == 'l2':
            criterion = nn.MSELoss(reduction='none')
            return criterion(x.view(x.size()[0], -1), y.view(y.size()[0], -1)).mean(dim=1)
        if self.hparams.loss_name == 'l1':
            criterion = nn.L1Loss(reduction='none')
            return criterion(x.view(x.size()[0], -1), y.view(y.size()[0], -1)).mean(dim=1)

    def training_step(self, batch, batch_idx):

        y, _ = batch

        sigma = random.uniform(self.hparams.min_sigma_train, self.hparams.max_sigma_train) / 255
        u = torch.randn(y.size(), device=self.device)
        noise_in = u * sigma
        x = y + noise_in
        x_hat, Dg = self.forward(x, sigma)
        loss = self.lossfn(x_hat, y)
        self.train_PSNR.update(x_hat, y)

        if self.hparams.jacobian_loss_weight > 0:
            if self.hparams.jacobian_compute_type == 'nonsymmetric':
                jacobian_norm = self.jacobian_spectral_norm(x[0:1], x_hat[0:1], sigma=sigma, interpolation=False, training=True)
            self.log('train/jacobian_norm_max', jacobian_norm.max(), prog_bar=True)
            if self.hparams.jacobian_loss_type == 'max':
                jacobian_loss = torch.maximum(jacobian_norm, torch.ones_like(jacobian_norm)-self.hparams.eps_jacobian_loss)
            elif self.hparams.jacobian_loss_type == 'exp':
                jacobian_loss = self.hparams.eps_jacobian_loss * torch.exp(jacobian_norm - torch.ones_like(jacobian_norm)*(1+self.hparams.eps_jacobian_loss))  / self.hparams.eps_jacobian_loss
            else:
                print("jacobian loss not available")
            jacobian_loss = torch.clip(jacobian_loss, 0, 1e3)
            self.log('train/jacobian_loss_max', jacobian_loss.max(), prog_bar=True)

            loss = (loss + self.hparams.jacobian_loss_weight * jacobian_loss)

        loss = loss.mean()

        psnr = self.train_PSNR.compute()
        self.log('train/train_loss', loss.detach())
        self.log('train/train_psnr', psnr.detach(), prog_bar=True)

        if batch_idx == 0:
            noisy_grid = torchvision.utils.make_grid(normalize_min_max(x.detach())[:1])
            clean_grid = torchvision.utils.make_grid(normalize_min_max(y.detach())[:1])
            denoised_grid = torchvision.utils.make_grid(normalize_min_max(x_hat.detach())[:1])
            self.logger.experiment.add_image('train/noisy', noisy_grid, self.current_epoch)
            self.logger.experiment.add_image('train/denoised', denoised_grid, self.current_epoch)
            self.logger.experiment.add_image('train/clean', clean_grid, self.current_epoch)

        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx: int, unused: int = 0):
        if ('project' in self.denoiser.model_name) and ('noproj' not in self.denoiser.model_name):
            self.denoiser.model.project_weights()
        return None

    def training_epoch_end(self, outputs):
        print('train PSNR updated')
        self.train_PSNR.reset()

    def validation_step(self, batch, batch_idx):
        y, _ = batch
        batch_dict = {}

        sigma_list = self.hparams.sigma_list_test
        for i, sigma in enumerate(sigma_list):
            x = y + torch.randn(y.size(), device=self.device) * sigma / 255.
            if self.hparams.use_sigma_model:  # Possibility to test with sigma model different than input sigma
                sigma_model = self.hparams.sigma_model / 255.
            else:
                sigma_model = sigma / 255.
            torch.set_grad_enabled(True)
            for n in range(self.hparams.n_step_eval):
                current_model = lambda v: self.forward(v, sigma / 255)[0]
                x_hat = x
                if x.size(2) % 8 == 0 and x.size(3) % 8 == 0:
                    x_hat = current_model(x_hat)
                elif x.size(2) % 8 != 0 or x.size(3) % 8 != 0:
                    x_hat = test_mode(current_model, x_hat, refield=64, mode=5)
            Dg = (x - x_hat)
            Dg_norm = torch.norm(Dg, p=2)
            l = self.lossfn(x_hat, y)
            self.val_PSNR.reset()
            p = self.val_PSNR(x_hat, y)

            if self.hparams.get_spectral_norm:
                jacobian_norm = self.jacobian_spectral_norm(x, x_hat, sigma_model)
                batch_dict["max_jacobian_norm_" + str(sigma)] = jacobian_norm.max().detach()
                batch_dict["mean_jacobian_norm_" + str(sigma)] = jacobian_norm.mean().detach()

            batch_dict["psnr_" + str(sigma)] = p.detach()
            batch_dict["loss_" + str(sigma)] = l.detach()
            batch_dict["Dg_norm_" + str(sigma)] = Dg_norm.detach()

        if batch_idx == 0:  # logging for tensorboard
            clean_grid = torchvision.utils.make_grid(normalize_min_max(y.detach())[:1])
            noisy_grid = torchvision.utils.make_grid(normalize_min_max(x.detach())[:1])
            denoised_grid = torchvision.utils.make_grid(normalize_min_max(x_hat.detach())[:1])
            self.logger.experiment.add_image('val/clean', clean_grid, self.current_epoch)
            self.logger.experiment.add_image('val/noisy', noisy_grid, self.current_epoch)
            self.logger.experiment.add_image('val/denoised', denoised_grid, self.current_epoch)

        if self.hparams.save_images:
            save_dir = 'images/' + self.hparams.model_name

            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
                os.mkdir(save_dir + '/noisy')
                os.mkdir(save_dir + '/denoised')
                os.mkdir(save_dir + '/denoised_no_noise')
                os.mkdir(save_dir + '/clean')
            for i in range(len(x)):
                clean = y[i].detach().cpu().numpy().transpose(1, 2, 0) * 255
                noisy = x[i].detach().cpu().numpy().transpose(1, 2, 0) * 255
                denoised = x_hat[i].detach().cpu().numpy().transpose(1, 2, 0) * 255
                clean = cv2.cvtColor(clean, cv2.COLOR_RGB2BGR)
                noisy = cv2.cvtColor(noisy, cv2.COLOR_RGB2BGR)
                denoised = cv2.cvtColor(denoised, cv2.COLOR_RGB2BGR)

                cv2.imwrite(save_dir + '/denoised/' + str(batch_idx) + '.png', denoised)
                cv2.imwrite(save_dir + '/clean/' + str(batch_idx) + '.png', clean)
                cv2.imwrite(save_dir + '/noisy/' + str(batch_idx) + '.png', noisy)

        return batch_dict

    def validation_epoch_end(self, outputs):

        self.val_PSNR.reset()

        sigma_list = self.hparams.sigma_list_test
        for i, sigma in enumerate(sigma_list):
            res_mean_SN = []
            res_max_SN = []
            res_psnr = []
            res_Dg = []
            if self.hparams.get_regularization:
                res_g = []
            for x in outputs:
                if x["psnr_" + str(sigma)] is not None:
                    res_psnr.append(x["psnr_" + str(sigma)])
                res_Dg.append(x["Dg_norm_" + str(sigma)])
                if self.hparams.get_regularization:
                    res_g.append(x["g_" + str(sigma)])
                if self.hparams.get_spectral_norm:
                    res_max_SN.append(x["max_jacobian_norm_" + str(sigma)])
                    res_mean_SN.append(x["mean_jacobian_norm_" + str(sigma)])
            avg_psnr_sigma = torch.stack(res_psnr).mean()
            avg_Dg_norm = torch.stack(res_Dg).mean()
            if self.hparams.get_regularization:
                avg_s = torch.stack(res_g).mean()
                self.log('val/val_g_sigma=' + str(sigma), avg_s)
            if self.hparams.get_spectral_norm:
                avg_mean_SN = torch.stack(res_mean_SN).mean()
                max_max_SN = torch.stack(res_max_SN).max()
                self.log('val/val_max_SN_sigma=' + str(sigma), max_max_SN)
                self.log('val/val_mean_SN_sigma=' + str(sigma), avg_mean_SN)
            self.log('val/val_psnr_sigma=' + str(sigma), avg_psnr_sigma)
            self.log('val/val_Dg_norm_sigma=' + str(sigma), avg_Dg_norm)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs)

    def configure_optimizers(self):
        optim_params = []
        for k, v in self.denoiser.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                print('Params [{:s}] will not optimize.'.format(k))
        optimizer = Adam(optim_params, lr=self.hparams.optimizer_lr, weight_decay=0)
        scheduler = lr_scheduler.MultiStepLR(optimizer,
                                             self.hparams.scheduler_milestones,
                                             self.hparams.scheduler_gamma)
        return [optimizer], [scheduler]

    def jacobian_spectral_norm(self, y_in, x_hat, sigma, interpolation=True, training=False):
        '''
        Jacobian spectral norm from Pesquet et al; computed with a power iteration method.
        Given a denoiser J, computes the spectral norm of Q = 2J-I where J is the denoising model.

        Inputs:
        :y_in: point where the jacobian is to be computed, typically a noisy image (torch Tensor)
        :x_hat: denoised image (unused if interpolation = False) (torch Tensor)
        :sigma: noise level
        :interpolation: whether to compute the jacobian only at y_in, or somewhere on the segment [x_hat, y_in].
        :training: set to True during training to retain grad appropriately
        Outputs:
        :z.view(-1): the square of the Jacobian spectral norm of (2J-Id)

        Beware: reversed usage compared to the original Pesquet et al code.
        '''

        if interpolation:
            eta = torch.rand(y_in.size(0), 1, 1, 1, requires_grad=True).to(self.device)
            x = eta * y_in.detach() + (1 - eta) * x_hat.detach()
            x = x.to(self.device)
        else:
            x = y_in

        x.requires_grad_()
        x_hat, _ = self.forward(x, sigma)

        y = 2.*x_hat-y_in  # Beware notation : y_in = input, x_hat = output network

        u = torch.randn_like(x)
        u = u / torch.norm(u, p=2)

        z_old = torch.zeros(u.shape[0])

        for it in range(self.hparams.power_method_nb_step):

            w = torch.ones_like(y, requires_grad=True)  # Double backward trick. From https://gist.github.com/apaszke/c7257ac04cb8debb82221764f6d117ad
            v = torch.autograd.grad(torch.autograd.grad(y, x, w, create_graph=True), w, u, create_graph=training)[0]  # Ju

            v, = torch.autograd.grad(y, x, v, retain_graph=True, create_graph=True)  # vtJt

            z = torch.matmul(u.reshape(u.shape[0], 1, -1), v.reshape(v.shape[0], -1, 1)) / torch.matmul(
                u.reshape(u.shape[0], 1, -1), u.reshape(u.shape[0], -1, 1))

            if it > 0:
                rel_var = torch.norm(z - z_old)
                if rel_var < self.hparams.power_method_error_threshold:
                    break
            z_old = z.clone()

            u = v / torch.norm(v, p=2)  # Modified

            if self.eval:
                w.detach_()
                v.detach_()
                u.detach_()

        return z.view(-1)


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--start_from_checkpoint', dest='start_from_checkpoint', action='store_true')
        parser.set_defaults(start_from_checkpoint=False)
        parser.add_argument('--pretrained_student', dest='pretrained_student', action='store_true')
        parser.set_defaults(pretrained_student=False)
        parser.add_argument('--n_channels', type=int, default=3)
        parser.add_argument('--nc', type=int, default=64)
        parser.add_argument('--nb', type=int, default=20)
        parser.add_argument('--no_bias', dest='no_bias', action='store_false')
        parser.set_defaults(use_bias=True)
        parser.add_argument('--power_method_nb_step', type=int, default=50)
        parser.add_argument('--power_method_error_threshold', type=float, default=1e-2)
        parser.add_argument('--power_method_error_momentum', type=float, default=0.)
        parser.add_argument('--power_method_mean_correction', dest='power_method_mean_correction', action='store_true')
        parser.add_argument('--DRUNet_nb', type=int, default=2)
        parser.set_defaults(power_method_mean_correction=False)
        parser.add_argument('--no_grad_matching', dest='grad_matching', action='store_false')
        parser.set_defaults(grad_matching=False)
        parser.add_argument('--weight_Ds', type=float, default=1.)
        return parser

    @staticmethod
    def add_optim_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--optimizer_type', type=str, default='adam')
        parser.add_argument('--scheduler_type', type=str, default='MultiStepLR')
        parser.add_argument('--early_stopping_patiente', type=str, default=5)
        parser.add_argument('--val_check_interval', type=float, default=1.0)
        parser.add_argument('--check_val_every_n_epoch', type=int, default=20)
        parser.add_argument('--min_sigma_test', type=int, default=0)
        parser.add_argument('--max_sigma_test', type=int, default=50)
        parser.add_argument('--sigma_step', dest='sigma_step', action='store_true')
        parser.set_defaults(sigma_step=False)
        parser.add_argument('--get_spectral_norm', dest='get_spectral_norm', action='store_true')
        parser.set_defaults(get_spectral_norm=True)
        parser.add_argument('--jacobian_loss_weight', type=float, default=0)
        parser.add_argument('--jacobian_compute_type', type=str, default='nonsymmetric')
        parser.add_argument('--eps_jacobian_loss', type=float, default=0.1)
        parser.add_argument('--jacobian_loss_type', type=str, default='max')
        parser.add_argument('--n_step_eval', type=int, default=1)
        parser.add_argument('--use_post_forward_clip', dest='use_post_forward_clip', action='store_true')
        parser.set_defaults(use_post_forward_clip=False)
        parser.add_argument('--use_sigma_model', dest='use_sigma_model', action='store_true')
        parser.set_defaults(use_sigma_model=False)
        parser.add_argument('--sigma_model', type=int, default=25)
        parser.add_argument('--get_regularization', dest='get_regularization', action='store_true')
        parser.set_defaults(get_regularization=False)
        return parser
