# Training firmly nonexpansive denoisers

This repository is an updated version of [our training codes](https://github.com/basp-group/PnP-MMO-imaging) based on the training framework and dataset of [Hurault et al](https://github.com/samuro95/GSPnP).

## Summary
We train a denoiser while imposing a firm-nonexpansiveness penalization through a Jacobian regularization term.
Given the denoiser `J`, we impose a 1-Lipschitz regularization on `Q = 2J-I`, where `I` is the identity.
The Lipschitz constant of `Q` is estimated via the spectral norm of its Jacobian.

New weights for a DnCNN trained with small noise level (2.0/255.) are currently available [here](https://drive.google.com/drive/folders/1A6SN5yZEiXxzdp-NKtEGB4ZutBI7t0ea).

This repository only focuses on training denoisers; for the PnP algorithm, we refer you to [our older code](https://github.com/basp-group/PnP-MMO-imaging). 
However, a small **google colab notebook** with a demo PnP-FB example is available online. Try it out [here](https://colab.research.google.com/drive/1pVNl4VhDLaYMC7KOyL8f7Zyv-zygM4vK#scrollTo=QlNANCmQkbUu)!

**Link to paper: [https://arxiv.org/pdf/2012.13247.pdf](https://arxiv.org/pdf/2012.13247.pdf)**

### Requirements

This code was tested in the following configuration:
```bash
- numpy = 1.23.4
- torch = 1.13.0
- torchvision = 0.14.0
- pytorch-lightning = 1.8.1
- torchmetrics = 0.11.0
- opencv-python = 4.6.0.66
```

## Training 

- Download training dataset from https://drive.google.com/file/d/1WVTgEBZgYyHNa2iVLUYwcrGWZ4LcN4--/view?usp=sharing and unzip ```DRUNET``` in a ```somewhere/datasets``` folder. *Important:* Update the path accordingly [here](https://github.com/matthieutrs/training_FNE_denoisers_dana/blob/main/train_denoisers/data_module.py#L120).

- To train with the regularised loss:
```
python3 main_train.py --model_name DNCNN \
                      --nc_in 3 --nc_out 3 \
                      --min_sigma_train 15 --max_sigma_train 15 --sigma_list_test 15 \
                      --jacobian_loss_weight 1e-2 --jacobian_compute_type 'nonsymmetric' --power_method_nb_step 20 \
                      --act_mode 'R' --optimizer_lr 1e-4 --loss_name 'l1'
```

### Monitoring training with `tensorboard`
Training logs are saved in `hparams.log_folder` (`log_folder` argument of the parser). `cd` to this folder and launch
```bash
tensorboard --logdir .
```
If launching on a server, forward this to the appropriate port with the `--port ` argument in the above command.

### Exporting the trained weights to a standard state_dict
Saved checkpoints (`.ckpt` files) follow the structure of a (DenoisingModel)[] class. In order to export them to a standard `state_dict` torch nomenclature (more versatile), you can use the template `save_dict.py` code.

## Acknowledgments
This repo contains parts of code taken from : 
- Prox-PnP : https://github.com/samuro95/Prox-PnP
- Deep Plug-and-Play Image Restoration (DPIR) : https://github.com/cszn/DPIR 
- Gradient Step Denoiser for convergent Plug-and-Play (GS-PnP) : https://github.com/samuro95/GSPnP
- Learning Maximally Monotone Operators for image recovery https://github.com/basp-group/PnP-MMO-imaging

### Citation 
This code is based on the following work:
```
@article{pesquet2021learning,
  title={Learning maximally monotone operators for image recovery},
  author={Pesquet, Jean-Christophe and Repetti, Audrey and Terris, Matthieu and Wiaux, Yves},
  journal={SIAM Journal on Imaging Sciences},
  volume={14},
  number={3},
  pages={1206--1237},
  year={2021},
  publisher={SIAM}
}
```

Don't hesitate to contact me if you have any question!

**License:** GNU General Public License v3.0.
