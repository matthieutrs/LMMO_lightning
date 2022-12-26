import torch

pth_ckpt_orig = 'my/logs/folder/logs/DRUNet/lightning_logs/'
pth_target = '/some/folder/logs_experiments/currated/'

# Initial checkpoint file (.ckpt extension)
pth_ckpt_orig_spe = pth_ckpt_orig+'checkpoints/epoch=2999-step=822000.ckpt'

# Load checkpoint
ckpt_orig = torch.load(pth_ckpt_orig_spe, map_location=torch.device('cpu'))

# Remove 'denoiser.model.' in the keys of the dict
new_weights = dict((key.replace('denoiser.model.', ''), value) for (key, value) in ckpt_orig['state_dict'].items())
new_dict = {'state_dict': new_weights}

# Save the new dictionary
torch.save(new_dict, pth_target+'out.ckpt')
