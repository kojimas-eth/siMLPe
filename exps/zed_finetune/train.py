from pathlib import Path
import os, sys
import argparse
import torch
import numpy as np
import json
import copy
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# 1. Get the path of the current file
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--exp-name', type=str, default='zed_finetune', help='experiment name')
parser.add_argument('--seed', type=int, default=888, help='=seed')
parser.add_argument('--temporal-only', action='store_true', help='=temporal only')
parser.add_argument('--layer-norm-axis', type=str, default='spatial', help='=layernorm axis')
parser.add_argument('--with-normalization', action='store_true', help='=use layernorm')
parser.add_argument('--spatial-fc', action='store_true', help='=use only spatial fc')
parser.add_argument('--num', type=int, default=64, help='=num of blocks')
parser.add_argument('--weight', type=float, default=1., help='=loss weight')

args = parser.parse_args()
# torch.use_deterministic_algorithms(True)
acc_log = open(args.exp_name, 'a')
# torch.manual_seed(args.seed)
writer = SummaryWriter()
acc_log.write(''.join('Seed : ' + str(args.seed) + '\n'))

from exps.baseline_h36m.model import siMLPe as Model
from exps.baseline_h36m.train import get_dct_matrix, update_lr_multistep , gen_velocity



from config import config
config.motion.h36m_target_length = config.motion.h36m_target_length_eval

from datasets.zed import ZEDDataset 
from utils.logger import get_logger, print_and_log_info
from utils.pyt_utils import ensure_dir

dct_m,idct_m = get_dct_matrix(config.motion.h36m_input_length_dct)
dct_m = torch.tensor(dct_m).float().cuda().unsqueeze(0)
idct_m = torch.tensor(idct_m).float().cuda().unsqueeze(0)


def train_step(h36m_motion_input, h36m_motion_target, model, optimizer, nb_iter, total_iter, max_lr, min_lr) :

    if config.deriv_input:
        b,n,c = h36m_motion_input.shape
        h36m_motion_input_ = h36m_motion_input.clone()
        h36m_motion_input_ = torch.matmul(dct_m[:, :, :config.motion.h36m_input_length], h36m_motion_input_.cuda())
    else:
        h36m_motion_input_ = h36m_motion_input.clone()

    motion_pred = model(h36m_motion_input_.cuda())
    motion_pred = torch.matmul(idct_m[:, :config.motion.h36m_input_length, :], motion_pred)

    if config.deriv_output:
        offset = h36m_motion_input[:, -1:].cuda()
        motion_pred = motion_pred[:, :config.motion.h36m_target_length] + offset
    else:
        motion_pred = motion_pred[:, :config.motion.h36m_target_length]

    b,n,c = h36m_motion_target.shape
    motion_pred = motion_pred.reshape(b,n,22,3).reshape(-1,3)
    h36m_motion_target = h36m_motion_target.cuda().reshape(b,n,22,3).reshape(-1,3)
    loss = torch.mean(torch.norm(motion_pred - h36m_motion_target, 2, 1))

    if config.use_relative_loss:
        motion_pred = motion_pred.reshape(b,n,22,3)
        dmotion_pred = gen_velocity(motion_pred)
        motion_gt = h36m_motion_target.reshape(b,n,22,3)
        dmotion_gt = gen_velocity(motion_gt)
        dloss = torch.mean(torch.norm((dmotion_pred - dmotion_gt).reshape(-1,3), 2, 1))
        loss = loss + dloss
    else:
        loss = loss.mean()

    writer.add_scalar('Loss/angle', loss.detach().cpu().numpy(), nb_iter)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    optimizer, current_lr = update_lr_multistep(nb_iter, total_iter, max_lr, min_lr, optimizer)
    writer.add_scalar('LR/train', current_lr, nb_iter)

    return loss.item(), optimizer, current_lr

# 1. Initialize Model
model = Model(config)
model.cuda()

# 2. Load Pretrained Weights
if config.model_pth is not None:
    print(f"Loading pretrained model from {config.model_pth}")
    state_dict = torch.load(config.model_pth, weights_only=True, map_location="cpu")
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    
    for k, v in state_dict.items():
        # Step A: Remove 'module.' prefix (from DataParallel training)
        name = k.replace("module.", "")
        
        # Step B: Rename 'motion_transformer' to 'motion_mlp' (Legacy compatibility)
        if name.startswith("motion_transformer.transformer"):
            name = name.replace("motion_transformer.transformer", "motion_mlp.mlps")
            
        new_state_dict[name] = v
    try:
        model.load_state_dict(new_state_dict, strict=True)
        print("Successfully loaded model weights with strict=True.")
    except RuntimeError as e:
        print(f"Strict loading failed: {e}")
        print("Attempting to load with strict=False...")
        model.load_state_dict(new_state_dict, strict=False)

else:
    print("WARNING: No pretrained model path specified in config. Training from scratch.")

# 3. Setup ZED Dataset
dataset = ZEDDataset(config, 'train', data_aug=True)
dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)

# 4. Optimizer (Fine-tuning usually requires smaller LR, set in config)
optimizer = torch.optim.Adam(model.parameters(), lr=config.cos_lr_max, weight_decay=config.weight_decay)

# 5. Training Loop
model.train()
nb_iter = 0

while nb_iter < config.cos_lr_total_iters:
    for (input_motion, target_motion) in dataloader:
        # Reuse the train_step logic from the original script
        # Ensure 'dct_m' and 'idct_m' are defined or passed correctly if using the imported function
        #Trains step expects input (b,n,22,3)
        loss, optimizer, current_lr = train_step(input_motion, target_motion, model, optimizer, nb_iter, config.cos_lr_total_iters, config.cos_lr_max, config.cos_lr_min)
        
        if nb_iter % config.print_every == 0:
            print(f"Iter {nb_iter}: Loss = {loss}, LR = {current_lr}")

        if nb_iter % config.save_every == 0:
            if not os.path.exists(config.snapshot_dir):
                os.makedirs(config.snapshot_dir, exist_ok=True)
            torch.save(model.state_dict(), config.snapshot_dir + f'/zed_finetuned_{nb_iter}.pth')
        
        nb_iter += 1
        if nb_iter >= config.cos_lr_total_iters:
            break
    
