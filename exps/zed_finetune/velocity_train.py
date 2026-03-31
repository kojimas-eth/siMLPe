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

from exps.baseline_h36m.test import test
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--exp-name', 
type=str, default='zed_finetune', help='experiment name')
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

from exps.zed_finetune.velocity_model import siMLPe as Model
from exps.baseline_h36m.train import get_dct_matrix , gen_velocity

from vel_config import config
config.motion.h36m_target_length = config.motion.h36m_target_length_train

from datasets.zed_vel import ZEDDataset 
from utils.logger import get_logger, print_and_log_info
from utils.pyt_utils import ensure_dir

dct_m,idct_m = get_dct_matrix(config.motion.h36m_input_length_dct)
dct_m = torch.tensor(dct_m).float().cuda().unsqueeze(0)
idct_m = torch.tensor(idct_m).float().cuda().unsqueeze(0)

def update_lr_multistep(nb_iter, total_iter, max_lr, min_lr, optimizer) :
    if nb_iter > 25000:
        current_lr = 1e-5
    else:
        current_lr = 5e-4

    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr

    return optimizer, current_lr

def train_step(h36m_motion_input, h36m_motion_target, velocity_target, yaw_target, model, optimizer, nb_iter, total_iter, max_lr, min_lr) :

    if config.deriv_input:
        b,n,c = h36m_motion_input.shape
        h36m_motion_input_ = h36m_motion_input.clone()
        h36m_motion_input_ = torch.matmul(dct_m[:, :, :config.motion.h36m_input_length], h36m_motion_input_.cuda())
    else:
        h36m_motion_input_ = h36m_motion_input.clone()

    motion_pred, velocity_pred,yaw_pred = model(h36m_motion_input_.cuda())
    
    # Apply IDCT smoothing to both
    motion_pred = torch.matmul(idct_m[:, :config.motion.h36m_input_length, :], motion_pred)
    velocity_pred = torch.matmul(idct_m[:, :config.motion.h36m_input_length, :], velocity_pred)
    yaw_pred = torch.matmul(idct_m[:, :config.motion.h36m_input_length, :], yaw_pred)

    if config.deriv_output:
        offset = h36m_motion_input[:, -1:].cuda()
        motion_pred = motion_pred[:, :config.motion.h36m_target_length] + offset
        velocity_pred = velocity_pred[:, :config.motion.h36m_target_length]
        yaw_pred = yaw_pred[:, :config.motion.h36m_target_length]
    else:
        motion_pred = motion_pred[:, :config.motion.h36m_target_length]
        velocity_pred = velocity_pred[:, :config.motion.h36m_target_length]
        yaw_pred = yaw_pred[:, :config.motion.h36m_target_length]

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

    # --- Velocity Loss ---
    velocity_target = velocity_target.cuda() # Shape: (Batch, Time, 2)
    loss_vel = torch.mean(torch.norm(velocity_pred - velocity_target, 2, 2))
    
    yaw_target = yaw_target.cuda() # Shape: (Batch, Time, 1)
    loss_yaw = torch.mean(torch.norm(yaw_pred - yaw_target, 2, 2))

    loss_total = loss_vel + loss_yaw
    writer.add_scalar('Loss/velocity', loss_vel.detach().cpu().numpy(), nb_iter)
    writer.add_scalar('Loss/yaw', loss_yaw.detach().cpu().numpy(), nb_iter)
    
    optimizer.zero_grad()
    loss_total.backward() 
    optimizer.step()
    optimizer, current_lr = update_lr_multistep(nb_iter, total_iter, max_lr, min_lr, optimizer)
    writer.add_scalar('LR/train', current_lr, nb_iter)

    return loss_total.item(), optimizer, current_lr

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
        name = k.replace("module.", "")

        #Rename 'motion_transformer' to 'motion_mlp' (Legacy compatibility)
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

for name, param in model.named_parameters():
    if 'velocity' not in name and 'rotation' not in name:
        param.requires_grad = False # Freeze everything else!

# 3. Setup ZED Dataset
dataset = ZEDDataset(config, 'train', data_aug=True)
dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)

eval_config = copy.deepcopy(config)
eval_config.motion.h36m_target_length = eval_config.motion.h36m_target_length_eval
eval_dataset = ZEDDataset(eval_config, 'test', data_aug = False)
eval_dataloader = DataLoader(eval_dataset, batch_size=config.batch_size, shuffle=False, drop_last=True)


optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                             lr=config.cos_lr_max, 
                             weight_decay=config.weight_decay)
# 5. Training Loop
model.train()
nb_iter = 0

while nb_iter < config.cos_lr_total_iters:
    for (input_motion, target_motion, velocity_target, yaw_target) in dataloader:
        #Trains step expects input (b,n,22,3)
        loss, optimizer, current_lr = train_step(input_motion, target_motion, velocity_target, yaw_target, model, optimizer, nb_iter, config.cos_lr_total_iters, config.cos_lr_max, config.cos_lr_min)
        
        if nb_iter % config.print_every == 0:
            print(f"Iter {nb_iter}: Loss = {loss}, LR = {current_lr}")

        if nb_iter % config.save_every == 0:
            if not os.path.exists(config.snapshot_dir):
                os.makedirs(config.snapshot_dir, exist_ok=True)
            torch.save(model.state_dict(), config.snapshot_dir + f'/velocity_model_{nb_iter}.pth')
        
        #Evaluation 
        if (nb_iter + 1) % config.save_every ==  0 :
            model.eval()
            val_losses = []
            val_vel_losses=[]
            val_yaw_losses = []
        
            with torch.no_grad(): # No gradients needed for testing
                for (val_input, val_target, val_velocity_target, val_yaw_target) in eval_dataloader:
                    if config.deriv_input:
                        val_input_ = torch.matmul(dct_m[:, :, :config.motion.h36m_input_length], val_input.cuda())
                    else:
                        val_input_ = val_input.clone()
                    val_pred, val_vel_pred, val_yaw_pred = model(val_input_.cuda())

                    val_pred = torch.matmul(idct_m[:, :config.motion.h36m_input_length, :], val_pred)
                    val_vel_pred = torch.matmul(idct_m[:, :config.motion.h36m_input_length, :], val_vel_pred)
                    val_yaw_pred = torch.matmul(idct_m[:, :config.motion.h36m_input_length, :], val_yaw_pred)

                    #compute loss
                    b, n, c = val_target.shape
                    num_joints = c // 3

                    if config.deriv_output:
                        offset = val_input[:, -1:].cuda()
                        val_pred = val_pred[:, :n] + offset
                    else:
                        val_pred = val_pred[:, :n]

                    # Reshape to (Batch, Time, Joints, 3)
                    pred_phys = val_pred.reshape(b, n, num_joints, 3)
                    gt_phys = val_target.cuda().reshape(b, n, num_joints, 3)
                    dist = torch.norm(pred_phys - gt_phys, dim=3) # (B, T, J)
                    val_losses.append(dist.mean().item())


                    val_vel_pred = val_vel_pred[:, :n] 
                    val_yaw_pred = val_yaw_pred[:, :n]

                    vel_dist = torch.norm(val_vel_pred - val_velocity_target.cuda(), dim=2)
                    val_vel_losses.append(vel_dist.mean().item())

                    yaw_dist = torch.norm(val_yaw_pred - val_yaw_target.cuda(), dim=2)
                    val_yaw_losses.append(yaw_dist.mean().item())

            avg_val_loss = np.mean(val_losses)
            avg_val_vel_loss = np.mean(val_vel_losses)
            avg_val_yaw_loss = np.mean(val_yaw_losses)
            
            print(f"\n[Validation] Iter {nb_iter}: MPJPE = {avg_val_loss:.4f}m | Vel Error = {avg_val_vel_loss:.4f} m/s |  Yaw Error = {avg_val_yaw_loss:.4f} rad/s\n")
            writer.add_scalar('Loss/validation_pose', avg_val_loss, nb_iter)
            writer.add_scalar('Loss/validation_vel', avg_val_vel_loss, nb_iter)
            writer.add_scalar('Loss/validation_yaw', avg_val_yaw_loss, nb_iter)
            model.train()
        
        nb_iter += 1
        if nb_iter >= config.cos_lr_total_iters:
            break
    
