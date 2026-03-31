import copy
import sys
import os.path as osp
dir_path = osp.dirname(osp.abspath(__file__))
root_path = osp.abspath(osp.join(dir_path, '../../'))
if root_path not in sys.path:
    sys.path.append(root_path)
    
import torch
from torch import nn
from exps.baseline_h36m.mlp import build_mlps
# from mlp import build_mlps
from einops.layers.torch import Rearrange

class siMLPe(nn.Module):
    def __init__(self, config):
        self.config = copy.deepcopy(config)
        super(siMLPe, self).__init__()
        seq = self.config.motion_mlp.seq_len
        self.arr0 = Rearrange('b n d -> b d n')
        self.arr1 = Rearrange('b d n -> b n d')

        self.motion_mlp = build_mlps(self.config.motion_mlp)

        self.temporal_fc_in = config.motion_fc_in.temporal_fc
        self.temporal_fc_out = config.motion_fc_out.temporal_fc
        if self.temporal_fc_in:
            self.motion_fc_in = nn.Linear(self.config.motion.h36m_input_length_dct, self.config.motion.h36m_input_length_dct)
        else:
            self.motion_fc_in = nn.Linear(self.config.motion.dim, self.config.motion.dim)
        
        if self.temporal_fc_out:
            self.motion_fc_out = nn.Linear(self.config.motion.h36m_input_length_dct, self.config.motion.h36m_input_length_dct)
            self.velocity_fc_out = nn.Linear(self.config.motion.h36m_input_length_dct, self.config.motion.h36m_input_length_dct)
            self.rotation_fc_out = nn.Linear(self.config.motion.h36m_input_length_dct, self.config.motion.h36m_input_length_dct) 
            
            self.velocity_proj = nn.Sequential(
                nn.Linear(self.config.motion.dim, 64),
                nn.GELU(),
                nn.Dropout(p=0.3),         
                nn.Linear(64, 32),
                nn.GELU(),
                nn.Dropout(p=0.3),             
                nn.Linear(32, 2)
            )
            
            self.rotation_proj = nn.Sequential(
                nn.Linear(self.config.motion.dim, 32),
                nn.GELU(),
                nn.Dropout(p=0.3),
                nn.Linear(64, 16),
                nn.GELU(),
                nn.Dropout(p=0.3),
                nn.Linear(16, 1) # Predicts a single number: Yaw Angular Velocity (rad/s)
            )


        else:
            self.motion_fc_out = nn.Linear(self.config.motion.dim, self.config.motion.dim)
            self.velocity_fc_out = nn.Sequential(
                nn.Linear(self.config.motion.dim, 64),
                nn.GELU(),
                nn.Dropout(p=0.3),
                nn.Linear(64, 32),
                nn.GELU(),
                nn.Dropout(p=0.3),
                nn.Linear(32, 2)
            )
            
            self.rotation_proj = nn.Sequential(
                nn.Linear(self.config.motion.dim, 64),
                nn.GELU(),
                nn.Dropout(p=0.3),
                nn.Linear(64, 16),
                nn.GELU(),
                nn.Dropout(p=0.3),
                nn.Linear(16, 1) # Predicts a single number: Yaw Angular Velocity (rad/s)
            )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.motion_fc_out.weight, gain=1e-8)
        nn.init.constant_(self.motion_fc_out.bias, 0)
        
        # Initialize the deep velocity head
        if hasattr(self, 'velocity_proj'):
            nn.init.constant_(self.velocity_proj[-1].weight, 0)
            nn.init.constant_(self.velocity_proj[-1].bias, 0)
        elif hasattr(self, 'velocity_fc_out') and isinstance(self.velocity_fc_out, nn.Sequential):
            nn.init.constant_(self.velocity_fc_out[-1].weight, 0)
            nn.init.constant_(self.velocity_fc_out[-1].bias, 0)

        if hasattr(self, 'rotation_proj'):
            nn.init.constant_(self.rotation_proj[-1].weight, 0)
            nn.init.constant_(self.rotation_proj[-1].bias, 0)

    def forward(self, motion_input):

        if self.temporal_fc_in:
            motion_feats = self.arr0(motion_input)
            motion_feats = self.motion_fc_in(motion_feats)
        else:
            motion_feats = self.motion_fc_in(motion_input)
            motion_feats = self.arr0(motion_feats)

        motion_feats = self.motion_mlp(motion_feats)


        if self.temporal_fc_out:
            motion_out = self.motion_fc_out(motion_feats)
            motion_pred = self.arr1(motion_out)
            
            vel_out = self.velocity_fc_out(motion_feats)
            vel_pred = self.velocity_proj(self.arr1(vel_out))

            rot_out = self.rotation_fc_out(motion_feats)
            rot_pred = self.rotation_proj(self.arr1(rot_out))
        else:
            motion_feats_arr = self.arr1(motion_feats)
            motion_pred = self.motion_fc_out(motion_feats_arr)
            vel_pred = self.velocity_fc_out(motion_feats_arr)
            rot_pred = self.rotation_proj(motion_feats_arr)
        return motion_pred, vel_pred, rot_pred

