import argparse
from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from copy import deepcopy
import os
import torch.backends.cudnn as cudnn

import models.vision_transformer as vits

class vit(nn.Module):
    
    def __init__(self, model_size="base", freeze_transformer=True, pretrained_weights=None):
        super(ibotvit, self).__init__()
        self.model_size = model_size
        self.freeze_transformer = freeze_transformer
        self.pretrained_weights = pretrained_weights

        # Loading a model with registers
        n_register_tokens = 4
        
        if model_size == "vit_small":
            self.embedding_size = 384
            
        elif model_size == "vit_base":
            self.embedding_size = 768

        elif model_size == "vit_large":
            self.embedding_size = 1024
            
        elif model_size == "giant":
            self.embedding_size = 1536

        # Load state_dict
        model = vits.__dict__[model_size](patch_size=16)
        self.transformer = deepcopy(model)

        # Freeze transformer if specified        
        if self.freeze_transformer:
            for param in self.transformer.parameters():
                param.requires_grad = False

        
        if self.pretrained_weights and os.path.isfile(self.pretrained_weights):
            state_dict = torch.load(self.pretrained_weights, map_location="cpu")
            if 'teacher' in state_dict:
                state_dict = state_dict['teacher']
            elif 'model' in state_dict:
                state_dict = state_dict['model']

            # remove `backbone.` prefix induced by multicrop wrapper
            state_dict = {
                (k[len("teacher."):] if k.startswith("teacher.") else k): v
                for k, v in state_dict.items()
            }
            state_dict = {
                (k[len("backbone."):] if k.startswith("backbone.") else k): v
                for k, v in state_dict.items()
            }
            msg = self.transformer.load_state_dict(state_dict, strict=False)
            print(model_size, msg)
        

    def forward(self, x):
        x = self.transformer(x)

        return x



def build_model(args):
    
    net = vit("vit_base", freeze_transformer=True, pretrained_weights=args.pretrained_weights)
    net.cuda()
    


    return net
