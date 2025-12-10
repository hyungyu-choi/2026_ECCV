# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import sys
import datetime
import time
import math
import json
import numpy as np
import utils
import models
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from pathlib import Path
from PIL import Image
from torchvision import datasets, transforms
from torchvision import models as torchvision_models
from tensorboardX import SummaryWriter
from models.head import iBOTHead, TemporalHead
from models import puzzle_decoder
from models.puzzle_decoder import PuzzleDecoder
from loader import ImageFolderMask_puzzle, Temporal_RandStep_dataset, RepeatDataset

def get_args_parser():
    parser = argparse.ArgumentParser('iBOT', add_help=False)

    # Model parameters
    parser.add_argument('--arch', default='vit_small', type=str,
        choices=['vit_tiny', 'vit_small', 'vit_base', 'vit_large', 'deit_tiny', 'deit_small',
                 'swin_tiny','swin_small', 'swin_base', 'swin_large'],
        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using vit_tiny or vit_small.""")
    parser.add_argument('--patch_size', default=16, type=int, help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory. Applies only
        for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
        mixed precision training (--use_fp16 false) to avoid unstabilities.""")
    parser.add_argument('--window_size', default=7, type=int, help="""Size of window - default 7.
        This config is only valid for Swin Transofmer and is ignoired for vanilla ViT architectures.""")
    parser.add_argument('--out_dim', default=8192, type=int, help="""Dimensionality of
        output for [CLS] token.""")
    parser.add_argument('--patch_out_dim', default=8192, type=int, help="""Dimensionality of
        output for patch tokens.""")
    parser.add_argument('--shared_head', default=False, type=utils.bool_flag, help="""Whether to share 
        the same head for [CLS] token output and patch tokens output. When set to false, patch_out_dim
        is ignored and enforced to be same with out_dim. (Default: False)""")
    parser.add_argument('--shared_head_teacher', default=True, type=utils.bool_flag, help="""See above.
        Only works for teacher model. (Defeault: True)""")
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag,
        help="""Whether or not to weight normalize the last layer of the head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_argument('--norm_in_head', default=None,
        help="Whether to use batch normalizations in projection head (Default: None)")
    parser.add_argument('--act_in_head', default='gelu',
        help="Whether to use batch normalizations in projection head (Default: gelu)")
    parser.add_argument('--use_masked_im_modeling', default=True, type=utils.bool_flag,
        help="Whether to use masked image modeling (mim) in backbone (Default: True)")
    parser.add_argument('--pred_ratio', default=0.3, type=float, nargs='+', help="""Ratio of partial prediction.
        If a list of ratio is specified, one of them will be randomly choosed for each patch.""")
    parser.add_argument('--pred_ratio_var', default=0, type=float, nargs='+', help="""Variance of partial prediction
        ratio. Length should be indentical to the length of pred_ratio. 0 for disabling. """)
    parser.add_argument('--pred_shape', default='block', type=str, help="""Shape of partial prediction.""")
    parser.add_argument('--pred_start_epoch', default=0, type=int, help="""Start epoch to perform masked
        image prediction. We typically set this to 50 for swin transformer. (Default: 0)""")
    parser.add_argument('--lambda1', default=1.0, type=float, help="""loss weight for dino
        loss over [CLS] tokens (Default: 1.0)""")
    parser.add_argument('--lambda2', default=1.0, type=float, help="""loss weight for beit 
        loss over masked patch tokens (Default: 1.0)""")
    parser.add_argument('--lambda_video', default=1.0, type=float)
    parser.add_argument('--lambda_puzzle', default=0.4, type=float)
    parser.add_argument('--lambda_temporal', default=1, type=float, help="""loss weight for beit 
        loss over masked patch tokens (Default: 1.0)""")
        
    # Temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_patch_temp', default=0.04, type=float, help="""See 
        `--warmup_teacher_temp`""")
    parser.add_argument('--teacher_patch_temp', default=0.07, type=float, help=""""See 
        `--teacher_temp`""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=3, type=int,
        help='Number of warmup epochs for the teacher temperature (Default: 30).')

    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=True, help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--batch_size_per_gpu', default=64, type=int,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--batch_size_temporal_per_gpu', default=64, type=int,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=30, type=int, help='Number of epochs of training.')
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument("--lr", default=0.00005, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument("--lr_head", default=0.00035, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    
    parser.add_argument("--warmup_epochs", default=3, type=int,
        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument("--start_epoch_video", default=40, type=int,
        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument("--start_epoch_puzzle", default=20, type=int,
        help="Number of epochs for the linear learning-rate warm up.")
    
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--optimizer', default='adamw', type=str,
        choices=['adamw', 'sgd', 'lars'], help="""Type of optimizer. We recommend using adamw with ViTs.""")
    parser.add_argument('--load_from', default=None, help="""Path to load checkpoints to resume training.""")
    parser.add_argument('--drop_path', type=float, default=0.1, help="""Drop path rate for student network.""")

    # Multi-crop parameters
    parser.add_argument('--global_crops_number', type=int, default=2, help="""Number of global
        views to generate. Default is to use two global crops. """)
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.6, 1.),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
        recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
    parser.add_argument('--local_crops_number', type=int, default=0, help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.
        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.4),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for small local view cropping of multi-crop.""")

    # Misc
    parser.add_argument('--data_path', default='/path/to/imagenet/train/', type=str,
        help='Please specify path to the ImageNet training data.')
    parser.add_argument('--output_dir', default=".", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=20, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--no-pe', default=False, type=utils.bool_flag,
        help="Whether to use masked image modeling (mim) in backbone (Default: True)")

    return parser

def train_ibot(args):
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ preparing data ... ============
    transform = DataAugmentationiBOT(
        args.global_crops_scale,
        args.local_crops_scale,
        args.global_crops_number,
        args.local_crops_number,
    )
    pred_size = args.patch_size * 8 if 'swin' in args.arch else args.patch_size
    dataset_image = ImageFolderMask_puzzle(
        args.data_path, 
        transform=transform,
        patch_size=pred_size,
        pred_ratio=args.pred_ratio,
        pred_ratio_var=args.pred_ratio_var,
        pred_aspect_ratio=(0.3, 1/0.3),
        pred_shape=args.pred_shape,
        pred_start_epoch=args.pred_start_epoch)
      
    print(f"Data loaded: there are {len(dataset_image)} images.")
    
    dataset_temporal = Temporal_RandStep_dataset(lmdb_path=args.data_path, img_size=224)

    N_img,  B_img = len(dataset_image),  args.batch_size_per_gpu
    N_vid0, B_vid = len(dataset_temporal), args.batch_size_temporal_per_gpu

    iter_img = max(1, N_img  // B_img)
    iter_vid = max(1, N_vid0 // B_vid)
    repeat_factor = math.ceil(iter_img / iter_vid)

    print(f"Data loaded: there are {len(dataset_temporal)} video clips.")
    dataset_temporal = RepeatDataset(dataset_temporal, repeat_factor)
    
    sampler_image = torch.utils.data.DistributedSampler(dataset_image, shuffle=True)
    sampler_temporal = torch.utils.data.DistributedSampler(dataset_temporal, shuffle=True)
    
    data_loader_image = torch.utils.data.DataLoader(
        dataset_image,
        sampler=sampler_image,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    data_loader_temporal = torch.utils.data.DataLoader(
        dataset_temporal,
        sampler=sampler_temporal,
        batch_size=args.batch_size_temporal_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    

    # ============ building student and teacher networks ... ============
    # we changed the name DeiT-S for ViT-S to avoid confusions
    args.arch = args.arch.replace("deit", "vit")
    # if the network is of hierechical features (i.e. swin_tiny, swin_small, swin_base)
    if args.arch in models.__dict__.keys() and 'swin' in args.arch:
        student = models.__dict__[args.arch](
            window_size=args.window_size,
            return_all_tokens=True, 
            masked_im_modeling=args.use_masked_im_modeling,
        )
        teacher = models.__dict__[args.arch](
            window_size=args.window_size,
            drop_path_rate=0.0,
            return_all_tokens=True,
        )
        embed_dim = student.num_features
    # if the network is a vision transformer (i.e. vit_tiny, vit_small, vit_base, vit_large)
    elif args.arch in models.__dict__.keys():
        student = models.__dict__[args.arch](
            patch_size=args.patch_size,
            drop_path_rate=args.drop_path,
            return_all_tokens=True,
            masked_im_modeling=args.use_masked_im_modeling,
        )
        teacher = models.__dict__[args.arch](
            patch_size=args.patch_size,
            return_all_tokens=True,
        )
        embed_dim = student.embed_dim
    # otherwise, we check if the architecture is in torchvision models
    elif args.arch in torchvision_models.__dict__.keys():
        student = torchvision_models.__dict__[args.arch]()
        teacher = torchvision_models.__dict__[args.arch]()
        embed_dim = student.fc.weight.shape[1]
    else:
        print(f"Unknow architecture: {args.arch}")

    # multi-crop wrapper handles forward with inputs of different resolutions
    student = utils.MultiCropWrapper(student, iBOTHead(
        embed_dim,
        args.out_dim,
        patch_out_dim=args.patch_out_dim,
        norm=args.norm_in_head,
        act=args.act_in_head,
        norm_last_layer=args.norm_last_layer,
        shared_head=args.shared_head,
    ))
    teacher = utils.MultiCropWrapper(
        teacher,
        iBOTHead(
            embed_dim, 
            args.out_dim,
            patch_out_dim=args.patch_out_dim,
            norm=args.norm_in_head,
            act=args.act_in_head,
            shared_head=args.shared_head_teacher,
        ),
    )

    embed_dim = student.backbone.embed_dim  # or set from config

    puzzle_decoder = PuzzleDecoder(attn_mul=4, num_blocks=1, embed_dim=embed_dim, num_heads=12 if embed_dim==768 else 8).cuda()
    puzzle_decoder = nn.parallel.DistributedDataParallel(puzzle_decoder, device_ids=[args.gpu], broadcast_buffers=False)

    temporal_head = TemporalHead(backbone_dim = student.backbone.embed_dim if hasattr(student.backbone,'embed_dim')
                      else student.backbone.num_features).cuda()
    temporal_head = nn.parallel.DistributedDataParallel(temporal_head, device_ids=[args.gpu], broadcast_buffers=False)
    
    # move networks to gpu
    student, teacher = student.cuda(), teacher.cuda()
    # synchronize batch norms (if any)
    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

        # we need DDP wrapper to have synchro batch norms working...
        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu], broadcast_buffers=False) if \
            'swin' in args.arch else nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu])
        teacher_without_ddp = teacher.module
    else:
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher
    student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu], broadcast_buffers=False) if \
        'swin' in args.arch else nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu])
    # teacher and student start with the same weights
    teacher_without_ddp.load_state_dict(student.module.state_dict(), strict=False)
    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both {args.arch} network.")

    # ============ preparing loss ... ============
    same_dim = args.shared_head or args.shared_head_teacher
    ibot_loss = utils.iBOTLoss(
        args.out_dim,
        args.out_dim if same_dim else args.patch_out_dim,
        args.global_crops_number,
        args.local_crops_number,
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_patch_temp,
        args.teacher_patch_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs,
        lambda1=args.lambda1,
        lambda2=args.lambda2,
        mim_start_epoch=args.pred_start_epoch,
    ).cuda()

    pl_loss_fn = utils.PlackettLuceLoss(sample = False).cuda()
    #pl_loss_fn = utils.WindowedPlackettLuce(H=14, W=14, win=6, windows_per_img=4, dirs_per_win=4, mode="rowcol").cuda()

    if utils.is_main_process(): # Tensorboard configuration
        local_runs = os.path.join(args.output_dir, 'tf_logs')
        writer = SummaryWriter(logdir=local_runs)
        
    # ============ preparing optimizer ... ============
    params_groups = utils.get_params_groups(student)
    puzzle_groups = utils.get_params_groups(puzzle_decoder)
    head_groups = utils.get_params_groups(temporal_head)
    params_groups.extend(puzzle_groups)
    params_groups.extend(head_groups)
    
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)  # to use with convnet and large batches
    # for mixed precision training
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ============ init schedulers ... ============
    lr_schedule = utils.cosine_scheduler(
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,  # linear scaling rule
        args.min_lr,
        args.epochs, len(data_loader_image),
        warmup_epochs=args.warmup_epochs,
    )

    lr_schedule_head = utils.cosine_scheduler(
        args.lr_head * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,  # linear scaling rule
        args.min_lr * args.lr_head / args.lr,
        args.epochs, len(data_loader_image),
        warmup_epochs=args.warmup_epochs)
    
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(data_loader_image),
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1,
                                            args.epochs, len(data_loader_image))
                  
    print(f"Loss, optimizer and schedulers ready.")

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}
    if args.load_from:
        utils.restart_from_checkpoint(
            os.path.join(args.output_dir, args.load_from),
            run_variables=to_restore,
            student=student,
            teacher=teacher,
            temporal_head=temporal_head,
            puzzle_decoder=puzzle_decoder,
            optimizer=optimizer,
            fp16_scaler=fp16_scaler,
            ibot_loss=ibot_loss,
        )
    start_epoch = to_restore["epoch"]

    start_time = time.time()
    print("Starting iBOT training!")
    for epoch in range(start_epoch, args.epochs):
        data_loader_image.sampler.set_epoch(epoch)
        data_loader_image.dataset.set_epoch(epoch)
        dataset_temporal.dataset.set_epoch(epoch)
        data_loader_temporal.sampler.set_epoch(epoch)

        # ============ training one epoch of iBOT ... ============
        train_stats = train_one_epoch(student, teacher, teacher_without_ddp, ibot_loss, temporal_head, puzzle_decoder, pl_loss_fn,
            data_loader_image, data_loader_temporal, optimizer,lr_schedule, lr_schedule_head, wd_schedule, momentum_schedule, epoch, fp16_scaler, args)

        # ============ writing logs ... ============
        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            "temporal_head": temporal_head.state_dict(),
            'puzzle_decoder':puzzle_decoder.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'ibot_loss': ibot_loss.state_dict(),
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        if args.saveckp_freq and (epoch % args.saveckp_freq == 0) and epoch:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
                for k, v in train_stats.items():
                    writer.add_scalar(k, v, epoch)
        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train_one_epoch(student, teacher, teacher_without_ddp, ibot_loss, temporal_head, puzzle_decoder, pl_loss_fn, data_loader_image, data_loader_temporal, optimizer, lr_schedule, lr_schedule_head, wd_schedule, momentum_schedule, epoch, fp16_scaler, args):
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)

    iter_temporal = iter(data_loader_temporal)

    lambda_video = args.lambda_video
    start_epoch_video = args.start_epoch_video
    lambda_puzzle = args.lambda_puzzle
    start_epoch_puzzle = args.start_epoch_puzzle
    
    # common params
    names_q, params_q, names_k, params_k = [], [], [], []
    for name_q, param_q in student.module.named_parameters():
        names_q.append(name_q)
        params_q.append(param_q)
    for name_k, param_k in teacher_without_ddp.named_parameters():
        names_k.append(name_k)
        params_k.append(param_k)
    names_common = list(set(names_q) & set(names_k))
    params_q = [param_q for name_q, param_q in zip(names_q, params_q) if name_q in names_common]
    params_k = [param_k for name_k, param_k in zip(names_k, params_k) if name_k in names_common]

    pred_labels, real_labels = [], []
    for it_curr, (images, masks, puzzle_images) in enumerate(metric_logger.log_every(data_loader_image, 100, header)):
        clips = next(iter_temporal)
        it = len(data_loader_image) * epoch + it_curr  # global training iteration
        
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it] if i < 4 else lr_schedule_head[it]
            if i in (0, 2, 4):  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        # move images to gpu
        images = [im.cuda(non_blocking=True) for im in images]
        masks = [msk.cuda(non_blocking=True) for msk in masks]   
        clips  = clips.cuda(non_blocking=True)

        current_image = puzzle_images[0].cuda(non_blocking=True)
        past_image = puzzle_images[1].cuda(non_blocking=True)
        future_image = puzzle_images[2].cuda(non_blocking=True)
        
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            # get global views
            
            teacher_output = teacher(images[:args.global_crops_number])
            student_output = student(images[:args.global_crops_number], mask=masks[:args.global_crops_number])  

            # get current image puzzle patch embeddings 
            current_patch = student.module.backbone(current_image, mask=masks[2+args.local_crops_number], no_pe = True)[:,1:,:]
        
            student.module.backbone.masked_im_modeling = False            
            # get local views
            student_local_cls = student(images[args.global_crops_number:])[0] if len(images) > args.global_crops_number else None

            # get neighbor image puzzle patch embeddings       
            with torch.no_grad():  # <- better than .detach() afterwards
                past_neighbor_patch = student.module.backbone(past_image, no_pe = True)[:,1:,:]   
                future_neighbor_patch = student.module.backbone(future_image, no_pe = True)[:,1:,:]   
                neighbor_patch = torch.cat((past_neighbor_patch, future_neighbor_patch), dim=1)
                
            student.module.backbone.masked_im_modeling = args.use_masked_im_modeling

            # ----- Image branch-----
            puzzle_scores = puzzle_decoder(current_patch, neighbor_patch)
            loss_puzzle = pl_loss_fn(puzzle_scores)
            loss_puzzle = cosine_lambda(lambda_puzzle, start_epoch_puzzle, epoch, it_curr, len(data_loader_image)) * loss_puzzle
            
            loss_dict_ibot = ibot_loss(student_output, teacher_output, student_local_cls, masks, epoch)
            loss_ibot = loss_dict_ibot.pop('loss')

            # ----- video branch (temporal) -----
            B,T,C,H,W = clips.shape
            flat = clips.view(B*T, C, H, W)
            student.module.backbone.masked_im_modeling = False
            feat = student.module.backbone(flat)[:, 0, :].view(B, T, -1)
            student.module.backbone.masked_im_modeling = True
            logits = temporal_head(feat).squeeze(-1)
            loss_temporal = pl_loss_fn(logits.float())
            #loss_temporal = cosine_lambda(lambda_video, start_epoch_video, epoch, it_curr, len(data_loader_image)) * loss_temporal
            
            # total loss
            loss = loss_ibot + loss_puzzle + loss_temporal

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        # log statistics
        probs1 = teacher_output[0].chunk(args.global_crops_number)
        probs2 = student_output[0].chunk(args.global_crops_number)
        pred1 = utils.concat_all_gather(probs1[0].max(dim=1)[1]) 
        pred2 = utils.concat_all_gather(probs2[1].max(dim=1)[1])
        acc = (pred1 == pred2).sum() / pred1.size(0)
        #pred_labels.append(pred1)
        #real_labels.append(utils.concat_all_gather(labels.to(pred1.device)))

        # student update
        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(params_q, params_k):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        for key, value in loss_dict_ibot.items():
            metric_logger.update(**{key: value.item()})
        metric_logger.update(loss_puzzle=loss_puzzle.item())
        metric_logger.update(loss_temporal=loss_temporal.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
        metric_logger.update(acc=acc)

    #pred_labels = torch.cat(pred_labels).cpu().detach().numpy()
    #real_labels = torch.cat(real_labels).cpu().detach().numpy()
    #nmi, ari, fscore, adjacc = eval_pred(real_labels, pred_labels, calc_acc=False)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    #print("NMI: {}, ARI: {}, F: {}, ACC: {}".format(nmi, ari, fscore, adjacc))
    print("Averaged stats:", metric_logger)
    return_dict = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    #return_dict.update({"nmi": nmi, "ari": ari, "fscore": fscore, "adjacc": adjacc})
    return return_dict


def cosine_lambda(lambda_v, start_epoch, epoch, it, steps_per_epoch):
    if epoch < start_epoch:
        return 0.0
    # cosine ramp to 0.1 over ~1 epoch
    prog = min(1.0, (epoch-start_epoch + it/steps_per_epoch) / 1.0)
    return lambda_v * (0.5 - 0.5 * math.cos(math.pi * prog))

'''
def lambda_puzzle(lambda_p, warmup, epoch, it, steps_per_epoch):
    warmup = 20
    if epoch < warmup:
        return 0.0
    # cosine ramp to 0.1 over ~1 epoch
    prog = min(1.0, (epoch-warmup + it/steps_per_epoch) / 20.0)
    return lambda_p * (0.5 - 0.5 * math.cos(math.pi * prog))
'''


class DataAugmentationiBOT(object):
    def __init__(self, global_crops_scale, local_crops_scale, global_crops_number, local_crops_number):
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        self.puzzle_transf1 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.3
            ),
            transforms.RandomGrayscale(p=0.2),            
            normalize,
        ])

        self.puzzle_transf2 = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.3
            ),
            transforms.RandomGrayscale(p=0.2),            
            normalize,
        ])

        self.global_crops_number = global_crops_number
        # transformation for the first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(1.0),
            normalize,
        ])
        # transformation for the rest of global crops
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(0.1),
            utils.Solarization(0.2),
            normalize,
        ])
        # transformation for the local crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(p=0.5),
            normalize,
        ])

    def __call__(self, image, time_frame=None):
        crops = []
        crops.append(self.global_transfo1(image))
        for _ in range(self.global_crops_number - 1):
            crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        
        if time_frame:
            puzzles = [self.puzzle_transf1(image), self.puzzle_transf2(time_frame[0]), self.puzzle_transf2(time_frame[1])]

        
        return (crops, puzzles)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('iBOT', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_ibot(args)
