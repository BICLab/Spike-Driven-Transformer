import argparse
import time
import yaml
import os
import logging
import numpy as np
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime
from spikingjelly.clock_driven import functional
from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
import torch
import torch.nn.utils.prune as prune
import torch.nn as nn
import torchvision.utils
import torchvision.transforms as transforms
from torch.nn.parallel import DistributedDataParallel as NativeDDP
from torch import quantization,quantize_per_tensor,floor
import torchinfo
from timm.data import (
    create_dataset,
    create_loader,
    resolve_data_config,
    Mixup,
    FastCollateMixup,
    AugMixDataset,
)
from timm.models import (
    create_model,
    safe_model_name,
    resume_checkpoint,
    load_checkpoint,
    convert_splitbn_model,
    model_parameters,
)
from timm.models.helpers import clean_state_dict
from timm.utils import *
from timm.loss import (
    LabelSmoothingCrossEntropy,
    SoftTargetCrossEntropy,
    JsdCrossEntropy,
    BinaryCrossEntropy,
)
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler
from timm.utils import ApexScaler, NativeScaler
import model, dvs_utils, criterion


try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model

    has_apex = True
except ImportError:
    has_apex = False

has_native_amp = False
try:
    if getattr(torch.cuda.amp, "autocast") is not None:
        has_native_amp = True
except AttributeError:
    pass

try:
    import wandb

    has_wandb = True
except ImportError:
    has_wandb = False


def resume_checkpoint(
    model, checkpoint_path, optimizer=None, loss_scaler=None, log_info=True
):
    resume_epoch = None
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cuda:0")
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            if log_info:
                _logger.info("Restoring model state from checkpoint...")
            state_dict = clean_state_dict(checkpoint["state_dict"])
            model.load_state_dict(state_dict, strict=False)

            if optimizer is not None and "optimizer" in checkpoint:
                if log_info:
                    _logger.info("Restoring optimizer state from checkpoint...")
                optimizer.load_state_dict(checkpoint["optimizer"])

            if loss_scaler is not None and loss_scaler.state_dict_key in checkpoint:
                if log_info:
                    _logger.info("Restoring AMP loss scaler state from checkpoint...")
                loss_scaler.load_state_dict(checkpoint[loss_scaler.state_dict_key])

            if "epoch" in checkpoint:
                resume_epoch = checkpoint["epoch"]
                if "version" in checkpoint and checkpoint["version"] > 1:
                    resume_epoch += 1  # start at the next epoch, old checkpoints incremented before save

            if log_info:
                _logger.info(
                    "Loaded checkpoint '{}' (epoch {})".format(
                        checkpoint_path, checkpoint["epoch"]
                    )
                )
        else:
            model.load_state_dict(checkpoint)
            if log_info:
                _logger.info("Loaded checkpoint '{}'".format(checkpoint_path))
        return resume_epoch
    else:
        _logger.error("No checkpoint found at '{}'".format(checkpoint_path))
        raise FileNotFoundError()


torch.backends.cudnn.benchmark = True
# The first arg parser parses out only the --config argument, this argument is used to
# load a yaml file containing key-values that override the defaults for the main parser below
config_parser = parser = argparse.ArgumentParser(
    description="Training Config", add_help=False
)
parser.add_argument(
    "-c",
    "--config",
    default="cifar100_config.yml",
    type=str,
    metavar="FILE",
    help="YAML config file specifying default arguments",
)

parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")

# Dataset / Model parameters
parser.add_argument(
    "-data-dir",
    metavar="DIR",
    default="",
    help="path to dataset",
)
parser.add_argument(
    "--dataset",
    "-d",
    metavar="NAME",
    default="torch/cifar10",
    help="dataset type (default: ImageFolder/ImageTar if empty)",
)
parser.add_argument(
    "--train-split",
    metavar="NAME",
    default="train",
    help="dataset train split (default: train)",
)
parser.add_argument(
    "--val-split",
    metavar="NAME",
    default="validation",
    help="dataset validation split (default: validation)",
)
parser.add_argument(
    "--train-split-path",
    type=str,
    default=None,
    metavar="N",
    help="",
)
parser.add_argument(
    "--model",
    default="sdt",
    type=str,
    metavar="MODEL",
    help='Name of model to train (default: "sdt")',
)
parser.add_argument(
    "--pooling-stat",
    default="1111",
    type=str,
    help="pooling layers in SPS moduls",
)
parser.add_argument(
    "--TET",
    default=False,
    type=bool,
    help="",
)
parser.add_argument(
    "--TET-means",
    default=1.0,
    type=float,
    help="",
)
parser.add_argument(
    "--TET-lamb",
    default=0.0,
    type=float,
    help="",
)
parser.add_argument(
    "--spike-mode",
    default="lif",
    type=str,
    help="",
)
parser.add_argument(
    "--layer",
    default=4,
    type=int,
    help="",
)
parser.add_argument(
    "--in-channels",
    default=3,
    type=int,
    help="",
)
parser.add_argument(
    "--pretrained",
    action="store_true",
    default=False,
    help="Start with pretrained version of specified network (if avail)",
)
parser.add_argument(
    "--initial-checkpoint",
    default="",
    type=str,
    metavar="PATH",
    help="Initialize model from this checkpoint (default: none)",
)
parser.add_argument(
    "--resume",
    default="",
    type=str,
    metavar="PATH",
    help="Resume full model and optimizer state from checkpoint (default: none)",
)
parser.add_argument(
    "--no-resume-opt",
    action="store_true",
    default=False,
    help="prevent resume of optimizer state when resuming model",
)
parser.add_argument(
    "--num-classes",
    type=int,
    default=1000,
    metavar="N",
    help="number of label classes (Model default if None)",
)
parser.add_argument(
    "--time-steps",
    type=int,
    default=4,
    metavar="N",
    help="",
)
parser.add_argument(
    "--num-heads",
    type=int,
    default=8,
    metavar="N",
    help="",
)
parser.add_argument(
    "--patch-size", type=int, default=None, metavar="N", help="Image patch size"
)
parser.add_argument(
    "--mlp-ratio",
    type=int,
    default=4,
    metavar="N",
    help="expand ration of embedding dimension in MLP block",
)
parser.add_argument(
    "--gp",
    default=None,
    type=str,
    metavar="POOL",
    help="Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.",
)
parser.add_argument(
    "--img-size",
    type=int,
    default=None,
    metavar="N",
    help="Image patch size (default: None => model default)",
)
parser.add_argument(
    "--input-size",
    default=None,
    nargs=3,
    type=int,
    metavar="N N N",
    help="Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty",
)
parser.add_argument(
    "--crop-pct",
    default=None,
    type=float,
    metavar="N",
    help="Input image center crop percent (for validation only)",
)
parser.add_argument(
    "--mean",
    type=float,
    nargs="+",
    default=None,
    metavar="MEAN",
    help="Override mean pixel value of dataset",
)
parser.add_argument(
    "--std",
    type=float,
    nargs="+",
    default=None,
    metavar="STD",
    help="Override std deviation of of dataset",
)
parser.add_argument(
    "--interpolation",
    default="",
    type=str,
    metavar="NAME",
    help="Image resize interpolation type (overrides model)",
)
parser.add_argument(
    "-b",
    "--batch-size",
    type=int,
    default=32,
    metavar="N",
    help="input batch size for training (default: 32)",
)
parser.add_argument(
    "-vb",
    "--val-batch-size",
    type=int,
    default=16,
    metavar="N",
    help="input val batch size for training (default: 32)",
)

# Optimizer parameters
parser.add_argument(
    "--opt",
    default="sgd",
    type=str,
    metavar="OPTIMIZER",
    help='Optimizer (default: "sgd")',
)
parser.add_argument(
    "--opt-eps",
    default=None,
    type=float,
    metavar="EPSILON",
    help="Optimizer Epsilon (default: None, use opt default)",
)
parser.add_argument(
    "--opt-betas",
    default=None,
    type=float,
    nargs="+",
    metavar="BETA",
    help="Optimizer Betas (default: None, use opt default)",
)
parser.add_argument(
    "--momentum",
    type=float,
    default=0.9,
    metavar="M",
    help="Optimizer momentum (default: 0.9)",
)
parser.add_argument(
    "--weight-decay", type=float, default=0.0001, help="weight decay (default: 0.0001)"
)
parser.add_argument(
    "--clip-grad",
    type=float,
    default=None,
    metavar="NORM",
    help="Clip gradient norm (default: None, no clipping)",
)
parser.add_argument(
    "--clip-mode",
    type=str,
    default="norm",
    help='Gradient clipping mode. One of ("norm", "value", "agc")',
)

# Learning rate schedule parameters
parser.add_argument(
    "--sched",
    default="step",
    type=str,
    metavar="SCHEDULER",
    help='LR scheduler (default: "step"',
)
parser.add_argument(
    "--lr", type=float, default=0.01, metavar="LR", help="learning rate (default: 0.01)"
)
parser.add_argument(
    "--lr-noise",
    type=float,
    nargs="+",
    default=None,
    metavar="pct, pct",
    help="learning rate noise on/off epoch percentages",
)
parser.add_argument(
    "--lr-noise-pct",
    type=float,
    default=0.67,
    metavar="PERCENT",
    help="learning rate noise limit percent (default: 0.67)",
)
parser.add_argument(
    "--lr-noise-std",
    type=float,
    default=1.0,
    metavar="STDDEV",
    help="learning rate noise std-dev (default: 1.0)",
)
parser.add_argument(
    "--lr-cycle-mul",
    type=float,
    default=1.0,
    metavar="MULT",
    help="learning rate cycle len multiplier (default: 1.0)",
)
parser.add_argument(
    "--lr-cycle-limit",
    type=int,
    default=1,
    metavar="N",
    help="learning rate cycle limit",
)
parser.add_argument(
    "--warmup-lr",
    type=float,
    default=0.0001,
    metavar="LR",
    help="warmup learning rate (default: 0.0001)",
)
parser.add_argument(
    "--min-lr",
    type=float,
    default=1e-5,
    metavar="LR",
    help="lower lr bound for cyclic schedulers that hit 0 (1e-5)",
)
parser.add_argument(
    "--epochs",
    type=int,
    default=200,
    metavar="N",
    help="number of epochs to train (default: 2)",
)
parser.add_argument(
    "--epoch-repeats",
    type=float,
    default=0.0,
    metavar="N",
    help="epoch repeat multiplier (number of times to repeat dataset epoch per train epoch).",
)
parser.add_argument(
    "--start-epoch",
    default=None,
    type=int,
    metavar="N",
    help="manual epoch number (useful on restarts)",
)
parser.add_argument(
    "--decay-epochs",
    type=float,
    default=30,
    metavar="N",
    help="epoch interval to decay LR",
)
parser.add_argument(
    "--warmup-epochs",
    type=int,
    default=3,
    metavar="N",
    help="epochs to warmup LR, if scheduler supports",
)
parser.add_argument(
    "--cooldown-epochs",
    type=int,
    default=10,
    metavar="N",
    help="epochs to cooldown LR at min_lr, after cyclic schedule ends",
)
parser.add_argument(
    "--patience-epochs",
    type=int,
    default=10,
    metavar="N",
    help="patience epochs for Plateau LR scheduler (default: 10",
)
parser.add_argument(
    "--decay-rate",
    "--dr",
    type=float,
    default=0.1,
    metavar="RATE",
    help="LR decay rate (default: 0.1)",
)

# Augmentation & regularization parameters
parser.add_argument(
    "--no-aug",
    action="store_true",
    default=False,
    help="Disable all training augmentation, override other train aug args",
)
parser.add_argument(
    "--scale",
    type=float,
    nargs="+",
    default=[0.08, 1.0],
    metavar="PCT",
    help="Random resize scale (default: 0.08 1.0)",
)
parser.add_argument(
    "--ratio",
    type=float,
    nargs="+",
    default=[3.0 / 4.0, 4.0 / 3.0],
    metavar="RATIO",
    help="Random resize aspect ratio (default: 0.75 1.33)",
)
parser.add_argument(
    "--hflip", type=float, default=0.5, help="Horizontal flip training aug probability"
)
parser.add_argument(
    "--vflip", type=float, default=0.0, help="Vertical flip training aug probability"
)
parser.add_argument(
    "--color-jitter",
    type=float,
    default=0.4,
    metavar="PCT",
    help="Color jitter factor (default: 0.4)",
)
parser.add_argument(
    "--aa",
    type=str,
    default=None,
    metavar="NAME",
    help='Use AutoAugment policy. "v0" or "original". (default: None)',
),
parser.add_argument(
    "--aug-splits",
    type=int,
    default=0,
    help="Number of augmentation splits (default: 0, valid: 0 or >=2)",
)
parser.add_argument(
    "--jsd",
    action="store_true",
    default=False,
    help="Enable Jensen-Shannon Divergence + CE loss. Use with `--aug-splits`.",
)
parser.add_argument(
    "--bce-loss",
    action="store_true",
    default=False,
    help="Enable BCE loss w/ Mixup/CutMix use.",
)
parser.add_argument(
    "--bce-target-thresh",
    type=float,
    default=None,
    help="Threshold for binarizing softened BCE targets (default: None, disabled)",
)
parser.add_argument(
    "--reprob",
    type=float,
    default=0.0,
    metavar="PCT",
    help="Random erase prob (default: 0.)",
)
parser.add_argument(
    "--remode", type=str, default="const", help='Random erase mode (default: "const")'
)
parser.add_argument(
    "--recount", type=int, default=1, help="Random erase count (default: 1)"
)
parser.add_argument(
    "--resplit",
    action="store_true",
    default=False,
    help="Do not random erase first (clean) augmentation split",
)
parser.add_argument(
    "--mixup",
    type=float,
    default=0.0,
    help="mixup alpha, mixup enabled if > 0. (default: 0.)",
)
parser.add_argument(
    "--cutmix",
    type=float,
    default=0.0,
    help="cutmix alpha, cutmix enabled if > 0. (default: 0.)",
)
parser.add_argument(
    "--cutmix-minmax",
    type=float,
    nargs="+",
    default=None,
    help="cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)",
)
parser.add_argument(
    "--mixup-prob",
    type=float,
    default=1.0,
    help="Probability of performing mixup or cutmix when either/both is enabled",
)
parser.add_argument(
    "--mixup-switch-prob",
    type=float,
    default=0.5,
    help="Probability of switching to cutmix when both mixup and cutmix enabled",
)
parser.add_argument(
    "--mixup-mode",
    type=str,
    default="batch",
    help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"',
)
parser.add_argument(
    "--mixup-off-epoch",
    default=0,
    type=int,
    metavar="N",
    help="Turn off mixup after this epoch, disabled if 0 (default: 0)",
)
parser.add_argument(
    "--smoothing", type=float, default=0.1, help="Label smoothing (default: 0.1)"
)
parser.add_argument(
    "--train-interpolation",
    type=str,
    default="random",
    help='Training interpolation (random, bilinear, bicubic default: "random")',
)
parser.add_argument(
    "--drop", type=float, default=0.0, metavar="PCT", help="Dropout rate (default: 0.)"
)
parser.add_argument(
    "--drop-connect",
    type=float,
    default=None,
    metavar="PCT",
    help="Drop connect rate, DEPRECATED, use drop-path (default: None)",
)
parser.add_argument(
    "--drop-path",
    type=float,
    default=0.2,
    metavar="PCT",
    help="Drop path rate (default: None)",
)
parser.add_argument(
    "--drop-block",
    type=float,
    default=None,
    metavar="PCT",
    help="Drop block rate (default: None)",
)

# Batch norm parameters (only works with gen_efficientnet based models currently)
parser.add_argument(
    "--bn-tf",
    action="store_true",
    default=False,
    help="Use Tensorflow BatchNorm defaults for models that support it (default: False)",
)
parser.add_argument(
    "--bn-momentum",
    type=float,
    default=None,
    help="BatchNorm momentum override (if not None)",
)
parser.add_argument(
    "--bn-eps",
    type=float,
    default=None,
    help="BatchNorm epsilon override (if not None)",
)
parser.add_argument(
    "--sync-bn",
    action="store_true",
    help="Enable NVIDIA Apex or Torch synchronized BatchNorm.",
)
parser.add_argument(
    "--dist-bn",
    type=str,
    default="",
    help='Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")',
)
parser.add_argument(
    "--split-bn",
    action="store_true",
    help="Enable separate BN layers per augmentation split.",
)
parser.add_argument(
    "--linear-prob",
    action="store_true",
    help="",
)
# Model Exponential Moving Average
parser.add_argument(
    "--model-ema",
    action="store_true",
    default=False,
    help="Enable tracking moving average of model weights",
)
parser.add_argument(
    "--model-ema-force-cpu",
    action="store_true",
    default=False,
    help="Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.",
)
parser.add_argument(
    "--model-ema-decay",
    type=float,
    default=0.9998,
    help="decay factor for model weights moving average (default: 0.9998)",
)

# Misc
parser.add_argument(
    "--seed", type=int, default=42, metavar="S", help="random seed (default: 42)"
)
parser.add_argument(
    "--log-interval",
    type=int,
    default=100,
    metavar="N",
    help="how many batches to wait before logging training status",
)
parser.add_argument(
    "--recovery-interval",
    type=int,
    default=0,
    metavar="N",
    help="how many batches to wait before writing recovery checkpoint",
)
parser.add_argument(
    "--checkpoint-hist",
    type=int,
    default=10,
    metavar="N",
    help="number of checkpoints to keep (default: 10)",
)
parser.add_argument(
    "-j",
    "--workers",
    type=int,
    default=4,
    metavar="N",
    help="how many training processes to use (default: 1)",
)
parser.add_argument(
    "--save-images",
    action="store_true",
    default=False,
    help="save images of input bathes every log interval for debugging",
)
parser.add_argument(
    "--amp",
    action="store_true",
    default=False,
    help="use NVIDIA Apex AMP or Native AMP for mixed precision training",
)
parser.add_argument(
    "--apex-amp",
    action="store_true",
    default=False,
    help="Use NVIDIA Apex AMP mixed precision",
)
parser.add_argument(
    "--native-amp",
    action="store_true",
    default=False,
    help="Use Native Torch AMP mixed precision",
)
parser.add_argument(
    "--channels-last",
    action="store_true",
    default=False,
    help="Use channels_last memory layout",
)
parser.add_argument(
    "--pin-mem",
    action="store_true",
    default=False,
    help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
)
parser.add_argument(
    "--no-prefetcher",
    action="store_true",
    default=False,
    help="disable fast prefetcher",
)
parser.add_argument(
    "--dvs-aug",
    action="store_true",
    default=False,
    help="disable fast prefetcher",
)
parser.add_argument(
    "--dvs-trival-aug",
    action="store_true",
    default=False,
    help="disable fast prefetcher",
)
parser.add_argument(
    "--output",
    default="",
    type=str,
    metavar="PATH",
    help="path to output folder (default: none, current dir)",
)
parser.add_argument(
    "--experiment",
    default="",
    type=str,
    metavar="NAME",
    help="name of train experiment, name of sub-folder for output",
)
parser.add_argument(
    "--eval-metric",
    default="top1",
    type=str,
    metavar="EVAL_METRIC",
    help='Best metric (default: "top1")',
)
parser.add_argument(
    "--tta",
    type=int,
    default=0,
    metavar="N",
    help="Test/inference time augmentation (oversampling) factor. 0=None (default: 0)",
)
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument(
    "--use-multi-epochs-loader",
    action="store_true",
    default=False,
    help="use the multi-epochs-loader to save time at the beginning of every epoch",
)
parser.add_argument(
    "--torchscript",
    dest="torchscript",
    action="store_true",
    help="convert model torchscript for inference",
)
parser.add_argument(
    "--log-wandb",
    action="store_true",
    default=False,
    help="log training and validation metrics to wandb",
)

_logger = logging.getLogger("train")
stream_handler = logging.StreamHandler()
format_str = "%(asctime)s %(levelname)s: %(message)s"
stream_handler.setFormatter(logging.Formatter(format_str))
_logger.addHandler(stream_handler)
_logger.propagate = False


def _parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, "r") as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


def main():
    setup_default_logging()
    args, args_text = _parse_args()

    if args.log_wandb:
        if has_wandb:
            wandb.init(project=args.experiment, config=args)
        else:
            _logger.warning(
                "You've requested to log metrics to wandb but package not found. "
                "Metrics not being logged to wandb, try `pip install wandb`"
            )

    args.prefetcher = not args.no_prefetcher
    args.distributed = False
    if "WORLD_SIZE" in os.environ:
        args.distributed = int(os.environ["WORLD_SIZE"]) > 1
    args.device = "cuda:1"
    args.world_size = 1
    args.rank = 0  # global rank
    if args.distributed:
        args.device = "cuda:%d" % args.local_rank
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
        _logger.info(
            "Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d."
            % (args.rank, args.world_size)
        )
    else:
        _logger.info("Training with a single process on 1 GPUs.")
    assert args.rank >= 0

    # resolve AMP arguments based on PyTorch / Apex availability
    use_amp = None
    if args.amp:
        # `--amp` chooses native amp before apex (APEX ver not actively maintained)
        if has_native_amp:
            args.native_amp = True
        elif has_apex:
            args.apex_amp = True
    if args.apex_amp and has_apex:
        use_amp = "apex"
    elif args.native_amp and has_native_amp:
        use_amp = "native"
    elif args.apex_amp or args.native_amp:
        _logger.warning(
            "Neither APEX or native Torch AMP is available, using float32. "
            "Install NVIDA apex or upgrade to PyTorch 1.6"
        )

    torch.backends.cudnn.benchmark = True
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    os.environ['CUDA_VISIBLE_DEVICE']='0'       #做gpu运算
    torch.cuda.set_device(0)                    #做gpu运算
    np.random.seed(args.seed)
    torch.initial_seed()  # dataloader multi processing
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random_seed(args.seed, args.rank)

    args.dvs_mode = False
    if args.dataset in ["cifar10-dvs-tet", "cifar10-dvs"]:
        args.dvs_mode = True

    model = create_model(
        args.model,
        T=args.time_steps,
        pretrained=args.pretrained,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=args.drop_block,
        num_heads=args.num_heads,
        num_classes=args.num_classes,
        pooling_stat=args.pooling_stat,
        img_size_h=args.img_size,
        img_size_w=args.img_size,
        patch_size=args.patch_size,
        embed_dims=args.dim,
        mlp_ratios=args.mlp_ratio,
        in_channels=args.in_channels,
        qkv_bias=False,
        depths=args.layer,
        sr_ratios=1,
        spike_mode=args.spike_mode,
        dvs_mode=args.dvs_mode,
        TET=args.TET,
    )
    if args.local_rank == 0:
        _logger.info(f"Creating model {args.model}")
        _logger.info(
            str(
                torchinfo.summary(
                    model, (2, args.in_channels, args.img_size, args.img_size)
                )
            )
        )

    if args.num_classes is None:
        assert hasattr(
            model, "num_classes"
        ), "Model must have `num_classes` attr if not set on cmd line/config."
        args.num_classes = (
            model.num_classes
        )  # FIXME handle model default vs config num_classes more elegantly

    data_config = resolve_data_config(
        vars(args), model=model, verbose=args.local_rank == 0
    )
    output_dir = None
    if args.rank == 0:
        if args.experiment:
            exp_name = args.experiment
        else:
            exp_name = "-".join(
                [
                    datetime.now().strftime("%Y%m%d-%H%M%S"),
                    safe_model_name(args.model),
                    "data-" + args.dataset.split("/")[-1],
                    f"t-{args.time_steps}",
                    f"spike-{args.spike_mode}",
                ]
            )
        output_dir = get_outdir(
            args.output if args.output else "./output/train", exp_name
        )
        file_handler = logging.FileHandler(
            os.path.join(output_dir, f"{args.model}.log"), "w"
        )
        file_handler.setFormatter(logging.Formatter(format_str))
        file_handler.setLevel(logging.INFO)
        _logger.addHandler(file_handler)

    if args.local_rank == 0:
        _logger.info(
            f"Model {safe_model_name(args.model)} created, param count:{sum([m.numel() for m in model.parameters()])}"
        )

    # setup augmentation batch splits for contrastive loss or split bn
    num_aug_splits = 0
    if args.aug_splits > 0:
        assert args.aug_splits > 1, "A split of 1 makes no sense"
        num_aug_splits = args.aug_splits

    # enable split bn (separate bn stats per batch-portion)
    if args.split_bn:
        assert num_aug_splits > 1 or args.resplit
        model = convert_splitbn_model(model, max(num_aug_splits, 2))

    # move model to GPU, enable channels last layout if set
    model.cuda()
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    # setup synchronized BatchNorm for distributed training
    if args.distributed and args.sync_bn:
        assert not args.split_bn
        if has_apex and use_amp != "native":
            # Apex SyncBN preferred unless native amp is activated
            model = convert_syncbn_model(model)
        else:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if args.local_rank == 0:
            _logger.info(
                "Converted model to use Synchronized BatchNorm. WARNING: You may have issues if using "
                "zero initialized BN layers (enabled by default for ResNets) while sync-bn enabled."
            )

    if args.torchscript:
        assert not use_amp == "apex", "Cannot use APEX AMP with torchscripted model"
        assert not args.sync_bn, "Cannot use SyncBatchNorm with torchscripted model"
        model = torch.jit.script(model)

    optimizer = create_optimizer_v2(model, **optimizer_kwargs(cfg=args))

    # setup automatic mixed-precision (AMP) loss scaling and op casting
    amp_autocast = suppress  # do nothing
    loss_scaler = None
    if use_amp == "apex":
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
        loss_scaler = ApexScaler()
        if args.local_rank == 0:
            _logger.info("Using NVIDIA APEX AMP. Training in mixed precision.")
    elif use_amp == "native":
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()
        if args.local_rank == 0:
            _logger.info("Using native Torch AMP. Training in mixed precision.")
    else:
        if args.local_rank == 0:
            _logger.info("AMP not enabled. Training in float32.")

    # optionally resume from a checkpoint
    resume_epoch = None
    if args.resume:
        resume_epoch = resume_checkpoint(
            model,
            args.resume,
            optimizer=None if args.no_resume_opt else optimizer,
            loss_scaler=None if args.no_resume_opt else loss_scaler,
            log_info=args.local_rank == 0,
        )

    # setup exponential moving average of model weights, SWA could be used here too
    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEmaV2(
            model,
            decay=args.model_ema_decay,
            device="cuda:0" if args.model_ema_force_cpu else None,
        )
        if args.resume:
            load_checkpoint(model_ema.module, args.resume, use_ema=True)

    # setup distributed training
    if args.distributed:
        if has_apex and use_amp != "native":
            # Apex DDP preferred unless native amp is activated
            if args.local_rank == 0:
                _logger.info("Using NVIDIA APEX DistributedDataParallel.")
            model = ApexDDP(model, delay_allreduce=True, find_unused_parameters=True)
        else:
            if args.local_rank == 0:
                _logger.info("Using native Torch DistributedDataParallel.")
            model = NativeDDP(
                model, device_ids=[args.local_rank], find_unused_parameters=True
            )  # can use device str in Torch >= 1.1
        # NOTE: EMA model does not need to be wrapped by DDP

    # for linear prob
    if args.linear_prob:
        for n, p in model.module.named_parameters():
            if "patch_embed" in n:
                p.requires_grad = False
            # if "block" in n:
            #     p.requires_grad = False

    # setup learning rate schedule and starting epoch
    lr_scheduler, num_epochs = create_scheduler(args, optimizer)
    start_epoch = 0
    if args.start_epoch is not None:
        # a specified start_epoch will always override the resume epoch
        start_epoch = args.start_epoch
    elif resume_epoch is not None and (not args.linear_prob):
        start_epoch = resume_epoch
    if lr_scheduler is not None and start_epoch > 0:
        lr_scheduler.step(start_epoch)

    if args.local_rank == 0:
        _logger.info("Scheduled epochs: {}".format(num_epochs))

    transforms_train, transforms_eval = None, None

    # create the train and eval datasets
    dataset_train, dataset_eval = None, None
    if args.dataset == "cifar10-dvs-tet":
        dataset_train = dvs_utils.DVSCifar10(
            root=os.path.join(args.data_dir, "train"),
            train=True,
        )
        dataset_eval = dvs_utils.DVSCifar10(
            root=os.path.join(args.data_dir, "test"),
            train=False,
        )
    elif args.dataset == "cifar10-dvs":
        dataset = CIFAR10DVS(
            args.data_dir,
            data_type="frame",
            frames_number=args.time_steps,
            split_by="number",
            transform=dvs_utils.Resize(128),
        )
        dataset_train, dataset_eval = dvs_utils.split_to_train_test_set(
            0.9, dataset, 10
        )
    elif args.dataset == "gesture":
        dataset_train = DVS128Gesture(
            args.data_dir,
            train=True,
            data_type="frame",
            frames_number=args.time_steps,
            split_by="number",
        )
        dataset_eval = DVS128Gesture(
            args.data_dir,
            train=False,
            data_type="frame",
            frames_number=args.time_steps,
            split_by="number",
        )
    else:
        dataset_train = create_dataset(
            args.dataset,
            root=args.data_dir,
            split=args.train_split,
            is_training=True,
            batch_size=args.batch_size,
            repeats=args.epoch_repeats,
            transform=transforms_train,
            download=True,
        )
        dataset_eval = create_dataset(
            args.dataset,
            root=args.data_dir,
            split=args.val_split,
            is_training=False,
            batch_size=args.batch_size,
            transform=transforms_eval,
            # download=True,
        )

    # setup mixup / cutmix
    collate_fn = None
    train_dvs_aug, train_dvs_trival_aug = None, None
    if args.dvs_aug:
        train_dvs_aug = dvs_utils.Cutout(n_holes=1, length=16)
    if args.dvs_trival_aug:
        train_dvs_trival_aug = dvs_utils.SNNAugmentWide()
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0.0 or args.cutmix_minmax is not None
    if mixup_active:
        mixup_args = dict(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob,
            switch_prob=args.mixup_switch_prob,
            mode=args.mixup_mode,
            label_smoothing=args.smoothing,
            num_classes=args.num_classes,
        )
        if args.prefetcher and args.dataset not in dvs_utils.DVS_DATASET:
            assert (
                not num_aug_splits
            )  # collate conflict (need to support deinterleaving in collate mixup)
            collate_fn = FastCollateMixup(**mixup_args)
        else:
            mixup_fn = Mixup(**mixup_args)

    # wrap dataset in AugMix helper
    if num_aug_splits > 1 and args.dataset not in dvs_utils.DVS_DATASET:
        dataset_train = AugMixDataset(dataset_train, num_splits=num_aug_splits)

    # create data loaders w/ augmentation pipeiine
    train_interpolation = args.train_interpolation
    if args.no_aug or not train_interpolation:
        train_interpolation = data_config["interpolation"]

    loader_train, loader_eval, train_idx = None, None, None
    # NOTE(hujiakui): only for ImageNet
    if args.train_split_path is not None:
        train_idx = np.load(args.train_split_path).tolist()

    if args.dataset in dvs_utils.DVS_DATASET:
        loader_train = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=args.batch_size,
            shuffle=True,           #first is true
            num_workers=args.workers,
            pin_memory=True,
        )
        loader_eval = torch.utils.data.DataLoader(
            dataset_eval,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
        )
    else:
        loader_train = create_loader(
            dataset_train,
            input_size=data_config["input_size"],
            batch_size=args.batch_size,
            is_training=True,
            use_prefetcher=args.prefetcher,
            no_aug=args.no_aug,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            re_split=args.resplit,
            scale=args.scale,
            ratio=args.ratio,
            hflip=args.hflip,
            vflip=args.vflip,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            num_aug_splits=num_aug_splits,
            interpolation=train_interpolation,
            mean=data_config["mean"],
            std=data_config["std"],
            num_workers=args.workers,
            distributed=args.distributed,
            collate_fn=collate_fn,
            pin_memory=args.pin_mem,
            use_multi_epochs_loader=args.use_multi_epochs_loader,
            # train_idx=train_idx,
        )
        # NOTE(hujiakui): train_idx should modify the code of timm

        loader_eval = create_loader(
            dataset_eval,
            input_size=data_config["input_size"],
            batch_size=args.val_batch_size,
            is_training=False,
            use_prefetcher=args.prefetcher,
            interpolation=data_config["interpolation"],
            mean=data_config["mean"],
            std=data_config["std"],
            num_workers=args.workers,
            distributed=args.distributed,
            crop_pct=data_config["crop_pct"],
            pin_memory=args.pin_mem,
        )
    if args.local_rank == 0:
        _logger.info("Create dataloader: {}".format(args.dataset))

    # setup loss function
    if args.jsd:
        assert num_aug_splits > 1  # JSD only valid with aug splits set
        train_loss_fn = JsdCrossEntropy(
            num_splits=num_aug_splits, smoothing=args.smoothing
        ).cuda()
    elif mixup_active:
        # smoothing is handled with mixup target transform
        if args.bce_loss:
            train_loss_fn = BinaryCrossEntropy(target_threshold=args.bce_target_thresh)
        else:
            train_loss_fn = SoftTargetCrossEntropy()
    elif args.smoothing:
        if args.bce_loss:
            train_loss_fn = BinaryCrossEntropy(
                smoothing=args.smoothing, target_threshold=args.bce_target_thresh
            )
        else:
            train_loss_fn = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        train_loss_fn = nn.CrossEntropyLoss()

    train_loss_fn = train_loss_fn.cuda()
    validate_loss_fn = nn.CrossEntropyLoss().cuda()

    # setup checkpoint saver and eval metric tracking
    eval_metric = args.eval_metric
    best_metric = None
    best_epoch = None
    saver = None
    if args.rank == 0:
        decreasing = False
        saver = CheckpointSaver(
            model=model,                       #初始为model
            optimizer=optimizer,
            args=args,
            model_ema=model_ema,
            amp_scaler=loss_scaler,
            checkpoint_dir=output_dir,
            recovery_dir=output_dir,
            decreasing=decreasing,
            max_history=args.checkpoint_hist,
        )
        with open(os.path.join(output_dir, "args.yaml"), "w") as f:
            f.write(args_text)

    try:
        for epoch in range(1, 2):

            model_best = torch.load('./output/train/20231113-094928-sdt-data-cifar10-t-4-spike-lif/model_best.pth.tar')
            model.load_state_dict(model_best['state_dict'])

            fuse_module(model)

            state_dict = model.state_dict

            #模型剪枝
            
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    prune.l1_unstructured(module, name='weight', amount=0.15)
                    prune.remove(module,'weight')  
                elif isinstance(module, torch.nn.Linear):
                    prune.l1_unstructured(module, name='weight', amount=0.2)
                    prune.remove(module,'weight')  

            print(
                    "稀疏性 in head.weight: {:.2f}%".format(
                      100. * float(torch.sum(model.head.weight == 0))
                           / float(model.head.weight.nelement())
                                     )
                )   

            state_dict = model.state_dict()
            state_dict['block.0.attn.q_conv.weight'].data = floor(128 * state_dict['block.0.attn.q_conv.weight'].data)/128   #权重量化
            state_dict['block.0.attn.q_conv.bias'].data = floor(128 * state_dict['block.0.attn.q_conv.bias'].data)/128   #权重量化
            state_dict['block.0.attn.k_conv.weight'].data = floor(128 * state_dict['block.0.attn.k_conv.weight'].data)/128   #权重量化
            state_dict['block.0.attn.k_conv.bias'].data = floor(128 * state_dict['block.0.attn.k_conv.bias'].data)/128   #权重量化
            state_dict['block.0.attn.v_conv.weight'].data = floor(128 * state_dict['block.0.attn.v_conv.weight'].data)/128   #权重量化
            state_dict['block.0.attn.v_conv.bias'].data = floor(128 * state_dict['block.0.attn.v_conv.bias'].data)/128   #权重量化
            state_dict['block.0.attn.talking_heads.weight'].data = floor(128 * state_dict['block.0.attn.talking_heads.weight'].data)/128   #权重量化
            state_dict['block.0.attn.proj_conv.weight'].data = floor(128 * state_dict['block.0.attn.proj_conv.weight'].data)/128   #权重量化
            state_dict['block.0.attn.proj_conv.bias'].data = floor(128 * state_dict['block.0.attn.proj_conv.bias'].data)/128   #权重量化
            state_dict['block.0.mlp.fc1_conv.weight'].data = floor(128 * state_dict['block.0.mlp.fc1_conv.weight'].data)/128   #权重量化
            state_dict['block.0.mlp.fc1_conv.bias'].data = floor(128 * state_dict['block.0.mlp.fc1_conv.bias'].data)/128   #权重量化
            state_dict['block.0.mlp.fc2_conv.weight'].data = floor(128 * state_dict['block.0.mlp.fc2_conv.weight'].data)/128   #权重量化
            state_dict['block.0.mlp.fc2_conv.bias'].data = floor(128 * state_dict['block.0.mlp.fc2_conv.bias'].data)/128   #权重量化
            state_dict['head.weight'].data = floor(1024 * state_dict['head.weight'].data)/1024   #权重量化
            state_dict['head.bias'].data = floor(1024 * state_dict['head.bias'].data)/1024   #权重量化
            model.load_state_dict(state_dict)

            print(torch.max(state_dict['block.0.attn.q_conv.weight'].data))
            print(torch.max(state_dict['block.0.attn.k_conv.weight'].data)) 
            print(torch.max(state_dict['block.0.attn.v_conv.weight'].data))
            print(torch.max(state_dict['block.0.attn.proj_conv.weight'].data))
            print(torch.max(state_dict['block.0.mlp.fc1_conv.weight'].data))
            print(torch.max(state_dict['block.0.mlp.fc2_conv.weight'].data))
            print(torch.min(state_dict['block.0.attn.q_conv.weight'].data))
            print(torch.min(state_dict['block.0.attn.k_conv.weight'].data)) 
            print(torch.min(state_dict['block.0.attn.v_conv.weight'].data))
            print(torch.min(state_dict['block.0.attn.proj_conv.weight'].data))
            print(torch.min(state_dict['block.0.mlp.fc1_conv.weight'].data))
            print(torch.min(state_dict['block.0.mlp.fc2_conv.weight'].data))
            print(torch.max(state_dict['head.weight'].data))
            print(torch.max(state_dict['head.bias'].data))
            print(torch.min(state_dict['head.weight'].data))
            print(torch.min(state_dict['head.bias'].data))
            print(torch.max(state_dict['block.0.attn.q_conv.bias'].data))


            eval_metrics = validate(
                model, loader_eval, validate_loss_fn, args, amp_autocast=amp_autocast
            )

            if model_ema is not None and not args.model_ema_force_cpu:
                if args.distributed and args.dist_bn in ("broadcast", "reduce"):
                    distribute_bn(model_ema, args.world_size, args.dist_bn == "reduce")
                ema_eval_metrics = validate(
                    model_ema.module,
                    loader_eval,
                    validate_loss_fn,
                    args,
                    amp_autocast=amp_autocast,
                    log_suffix=" (EMA)",
                )
                eval_metrics = ema_eval_metrics

            print(eval_metrics['top1'])  


    except KeyboardInterrupt:
        pass
    if best_metric is not None:
        _logger.info("*** Best metric: {0} (epoch {1})".format(best_metric, best_epoch))


def train_one_epoch(
    epoch,
    model,
    loader,
    optimizer,
    loss_fn,
    args,
    lr_scheduler=None,
    saver=None,
    output_dir=None,
    amp_autocast=suppress,
    loss_scaler=None,
    model_ema=None,
    mixup_fn=None,
    dvs_aug=None,
    dvs_trival_aug=None,
):
    if args.mixup_off_epoch and epoch >= args.mixup_off_epoch:
        if args.prefetcher:
            if hasattr(loader, "mixup_enabled"):
                loader.mixup_enabled = False
        elif mixup_fn is not None:
            mixup_fn.mixup_enabled = False

    sample_number = 0
    start_time = time.time()

    second_order = hasattr(optimizer, "is_second_order") and optimizer.is_second_order
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()

    model.train()
    functional.reset_net(model)

    end = time.time()
    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)
    for batch_idx, (input, target) in enumerate(loader):
        last_batch = batch_idx == last_idx
        data_time_m.update(time.time() - end)
        input = input.float()
        if not args.prefetcher or args.dataset in dvs_utils.DVS_DATASET:
            if args.amp and not isinstance(input, torch.cuda.HalfTensor):
                input = input.half()
            input, target = input.cuda(), target.cuda()
            if dvs_aug is not None:
                input = dvs_aug(input)
            if dvs_trival_aug is not None:
                output = []
                for i in range(input.shape[0]):
                    output.append(dvs_trival_aug(input[i]))
                input = torch.stack(output)
                del output
            if mixup_fn is not None:
                input, target = mixup_fn(input, target)

        if args.channels_last:
            input = input.contiguous(memory_format=torch.channels_last)

        with amp_autocast():
            output = model(input)[0]
            if args.TET:
                loss = criterion.TET_loss(
                    output, target, loss_fn, means=args.TET_means, lamb=args.TET_lamb
                )
            else:
                loss = loss_fn(output, target)
        sample_number += input.shape[0]
        if not args.distributed:
            losses_m.update(loss.item(), input.size(0))

        optimizer.zero_grad()
        if loss_scaler is not None:
            loss_scaler(
                loss,
                optimizer,
                clip_grad=args.clip_grad,
                clip_mode=args.clip_mode,
                parameters=model_parameters(
                    model, exclude_head="agc" in args.clip_mode
                ),
                create_graph=second_order,
            )
        else:
            # loss.backward()
            loss.backward(create_graph=second_order)
            if args.clip_grad is not None:
                dispatch_clip_grad(
                    model_parameters(model, exclude_head="agc" in args.clip_mode),
                    value=args.clip_grad,
                    mode=args.clip_mode,
                )
            optimizer.step()

        functional.reset_net(model)
        if model_ema is not None:
            model_ema.update(model)
            functional.reset_net(model_ema)

        torch.cuda.synchronize()
        num_updates += 1
        batch_time_m.update(time.time() - end)
        if last_batch or batch_idx % args.log_interval == 0:
            lrl = [param_group["lr"] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                losses_m.update(reduced_loss.item(), input.size(0))

            if args.local_rank == 0:
                _logger.info(
                    "Train: {} [{:>4d}/{} ({:>3.0f}%)]  "
                    "Loss: {loss.val:>9.6f} ({loss.avg:>6.4f})  "
                    "Time: {batch_time.val:.3f}s, {rate:>7.2f}/s  "
                    "({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  "
                    "LR: {lr:.3e}  "
                    "Data: {data_time.val:.3f} ({data_time.avg:.3f})".format(
                        epoch,
                        batch_idx,
                        len(loader),
                        100.0 * batch_idx / last_idx,
                        loss=losses_m,
                        batch_time=batch_time_m,
                        rate=input.size(0) * args.world_size / batch_time_m.val,
                        rate_avg=input.size(0) * args.world_size / batch_time_m.avg,
                        lr=lr,
                        data_time=data_time_m,
                    )
                )

                if args.save_images and output_dir:
                    torchvision.utils.save_image(
                        input,
                        os.path.join(output_dir, "train-batch-%d.jpg" % batch_idx),
                        padding=0,
                        normalize=True,
                    )

        if (
            saver is not None
            and args.recovery_interval
            and (last_batch or (batch_idx + 1) % args.recovery_interval == 0)
        ):
            saver.save_recovery(epoch, batch_idx=batch_idx)

        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

        end = time.time()
        # end for

    if hasattr(optimizer, "sync_lookahead"):
        optimizer.sync_lookahead()
    if args.local_rank == 0:
        _logger.info(f"samples / s = {sample_number / (time.time() - start_time): .3f}")
    return OrderedDict([("loss", losses_m.avg)])


def validate(model, loader, loss_fn, args, amp_autocast=suppress, log_suffix=""):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()

    model.eval()
    # functional.reset_net(model)

    end = time.time()
    last_idx = len(loader) - 1
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            input = input.float()
            if (target >= 1000).sum() != 0 or (target < 0).sum() != 0:
                print(target)

            last_batch = batch_idx == last_idx
            if not args.prefetcher or args.dataset in dvs_utils.DVS_DATASET:
                if args.amp and not isinstance(input, torch.cuda.HalfTensor):
                    input = input.half()
                input = input.cuda()
                target = target.cuda()
            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)

            with amp_autocast():
                output = model(input)
            if isinstance(output, (tuple, list)):
                output = output[0]
            if args.TET:
                output = output.mean(0)

            # augmentation reduction
            reduce_factor = args.tta
            if reduce_factor > 1:
                output = output.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                target = target[0 : target.size(0) : reduce_factor]

            if (target >= 1000).sum() != 0 or (target < 0).sum() != 0:
                print(target)
            loss = loss_fn(output, target)
            functional.reset_net(model)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                acc1 = reduce_tensor(acc1, args.world_size)
                acc5 = reduce_tensor(acc5, args.world_size)
            else:
                reduced_loss = loss.data

            torch.cuda.synchronize()

            losses_m.update(reduced_loss.item(), input.size(0))
            top1_m.update(acc1.item(), output.size(0))
            top5_m.update(acc5.item(), output.size(0))

            batch_time_m.update(time.time() - end)
            end = time.time()
            if args.local_rank == 0 and (
                last_batch or batch_idx % args.log_interval == 0
            ):
                log_name = "Test" + log_suffix
                _logger.info(
                    "{0}: [{1:>4d}/{2}]  "
                    "Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  "
                    "Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  "
                    "Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  "
                    "Acc@5: {top5.val:>7.4f} ({top5.avg:>7.4f})".format(
                        log_name,
                        batch_idx,
                        last_idx,
                        batch_time=batch_time_m,
                        loss=losses_m,
                        top1=top1_m,
                        top5=top5_m,
                    )
                )

    metrics = OrderedDict(
        [("loss", losses_m.avg), ("top1", top1_m.avg), ("top5", top5_m.avg)]
    )

    return metrics

class DummyModule(nn.Module):
    def __init__(self):
        super(DummyModule, self).__init__()

    def forward(self, x):
        return x

def fuse(conv,bn):
    w = conv.weight
    mean = bn.running_mean
    var_sqrt = torch.sqrt(bn.running_var + bn.eps)

    beta = bn.weight
    gamma = bn.bias

    if conv.bias is not None:
        b = conv.bias
    else:
        b = mean.new_zeros(mean.shape)

    w = w * (beta / var_sqrt).reshape([conv.out_channels, 1, 1, 1])
    b = (b - mean)/var_sqrt * beta + gamma
    fused_conv = nn.Conv2d(conv.in_channels,
                         conv.out_channels,
                         conv.kernel_size,
                         conv.stride,
                         conv.padding,
                         bias=True)
    fused_conv.weight = nn.Parameter(w)
    fused_conv.bias = nn.Parameter(b)
    return fused_conv

def fuse_module(m):
    children = list(m.named_children())
    c = None
    cn = None

    for name, child in children:
        if isinstance(child, nn.BatchNorm2d):
            bc = fuse(c, child)
            m._modules[cn] = bc
            m._modules[name] = DummyModule()
            c = None
        elif isinstance(child, nn.Conv2d):
            c = child
            cn = name
        else:
            fuse_module(child)


if __name__ == "__main__":
    main()
