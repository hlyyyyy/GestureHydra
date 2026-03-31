import random
import warnings

import numpy as np
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (DistSamplerSeedHook, Fp16OptimizerHook, OptimizerHook,
                         build_runner)
from mmcv.runner import load_checkpoint

from gesturehydra.core.distributed_wrapper import DistributedDataParallelWrapper
from gesturehydra.core.optimizer import build_optimizers
from gesturehydra.datasets import build_dataloader, build_dataset
from gesturehydra.utils import get_root_logger
from gesturehydra.runner import BF16OptimizerHook


def set_random_seed(seed, deterministic=False):
    """Set random seed.
    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_model(model,
                dataset,
                cfg,
                distributed=False,
                timestamp=None,
                device='cuda',
                meta=None):
    """Main api for training model."""
    logger = get_root_logger(cfg.log_level)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    if cfg.data.get('use_webdataset', False):
        # data_loaders = [dataset.make_loader().with_epoch(cfg.runner['max_epochs'])]
        batch_size = cfg.data.train.batch_size
        nbatches = cfg.data.ntrain // (batch_size * len(cfg.gpu_ids)) * 25
        data_loaders = [ds.make_loader().with_epoch(nbatches).with_length(cfg.data.ntrain) for ds in dataset]
    else:
        data_loaders = [
        build_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            # cfg.gpus will be ignored if distributed
            num_gpus=len(cfg.gpu_ids),
            dist=distributed,
            round_up=True,
            seed=cfg.seed) for ds in dataset
    ]

    # determine whether use adversarial training precess or not
    use_adverserial_train = cfg.get('use_adversarial_train', False)

    if cfg.load_from:
        load_checkpoint(model, cfg.load_from, map_location='cpu')

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f'Total parameters: {total_params//1e6:.2f}M')
    # print(f'Trainable parameters: {trainable_params//1e6:.2f}M')

    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', True)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        if use_adverserial_train:
            # Use DistributedDataParallelWrapper for adversarial training
            model = DistributedDataParallelWrapper(
                model,
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False,
                find_unused_parameters=find_unused_parameters)
        else:
            model = MMDistributedDataParallel(
                model.cuda(),
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False,
                find_unused_parameters=find_unused_parameters)
    else:
        if device == 'cuda':
            model = MMDataParallel(model.cuda(cfg.gpu_ids[0]),
                                   device_ids=cfg.gpu_ids)
        elif device == 'cpu':
            model = model.cpu()
        else:
            raise ValueError(F'unsupported device name {device}.')

    # build runner
    optimizer = build_optimizers(model, cfg.optimizer)

    if cfg.get('runner') is None:
        cfg.runner = {
            'type': 'EpochBasedRunner',
            'max_epochs': cfg.total_epochs
        }
        warnings.warn(
            'config is now expected to have a `runner` section, '
            'please set `runner` in your config.', UserWarning)

    runner = build_runner(cfg.runner,
                          default_args=dict(model=model,
                                            batch_processor=None,
                                            optimizer=optimizer,
                                            work_dir=cfg.work_dir,
                                            logger=logger,
                                            meta=meta))

    # an ugly walkaround to make the .log and .log.json filenames the same
    runner.timestamp = timestamp

    if use_adverserial_train:
        # The optimizer step process is included in the train_step function
        # of the model, so the runner should NOT include optimizer hook.
        optimizer_config = None
    else:
        # fp16 setting
        fp16_cfg = cfg.get('fp16', None)
        bf16_cfg = cfg.get('bf16', None)
        if fp16_cfg is not None:
            optimizer_config = Fp16OptimizerHook(**cfg.optimizer_config,
                                                 **fp16_cfg,
                                                 distributed=distributed)
        elif bf16_cfg is not None:
            optimizer_config = BF16OptimizerHook(**cfg.optimizer_config,
                                                 **bf16_cfg,
                                                 distributed=distributed)
        elif distributed and 'type' not in cfg.optimizer_config:
            optimizer_config = OptimizerHook(**cfg.optimizer_config)
        else:
            optimizer_config = cfg.optimizer_config

    # register hooks
    runner.register_training_hooks(cfg.lr_config,
                                   optimizer_config,
                                   cfg.checkpoint_config,
                                   cfg.log_config,
                                   cfg.get('momentum_config', None),
                                   custom_hooks_config=cfg.get(
                                       'custom_hooks', None))
    if distributed:
        if cfg.runner['type'] == "EpochBasedRunner":
            runner.register_hook(DistSamplerSeedHook())

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    # elif cfg.load_from:
    #     runner.load_checkpoint(cfg.load_from)
    
    runner.run(data_loaders, cfg.workflow)
