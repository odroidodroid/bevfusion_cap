import torch
from mmcv.parallel import MMDistributedDataParallel
from mmcv.runner import (
    DistSamplerSeedHook,
    EpochBasedRunner,
    GradientCumulativeFp16OptimizerHook,
    Fp16OptimizerHook,
    OptimizerHook,
    build_optimizer,
    build_runner,
)
from .runner import Fp16PruningOptimizerHook, CustomPruningEpochBasedRunner
from mmdet3d.utils import get_root_logger
from mmdet.core import DistEvalHook
from mmdet.datasets import build_dataloader, build_dataset, replace_ImageToTensor
from .pruner import prune_to_target_flops
from . import torch_pruning as tp

def prune_model(
    model,
    pruner,
    dummy,
    dataset,
    cfg,
    prune_cfg,
    distributed=False,
    validate=False,
    timestamp=None,
):
    logger = get_root_logger()

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]

    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            None,
            dist=distributed,
            seed=cfg.seed,
        )
        for ds in dataset
    ]

    # put model on gpus
    find_unused_parameters = cfg.get("find_unused_parameters", False)
    # Sets the `find_unused_parameters` parameter in
    # torch.nn.parallel.DistributedDataParallel
    model = MMDistributedDataParallel(
        model.cuda(),
        device_ids=[torch.cuda.current_device()],
        broadcast_buffers=False,
        find_unused_parameters=find_unused_parameters,
    )

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)
    if pruner is not None :
        runner = build_runner(
            cfg.runner,
            default_args=dict(
                model=model,
                pruner=pruner,
                prune_cfg=prune_cfg,
                optimizer=optimizer,
                work_dir=cfg.run_dir,
                logger=logger,
                meta={},
            ),
        )
    else :
        runner = build_runner(
            cfg.runner,
            default_args=dict(
                model=model,
                optimizer=optimizer,
                work_dir=cfg.run_dir,
                logger=logger,
                meta={},
            ),
        )
    
    if hasattr(runner, "set_dataset"):
        runner.set_dataset(dataset)

    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp

    # fp16 setting
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        if "cumulative_iters" in cfg.optimizer_config:
            optimizer_config = GradientCumulativeFp16OptimizerHook(
                **cfg.optimizer_config, **fp16_cfg, distributed=distributed
            )
        else: 
            if pruner is not None :
                optimizer_config = Fp16PruningOptimizerHook(
                    **cfg.optimizer_config, **fp16_cfg, distributed=distributed
                )
            else :
                optimizer_config = Fp16OptimizerHook(
                    **cfg.optimizer_config, **fp16_cfg, distributed=distributed
                )
    elif distributed and "type" not in cfg.optimizer_config:
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # register hooks
    runner.register_training_hooks(
        cfg.lr_config,
        optimizer_config,
        cfg.checkpoint_config,
        cfg.log_config,
        cfg.get("momentum_config", None),
    )
    if isinstance(runner, EpochBasedRunner):
        runner.register_hook(DistSamplerSeedHook())

    # register eval hooks
    if validate:
        # Support batch_size > 1 in validation
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        val_dataloader = build_dataloader(
            val_dataset,
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=False,
            shuffle=False,
        )
        eval_cfg = cfg.get("evaluation", {})
        eval_cfg["by_epoch"] = cfg.runner["type"] != "IterBasedRunner"
        eval_hook = DistEvalHook
        runner.register_hook(eval_hook(val_dataloader, **eval_cfg))

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)

    # TODO print log with logger
    # TODO add prune to workflow
    print("Pruning model...")
    runner.prune(data_loaders)
    print("Finetuning...")
    runner.run(data_loaders, [("finetune", 1)])

    