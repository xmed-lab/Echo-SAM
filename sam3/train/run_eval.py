import logging
import os
import random
import sys
from argparse import ArgumentParser

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from iopath.common.file_io import g_pathmgr
from omegaconf import OmegaConf
from sam3.train.utils.train_utils import makedir, register_omegaconf_resolvers


os.environ["HYDRA_FULL_ERROR"] = "1"


def single_proc_run(local_rank, main_port, cfg, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(main_port)
    os.environ["RANK"] = str(local_rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    try:
        register_omegaconf_resolvers()
    except Exception as e:
        logging.info(e)

    from sam3.train.eval import Evaluator
    
    trainer_cfg = OmegaConf.to_container(cfg.trainer, resolve=True)
    
    evaluator_cfg = {
        "data": trainer_cfg.get("data"),
        "model": trainer_cfg.get("model"),
        "logging": trainer_cfg.get("logging"),
        "checkpoint": trainer_cfg.get("checkpoint"),
        "accelerator": trainer_cfg.get("accelerator", "cuda"),
        "seed_value": trainer_cfg.get("seed_value", 123),
        "distributed": trainer_cfg.get("distributed", {}),
        "cuda": trainer_cfg.get("cuda", {}),
        "env_variables": trainer_cfg.get("env_variables"),
        "meters": trainer_cfg.get("meters"),
        "amp": trainer_cfg.get("optim", {}).get("amp", {}),
    }
    
    evaluator = Evaluator(**evaluator_cfg)
    evaluator.run()


def single_node_runner(cfg, main_port: int):
    num_proc = cfg.launcher.gpus_per_node
    torch.multiprocessing.set_start_method("spawn")
    if num_proc == 1:
        single_proc_run(local_rank=0, main_port=main_port, cfg=cfg, world_size=num_proc)
    else:
        mp_runner = torch.multiprocessing.start_processes
        args = (main_port, cfg, num_proc)
        mp_runner(single_proc_run, args=args, nprocs=num_proc, start_method="spawn")


def add_pythonpath_to_sys_path():
    if "PYTHONPATH" not in os.environ or not os.environ["PYTHONPATH"]:
        return
    sys.path = os.environ["PYTHONPATH"].split(":") + sys.path


def main(args) -> None:
    import sam3.train
    config_dir = os.path.join(os.path.dirname(sam3.train.__file__), "configs")
    
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name=args.config)
    
    if cfg.launcher.experiment_log_dir is None:
        cfg.launcher.experiment_log_dir = os.path.join(
            os.getcwd(), "sam3_logs", args.config
        )
    
    print("###################### Eval Config ####################")
    print(OmegaConf.to_yaml(cfg))
    print("#######################################################")

    add_pythonpath_to_sys_path()
    makedir(cfg.launcher.experiment_log_dir)
    with g_pathmgr.open(
        os.path.join(cfg.launcher.experiment_log_dir, "eval_config.yaml"), "w"
    ) as f:
        f.write(OmegaConf.to_yaml(cfg))

    experiment_log_dir = cfg.launcher.experiment_log_dir
    print(f"Experiment Log Dir:\n{experiment_log_dir}")

    cfg.launcher.gpus_per_node = (
        args.num_gpus if args.num_gpus is not None else cfg.launcher.gpus_per_node
    )
    
    main_port = random.randint(10000, 20000)
    single_node_runner(cfg, main_port)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        type=str,
        help="path to config file (e.g. ultrasound_cardiac/ultrasound_cardiac_zero_shot_lv_myo.yaml)",
    )
    parser.add_argument(
        "--num-gpus", type=int, default=None, help="number of GPUS per node"
    )
    args = parser.parse_args()
    
    config_name = args.config
    if config_name.startswith("configs/"):
        config_name = config_name[len("configs/"):]
    args.config = config_name
    
    register_omegaconf_resolvers()
    main(args)
