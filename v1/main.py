import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

from personal_ppg2ecg.v1.engine.logger import Logger
from personal_ppg2ecg.v1.engine.solver import Trainer
from personal_ppg2ecg.v1.utils.io_utils import (
    load_yaml_config,
    seed_everything,
    merge_opts_to_config,
    instantiate_from_config,
)
from personal_ppg2ecg.v1.utils.ppgecg_dataset import PPGECGDataset, PPGECGPairDataset
from torch.utils.data import Dataset, DataLoader
# huggingface offline mode
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch Training Script")
    parser.add_argument("--name", type=str, default="latent_rectified_flow")

    parser.add_argument(
        "--config_file", type=str, default="config/latent_rectified_flow.yaml", help="path of config file"
    )
    parser.add_argument(
        "--output", type=str, default="baseline", help="directory to save the results"
    )
    parser.add_argument(
        "--tensorboard", action="store_true", help="use tensorboard for logging"
    )

    # args for random

    parser.add_argument(
        "--cudnn_deterministic",
        action="store_true",
        default=True,
        help="set cudnn.deterministic True",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="seed for initializing training."
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU id to use. If given, only the specific gpu will be"
        " used, and ddp will be disabled",
    )

    # args for training
    parser.add_argument(
        "--train", action="store_true", default=False, help="Train or Test."
    )
    parser.add_argument(
        "--condition_type",
        type=int,
        default=1,
        choices=[0, 1, 2, 3],
        help="sampling options",
    )
    parser.add_argument("--model_type", type=str, choices=['pf', 'ppf'], default='pf')
    parser.add_argument(
        "--mode",
        type=str,
        default="synthesis",
        help="Infilling, Forecasting or Synthesis.",
    )
    parser.add_argument("--milestone", type=int, default=10)

    parser.add_argument(
        "--synthesis_channels",
        type=lambda x: list(map(int, x.split(","))),
        default=list(range(1, 2)),
        help="List of synthesis channels (default is [1, 2, 3, ..., 11]).",
    )

    # args for modify config
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    # args.save_dir = os.path.join(args.output, f"{args.name}")

    return args


def main():
    args = parse_args()

    if args.seed is not None:
        seed_everything(args.seed, args.cudnn_deterministic)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)

    config = load_yaml_config(args.config_file)
    config = merge_opts_to_config(config, args.opts)

    print(config)

    solver_cfg = config.get("solver", {})
    args.save_dir = solver_cfg.get("results_folder", {})

    logger = Logger(args)
    logger.save_config(config)

    # --- read optional data & sampling configs from YAML ---
    data_cfg = config.get("data", {})
    train_data_cfg = data_cfg.get("train", {})
    test_data_cfg = data_cfg.get("test", {})
    sample_cfg = config.get("sample", {})
    default_data_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "mimic-iv-aligned-ppg_ecgII-processed-filtered"
    )
    train_data_dir = os.path.abspath(train_data_cfg.get("dir") or default_data_dir)
    test_data_dir = os.path.abspath(test_data_cfg.get("saved_dir") or train_data_dir)


    if args.train:
        model = instantiate_from_config(config["model"]).cuda()
        train_dataset = PPGECGPairDataset(
            data_dir=train_data_dir,
            train=True,
            return_type=args.model_type
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=int(train_data_cfg.get("batch_size", 4)),
            shuffle=True,
            num_workers=int(train_data_cfg.get("num_workers", 32)),
            pin_memory=bool(train_data_cfg.get("pin_memory", True)),
        )
        trainer = Trainer(
            config=config,
            args=args,
            model=model,
            dataloader=train_dataloader,
            logger=logger,
        )
        trainer.train()

    elif args.condition_type == 1 and args.mode in ["synthesis"]:
        model = instantiate_from_config(config["model"]).cuda()
        # Load test dataset 
        test_dataset = PPGECGPairDataset(
            data_dir=test_data_dir,
            train=False,
            return_type=args.model_type
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=int(test_data_cfg.get('batch_size', 128)),
            shuffle=False,
            num_workers=int(test_data_cfg.get('num_workers', 32)),
            pin_memory=bool(test_data_cfg.get('pin_memory', True))
        )
        trainer = Trainer(
            config=config,
            args=args,
            model=model,
            dataloader=test_dataloader,
            logger=logger,
        )

        trainer.load(args.milestone)

        sampling_steps = int(sample_cfg.get('sampling_steps', 10))
        subset_save_threshold = int(sample_cfg.get('subset_save_threshold', 100000))
        samples, *_ = trainer.sample_shift(
            test_dataloader,
            # ours
            [1250,1],
            sampling_steps,
            subset_save_threshold=subset_save_threshold,
            save_dir=os.path.join(args.save_dir, "samples"),
            num_samples=int(sample_cfg.get('num_samples', 1)),
        )

    # elif args.condition_type == 2 and args.mode in ["synthesis"]:
    #     model = instantiate_from_config(config["model"]).cuda()
    #     # Load test dataset 
    #     root_dir = "/root/autodl-tmp/fengyuan/data/mimic-iv-aligned-ppg_ecgII-processed-filtered/test_pt"
    #     data_dirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if d.endswith('.pt')]
    #     for i, data_dir in enumerate(data_dirs):
    #         test_dataset = PPGECGPairDataset(
    #             data_dir=data_dir,
    #             train=False,
    #             return_type=args.model_type
    #         )
    #         test_dataloader = DataLoader(
    #             test_dataset,
    #             batch_size=int(test_data_cfg.get('batch_size', 128)),
    #             shuffle=False,
    #             num_workers=int(test_data_cfg.get('num_workers', 32)),
    #             pin_memory=bool(test_data_cfg.get('pin_memory', True))
    #         )
    #         trainer = Trainer(
    #             config=config,
    #             args=args,
    #             model=model,
    #             dataloader=test_dataloader,
    #             logger=logger,
    #         )

    #         trainer.load(args.milestone)

    #         sampling_steps = int(sample_cfg.get('sampling_steps', 10))
    #         subset_save_threshold = int(sample_cfg.get('subset_save_threshold', 100000))

    #         if args.model_type == 'pf':
    #             ppg, ecg = test_dataset[0]
    #             ppg_ref, ecg_ref = None, None
    #         elif args.model_type == 'ppf':
    #             ppg, ecg, ppg_ref, ecg_ref = test_dataset[0]

    #         num_repeats = 10
    #         print(ppg.shape)
    #         ppg = ppg.repeat(num_repeats, 1)
    #         ecg = ecg.repeat(num_repeats, 1)
    #         ppg_ref = ppg_ref.repeat(num_repeats, 1) if ppg_ref is not None else None
    #         ecg_ref = ecg_ref.repeat(num_repeats, 1) if ecg_ref is not None else None

    #         samples = trainer.sample(
    #             ppg=ppg,
    #             ecg=ecg,
    #             ppg_ref=ppg_ref,
    #             ecg_ref=ecg_ref,
    #             # ours
    #             shape=[1250,1],
    #             sampling_steps=sampling_steps,
    #         )

    #         samples_np = samples.squeeze(-1).detach().cpu().numpy()

    #         # 计算均值和方差 (沿着 batch 维度，即 dim=0)
    #         sample_mean = np.mean(samples_np, axis=0)
    #         sample_var = np.var(samples_np, axis=0)
    #         sample_std = np.std(samples_np, axis=0)

    #         print(f"Samples shape: {samples_np.shape}") # (10, 1250)
    #         print(f"Average Variance: {np.mean(sample_var):.6f}")

    elif args.condition_type == 2 and args.mode in ["synthesis"]:
        model = instantiate_from_config(config["model"]).cuda()
        # Load test dataset 
        root_dir = "/root/autodl-tmp/fengyuan/data/mimic-iv-aligned-ppg_ecgII-processed-filtered/test_pt"
        data_dirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if d.endswith('.pt')]

        all_subject_vars = []  # 存储每个subject的平均方差

        for i, data_dir in enumerate(data_dirs):
            print(f"\n===== Processing subject {i+1}/{len(data_dirs)}: {data_dir} =====")

            # ---- 数据加载 ----
            test_dataset = PPGECGPairDataset(
                data_dir=data_dir,
                train=False,
                return_type=args.model_type
            )
            test_dataloader = DataLoader(
                test_dataset,
                batch_size=int(test_data_cfg.get('batch_size', 128)),
                shuffle=False,
                num_workers=int(test_data_cfg.get('num_workers', 32)),
                pin_memory=bool(test_data_cfg.get('pin_memory', True))
            )

            # ---- 初始化 trainer ----
            trainer = Trainer(
                config=config,
                args=args,
                model=model,
                dataloader=test_dataloader,
                logger=logger,
            )
            trainer.load(args.milestone)

            sampling_steps = int(sample_cfg.get('sampling_steps', 10))
            subset_save_threshold = int(sample_cfg.get('subset_save_threshold', 100000))

            # ---- 从当前subject中取前10条样本 ----
            num_samples_per_subject = min(10, len(test_dataset))
            per_sample_vars = []  # 当前subject内，每个样本的方差

            for j in range(num_samples_per_subject):
                if args.model_type == 'pf':
                    ppg, ecg = test_dataset[j]
                    ppg_ref, ecg_ref = None, None
                elif args.model_type == 'ppf':
                    ppg, ecg, ppg_ref, ecg_ref = test_dataset[j]

                # ---- repeat 10次，用于采样 ----
                num_repeats = 10
                ppg = ppg.unsqueeze(0).repeat(num_repeats, 1)
                ecg = ecg.unsqueeze(0).repeat(num_repeats, 1)
                ppg_ref = ppg_ref.unsqueeze(0).repeat(num_repeats, 1) if ppg_ref is not None else None
                ecg_ref = ecg_ref.unsqueeze(0).repeat(num_repeats, 1) if ecg_ref is not None else None

                # ---- 执行采样 ----
                samples = trainer.sample(
                    ppg=ppg,
                    ppg_ref=ppg_ref,
                    ecg_ref=ecg_ref,
                    shape=[1250, 1],
                    sampling_steps=sampling_steps,
                )

                samples_np = samples.squeeze(-1).detach().cpu().numpy()  # (10, 1250)

                # ---- 计算方差 ----
                sample_var = np.var(samples_np, axis=0)        # 每个时间点的方差 (1250,)
                mean_var = np.mean(sample_var)                 # 当前样本的平均方差（标量）
                per_sample_vars.append(mean_var)

                print(f"  Sample {j+1}/{num_samples_per_subject} | Mean variance: {mean_var:.6f}")

            # ---- 当前subject的平均方差 ----
            subject_mean_var = np.mean(per_sample_vars)
            if subject_mean_var is not None:
                all_subject_vars.append(subject_mean_var)

            print(f"✅ Subject {i+1}: Mean sample variance = {subject_mean_var:.6f}")

        # ---- 全部subject的平均 ----
        print(all_subject_vars)
        global_mean_var = np.mean(all_subject_vars)
        print("\n===========================")
        print(f"Final Average Variance across {len(all_subject_vars)} subjects: {global_mean_var:.6f}")
        print("===========================")

    elif args.condition_type == 3 and args.mode in ["synthesis"]:
        model = instantiate_from_config(config["model"]).cuda()
        # Load test dataset 
        root_dir = "/root/autodl-tmp/fengyuan/data/mimic-iv-aligned-ppg_ecgII-processed-filtered/test_pt/"
        data_dirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if d.endswith('.pt')]
        data_dirs = [data_dirs[0]]
        # print(len(data_dirs), data_dirs[0])
        for i, data_dir in enumerate(data_dirs):
            print(f"\n===== Processing subject {i+1}/{len(data_dirs)}: {data_dir} =====")

            # ---- 数据加载 ----
            print(data_dir)
            test_dataset = PPGECGPairDataset(
                data_dir=data_dir,
                train=False,
                return_type=args.model_type
            )
            test_dataloader = DataLoader(
                test_dataset,
                batch_size=int(test_data_cfg.get('batch_size', 128)),
                shuffle=False,
                num_workers=int(test_data_cfg.get('num_workers', 32)),
                pin_memory=bool(test_data_cfg.get('pin_memory', True))
            )

            # ---- 初始化 trainer ----
            trainer = Trainer(
                config=config,
                args=args,
                model=model,
                dataloader=test_dataloader,
                logger=logger,
            )
            trainer.load(args.milestone)

            sampling_steps = int(sample_cfg.get('sampling_steps', 10))
            subset_save_threshold = int(sample_cfg.get('subset_save_threshold', 100000))

            # ---- 从当前subject中取前1条样本 ----
            num_samples_per_subject = 1

            for j in range(num_samples_per_subject):
                if args.model_type == 'pf':
                    ppg, ecg = test_dataset[0]
                    ppg_ref, ecg_ref = None, None
                elif args.model_type == 'ppf':
                    ppg, ecg, ppg_ref, ecg_ref = test_dataset[0]

                # ---- repeat 5次，用于采样 ----
                num_repeats = 5
                ppg = ppg.unsqueeze(0).repeat(num_repeats, 1)
                ecg = ecg.unsqueeze(0).repeat(num_repeats, 1)
                ppg_ref = ppg_ref.unsqueeze(0).repeat(num_repeats, 1) if ppg_ref is not None else None
                ecg_ref = ecg_ref.unsqueeze(0).repeat(num_repeats, 1) if ecg_ref is not None else None

                # ---- 执行采样 ----
                samples = trainer.sample(
                    ppg=ppg,
                    ppg_ref=ppg_ref,
                    ecg_ref=ecg_ref,
                    shape=[1250, 1],
                    sampling_steps=sampling_steps,
                )

                samples_np = samples.squeeze(-1).detach().cpu().numpy()  # (10, 1250)

                plt.figure(figsize=(10, 6))
                plt.plot(ecg[0][:500], alpha=0.6, label=f"Ground Truth" )
                for i in range(num_repeats):
                    plt.plot(samples_np[i][:500], alpha=0.6, label=f"Sample {i+1}")
                plt.title(f"Repeated Sampling Results for One Sample ({num_repeats} Repeats)")
                plt.xlabel("Samples")
                plt.ylabel("Amplitude (normalized)")
                plt.legend(loc="upper right", fontsize=8, ncol=2)
                plt.tight_layout()
                plt.savefig(f"sampling_variability_subject1_sample1_{args.model_type}.png")


if __name__ == "__main__":
    main()


'''
python main.py --train --config_file config/latent_rectified_flow.yaml --model_type pf 
python main.py --config_file config/latent_rectified_flow.yaml --model_type pf --milestone 3
python main.py --config_file config/latent_rectified_flow.yaml --model_type pf --milestone 3 --condition_type 2
python main.py --config_file config/latent_rectified_flow.yaml --model_type pf --milestone 3 --condition_type 3

python main.py --train --config_file config/personal_latent_rectified_flow.yaml --model_type ppf
python main.py --config_file config/personal_latent_rectified_flow.yaml --model_type ppf --milestone 3
python main.py --config_file config/personal_latent_rectified_flow.yaml --model_type ppf --milestone 3 --condition_type 2
python main.py --config_file config/personal_latent_rectified_flow.yaml --model_type ppf --milestone 3 --condition_type 3
'''