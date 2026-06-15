import sys
import torch
import argparse
import logging
import os
import gc
import numpy as np
import torch.nn.functional as F


project_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root_dir)
os.chdir(project_root_dir)

from model.Individual_base_extractor.ib_extractor import IBExtractor
from utils.io_utils import load_yaml_config, seed_everything
from utils.ppgecg_dataset import PPGECGDataset
from torch.utils.data import Dataset, DataLoader

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config/Individual_base_extractor.yaml",
        help="path to config file",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="/root/autodl-tmp/personal_ppg2ecg/results/Individual_base_extractor/",
        help="directory to save checkpoints",
    )
    parser.add_argument(
        "--cudnn_deterministic",
        action="store_true",
        default=True,
        help="set cudnn.deterministic True",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="seed for initializing training."
    )
    return parser.parse_args()


def contrastive_loss(features, subject_ids, temperature=0.07):
    # (B, D)
    features = F.normalize(features, dim=-1)  # L2 normalize
    sim_matrix = features @ features.T        # (B, B)

    # 除以温度
    sim_matrix = sim_matrix / temperature

    # 构造正样本 mask
    positive_mask = subject_ids.unsqueeze(0) == subject_ids.unsqueeze(1)  # (B, B)
    # 去掉自己（i=j）
    self_mask = torch.eye(subject_ids.shape[0], dtype=torch.bool, device=features.device)
    positive_mask = positive_mask.to(features.device) & ~self_mask

    # 对每一行，计算：
    # numerator = sum(exp(sim_ij)) over j in same-subject
    # denominator = sum(exp(sim_ij)) over j != i
    exp_sim = torch.exp(sim_matrix)

    numerator = (exp_sim * positive_mask).sum(dim=1)
    denominator = (exp_sim * (~self_mask)).sum(dim=1)

    # 处理没有正样本的情况（numerator=0），防止log(0)
    loss_i = -torch.log((numerator + 1e-8) / (denominator + 1e-8))

    # 只对有正样本的样本计算平均
    valid = (positive_mask.sum(dim=1) > 0)
    loss = loss_i[valid].mean()

    return loss


def train_loop(
    dataloader,
    model,
    loss_fn,
    optimizer,
    scheduler,
    device,
    save_weights_path=None,
    logger=None,
    total_iterations=100000,
    save_interval=5000,
    log_interval=100,
):
    size = len(dataloader.dataset)
    model.train()

    iteration = 0
    dataloader_iterator = iter(dataloader)

    while iteration < total_iterations:
        # print("iter:", iteration)
        try:
            ppg, ecg, subject_ids = next(dataloader_iterator)
        except StopIteration:
            dataloader_iterator = iter(dataloader)
            ppg, ecg, subject_ids = next(dataloader_iterator)

        iteration += 1
        ppg = ppg.unsqueeze(-1).to(device).float()
        ecg = ecg.unsqueeze(-1).to(device).float()

        features = model(ppg, ecg)

        loss = loss_fn(features, subject_ids)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if scheduler:
            scheduler.step()

        if iteration % log_interval == 0:
            logger.info(
                f"Iteration {iteration}/{total_iterations} - "
                f"total: {loss.item():>7f} | "
            )

        if save_weights_path and iteration % save_interval == 0:
            model_states = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler else None,
                "iteration": iteration,
            }
            save_path = os.path.join(save_weights_path, f"IBExtractor-iter-{iteration}.pth")
            torch.save(model_states, save_path)
            logger.info(f"Saved checkpoint at iteration {iteration}")


if __name__ == "__main__":
    args = parse_args()
    config = load_yaml_config(args.config)
    seed_everything(args.seed, args.cudnn_deterministic)

    args.save_dir = os.path.join(args.save_dir, "mimic-iv-waveform")
    save_weights_path = os.path.join(args.save_dir, "checkpoints")
    os.makedirs(save_weights_path, exist_ok=True)

    logger = logging.getLogger("ib_extractor")
    logger.setLevel("INFO")
    fh = logging.FileHandler(
        os.path.join(args.save_dir, "train.log"),
        encoding="utf-8",
    )
    ch = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    train_cfg = config.get("train", {})
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})

    # Hyperparameters with sensible defaults
    H_ = {
        "lr": float(train_cfg.get("lr", 2e-5)),
        "batch_size": int(train_cfg.get("batch_size", 4)),
        "total_iterations": int(train_cfg.get("total_iterations", 40000)),
        "save_interval": int(train_cfg.get("save_interval", 10000)),
        "log_interval": int(train_cfg.get("log_interval", 20)),
        "num_workers": int(train_cfg.get("num_workers", 32)),
        "pin_memory": bool(train_cfg.get("pin_memory", True)),
    }
    logger.info({"train": H_, "data": data_cfg})
    
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    train_dataset = PPGECGDataset(
        data_dir="/root/autodl-tmp/personal_ppg2ecg/mimic-iv-aligned-ppg_ecgII-processed-filtered",
        train=True,
        return_type='rm'
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=H_["batch_size"],
        shuffle=True,
        num_workers=H_["num_workers"],
        pin_memory=H_["pin_memory"],
    )

    model = IBExtractor(**model_cfg).to(device)

    loss_fn = contrastive_loss
    parameters = list(model.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=H_["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=H_["total_iterations"],
        eta_min=1e-9
    )

    train_loop(
        train_dataloader,
        model,
        loss_fn,
        optimizer,
        scheduler,
        device,
        save_weights_path=save_weights_path,
        logger=logger,
        total_iterations=H_["total_iterations"],
        save_interval=H_["save_interval"],
        log_interval=H_["log_interval"],
    )
    logger.info("Training completed!")

'''
python model/Individual_base_extractor/train.py --config config/ib_extractor.yaml --save_dir results/IBExtractor
'''