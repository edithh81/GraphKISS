import argparse
import csv
import os

import torch
import yaml

from base_model import BaseModel
from load_data import DataLoader
from utils import seed_everything

parser = argparse.ArgumentParser(description="Parser for Slot-Attention-based Adaptive Subgraph Model")
parser.add_argument("--data_path", type=str, default="data/last-fm/")
parser.add_argument("--config", type=str, default=None, help="Path to YAML config file (e.g. configs/last-fm.yaml)")
parser.add_argument("--seed", type=int, default=None,
                    help="If set, overrides the seed from config; else config.seed (default 42).")
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--tag", type=str, default="", help="Experiment tag; appended to perf/CSV/checkpoint filenames")
parser.add_argument("--override", action="append", default=[],
                    help="Override config value, repeatable. Format: KEY=VALUE (VALUE parsed as YAML scalar)")
parser.add_argument("--deterministic", action="store_true",
                    help="Force cuDNN deterministic + torch deterministic algorithms (slower).")
args = parser.parse_args()


class Options(object):
    pass


def _apply_overrides(cfg, overrides):
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"--override expects KEY=VALUE, got: {item!r}")
        key, raw = item.split("=", 1)
        key = key.strip()
        cfg[key] = yaml.safe_load(raw)
    return cfg


if __name__ == "__main__":
    dataset = args.data_path.split("/")
    dataset = dataset[-1] if len(dataset[-1]) > 0 else dataset[-2]

    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    run_tag = f"{dataset}_{args.tag}" if args.tag else dataset

    opts = Options()
    opts.perf_file = os.path.join(results_dir, run_tag + "_perf.txt")
    opts.epochs_csv = os.path.join(results_dir, run_tag + "_epochs.csv")

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)

    config_path = args.config if args.config is not None else os.path.join("configs", dataset + ".yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg = _apply_overrides(cfg, args.override)

    seed = args.seed if args.seed is not None else int(cfg.get("seed", 42))
    cfg["seed"] = seed
    seed_everything(seed, deterministic=True)
    print(f"# seed: {seed}")

    for key, value in cfg.items():
        setattr(opts, key, value)

    loader = DataLoader(
        args.data_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        K_neg=opts.K_neg,
    )
    opts.n_ent = loader.n_ent
    opts.n_rel = loader.n_rel
    opts.n_users = loader.n_users
    opts.n_items = loader.n_items
    opts.n_nodes = loader.n_nodes

    config_str = yaml.safe_dump(cfg, sort_keys=True) + "\n"
    print(config_str)
    with open(opts.perf_file, "a+") as f:
        f.write(config_str)

    model = BaseModel(opts, loader)
    n_params_total = sum(p.numel() for p in model.model.parameters())
    n_params_trainable = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    print(f"# model params: {n_params_total:,} (trainable: {n_params_trainable:,})")
    with open(opts.perf_file, "a+") as f:
        f.write(f"# model_params_total: {n_params_total}\n")
        f.write(f"# model_params_trainable: {n_params_trainable}\n")

    csv_exists = os.path.exists(opts.epochs_csv)
    csv_file = open(opts.epochs_csv, "a", newline="")
    csv_writer = csv.writer(csv_file)
    if not csv_exists:
        csv_writer.writerow([
            "epoch", "recall", "ndcg",
            "train_s", "infer_s",
            "train_peak_gib", "infer_peak_gib",
            "msgs_fwd_train", "msgs_fwd_test",
            "lr", "model_params_total", "model_params_trainable",
        ])
        csv_file.flush()

    ckpt_dir = os.path.join(results_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, run_tag + "_best.pt")

    best_recall = 0.0
    best_str = ""
    for epoch in range(opts.epochs):
        print("epoch ", epoch)
        recall, ndcg, out_str = model.train_batch()

        with open(opts.perf_file, "a+") as f:
            f.write(str(epoch) + out_str)

        csv_writer.writerow([
            epoch,
            f"{recall:.6f}", f"{ndcg:.6f}",
            f"{getattr(model, 't_time', 0.0):.4f}",
            f"{getattr(model, 'i_time', 0.0):.4f}",
            f"{getattr(model, 'train_peak_gpu', 0.0):.4f}",
            f"{getattr(model, 'test_peak_gpu', 0.0):.4f}",
            int(getattr(model, 'msgs_fwd_train', 0)),
            int(getattr(model, 'msgs_fwd_test', 0)),
            f"{model.scheduler.get_last_lr()[0]:.8f}",
            n_params_total, n_params_trainable,
        ])
        csv_file.flush()

        if recall > best_recall:
            best_recall = recall
            best_str = out_str
            print("[BEST]" + str(epoch) + "\t" + best_str)
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.model.state_dict(),
                "optimizer_state_dict": model.optimizer.state_dict(),
                "scheduler_state_dict": model.scheduler.state_dict(),
                "recall": recall,
                "ndcg": ndcg,
            }, ckpt_path)
            print(f"Saved best checkpoint to: {ckpt_path}")

    csv_file.close()

    with open(opts.perf_file, "a+") as f:
        f.write("best:\n" + best_str)

    print(best_str)
