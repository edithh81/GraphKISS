import time

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm

from models.subgraph_model import AdaptiveSubgraphModel
from utils import cal_bpr_loss, ndcg_at_k, recall_at_k


class BaseModel(object):
    def __init__(self, args, loader):
        self.model = AdaptiveSubgraphModel(args, loader)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.loader = loader
        self.n_ent = loader.n_ent
        self.n_rel = loader.n_rel
        self.n_users = loader.n_users
        self.n_items = loader.n_items
        self.n_nodes = loader.n_nodes

        self.n_batch = args.n_batch
        self.n_tbatch = args.n_tbatch

        self.known_user_set = loader.known_user_set
        self.test_user_set = loader.test_user_set

        self.n_train = loader.n_train
        self.n_test = loader.n_test
        self.n_layer = args.n_layer
        self.args = args
        self.K_neg = args.K_neg

        self.optimizer = AdamW(self.model.parameters(), lr=args.lr, weight_decay=args.lamb)
        self.scheduler = ExponentialLR(self.optimizer, gamma=args.decay_rate)

        self.t_time = 0

    def train_batch(self, epoch: int = 0):
        epoch_loss = 0.0
        epoch_bpr = 0.0
        batch_size = self.n_batch
        n_batch = self.loader.n_train // batch_size + (self.loader.n_train % batch_size > 0)

        # Anneal temperature once per epoch before any forward passes.
        if hasattr(self.model, "set_temperature"):
            current_tau = self.model.set_temperature(epoch)
        else:
            current_tau = getattr(self.model, "tau_p", 1.0)
        self.current_tau = current_tau
        print(f"[Epoch {epoch}] tau = {current_tau:.4f}")
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t_time = time.time()
        self.model.train()
        edges_per_fwd_train = []
        pbar = tqdm(range(n_batch), desc="Train", unit="batch")
        for i in pbar:
            start = i * batch_size
            end = min(self.loader.n_train, (i + 1) * batch_size)
            batch_idx = np.arange(start, end)
            subs, rels, pos, neg = self.loader.get_batch(batch_idx)

            self.optimizer.zero_grad()
            if hasattr(self.model, "edge_counts_layer"):
                self.model.edge_counts_layer = []
            scores, aux = self.model(subs, rels, return_aux=True)
            if hasattr(self.model, "edge_counts_layer"):
                edges_per_fwd_train.append(int(sum(self.model.edge_counts_layer)))
            bpr_loss = cal_bpr_loss(self.n_users, pos, neg, scores)

            loss = (
                bpr_loss
                + float(getattr(self.args, "lambda_intent_div", 1e-4)) * aux["intent_div_loss"]
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            pbar.set_postfix(loss=f"{loss.item():.4f}", bpr=f"{bpr_loss.item():.4f}", tau=f"{current_tau:.3f}")

            epoch_loss += loss.item()
            epoch_bpr += bpr_loss.item()

        self.t_time = time.time() - t_time
        self.train_peak_gpu = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
        self.msgs_fwd_train = int(np.mean(edges_per_fwd_train)) if edges_per_fwd_train else 0
        self.scheduler.step()
        print(
            f"epoch_loss: {epoch_loss:.4f}  epoch_bpr: {epoch_bpr:.4f}  "
            f"tau: {current_tau:.4f}  lr: {self.scheduler.get_last_lr()[0]:.6f}"
        )
        print("start test")
        recall, ndcg, out_str = self.test_batch()

        self.loader.shuffle_train()
        print(out_str)
        return recall, ndcg, out_str

    def test_batch(self, K=20):
        batch_size = self.n_tbatch
        n_data = self.n_test
        n_batch = n_data // batch_size + (n_data % batch_size > 0)
        self.model.eval()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        i_time = time.time()
        total_recall = 0.0
        total_ndcg = 0.0
        edges_per_fwd_test = []

        pbar = tqdm(range(n_batch), desc="Test", unit="batch")
        with torch.no_grad():
            for idx in pbar:
                start = idx * batch_size
                end = min(n_data, (idx + 1) * batch_size)
                batch_idx = np.arange(start, end)
                subs, rels, _ = self.loader.get_batch(batch_idx, data="test")

                if hasattr(self.model, "edge_counts_layer"):
                    self.model.edge_counts_layer = []
                scores = self.model(subs, rels, mode="test")  # [B, n_items] on GPU
                if hasattr(self.model, "edge_counts_layer"):
                    edges_per_fwd_test.append(int(sum(self.model.edge_counts_layer)))

                known_mask, pos_padded, pos_counts = self.loader.get_eval_tensors(subs, self.device)
                scores = scores.masked_fill(known_mask, float("-inf"))
                _, topk_idx = torch.topk(scores, K, dim=-1)

                batch_recall = recall_at_k(topk_idx, pos_padded, pos_counts)
                batch_ndcg = ndcg_at_k(topk_idx, pos_padded, pos_counts)

                total_recall += batch_recall.sum().item()
                total_ndcg += batch_ndcg.sum().item()
                b = batch_recall.numel()
                pbar.set_postfix(
                    recall=f"{batch_recall.sum().item() / max(b, 1):.4f}",
                    ndcg=f"{batch_ndcg.sum().item() / max(b, 1):.4f}",
                )

        recall = total_recall / n_data
        ndcg = total_ndcg / n_data

        i_time = time.time() - i_time
        self.i_time = i_time
        self.test_peak_gpu = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
        self.msgs_fwd_test = int(np.mean(edges_per_fwd_test)) if edges_per_fwd_test else 0
        out_str = (
            "[TEST] recall:%.4f  ndcg:%.4f   [TIME] train:%.4f inference:%.4f  "
            "[GPU] train_peak:%.2fGiB infer_peak:%.2fGiB  "
            "[MSG] msgs/fwd:%d msgs/fwd_test:%d\n"
        ) % (
            recall, ndcg, self.t_time, i_time,
            getattr(self, "train_peak_gpu", 0.0), self.test_peak_gpu,
            getattr(self, "msgs_fwd_train", 0), self.msgs_fwd_test,
        )
        return recall, ndcg, out_str
