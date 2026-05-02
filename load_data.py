import os
import random
from pathlib import Path
import time
from collections import defaultdict

import torch
from scipy.sparse import csr_matrix
import numpy as np

NEW_DATASETS = [
            'new_last-fm_subset',
            'new_last-fm',
            'new_amazon-book_subset',
            'new_amazon-book',
            'new_alibaba-fashion_subset',
            'new_alibaba-fashion'
        ]
class DataLoader:
    def __init__(self, task_dir, K_neg, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.task_dir = task_dir
        self.device = device
        self.K_neg = K_neg

        data_folder = Path(task_dir).name
        if  data_folder in NEW_DATASETS:
            self.all_cf = self.read_cf(self.task_dir + 'train_1.txt')  # all_cf  (np.array)
            self.test_cf = self.read_cf(self.task_dir + 'test_1.txt')
        else:
            self.all_cf = self.read_cf(self.task_dir + 'train.txt') 
            self.test_cf = self.read_cf(self.task_dir + 'test.txt')

        self.n_users = max(max(self.all_cf[:,0]),max(self.test_cf[:,0])) + 1    
        self.n_items = max(max(self.all_cf[:,1]),max(self.test_cf[:,1])) + 1
        self.known_user_set = self.cf_to_set(self.all_cf) 
        self.test_user_set = self.cf_to_set(self.test_cf)
       
        n_all = self.all_cf.shape[0]
        rand_idx = np.random.permutation(n_all)
        self.all_cf = self.all_cf[rand_idx]

        self.triple = self.read_triples('kg.txt')   

        self.arraytriple = np.asarray(self.triple)   
        self.n_ent = max(max(self.arraytriple[:, 0]), max(self.arraytriple[:, 2])) + 1  
        self.n_nodes = self.n_ent + self.n_users    # user + entity            
        self.n_rel = max(self.arraytriple[:,1]) + 1  
                
        data_folder = Path(task_dir).name
        if  data_folder in NEW_DATASETS:            
            self.item_set = self.cf_to_item_set(self.all_cf)
            self.facts_cf, self.train_cf = self.generate_inductive_train(self.all_cf)
        else:
            self.facts_cf = self.all_cf[0:n_all*8//9]
            self.train_cf = self.all_cf[n_all*8//9:]

        self.fact_triple = self.cf_to_triple(self.facts_cf)      
        self.train_triple = self.cf_to_triple(self.train_cf)  
        self.test_triple = self.cf_to_triple(self.test_cf)
        
        # add inverse
        self.d_triple  = self.double_triple(self.triple)
        
        # add interact and user into triplets -->
        # build collaborative knowledge graph from user-item interactions and knowledge graph triples
        self.fact_data, self.known_data = self.interact_triple(self.d_triple)   

        # use user-kg 
        if  task_dir == 'data/Dis_5fold_user/' or task_dir == 'data/Dis_5fold_item/' :
            self.readukg = 1
            self.ukg = self.read_user_kg()  
            self.n_rel += 1
            self.fact_data += self.ukg 
            self.known_data += self.ukg
            print('load user kg')
        else:
            self.readukg = 0

        self.load_graph(self.fact_data)  
        self.load_test_graph(self.known_data)

        self.train_q, self.train_a, self.train_w = self.load_train_query(self.train_triple)
        self.test_q,  self.test_a  = self.load_query(self.test_triple)

        self.n_train = len(self.train_q)
        self.n_test = len(self.test_q)

        self._build_eval_item_lists()

        print('n_facts:',len(self.facts_cf), 'n_test_cf:', len(self.test_cf), 'n_train:', self.n_train,  'n_test:', self.n_test)
        print('users:', self.n_users,'items:', self.n_items, 'other entities:', self.n_ent - self.n_items )        

    def read_cf(self, file_name):  
        inter_mat = list()
        lines = open(file_name, "r").readlines()
        for al in lines:
            tmps = al.strip()
            inters = [int(i) for i in tmps.split(" ")]

            u_id, pos_ids = inters[0], inters[1:]
            pos_ids = list(set(pos_ids))
            for i_id in pos_ids:
                inter_mat.append([u_id, i_id])

        return np.array(inter_mat)   #[[u1,i1],[u1,i2],……,[un,in]]   
    
    def read_triples(self, filename):
        triples = []
        with open(os.path.join(self.task_dir, filename)) as f:
            for line in f:
                h, r, t = line.strip().split()
                h, r, t = int(h), int(r), int(t)
                        
                triples.append([h,r,t])
                       
        return triples                                  

    def double_triple(self, triples):
        new_triples = []
        for triple in triples:
            h, r, t = triple
            new_triples.append([t, r+self.n_rel, h]) 

        return triples + new_triples 
    
    def interact_triple(self, triples):
        # consider  additional relations  'interact' and 'in-interact'.
        copy_tri = triples.copy()
        for id, [h, r, t] in enumerate(copy_tri):
            # Increase entity id and relation id to avoid conflict with user ids and interact relation id.
            # To build Collaborative KG
            copy_tri[id] = [h + self.n_users, r + 2, t + self.n_users] 
        
        fact_user_triple = []
        
        for uitriple in self.fact_triple:
            u , i = uitriple[0], uitriple[2]
            
            fact_user_triple.append([u, 0, i])  
            fact_user_triple.append([i, 1, u])    

        train_user_triple = []
        for uitriple in self.train_triple:
            u , i = uitriple[0], uitriple[2]
            
            train_user_triple.append([u, 0, i])  
            train_user_triple.append([i, 1, u])  
        return copy_tri + fact_user_triple,   copy_tri + fact_user_triple + train_user_triple
    
    def cf_to_triple(self, cf):
        cf = list(cf)
        newtriple = []
        for [u,i] in cf :
            if u >= self.n_users:
                continue
            i = i + self.n_users
            newtriple.append([u, 0, i ])
        return newtriple

    def cf_to_set(self, cf):
        cf = list(cf)
        user_set = defaultdict(set)
        for [u, i] in cf:
            if u >= self.n_users:
                print("unexpected users")
                continue
            user_set[u].add(i + self.n_users)
        return user_set
    
    def read_user_kg(self):
        ukg = []
        with open(os.path.join(self.task_dir, 'ukg.txt')) as f:
            for line in f:
                h, r, t = line.strip().split()
                h, r, t = int(h), int(r), int(t)
                if h >= self.n_users or t >= self.n_users:
                    continue
                ukg.append([h, 2*self.n_rel + 2, t])
                ukg.append([t, 2*self.n_rel + 3, h])
        return ukg
        

    def cf_to_item_set(self, cf):  # item_set[i] = [u1, u3, u4……]
        cf = list(cf)
        item_set = defaultdict(list)
        for [u, i] in cf :
            if u >= self.n_users:
                print("unexpected users")
                continue
            item_set[i].append(u)
        return item_set

    def check_item_inkg(self, triple):    # return d(item) in kg
        it_inkg = np.zeros((self.n_items,1))
        for h,r,t in triple:
            if h < self.n_items:
                it_inkg[h] += 1
            if t < self.n_items:
                it_inkg[t] += 1
        return it_inkg
    
    def generate_inductive_train(self, cf):
        print("Generating inductive training set...")
        # Use set for O(1) lookup and removal
        fcf_set = set(map(tuple, cf.tolist()))  # Convert to set of tuples
        n_train = 0
        train_cf = []
        ind_item = []
        
        target_train_size = len(cf) // 8
        
        while n_train < target_train_size:
            item = random.randint(0, self.n_items - 1)
            if item in ind_item:
                continue
            
            # Process all users for this item  <=> make item totally new for training
            for u in self.item_set[item]:
                pair = (u, item)
                if pair in fcf_set:
                    train_cf.append([u, item])
                    fcf_set.discard(pair)  # O(1) removal
            
            ind_item.append(item)
            n_train += len(self.item_set[item])
        
        # Convert back to numpy array
        fcf = np.array(list(fcf_set))
        train_cf = np.array(train_cf)
        
        print(f"Inductive training set generated with {len(ind_item)} new items.")
        return fcf, train_cf

    def load_graph(self, triples):      
        """
        Input:
            triples: list of triples
            - User-item interactions → [u, 0, i] (relation 0 = "interact")
            - Inverse interactions → [i, 1, u] (relation 1 = "in-interact")
            - KG facts → [h, r, t] from kg.txt
            - Inverse KG facts → [t, r+n_rel, h] via double_triple()
        """
        # add self-loop
        # idd = [(i, 2*self.n_rel+2, i) for i in range(self.n_nodes)]
        idd = np.concatenate([np.expand_dims(np.arange(self.n_nodes),1), (2*self.n_rel+2)*np.ones((self.n_nodes, 1)), np.expand_dims(np.arange(self.n_nodes),1)], 1)

        # Merge with existing triples
        self.KG = np.concatenate([np.array(triples), idd], 0)
        self.n_fact = len(self.KG)

        # build sparse matrix
        # Each row = one edge (triplet) from self.KG.
        # Each column = one node index.
        # Entry = 1 if that edge’s head node equals the column index.
        # So M_sub[i, j] = 1 if the i-th edge originates from node j.
        self.M_sub = csr_matrix(
            (np.ones((self.n_fact,)), # values to be placed in the sparse matrix
             (np.arange(self.n_fact), self.KG[:,0])), # row indices, column indices
            shape=(self.n_fact, self.n_nodes)) # facts and its head nodes

        # Invalidate any cached GPU CSR; rebuilt lazily on next get_neighbors_gpu call.
        self._train_gpu_csr = None

    def load_test_graph(self, triples):
        idd = np.concatenate([np.expand_dims(np.arange(self.n_nodes),1), (2*self.n_rel+2)*np.ones((self.n_nodes, 1)), np.expand_dims(np.arange(self.n_nodes),1)], 1)

        self.tKG = np.concatenate([np.array(triples), idd], 0)

        self.tn_fact = len(self.tKG)
        self.tM_sub = csr_matrix((np.ones((self.tn_fact,)), (np.arange(self.tn_fact), self.tKG[:,0])), shape=(self.tn_fact, self.n_nodes))

        self._test_gpu_csr = None

    # def load_train_query(self, triples):
    #     triples.sort(key=lambda x:(x[0], x[1]))
    #     pos_items = defaultdict(lambda:list())
    #     neg_items = defaultdict(list)
        
    #     for trip in triples:
    #         h, r, t = trip            
    #         pos_items[(h,r)].append(t)
    #         while True:
    #             neg_item = np.random.randint(low=self.n_users, high=self.n_users + self.n_items, size=1)[0]  #重要修改！low=n_users
    #             if neg_item not in self.known_user_set[h]:
    #                 break
    #         neg_items[(h,r)].append(neg_item)
        
    #     queries = []
    #     answers = []
    #     wrongs = []
    #     for key in pos_items:
    #         queries.append(key) # Each query (h,r)
    #         answers.append(np.array(pos_items[key])) # All positive items for (h,r)
    #         wrongs.append(np.array(neg_items[key]))  # Sampled negative items for positive items for (h,r)
        
    #     return queries, answers, wrongs

    def load_train_query(self, triples):
        """
        Build grouped train queries and sample K negatives per positive item.

        Output:
            queries: list of (h, r)
            answers: list of np.ndarray with shape [num_pos]
            wrongs:  list of np.ndarray with shape [num_pos, K_neg]
        """
        triples.sort(key=lambda x: (x[0], x[1]))

        pos_items = defaultdict(list)
        for h, r, t in triples:
            pos_items[(h, r)].append(t)

        queries = []
        answers = []
        wrongs = []

        item_low = self.n_users
        item_high = self.n_users + self.n_items

        for (h, r), pos_list in pos_items.items():
            pos_arr = np.asarray(pos_list, dtype=np.int64)
            num_pos = len(pos_arr)

            neg = np.empty((num_pos, self.K_neg), dtype=np.int64)
            known_items = self.known_user_set[h]  # a set for O(1) membership

            # Fill negatives in batches, only resampling invalid slots
            remaining_mask = np.ones((num_pos, self.K_neg), dtype=bool)

            while remaining_mask.any():
                n_need = remaining_mask.sum()
                candidates = np.random.randint(
                    low=item_low,
                    high=item_high,
                    size=n_need,
                    dtype=np.int64
                )

                valid = np.fromiter(
                    (c not in known_items for c in candidates),
                    dtype=bool,
                    count=n_need
                )

                flat_idx = np.flatnonzero(remaining_mask)
                good_idx = flat_idx[valid]

                neg.flat[good_idx] = candidates[valid]
                remaining_mask.flat[good_idx] = False

            queries.append((h, r))
            answers.append(pos_arr)
            wrongs.append(neg)

        return queries, answers, wrongs

    def load_query(self, triples):
        triples.sort(key=lambda x:(x[0], x[1]))
        trip_hr = defaultdict(lambda:list())

        for trip in triples:
            h, r, t = trip
            trip_hr[(h,r)].append(t)
        
        queries = []
        answers = []
        for key in trip_hr:
            queries.append(key)
            answers.append(np.array(trip_hr[key]))
        return queries, answers

    def get_neighbors(self, nodes, mode='train'):
        
        if mode=='train':
            KG = self.KG
            M_sub = self.M_sub # (n_edges, n_nodes) (edges and its one-hot head nodes )
        else:
            KG = self.tKG
            M_sub = self.tM_sub
        
        # nodes: n(node) x 2 with (batch_idx, node_idx)
        node_1hot = csr_matrix(
            (np.ones(len(nodes)), (nodes[:,1], nodes[:,0])), 
            shape=(self.n_nodes, nodes.shape[0])
        ) # [n_nodes x batch_size]
        edge_1hot = M_sub.dot(node_1hot) # [n_edges x batch_size] (edges and the batch they belong to)
        edges = np.nonzero(edge_1hot) # tuple (edge_idx, batch_idx) (indices of non-zero rows, cols)
        sampled_edges = np.concatenate(
            [np.expand_dims(edges[1],1), KG[edges[0]]],
            axis=1
        )     # (batch_idx, head, rela, tail)
        sampled_edges = torch.LongTensor(sampled_edges).to(self.device)

        # index to nodes
        # head_nodes: tuple of unique (batch_idx, head_node); head_index: index of head_nodes in sampled_edges
        head_nodes, head_index = torch.unique(sampled_edges[:,[0,1]], dim=0, sorted=True, return_inverse=True)
        tail_nodes, tail_index = torch.unique(sampled_edges[:,[0,3]], dim=0, sorted=True, return_inverse=True)

        # 
        sampled_edges = torch.cat([sampled_edges, head_index.unsqueeze(1), tail_index.unsqueeze(1)], 1)
       
        mask = sampled_edges[:,2] == (self.n_rel*2 + 2)
        _, old_idx = head_index[mask].sort()
        old_nodes_new_idx = tail_index[mask][old_idx]

        return tail_nodes, sampled_edges, old_nodes_new_idx

    def _build_gpu_csr(self, KG_np):
        """Build a head-sorted CSR (indptr, rels, tails) on self.device for fast neighbor expansion."""
        KG = torch.as_tensor(np.asarray(KG_np, dtype=np.int64), device=self.device)  # [E, 3]
        heads = KG[:, 0]
        order = torch.argsort(heads, stable=True)
        KG_sorted = KG.index_select(0, order)
        heads_sorted = KG_sorted[:, 0]
        rels = KG_sorted[:, 1].contiguous()
        tails = KG_sorted[:, 2].contiguous()
        counts = torch.bincount(heads_sorted, minlength=self.n_nodes)
        indptr = torch.zeros(self.n_nodes + 1, dtype=torch.long, device=self.device)
        indptr[1:] = torch.cumsum(counts, dim=0)
        return indptr, rels, tails

    def get_neighbors_gpu(self, nodes, mode='train'):
        """
        GPU-side frontier expansion.

        Args:
            nodes: [M, 2] LongTensor on self.device with columns (batch_idx, node_idx).
            mode:  'train' uses self.KG; anything else uses self.tKG.

        Returns the same tuple as ``get_neighbors``:
            tail_nodes:         [T, 2] sorted unique (batch_idx, tail_node) on device.
            sampled_edges:      [E, 6] on device — (batch_idx, head, rel, tail, head_idx, tail_idx).
            old_nodes_new_idx:  mapping old head-positions to new tail-positions via self-loop edges.
        """
        if mode == 'train':
            if getattr(self, '_train_gpu_csr', None) is None:
                self._train_gpu_csr = self._build_gpu_csr(self.KG)
            indptr, edge_rels, edge_tails = self._train_gpu_csr
        else:
            if getattr(self, '_test_gpu_csr', None) is None:
                self._test_gpu_csr = self._build_gpu_csr(self.tKG)
            indptr, edge_rels, edge_tails = self._test_gpu_csr

        if nodes.numel() == 0:
            empty = torch.empty(0, dtype=torch.long, device=self.device)
            return (
                torch.empty(0, 2, dtype=torch.long, device=self.device),
                torch.empty(0, 6, dtype=torch.long, device=self.device),
                empty,
            )

        node_batch = nodes[:, 0]
        node_heads = nodes[:, 1]
        starts = indptr[node_heads]
        ends = indptr[node_heads + 1]
        counts = ends - starts  # [M]

        total = int(counts.sum().item())
        group_id = torch.repeat_interleave(
            torch.arange(nodes.size(0), device=self.device), counts
        )  # [total]
        group_offsets = torch.cumsum(counts, dim=0) - counts  # [M]
        local_idx = torch.arange(total, device=self.device) - group_offsets[group_id]  # [total]
        edge_ids = starts[group_id] + local_idx  # [total]

        batch_ids = node_batch[group_id]
        heads_out = node_heads[group_id]
        rels_out = edge_rels[edge_ids]
        tails_out = edge_tails[edge_ids]

        sampled_edges = torch.stack([batch_ids, heads_out, rels_out, tails_out], dim=1)

        head_nodes, head_index = torch.unique(
            sampled_edges[:, [0, 1]], dim=0, sorted=True, return_inverse=True
        )
        tail_nodes, tail_index = torch.unique(
            sampled_edges[:, [0, 3]], dim=0, sorted=True, return_inverse=True
        )

        sampled_edges = torch.cat(
            [sampled_edges, head_index.unsqueeze(1), tail_index.unsqueeze(1)], dim=1
        )

        self_loop_rel = self.n_rel * 2 + 2
        mask = sampled_edges[:, 2] == self_loop_rel
        _, old_idx = head_index[mask].sort()
        old_nodes_new_idx = tail_index[mask][old_idx]

        return tail_nodes, sampled_edges, old_nodes_new_idx

    def get_batch(self, batch_idx, steps=2, data='train'):
        if data=='train':                                   
            query, answer, wrongs = np.array(self.train_q), self.train_a, self.train_w
        if data=='test':
            query, answer = np.array(self.test_q), self.test_a

        subs = []
        rels = []
        objs = []
        pos = []
        neg =[]
        subs = query[batch_idx, 0]
        rels = query[batch_idx, 1]
  
        if data =='train':
            pos = [answer[i] for i in batch_idx]
            neg = [wrongs[i] for i in batch_idx]
            return subs, rels, pos, neg
        else :
            objs = np.zeros((len(batch_idx), self.n_nodes))
            for i in range(len(batch_idx)):
                objs[i][answer[batch_idx[i]]] = 1
            return subs, rels, objs


    def _build_eval_item_lists(self):
        """Precompute per-user item-local tensors for vectorized eval."""
        self._known_items_local_per_user = [None] * self.n_users
        self._test_items_local_per_user = [None] * self.n_users
        empty = torch.empty(0, dtype=torch.long)
        for u in range(self.n_users):
            known = self.known_user_set.get(u, None)
            if known:
                self._known_items_local_per_user[u] = torch.tensor(
                    [i - self.n_users for i in known], dtype=torch.long
                )
            else:
                self._known_items_local_per_user[u] = empty
            test = self.test_user_set.get(u, None)
            if test:
                self._test_items_local_per_user[u] = torch.tensor(
                    [i - self.n_users for i in test], dtype=torch.long
                )
            else:
                self._test_items_local_per_user[u] = empty

    def get_eval_tensors(self, subs, device):
        """
        Build vectorized eval tensors for a batch of users.

        Args:
            subs: iterable of user ids (length B).
            device: torch device for the returned tensors.
        Returns:
            known_mask: [B, n_items] bool — True where the item was in training.
            pos_padded: [B, max_pos] long — item-local test positives, padded with -1.
            pos_counts: [B] long — number of positives per user.
        """
        B = len(subs)
        known_tensors = [self._known_items_local_per_user[int(u)] for u in subs]
        test_tensors = [self._test_items_local_per_user[int(u)] for u in subs]

        known_counts = torch.tensor([t.numel() for t in known_tensors], dtype=torch.long)
        test_counts_cpu = torch.tensor([t.numel() for t in test_tensors], dtype=torch.long)

        known_mask = torch.zeros(B, self.n_items, dtype=torch.bool)
        if int(known_counts.sum()) > 0:
            known_flat = torch.cat(known_tensors)
            known_rows = torch.repeat_interleave(torch.arange(B, dtype=torch.long), known_counts)
            known_mask[known_rows, known_flat] = True

        max_pos = int(test_counts_cpu.max().item()) if B > 0 else 0
        pos_padded = torch.full((B, max(max_pos, 1)), -1, dtype=torch.long)
        if max_pos > 0 and int(test_counts_cpu.sum()) > 0:
            test_flat = torch.cat(test_tensors)
            test_rows = torch.repeat_interleave(torch.arange(B, dtype=torch.long), test_counts_cpu)
            offsets = test_counts_cpu.cumsum(0) - test_counts_cpu
            slot = torch.arange(test_flat.numel(), dtype=torch.long) - offsets[test_rows]
            pos_padded[test_rows, slot] = test_flat

        return (
            known_mask.to(device, non_blocking=True),
            pos_padded.to(device, non_blocking=True),
            test_counts_cpu.to(device, non_blocking=True),
        )

    def shuffle_train(self,):
        
        if self.task_dir == 'data/Dis_5fold_item/' : 
            self.facts_cf, self.train_cf = self.generate_inductive_train(self.all_cf)
            self.fact_triple = self.cf_to_triple(self.facts_cf)      
            self.train_triple = self.cf_to_triple(self.train_cf) 
            
            self.fact_data,_ = self.interact_triple(self.d_triple)
            if self.readukg == 1:
                self.fact_data += self.ukg 
            self.load_graph(self.fact_data)
            self.train_q, self.train_a, self.train_w = self.load_train_query(self.train_triple)
            self.n_train = len(self.train_q)
        elif Path(self.task_dir).name in NEW_DATASETS:
            self.train_triple = np.array(self.train_triple)
            n_all = len(self.train_triple)
            rand_idx = np.random.permutation(n_all)
            self.train_triple = self.train_triple[rand_idx].tolist()
            self.train_q, self.train_a, self.train_w = self.load_train_query(self.train_triple)
            self.n_train = len(self.train_q)
        else:
            fact_triple = np.array(self.fact_triple)
            train_triple = np.array(self.train_triple)
            all_ui_triple = np.concatenate([fact_triple, train_triple], axis=0)
            n_all = len(all_ui_triple)
            rand_idx = np.random.permutation(n_all)
            all_ui_triple = all_ui_triple[rand_idx]

            self.fact_triple = all_ui_triple[:n_all*6//7].tolist()
            self.train_triple = all_ui_triple[n_all*6//7:].tolist()
            
            self.fact_data,_ = self.interact_triple(self.d_triple)
            if self.readukg == 1:
                self.fact_data += self.ukg 
            self.load_graph(self.fact_data)
            self.train_q, self.train_a, self.train_w = self.load_train_query(self.train_triple)
            self.n_train = len(self.train_q)

