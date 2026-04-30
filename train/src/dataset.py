########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import json, math, random, os, sys
import numpy as np
import torch
from torch.utils.data import Dataset
from pytorch_lightning.utilities import rank_zero_info
from .binidx import MMapIndexedDataset

def is_prime(n):
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

class MyDataset(Dataset):
    def __init__(self, args):
        self.args = args

        self.vocab_size = args.vocab_size
        rank_zero_info(f"Current vocab size = {self.vocab_size} (make sure it's correct)")

        self.data = MMapIndexedDataset(args.data_file)
        self.data_size = len(self.data._bin_buffer) // self.data._index._dtype_size
        rank_zero_info(f"Data has {self.data_size} tokens.")

        self.samples_per_epoch = args.epoch_steps * args.real_bsz
        assert self.samples_per_epoch == 40320
        rank_zero_info(f"########## train stage {args.train_stage} ##########")
        dataset_slot = self.data_size // args.ctx_len

        assert is_prime(args.magic_prime)
        assert args.magic_prime % 3 == 2
        assert args.magic_prime / dataset_slot > 0.9 and args.magic_prime / dataset_slot <= 1

    def __len__(self):
        return self.args.epoch_steps * self.args.micro_bsz

    def __getitem__(self, idx):
        args = self.args
        rank = self.global_rank
        epoch = self.real_epoch
        world_size = self.world_size
        # print(f"epoch {epoch} idx {idx} rank {rank}/{world_size}")

        ctx_len = args.ctx_len
        magic_prime = args.magic_prime

        ii = 1 + epoch * self.samples_per_epoch + (idx * world_size) + rank

        factor = (math.sqrt(5) - 1) / 2
        factor = int(magic_prime * factor)
        i = ((factor * ii * ii * ii) % magic_prime) * ctx_len
        # print(f"epoch {epoch} idx {idx} rank {rank}/{world_size} ii {ii} pos {round(i / self.data_size, 3)}")

        if getattr(args, "diffusion_mode", 0) == 1:
            block_size = args.diff_block_size
            n_blocks = ctx_len // (3 * block_size)
            raw_len = n_blocks * block_size

            dix = self.data.get(idx=0, offset=i, length=raw_len).astype(int)
            clean = torch.tensor(dix, dtype=torch.long)

            r_lo = float(args.diff_min_mask_ratio)
            r_hi = float(args.diff_max_mask_ratio)
            r = r_lo + (r_hi - r_lo) * float(torch.rand(1).item())
            mask_pos = torch.rand(raw_len) < r
            masked = torch.where(mask_pos, torch.full_like(clean, args.diff_mask_id), clean)

            x = torch.full((ctx_len,), args.diff_pad_id, dtype=torch.long)
            y = torch.full((ctx_len,), -100, dtype=torch.long)

            clean_blk = clean.view(n_blocks, block_size)
            masked_blk = masked.view(n_blocks, block_size)
            mask_blk = mask_pos.view(n_blocks, block_size)

            x_view = x[: n_blocks * 3 * block_size].view(n_blocks, 3, block_size)
            x_view[:, 0] = masked_blk
            x_view[:, 1] = masked_blk
            x_view[:, 2] = clean_blk

            y_view = y[: n_blocks * 3 * block_size].view(n_blocks, 3, block_size)
            y_view[:, 1] = torch.where(mask_blk, clean_blk, torch.full_like(clean_blk, -100))

            return x, y

        req_len = ctx_len + 1
        dix = self.data.get(idx=0, offset=i, length=req_len).astype(int)

        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)

        return x, y
