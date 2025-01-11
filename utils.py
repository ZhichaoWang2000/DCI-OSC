import torch
from torchmetrics import Metric
import os
import ast
import numpy as np
import pandas as pd
import random
import ffmpeg
import time


class StatePrec1(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("prec1", default=torch.tensor([0.0, 0.0, 0.0, 0.0]), dist_reduce_fx="sum")
        self.add_state("cnt", default=torch.tensor([0, 0, 0, 0]), dist_reduce_fx="sum")

    def update(self, idx, gt):
        unique_labels = torch.unique(gt).cpu().numpy().astype(int)
        unique_labels = unique_labels[unique_labels > 0]
        correct_cnt = 0
        for i, label in enumerate(unique_labels):
            state = label - 1
            is_correct = (gt[idx[state]].item() == label)
            self.prec1[state] += is_correct
            correct_cnt += is_correct
            self.cnt[state] += 1
        self.prec1[-1] += (correct_cnt / len(unique_labels))
        self.cnt[-1] += 1

    def compute(self):
        prec1 = self.prec1 / self.cnt
        return {
            "s0": prec1[0].item(),
            "s1": prec1[1].item(),
            "s2": prec1[2].item(),
            "avg": prec1[3].item()
        }


def derive_label(annotation, n_frames):
    state_to_label = {
        'initial_state': 1,
        'transitioning_state': 2,
        'end_state': 3,
    }
    gt = np.zeros(n_frames)
    for state in ['initial_state', 'transitioning_state', 'end_state']:
        for time_range in ast.literal_eval(annotation[state]):
            start, end = time_range
            gt[round(start):round(end)] = state_to_label[state]
    return gt
