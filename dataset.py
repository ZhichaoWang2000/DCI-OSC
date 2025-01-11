import os
import pickle

import torch
import numpy as np
import pandas as pd
import ast

from torch.utils.data import Dataset

from utils import derive_label


def build_vocab(args):
    df = pd.read_csv(os.path.join(args.ann_dir, 'howtochange_eval.csv'))

    df['verb'] = df['osc'].apply(lambda x: x.split('_')[0])
    if 'all' not in args.sc_list:
        df = df[df['verb'].isin(args.sc_list)]
    vocab = {'background': 0}
    key = 'verb'
    for i, k in enumerate(df[key].unique()):
        vocab[k] = i + 1
    sc_list = df['verb'].unique().tolist()
    print(f"Vocab len: {len(vocab)}")
    print(f"Vocab {vocab}")
    print(f"State Transition {sc_list}")
    return vocab, sc_list, df


class HowToChangeFeatDataset(Dataset):
    def __init__(self, args):
        self.args = args
        self.feat_dir = args.feat_dir
        self.vocab, _, self.df = build_vocab(args)
        print(f"HowToChange Eval: state transition = {args.sc_list} -> {len(self.df)} videos")
        self.max_seq_len = int(self.df['duration'].max())
        print(f"Max sequence length: {self.max_seq_len}")

    def load_feat(self, row):
        video_id = row['video_id']
        feat_path = os.path.join(self.feat_dir, 'feats', video_id + '.pth.tar')
        feat = torch.load(feat_path, weights_only=True)

        start = float(row['start_time'])
        duration = float(row['duration'])
        end = min(start + duration, feat.shape[0])
        feat = feat[int(start):int(end)]

        if self.args.det > 0:
            obj_features = torch.zeros_like(feat)
            file_name = row['video_name'] + '_obj.pth.tar'
            obj_feat_path = os.path.join(self.feat_dir, 'feats_handobj', row['osc'], file_name)

            if os.path.exists(obj_feat_path):
                obj_feat = torch.load(obj_feat_path, weights_only=True)
                obj_idx = np.load(obj_feat_path.replace('.pth.tar', '.npy'))
                obj_idx = obj_idx[obj_idx < len(obj_features)]
                obj_features[obj_idx] = obj_feat[0:len(obj_idx)]
            else:
                print(f'Warning! {obj_feat_path} do not exist')
            feat = torch.cat((feat, obj_features), dim=-1)

        return feat

    def load_bert(self, q):
        path = ""
        with open(path, 'rb') as file:
            data = pickle.load(file)
        bert = data[q]
        return bert

    def derive_label(self, annotation, n_frames):
        gt = np.zeros(n_frames)
        for state in ['s0', 's1', 's2']:
            for time_range in ast.literal_eval(annotation[state]):
                start, end = time_range
                gt[round(start):round(end)] = int(state[-1]) + 1
        return gt

    def __getitem__(self, index):
        row = self.df.iloc[index]
        osc = row['osc']
        q = osc.split('_')[0]
        bert = self.load_bert(q)
        feat = self.load_feat(row)
        label = torch.from_numpy(derive_label(row, feat.shape[0]))
        return feat, label, osc, row['is_novel_osc'], bert

    def __len__(self):
        return len(self.df)


class HowToChangeFeatCLIPLabelDataset(HowToChangeFeatDataset):
    def __init__(self, args):
        super().__init__(args)
        self.load_data()

    def load_data(self):
        df = pd.read_csv(os.path.join(self.args.ann_dir, 'howtochange_unlabeled_train.csv'))
        df['verb'] = df['osc'].apply(lambda x: x.split('_')[0])
        if 'all' not in self.args.sc_list:
            df = df[df['verb'].isin(self.args.sc_list)]
        print(f"HowToChange Train: state transition = {self.args.sc_list} -> {len(df)} videos")
        self.max_seq_len = int(df['duration'].max())
        self.data_list = []
        for i, row in df.iterrows():
            pl_path = os.path.join(self.args.pseudolabel_dir, row['osc'], row['video_name'] + '.npz')
            if not os.path.exists(pl_path):
                print(f'Missing pseudo label {pl_path}')
                continue
            pl = np.load(pl_path)['arr_0']
            self.data_list.append({
                'row': row,
                'pseudo_label': pl
            })
        print(f"{len(self.data_list)} training clips loaded")

    def pad_sequence(self, feat):
        t, dim = feat.shape
        padded_feat = torch.cat((feat, torch.zeros((self.max_seq_len - t, dim))), dim=0)
        return padded_feat

    def pad_label(self, label):
        t = label.shape[0]
        padded_label = torch.cat((label, -torch.ones(self.max_seq_len - t)), dim=0).long()
        return padded_label

    def __getitem__(self, index):
        data_dict = self.data_list[index]
        osc = data_dict['row']['osc']
        feat = self.load_feat(data_dict['row'])
        pseudo_label = torch.from_numpy(data_dict['pseudo_label'])
        pseudo_label = pseudo_label[0: feat.shape[0]]
        feat = self.pad_sequence(feat)
        pseudo_label = self.pad_label(pseudo_label)
        q = osc.split('_')[0]
        bert = self.load_bert(q)
        return feat, pseudo_label, bert

    def __len__(self):
        return len(self.data_list)
