import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
import torch.nn.functional as F
from torchmetrics.classification import MulticlassPrecision, MulticlassF1Score

from dataset import build_vocab
from loader import construct_loader
from utils import StatePrec1
from model import DCI_OSC


class DCI_task(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.vocab, self.sc_list, _ = build_vocab(args)
        self.category_num = len(self.vocab) - 1
        args.vocab_size = 3 * self.category_num + 1
        args.input_dim = 768 * (1 + self.args.det)
        self.model = DCI_OSC(args)
        self.global_dic = torch.load("/data/wzc/data/DCI-OSC/global_dic.pth.tar", weights_only=True)
        self.ce = nn.CrossEntropyLoss(ignore_index=-1)
        self.kl_mb = nn.KLDivLoss(reduction='batchmean')
        self.kl_b = nn.KLDivLoss(reduction='batchmean')
        self.a = args.a
        self.b = args.b
        self.eval_setting = ['known', 'novel', 'all']
        self.state_prec1 = {sc: {key: StatePrec1() for key in self.eval_setting} for sc in self.sc_list}
        self.state_prec = MulticlassPrecision(num_classes=4, average="none")
        self.f1_score = MulticlassF1Score(num_classes=4, average="none")
        self.metric_name_list = ['avg_f1_known', 'avg_f1_novel', 'avg_prec_known', 'avg_prec_novel', 'avg_prec1_known',
                                 'avg_prec1_novel'] if len(
            self.sc_list) > 1 else [f'{self.sc_list[0]}_avg_f1_known', f'{self.sc_list[0]}_avg_f1_novel',
                                    f'{self.sc_list[0]}_avg_prec_known', f'{self.sc_list[0]}_avg_prec_novel',
                                    f'{self.sc_list[0]}_avg_prec1_known', f'{self.sc_list[0]}_avg_prec1_novel']

    def training_step(self, batch, batch_idx):
        feat, pl, bert = batch
        out_fg, out_bg, out_m = self.model(feat, bert, self.global_dic)
        ce_loss = self.ce(out_fg, pl.view(-1))
        kl_loss = self.kl_mb(F.log_softmax(out_m, dim=1), F.softmax(out_fg, dim=1))
        klb_loss = self.kl_b(F.log_softmax(out_bg, dim=1), out_bg.new_ones(out_bg.size()) / self.args.vocab_size)
        loss = ce_loss + self.a * kl_loss + self.b * klb_loss

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=64)
        self.log("ce_loss", ce_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=64)
        self.log("kl_loss", kl_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=64)
        self.log("klb_loss", klb_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=64)
        return loss

    def infer_state_idx(self, prob):
        pred_idx = torch.argmax(prob, dim=0).cpu().numpy()
        pred_idx = pred_idx[1:]
        return pred_idx

    def validation_step(self, batch, batch_idx):
        feat, label, osc, is_novel, bert = batch
        osc = osc[0]
        sc_name = osc.split('_')[0]

        name = 'novel' if is_novel.item() else 'known'
        pred, _, _ = self.model(feat, bert, self.global_dic)
        prob = torch.softmax(pred, dim=-1)
        category_pred = prob[:, 1:].reshape(-1, self.category_num, 3).sum(dim=0).sum(dim=-1)
        inferred_catgeory_id = category_pred.argmax().item() + 1
        key = sc_name
        gt_category_id = self.vocab[key]

        self.log('category_acc', inferred_catgeory_id == gt_category_id, on_step=False, on_epoch=True, batch_size=1)
        category_id = inferred_catgeory_id
        st_prob = prob[:, [0, 3 * category_id - 2, 3 * category_id - 1, 3 * category_id]]

        pred_idx = self.infer_state_idx(st_prob)
        prec = self.state_prec(st_prob, label.view(-1))
        f1 = self.f1_score(st_prob, label.view(-1))
        self.state_prec1[sc_name][name].update(pred_idx, label.view(-1))
        self.state_prec1[sc_name]['all'].update(pred_idx, label.view(-1))
        unique_labels = torch.unique(label).cpu().numpy().astype(int)
        unique_labels = unique_labels[unique_labels > 0]
        avg_prec = prec[unique_labels].mean()
        avg_f1 = f1[unique_labels].mean()

        self.log(f'{sc_name}_avg_prec_{name}', avg_prec, on_step=False, on_epoch=True, batch_size=1)
        self.log(f'{sc_name}_avg_prec', avg_prec, on_step=False, on_epoch=True, prog_bar=True, batch_size=1)
        self.log(f'{sc_name}_avg_f1_{name}', avg_f1, on_step=False, on_epoch=True, batch_size=1)
        self.log(f'{sc_name}_avg_f1', avg_f1, on_step=False, on_epoch=True, prog_bar=True, batch_size=1)

    def on_validation_epoch_end(self):
        for i, sc_name in enumerate(self.sc_list):
            for key in self.eval_setting:
                val_prec1 = self.state_prec1[sc_name][key].compute()
                self.log(f'{sc_name}_avg_prec1_{key}', val_prec1['avg'], on_step=False, on_epoch=True, prog_bar=True,
                         batch_size=1)
                self.state_prec1[sc_name][key].reset()

        if len(self.sc_list) > 1:
            avg_result = np.zeros((len(self.sc_list), 6))
            value_name = ['avg_f1_known', 'avg_f1_novel', 'avg_prec_known', 'avg_prec_novel', 'avg_prec1_known',
                          'avg_prec1_novel']
            for i, sc_name in enumerate(self.sc_list):
                value_list = [self.trainer.callback_metrics.get(f'{sc_name}_{v}').item() for v in value_name]
                avg_result[i] = value_list
            avg_result = avg_result.mean(axis=0)
            for i, v in enumerate(value_name):
                self.log(f'{v}', avg_result[i], on_step=False, on_epoch=True, prog_bar=True, batch_size=1)

        val_name = [f'{self.sc_list[0]}_avg_f1_known', f'{self.sc_list[0]}_avg_f1_novel',
                    f'{self.sc_list[0]}_avg_prec_known', f'{self.sc_list[0]}_avg_prec_novel',
                    f'{self.sc_list[0]}_avg_prec1_known', f'{self.sc_list[0]}_avg_prec1_novel']
        value_list = [round(self.trainer.callback_metrics.get(v).item() * 100, 2) for v in val_name]

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.lr, weight_decay=self.args.wd)
        return optimizer

    def train_dataloader(self):
        return construct_loader(self.args, "train")

    def val_dataloader(self):
        return construct_loader(self.args, "val")
