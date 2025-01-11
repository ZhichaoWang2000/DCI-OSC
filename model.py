import math
import torch
import torch.nn as nn

from torch.nn.functional import gumbel_softmax


def fcim(global_dic, bg_mask, vid_feats):
    global_dic = global_dic.type_as(vid_feats).to(vid_feats.device)
    bs, v_len, hid_dim = vid_feats.size()
    num_samples_in_mem_bank = global_dic.size(0)
    sample_idx = torch.randint(0, num_samples_in_mem_bank, (bs, v_len), device=vid_feats.device)
    sampled_bg = global_dic[sample_idx.view(-1)].reshape(bs, v_len, hid_dim)
    vid_feats_new = vid_feats * ((~bg_mask).unsqueeze(-1)) + sampled_bg * (bg_mask.unsqueeze(-1))
    return vid_feats_new


def padding_mask_k(seq_q, seq_k):
    fake_q = torch.ones_like(seq_q)
    pad_mask = torch.bmm(fake_q, seq_k.transpose(1, 2))
    pad_mask = pad_mask.eq(0)
    return pad_mask


def padding_mask_q(seq_q, seq_k):
    fake_k = torch.ones_like(seq_k)
    pad_mask = torch.bmm(seq_q, fake_k.transpose(1, 2))
    pad_mask = pad_mask.eq(0)
    return pad_mask


class AttentionScore(nn.Module):
    def __init__(self, input_dim, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

        self.linear_q = nn.Linear(input_dim, input_dim)
        self.linear_k = nn.Linear(input_dim, input_dim)

    def forward(self, q, k, attn_mask=None, softmax_mask=None):
        if attn_mask is None:
            attn_mask = padding_mask_k(q, k)
        if softmax_mask is None:
            softmax_mask = padding_mask_q(q, k)

        q = self.linear_q(q)
        k = self.linear_k(k)

        scale = q.size(-1) ** -0.5

        attention = torch.bmm(q, k.transpose(-2, -1))

        if scale is not None:
            attention = attention * scale
        if attn_mask is not None:
            attention = attention.masked_fill(attn_mask.bool(), -float("inf"))
        attention = self.softmax(attention)
        attention = attention.masked_fill(softmax_mask, 0.)

        return attention


class BISC(nn.Module):

    def __init__(self, input_dim, tau=1, is_hard=False, dropout=0.5):
        super().__init__()
        self.tau = tau
        self.is_hard = is_hard
        self.fg_att = AttentionScore(input_dim, dropout)
        self.bg_att = AttentionScore(input_dim, dropout)

    def forward(self, q, v, attn_mask=None):
        fg_score = self.fg_att(q.unsqueeze(1), v, attn_mask=attn_mask)
        bg_score = self.bg_att(q.unsqueeze(1), v, attn_mask=attn_mask)
        score = torch.cat((fg_score, bg_score), 1)
        score = gumbel_softmax(score, tau=self.tau, hard=self.is_hard, dim=1)
        fg_mask = score[:, 0, :]
        bg_mask = score[:, 1, :]

        return fg_mask, bg_mask


class PositionalEncoding(nn.Module):
    """
    Designed for input of shape [batch_size, seq_len, d_model]
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class DCI_OSC(nn.Module):
    def __init__(self, args):
        super(DCI_OSC, self).__init__()
        self.args = args
        self.classes = args.vocab_size
        self.proj = nn.Linear(args.input_dim, args.transformer_dim)
        self.ln = nn.LayerNorm(args.transformer_dim)
        self.pos_encoder = PositionalEncoding(args.transformer_dim)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(d_model=args.transformer_dim,
                                                     nhead=args.transformer_heads,
                                                     dropout=args.transformer_dropout,
                                                     batch_first=True),
            num_layers=args.transformer_layers)
        self.classifier = BISC(args.transformer_dim, args.tau, False)
        self.head = nn.Linear(args.transformer_dim, self.classes)
        self.bert_encoder = nn.Linear(768, args.transformer_dim)

    def forward(self, input, q, global_dictionary=None):
        q = q.squeeze(1)
        x = self.ln(self.proj(input))
        x = self.pos_encoder(x)
        q = self.bert_encoder(q)

        fg_mask, bg_mask = self.classifier(q, x)
        fg_mask = fg_mask.bool()
        bg_mask = bg_mask.bool()
        fg_feats = x * fg_mask.unsqueeze(-1)
        out_fg = self.transformer_encoder(self.pos_encoder(fg_feats))
        out_fg = self.head(out_fg)
        out_fg = out_fg.view(-1, self.classes)

        bg_feats = x * bg_mask.unsqueeze(-1)
        out_bg = self.transformer_encoder(self.pos_encoder(bg_feats))
        out_bg = self.head(out_bg)
        out_bg = out_bg.view(-1, self.classes)

        vid_feats_m = fcim(global_dictionary, bg_mask, x)
        out_m = self.transformer_encoder(self.pos_encoder(vid_feats_m))
        out_m = self.head(out_m)
        out_m = out_m.view(-1, self.classes)

        return out_fg, out_bg, out_m
