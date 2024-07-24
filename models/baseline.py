"""
vision language model
the baseline model, transformer-nano (L3H8N512)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.modules4vlp import VisEmbed, TextEmbed, get_ht_mask, get_hv_mask
from modules.modules4transformer import Decoder, Encoder, PositionalEncoding, Embeddings, LayerNorm
from modules.beam_search import BeamSearch
from modules.pos_embed import get_2d_sincos_pos_embed

from .segmenter import ETRIS
from .config import load_cfg_from_cfg_file
from loguru import logger

class Baseline(nn.Module):
    def __init__(self, args, tokenizer):
        super(Baseline, self).__init__()
        self.args = args
        self.vocab_size = len(tokenizer.idx2token)
        self.tokenizer = tokenizer

        self.en_num_layers = args["en_num_layers"]
        self.de_num_layers = args["de_num_layers"]
        self.embed_dim = args["embed_dim"]
        self.num_heads = args["num_heads"]
        self.ff_dim = args["embed_dim"] * 4
        self.dropout = args["dropout"]
        self.max_seq_length = args["max_seq_length"]-1

        # ------------------------------------------
        # Embedding
        cfg = load_cfg_from_cfg_file("/home/ai1010/dc/code/mine/clip/pretrain/bridge_r101.yaml")
        self.encode_clip = ETRIS(cfg=cfg, wordlen=self.max_seq_length)

        self.dropout2 = nn.Dropout(p=self.dropout)
        self.proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.proj2 = nn.Linear(1024, self.embed_dim)
        self.norm = LayerNorm(self.embed_dim)

        self.pos_embed = nn.Parameter(torch.zeros(1, 196 + 1, self.embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding

        # self.vis_embed = VisEmbed(embed_dim=self.embed_dim, dropout=self.dropout)
        self.vis_embed = None
        self.text_embed = None

        # Encoder
        self.encoder = Encoder(embed_dim=self.embed_dim, num_layer=self.en_num_layers, num_heads=self.num_heads,
                               ff_dim=self.ff_dim, dropout=self.dropout)
        self.decoder = Decoder(embed_dim=self.embed_dim, num_layer=self.de_num_layers,
                               num_heads=self.num_heads, ff_dim=self.ff_dim, dropout=self.dropout)

        self.logit = nn.Linear(self.embed_dim, self.vocab_size + 1)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        torch.nn.init.normal_(self.cls_token, std=.02)

        self.beam_search = BeamSearch(args, self.vocab_size)

        # ------------------------------------------
        self.dataset_name = args["dataset_name"]
        if args["dataset_name"] == 'iu_xray':
            self.forward = self.forward_iu_xray
        elif args["dataset_name"] == 'mimic_cxr':
            self.forward = self.forward_mimic_cxr
        elif args["dataset_name"] == 'ffa_ir':
            self.forward = self.forward_ffa_ir
        elif args["dataset_name"] == 'mix':
            self.forward = self.forward_mimic_cxr
        else:
            raise ValueError

        self.initialize_weights()
        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def initialize_weights(self):
        # initialization
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(196 ** .5),
                                            cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    def forward_iu_xray(self, images, targets=None, mode='train'):
        B = images.size(0)
        outputs = self._forward(images, targets, mode, B)
        return outputs

    def forward_ffa_ir(self, images, targets=None, mode='train'):
        # B, N, C, H, W
        B, N, C, H, W = images.size()
        images = images.reshape([-1, C, H, W])
        hv = self.vis_embed(images)
        outputs = self._forward(hv, targets, mode, B)
        return outputs

    def forward_mimic_cxr(self, images, targets=None, mode='train'):
        outputs = self._forward(images, targets, mode, images.size(0))
        return outputs

    def _forward(self, hv, targets, mode, B):
        # append cls token
        hv = hv.reshape([B, -1, self.embed_dim])
        cls_token = self.cls_token + self.vis_embed.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(hv.shape[0], -1, -1)
        hv = torch.cat((cls_tokens, hv), dim=1)
        # encode
        hv_mask = get_hv_mask(hv)
        hv = self.encoder(hv, hv_mask)
        # decode
        if mode == 'train':
            ht_mask, targets = get_ht_mask(targets)
            ht = self.text_embed(targets)  # [B, L] -> [B, L, D]
            out = self.decoder(ht, hv, self_mask=ht_mask, cross_mask=hv_mask)
            outputs = [F.log_softmax(self.logit(out), dim=-1)]
        elif mode == 'sample':
            self.beam_search.load_model(self.sample_forward,
                                        self.logit)
            outputs, _ = self.beam_search.sample_beam(hv)
            self.beam_search.clean_model()
        else:
            raise ValueError
        return outputs

    def sample_forward(self, hv, ht, v_mask, t_mask):
        ht = self.text_embed(ht)
        return self.decoder(ht, hv, self_mask=t_mask, cross_mask=v_mask)
