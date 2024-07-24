"""
Radiology Report Generation (RRG)
Visual-Linguistics Causal Intervention framework for RRG
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.modules4vlp import get_ht_mask, get_hv_mask, get_cross_mask
from modules.modules4transformer import LayerNorm, Encoder, DecoderLayer
from models.baseline import Baseline
from utils import tensor_utils
import numpy as np


class MODEL(Baseline):
    def __init__(self, args, tokenizer):
        super(MODEL, self).__init__(args, tokenizer)
        # ------------------------------------------
        self.z = None
        self.fl = None
        self.vocab = torch.arange(0, self.vocab_size + 1).unsqueeze(-1).long().cuda() # torch.Size([4348, 1])
        self.z_norm = LayerNorm(self.embed_dim)
        self.fl_norm = LayerNorm(self.embed_dim)
        # ------------------------------------------
        self.encoder = CausalEncoder(embed_dim=self.embed_dim, num_layer=self.en_num_layers, num_heads=self.num_heads,
                                     ff_dim=self.ff_dim, dropout=self.dropout)

        self.decoder = CausalDecoder(embed_dim=self.embed_dim, num_layer=self.de_num_layers, num_heads=self.num_heads,
                                     ff_dim=self.ff_dim, dropout=self.dropout)

    def _forward(self, images, targets, mode, B):
        # append cls token
        # (bs/bs*2, 196, 512)
        if mode == 'train':
            ht_mask, targets2 = get_ht_mask(targets)  # [B, Lt(60/100)] --> [B, Lt-1, Lt-1], [B, Lt-1]

            if self.dataset_name == 'iu_xray':
                hv1, ht1 = self.encode_clip(images[:, 0, :],
                                            targets2)  # (b,3,224,224) --> (b,1024,14,14), (b, l-1) --> (b, l-1, d)
                hv2, ht2 = self.encode_clip(images[:, 1, :],
                                            targets2)  # (b,3,224,224) --> (b,1024,14,14), (b, l-1) --> (b, l-1, d)
                hv = torch.cat((hv1, hv2), dim=0)
                ht = ht1 + ht2
            else:
                hv, ht = self.encode_clip(images, targets2)  # (b,3,224,224) --> (b,1024,14,14), (b, l-1) --> (b, l-1, d)

            hv = hv.flatten(2).transpose(1, 2) # (b,1024,14,14) --> (b, 196, 1024)

            hv = self.proj(hv) + self.pos_embed[:, 1:, :]  # (bs/bs*2, 196, 1024) --> (bs/bs*2, 196, 512)
            hv = self.dropout2(self.norm(hv))  # (bs/bs*2, 196, 512)

            hv = hv.reshape([B, -1, self.embed_dim])
            cls_token = self.cls_token + self.pos_embed[:, :1, :]
            cls_tokens = cls_token.expand(hv.shape[0], -1, -1)
            hv = torch.cat((cls_tokens, hv), dim=1)  # ( b, l+1, n)
            # encode
            hv_mask = get_hv_mask(hv)  # ( b, 1, l+1)
            hv = self.encoder(hv, hv_mask, self.pos_embed)  # (b, 393, 512）/ (b, 197, 512）, # (b, 48, 512)


            cross_mask = get_cross_mask(hv, targets)  # (b, Lt-1, 393/197）

            out = self.decoder(ht, hv, self_mask=ht_mask, cross_mask=cross_mask)  # [B, Lt-1, D]
            outputs = F.log_softmax(self.logit(out), dim=-1)  # [16, Lt-1, 4348(vocab_size+1)]
            outputs = outputs

        elif mode == 'sample':
            self.B = B
            self.images = images
            if self.dataset_name == 'iu_xray':
                images = images.reshape(B * 2, 3, 224, 224)
            hv = self.encode_clip.backbone.encode_image(images)[1]
            hv = hv.flatten(2).transpose(1, 2)  # (b,1024,14,14) --> (b, 196, 1024)

            hv = self.proj2(hv) + self.pos_embed[:, 1:, :]  # (bs/bs*2, 196, 1024) --> (bs/bs*2, 196, 512)
            hv = self.dropout2(self.norm(hv))  # (bs/bs*2, 196, 512)

            hv = hv.reshape([B, -1, self.embed_dim])

            cls_token = self.cls_token + self.pos_embed[:, :1, :]
            cls_tokens = cls_token.expand(hv.shape[0], -1, -1)
            hv = torch.cat((cls_tokens, hv), dim=1)  # ( b, l+1, n)
            # encode
            hv_mask = get_hv_mask(hv)  # ( b, 1, l+1)
            hv = self.encoder(hv, hv_mask, self.pos_embed)  # (b, 393, 512）/ (b, 197, 512）, # (b, 48, 512)

            self.beam_search.load_model(self.sample_forward, self.logit)
            outputs, _ = self.beam_search.sample_beam(hv)
            self.beam_search.clean_model()
        else:
            raise ValueError

        return outputs

    def sample_forward(self, hv, ht_o, v_mask, t_mask):

        if self.images.size(0) != ht_o.size(0):
            self.images = tensor_utils.repeat_tensors(self.args["beam_size"], self.images)
            self.B *= self.args["beam_size"]

        if self.dataset_name == 'iu_xray':
            hv1, ht1 = self.encode_clip(self.images[:, 0, :],
                                        ht_o)  # (b,3,224,224) --> (b,1024,14,14), (b, l-1) --> (b, l-1, d)
            hv2, ht2 = self.encode_clip(self.images[:, 1, :],
                                        ht_o)  # (b,3,224,224) --> (b,1024,14,14), (b, l-1) --> (b, l-1, d)
            hv = torch.cat((hv1, hv2), dim=0)
            ht = ht1 + ht2
        else:
            hv, ht = self.encode_clip(self.images, ht_o)

        hv = hv.flatten(2).transpose(1, 2)  # (b,1024,14,14) --> (b, 196, 1024)

        hv = self.proj(hv) + self.pos_embed[:, 1:, :]  # (bs/bs*2, 196, 1024) --> (bs/bs*2, 196, 512)
        hv = self.dropout2(self.norm(hv))  # (bs/bs*2, 196, 512)

        hv = hv.reshape([self.B, -1, self.embed_dim])
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(hv.shape[0], -1, -1)
        hv = torch.cat((cls_tokens, hv), dim=1)  # ( b, l+1, n)
        # encode
        hv_mask = get_hv_mask(hv)  # ( b, 1, l+1)
        hv = self.encoder(hv, hv_mask, self.pos_embed)  # (b, 393, 512）/ (b, 197, 512）, # (b, 48, 512)

        v_mask = get_hv_mask(hv)
        out = self.decoder(ht, hv, self_mask=t_mask, cross_mask=v_mask)
        return out


class CausalEncoder(Encoder):
    def __init__(self, embed_dim, num_layer, num_heads, ff_dim, dropout):
        super(CausalEncoder, self).__init__(embed_dim, num_layer, num_heads, ff_dim, dropout)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, h, mask=None, pos=None, k=6):
        attn = []
        for layer in self.layers:
            h = layer(h, mask)
            attn.append(layer.attn.attn)

        # add pos embedding
        if h.size(1) > 197:
            h = h + torch.cat([pos, pos[:, 1:, :]], dim=1)
        else:
            h = h + pos # (b, 393, 512）

        h = self.norm(h) # (b, 393, 512）/ (b, 197, 512）
        return h


class CausalDecoder(nn.Module):
    def __init__(self, embed_dim, num_layer, num_heads, ff_dim, dropout):
        super(CausalDecoder, self).__init__()
        self.norm = LayerNorm(embed_dim)

        self.layers = nn.ModuleList(
            [DecoderLayer(embed_dim, num_heads, ff_dim, dropout) for _ in range(num_layer)])

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, output, h, self_mask=None, cross_mask=None):
        # [B, Lt-1, D], (b, 393/197, 512）,[B, Lt-1, Lt-1],(b, Lt-1, 393/197）,[b, 4348, 512], (b, 48, 512)
        for i in range(len(self.layers)):
            output = self.layers[i](output, h, self_mask, cross_mask)

        output = self.norm(output)
        return output
