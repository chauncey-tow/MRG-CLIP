import torch
import torch.nn as nn
import torch.nn.functional as F

from .clip import build_model

from .layers import HA, GA, Projector
from .bridger import Bridger_RN as Bridger_RL, Bridger_ViT as Bridger_VL
from .config import load_cfg_from_cfg_file
from loguru import logger

class ETRIS(nn.Module):
    def __init__(self, cfg, wordlen):
        super().__init__()
        # Vision & Text Encoder
        clip_model = torch.jit.load(cfg.clip_pretrain,
                                    map_location="cpu").eval()
        if "RN" in cfg.clip_pretrain:
            self.backbone = build_model(clip_model.state_dict(), wordlen).float()
            self.bridger = Bridger_RL(d_model=cfg.ladder_dim, nhead=cfg.nhead, fusion_stage=cfg.multi_stage)
        else:
            self.backbone = build_model(clip_model.state_dict(), wordlen, cfg.input_size).float()
            self.bridger = Bridger_VL(d_model=cfg.ladder_dim, nhead=cfg.nhead)

        self.neck = HA(in_channels=cfg.fpn_in, out_channels=cfg.fpn_out, stride=cfg.stride)

        # Fix Backbone
        for param_name, param in self.backbone.named_parameters():
            if 'positional_embedding' not in param_name:
                param.requires_grad = False


    def forward(self, img, word):
        '''
            img: b, 3, h, w
            word: b, words
            word_mask: b, words
            mask: b, 1, h, w
        '''

        # vis: C3 / C4 / C5
        # word: b, length, 1024
        # state: b, 1024
        vis, word, state = self.bridger(img, word, self.backbone)
        fq = self.neck(vis, state) # (b, 512, 14, 14)
        return fq, word


if __name__ == '__main__':
    cfg = load_cfg_from_cfg_file("/home/ai1010/dc/code/mine/clip/mrg/config/bridge_r101.yaml")
    model = ETRIS(cfg=cfg)
    
    backbone = []
    head = []
    fix = []
    for k, v in model.named_parameters():
        if (k.startswith('backbone') and 'positional_embedding' not in k or 'bridger' in k) and v.requires_grad:
            backbone.append(v)
        elif v.requires_grad:
            head.append(v)
        else:
            fix.append(v)
    logger.info('Backbone with decay={}, Head={}'.format(len(backbone), len(head)))
    param_list = [{
        'params': backbone,
        'initial_lr': cfg.lr_multi * cfg.base_lr
    }, {
        'params': head,
        'initial_lr': cfg.base_lr
    }]
    
    n_backbone_parameters = sum(p.numel() for p in backbone)
    logger.info(f'number of updated params (Backbone): {n_backbone_parameters}.')
    n_head_parameters = sum(p.numel() for p in head)
    logger.info(f'number of updated params (Head)    : {n_head_parameters}')
    n_fixed_parameters = sum(p.numel() for p in fix)
    logger.info(f'number of fixed params             : {n_fixed_parameters}')
    # model = ETRIS(cfg=cfg)
    # if cfg.sync_bn:
    #     model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    img = torch.rand(4, 3, 224, 224)
    text = torch.rand(4, 17)
    text = text.type(torch.long)
    mask = torch.rand(4, 1, 224, 224)
    y,x = model(img,text)
    print(len(y))
    print(y.shape)




