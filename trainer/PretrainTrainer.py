"""
Trainer for pretraining
"""
import os
from collections import OrderedDict

import matplotlib.pyplot as plt
from trainer.BaseTrainer import BaseTrainer
import torch
import time
import cv2
from utils import cvt_im_tensor
import numpy as np

import pandas as pd
from utils.loss import patchify


class Trainer(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, optimizer, args, lr_scheduler, train_dataloader, val_dataloader,
                 test_dataloader):
        super(Trainer, self).__init__(model, criterion, metric_ftns, optimizer, args)
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.best_score = 0.

    def _train_epoch(self, epoch):

        train_loss = 0
        self.model.train()
        start_time = time.time()
        for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.train_dataloader):
            images, reports_ids, reports_masks = images.cuda(), reports_ids.cuda(), reports_masks.cuda()
            output = self.model(images, reports_ids, mode='train')
            nll_loss = self.criterion(output, reports_ids, reports_masks)
            loss = nll_loss
            self.optimizer.zero_grad()
            loss.backward()
            train_loss += loss.item()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
            self.optimizer.step()

            print(f"\repoch: {epoch} {batch_idx}/{len(self.train_dataloader)}\tloss: {loss:.3f}\tmean loss: {train_loss/(batch_idx+1):.3f}",
                  flush=True, end='')

            if self.args["lr_scheduler"] != 'StepLR':
                self.lr_scheduler.step()
        if self.args["lr_scheduler"] == 'StepLR':
            self.lr_scheduler.step()

        log = {'train_loss': train_loss / len(self.train_dataloader)}
        print("\n")
        print("\tEpoch {}\tmean_loss: {:.4f}\ttime: {:.4f}s".format(epoch, log['train_loss'], time.time() - start_time))


    def test_epoch(self, mode='img'):
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.test_dataloader):
                images, reports_ids, reports_masks = images.cuda(), reports_ids.cuda(), reports_masks.cuda()

                if self.args.model == 'mae':
                    image = images[:, 1]
                    loss, output_img, mask = self.model(image)
                else:
                    output_img, image, mask = self.model(images, reports_ids, mode='img')

                out_img = unpatchify(output_img, mask)
                # re_img = get_img_from_phase(image, out_img)
                # vis_img(image[0], 'GT')
                vis_img(out_img[0], 'pred')
                # vis_img(re_img[0], 'pred_re_img')

    def vis_one_batch(self, epoch, mode='v2v'):
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.val_dataloader):
                images, reports_ids, reports_masks = images.cuda(), reports_ids.cuda(), reports_masks.cuda()
                if mode == 'v2v' or mode == 'vt2v':
                    output_img, image, mask = self.model(images, reports_ids, mode=mode)
                    self.vis_img(output_img, image, mask, epoch)
                break
                # elif mode == 'v2t' or mode == 'vt2t':
                #     outputs, postfix_text, mask = self.model(images, reports_ids, mode=mode)

        self.model.train()
        return

    def vis_img(self, output_img, image, mask, ep):
        """
        visualize the recon images, all input is tensor
        :param output_img: recon image of remove patch
        :param image: original image
        :param mask:
        :param ep:
        :return:
        """
        out_img = unpatchify(output_img, mask)
        val_out, val_in = out_img, image
        n = val_out.size(0)
        _img = OrderedDict()
        for i in range(n):
            out_img = val_out[i].detach().float().cpu()
            _img[f"img_out{i}"] = cvt_im_tensor.tensor2im(out_img)
            in_img = val_in[i].detach().float().cpu()
            _img[f"img_in{i}"] = cvt_im_tensor.tensor2im(in_img)
        self.monitor.display_current_images(_img, ep, True)

    def vis_report(self, outputs, image, target, ep):
        """
        visualize the output report, all input is tensor
        :param outputs: the prediction of report
        :param image:
        :param target: original report
        :param ep:
        :return:
        """
        n = outputs.size(0)
        _report = OrderedDict()
        _img = OrderedDict()
        for i in range(n):
            out_img = image[i].detach().float().cpu().reshape([-1, 3, 224, 224])
            out_img = out_img.permute([1, 2, 0, 3]).reshape(3, 224, -1)
            _img[f"img{i}"] = cvt_im_tensor.tensor2im(out_img)
            in_img = outputs[i].detach().cpu()
            _img[f"img_in{i}"] = cvt_im_tensor.tensor2im(in_img)
        self.monitor.display_current_results(_img, ep, True)


def vis_heatmap(img, hm, title):
    """
    img: (3, H, W), torch
    """
    img = img.permute(1, 2, 0).detach().cpu().numpy()
    hm = hm.permute(1, 2, 0).detach().cpu().numpy()
    img = np.uint8(255 * (img - img.min()) / (img.max() - img.min()))
    heatmap = 1 - (hm - hm.min()) / (hm.max() - hm.min())
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    attn_img = cv2.addWeighted(img, 0.4, heatmap, 0.6, 0)
    plt.imshow(attn_img)
    plt.title(title)
    plt.show()
    plt.imshow(img)
    plt.title('img')
    plt.show()
    # plt.imshow(heatmap)
    # plt.title("attn")
    # plt.show()


def unpatchify(x, mask=None):
    """
    x: (N, L, patch_size**2 *3)
    imgs: (N, 3, H, W)
    """
    p = 16
    h = w = 14
    if type(x) == list:
        x = x[0]

        imgs = torch.zeros([x.shape[0], h * w, p ** 2 * 3]).to(x.device)
        imgs[index] += x
        x = imgs

    if mask is not None:
        mask = mask.unsqueeze(-1)
        x = x * mask
    x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
    x = torch.einsum('nhwpqc->nchpwq', x)
    imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
    return imgs
