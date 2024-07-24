import os
from abc import abstractmethod
import time

import numpy as np
import torch
import pandas as pd
from numpy import inf
from utils.monitor import Monitor


class BaseTrainer(object):
    def __init__(self, model, criterion, metric_ftns, optimizer, args, monitor=True):
        self.args = args

        # setup GPU device if available, move model into configured device
        self.model = model
        # self.model = torch.nn.DataParallel(model)

        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer

        self.epochs = self.args["epochs"]
        self.save_period = self.args["save_period"]

        self.mnt_mode = args["monitor_mode"]
        self.mnt_metric = 'val_' + args["monitor_metric"]
        self.mnt_metric_test = 'test_' + args["monitor_metric"]
        assert self.mnt_mode in ['min', 'max']

        self.mnt_best = inf if self.mnt_mode == 'min' else -inf
        self.mnt_best_val = inf if self.mnt_mode == 'min' else -inf
        self.mnt_best_test = inf if self.mnt_mode == 'min' else -inf

        # self.early_stop = getattr(self.args, 'early_stop', inf)
        self.early_stop = self.args["early_stop"]

        self.start_epoch = 1
        self.checkpoint_dir = args["result_dir"]

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        if args["resume"] != "":
            self._resume_checkpoint(args["resume"])

        self.best_recorder = {'val': {self.mnt_metric: self.mnt_best},
                              'test': {self.mnt_metric_test: self.mnt_best}}

        # monitor
        if monitor:
            self.monitor = Monitor(args)

    @abstractmethod
    def _train_epoch(self, epoch):
        raise NotImplementedError

    def train(self):
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)
            if result is None:
                self._save_checkpoint(epoch)
                continue

            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)
            self._record_best(log)

            # print logged informations to the screen
            for key, value in log.items():
                print('\t{:15s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    cur_metric = log['test_BLEU_4'] + 0.33 * log['test_BLEU_1'] + 0.67 * log['test_METEOR']

                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and cur_metric <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and cur_metric >= self.mnt_best)
                except KeyError:
                    print("Warning: Metric '{}' is not found. " "Model performance monitoring is disabled.".format(
                        self.mnt_metric_test))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = cur_metric
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    print("Validation performance didn\'t improve for {} epochs. " "Training stops.".format(
                        self.early_stop))
                    break

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)
        self._print_best()
        self._print_best_to_file()

    def _print_best_to_file(self):
        crt_time = time.asctime(time.localtime(time.time()))
        self.best_recorder['val']['time'] = crt_time
        self.best_recorder['test']['time'] = crt_time
        self.best_recorder['val']['seed'] = self.args["seed"]
        self.best_recorder['test']['seed'] = self.args["seed"]
        self.best_recorder['val']['best_model_from'] = 'val'
        self.best_recorder['test']['best_model_from'] = 'test'

        if not os.path.exists(self.args["record_dir"]):
            os.makedirs(self.args["record_dir"])
        record_path = os.path.join(self.args["record_dir"], self.args["dataset_name"] + '.csv')
        if not os.path.exists(record_path):
            record_table = pd.DataFrame()
        else:
            record_table = pd.read_csv(record_path)
        record_table = record_table.append(self.best_recorder['val'], ignore_index=True)
        record_table = record_table.append(self.best_recorder['test'], ignore_index=True)
        record_table.to_csv(record_path, index=False)

    def _save_checkpoint(self, epoch, save_best=False):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'seed': self.args['seed']
        }
        filename = os.path.join(self.checkpoint_dir, 'current_checkpoint.pth')
        torch.save(state, filename)
        print("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = os.path.join(self.checkpoint_dir, 'model_best_' + str(epoch) + '.pth')
            torch.save(state, best_path)
            print("*************** Saving current best:{} ... ***************".format(best_path))

    def _save_checkpoint_step(self, epoch, step, save_best=False):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'seed': self.args['seed']
        }
        if save_best:
            best_path = os.path.join(self.checkpoint_dir, 'epoch_' + str(epoch) + 'step_' + str(step) + '.pth')
            torch.save(state, best_path)
            print("Saving best step file checkpoint: {} ...".format(best_path))

    def _resume_checkpoint(self, resume_path):
        resume_path = str(resume_path)
        print("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        print("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))


    def _record_best_test(self, log):

        cur_metric_test = log['test_BLEU_4'] + 0.33 * log['test_BLEU_1'] + 0.67 * log['test_METEOR']
        improved_test = (self.mnt_mode == 'min' and cur_metric_test <= self.mnt_best_test) or \
                        (self.mnt_mode == 'max' and cur_metric_test >= self.mnt_best_test)
        if improved_test:
            self.best_recorder['test'].update(log)
            self.mnt_best_test = cur_metric_test

    def _record_best(self, log):
        cur_metric_val = log['val_BLEU_4'] + 0.33 * log['val_BLEU_1'] + 0.67 * log['val_METEOR']
        improved_val = (self.mnt_mode == 'min' and cur_metric_val <= self.mnt_best_val) or \
                       (self.mnt_mode == 'max' and cur_metric_val >= self.mnt_best_val)
        if improved_val:
            self.best_recorder['val'].update(log)
            self.mnt_best_val = cur_metric_val

        cur_metric_test = log['test_BLEU_4'] + 0.33 * log['test_BLEU_1'] + 0.67 * log['test_METEOR']
        improved_test = (self.mnt_mode == 'min' and cur_metric_test <= self.mnt_best_test) or \
                        (self.mnt_mode == 'max' and cur_metric_test >= self.mnt_best_test)
        if improved_test:
            self.best_recorder['test'].update(log)
            self.mnt_best_test = cur_metric_test

    def _print_best(self):
        print('Best results (w.r.t {}) in validation set:'.format(self.args["monitor_metric"]))
        for key, value in self.best_recorder['val'].items():
            print('\t{:15s}: {}'.format(str(key), value))

        print('Best results (w.r.t {}) in test set:'.format(self.args["monitor_metric"]))
        for key, value in self.best_recorder['test'].items():
            print('\t{:15s}: {}'.format(str(key), value))


class Trainer(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, optimizer, args, lr_scheduler, train_dataloader, val_dataloader,
                 test_dataloader):
        super(Trainer, self).__init__(model, criterion, metric_ftns, optimizer, args)
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

    def inference(self):
        state = torch.load(self.args["load_model_path"], map_location='cuda')
        pretrained_dict = state['state_dict']
        self.model.load_state_dict(pretrained_dict, False)

        log = {'task_name': self.args['task_name']}
        self.model.eval()
        with torch.no_grad():
            val_gts, val_res = [], []
            p = torch.zeros([1, self.args["max_seq_length"]]).cuda()
            for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.val_dataloader):
                images, reports_ids, reports_masks = images.cuda(), reports_ids.cuda(), reports_masks.cuda()
                output = self.model(images, mode='sample')
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                val_res.extend(reports)
                val_gts.extend(ground_truths)
                p = torch.cat([p, output])
                print(f"\rVal Processing: [{int((batch_idx + 1) / len(self.val_dataloader) * 100)}%]", end='',
                      flush=True)
            val_met = self.metric_ftns({i: [gt] for i, gt in enumerate(val_gts)},
                                       {i: [re] for i, re in enumerate(val_res)})
            # record val metrics
            log.update(**{'val_' + k: v for k, v in val_met.items()})
        self.model.eval()
        with torch.no_grad():
            test_gts, test_res, p = [], [], []
            p = torch.zeros([1, self.args["max_seq_length"]]).cuda()
            for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.test_dataloader):
                images, reports_ids, reports_masks = images.cuda(), reports_ids.cuda(), reports_masks.cuda()
                output = self.model(images, mode='sample')
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                test_res.extend(reports)
                test_gts.extend(ground_truths)
                p = torch.cat([p, output])
                print(f"\rTest Processing: [{int((batch_idx + 1) / len(self.test_dataloader) * 100)}%]", end='',
                      flush=True)
            test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                        {i: [re] for i, re in enumerate(test_res)})

            log.update(**{'test_' + k: v for k, v in test_met.items()})

        record_path = os.path.join(self.args["record_dir"], self.args["dataset_name"] + '.csv')
        record_table = pd.DataFrame()
        record_table = record_table.append(log, ignore_index=True)
        record_table.to_csv(record_path, index=False)
        print(log)
        return log
