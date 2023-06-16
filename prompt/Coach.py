import copy
import logging
import time
from sklearn.metrics import cohen_kappa_score, classification_report

import torch
from tqdm import tqdm
from sklearn import metrics
from Prompt import Config, get_logger

log = get_logger(__name__)


class Coach:

    def __init__(self, train_data, dev_data, test_data, model, opt, args: Config):
        self.train_data = train_data
        self.dev_data = dev_data
        self.test_data = test_data
        self.model = model
        self.opt = opt
        self.args = args
        self.best_dev_f1 = None
        self.best_epoch = None
        self.best_state = None

    def load_ckpt(self, ckpt):
        self.best_dev_f1 = ckpt["best_dev_f1"]
        self.best_epoch = ckpt["best_epoch"]
        self.best_state = ckpt["best_state"]
        self.model.load_state_dict(self.best_state)

    def train(self):
        log.debug(self.model)
        # Early stopping.
        best_dev_f1, best_epoch, best_state = self.best_dev_f1, self.best_epoch, self.best_state

        # Train
        for epoch in range(1, self.args.epochs + 1):
            self.train_epoch(epoch)
            dev_f1, dev_acc, dev_kappa, class_focus_acc, _, _ = self.evaluate()
            log.info("[Dev set] [f1 {:.4f}] [acc {:.4f}] [kappa {:.4f}]".format(dev_f1, dev_acc, dev_kappa))
            if best_dev_f1 is None or dev_f1 > best_dev_f1:
                best_dev_f1 = dev_f1
                best_epoch = epoch
                best_state = copy.deepcopy(self.model.state_dict())
                log.info("Save the best model.")
            test_f1, test_acc, test_kappa, class_focus_acc, _, _ = self.evaluate(test=True)
            log.info(
                "[Test set] [f1 {:.4f}] [acc {:.4f}] [kappa {:.4f}]".format(test_f1, test_acc, test_kappa))

        # The best
        self.model.load_state_dict(best_state)
        log.info("")
        log.info("Best in epoch {}:".format(best_epoch))
        dev_f1, dev_acc, dev_kappa, class_focus_acc, _, _ = self.evaluate()
        log.info("[Dev set] [f1 {:.4f}] [acc {:.4f}] [kappa {:.4f}] [{}_acc {:.4f}]".format(dev_f1,
                                                                                            dev_acc,
                                                                                            dev_kappa,
                                                                                            self.args.class_focus,
                                                                                            class_focus_acc))
        test_f1, test_acc, test_kappa, class_focus_acc, _, _ = self.evaluate(test=True)
        log.info("[Test set] [f1 {:.4f}] [acc {:.4f}] [kappa {:.4f}] [{}_acc {:.4f}]".format(test_f1,
                                                                                             test_acc,
                                                                                             test_kappa,
                                                                                             self.args.class_focus,
                                                                                             class_focus_acc))
        out = [test_f1,
               test_acc,
               test_kappa,
               class_focus_acc]

        return best_dev_f1, best_epoch, best_state, out

    def train_epoch(self, epoch):
        start_time = time.time()
        epoch_loss = 0
        self.model.train()
        for idx, batch in tqdm(enumerate(self.train_data), desc="train epoch {}".format(epoch)):
            if self.args.use_cuda:
                batch = batch.to(self.args.device)
            logits = self.model(batch)
            labels = batch['label']
            if self.args.loss_weight is not None:
                weight = torch.tensor(self.args.loss_weight).to(self.args.device)
                loss_func = torch.nn.CrossEntropyLoss(weight=weight)
            else:
                loss_func = torch.nn.CrossEntropyLoss()
            loss = loss_func(logits, labels)
            loss.backward()
            epoch_loss += loss.item()
            self.opt.step()
            self.opt.zero_grad()

        end_time = time.time()
        log.info("")
        log.info("[Epoch %d] [Loss: %f] [Time: %f]" %
                 (epoch, epoch_loss, end_time - start_time))

    def evaluate(self, test=False):
        data = self.test_data if test else self.dev_data
        self.model.eval()
        with torch.no_grad():
            golds = []
            preds = []
            for idx, batch in tqdm(enumerate(data), desc="test" if test else "dev"):
                if self.args.use_cuda:
                    batch = batch.to(self.args.device)
                golds.extend(batch['label'].cpu().tolist())
                logits = self.model(batch)
                preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())

            f1 = metrics.f1_score(golds, preds, average="weighted")
            acc = metrics.accuracy_score(golds, preds)
            kappa = cohen_kappa_score(golds, preds)
            report = classification_report(golds, preds, target_names=self.args.classes, zero_division=0)
            log.debug(report)
            report_dict: dict = classification_report(golds, preds, target_names=self.args.classes, zero_division=0, output_dict=True)

        class_focus_acc = -1
        if self.args.class_focus is not None and self.args.class_focus in self.args.classes:
            class_focus_acc = report_dict[self.args.class_focus]['precision']

        return f1, acc, kappa, class_focus_acc, report, report_dict
