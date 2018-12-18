from tqdm import tqdm

import torch
from torch.autograd import Variable
import utils
import numpy as np

class Trainer(object):
    def __init__(self, args, model, criterion, optimizer, device):
        super(Trainer, self).__init__()
        self.args = args
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.epoch = 0

    # helper function for training
    def train(self, dataloader):
        self.model.train()
        self.optimizer.zero_grad()
        total_loss = 0.0
        # indices = torch.randperm(len(dataloader), dtype=torch.long, device='cpu')
        for data in tqdm(dataloader, desc='Training epoch  ' + str(self.epoch) + ''):
            # lsent, llen, rsent, rlen, label = data
            lsent, lgraph, llen, rsent, rgraph, rlen, label = data
            target = utils.map_label_to_target_classifcation(label, self.args.num_classes)
            linput, rinput = lsent.to(self.device), rsent.to(self.device)
            llen, rlen = llen.to(self.device), rlen.to(self.device)
            lgraph,rgraph = lgraph.to(self.device), rgraph.to(self.device)
            target = target.to(self.device)
            output = self.model(linput,llen,rinput,rlen,lgraph,rgraph)
            # output = self.model(linput, llen,rinput, rlen)

            loss = self.criterion(output, target)
            loss.backward()
            total_loss += loss.item()
            # if idx % self.args.batchsize == 0 and idx > 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
        self.epoch += 1
        return total_loss / len(dataloader.dataset)

    # helper function for testing
    def test(self, dataloader):
        self.model.eval()
        with torch.no_grad():
            total_loss = 0.0
            accuracy = 0
            for data in tqdm(dataloader, desc='Testing epoch  ' + str(self.epoch) + ''):
                lsent, lgraph, llen, rsent, rgraph, rlen, label = data
                # lsent, llen, rsent, rlen, label = data

                target = utils.map_label_to_target_classifcation(label, self.args.num_classes)
                linput, rinput = lsent.to(self.device), rsent.to(self.device)
                llen, rlen = llen.to(self.device), rlen.to(self.device)
                lgraph, rgraph = lgraph.to(self.device), rgraph.to(self.device)
                target = target.to(self.device)
                output = self.model(linput, llen,rinput, rlen, lgraph,rgraph )
                # output = self.model(linput, llen,rinput, rlen)

                label = label.to(self.device)
                loss = self.criterion(output, target)
                total_loss += loss.item()
                pred = output.data.max(1, keepdim=True)[1]
                accuracy += pred.eq(label.data.view_as(pred)).cpu().sum().numpy()
                self.optimizer.zero_grad()

        return total_loss / len(dataloader.dataset), 100*accuracy/len(dataloader.dataset)
