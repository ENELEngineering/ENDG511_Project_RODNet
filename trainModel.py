from sklearn.metrics import accuracy_score
from time import time
import torch

class modelHandler():
    def __init__(self, net, criterion, optimizer, device, scheduler=None, num_epochs=1):
        self.net = net
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.NUM_EPOCHS = num_epochs
        self.device = device

        self.history = {
            "1": {"train": {"loss": []}},
            "2": {"train": {"loss": []}},
            "T": {"train": {"loss": []}}
              }
        
    def train(self, trainLoader, model_dir= ''):

        for epoch in range(self.NUM_EPOCHS):
            total1loss, total2loss, totalloss = 0, 0, 0
            
            self.net.train()
            for iter, data_dict in enumerate(trainLoader):
                data = data_dict['radar_data'].to(self.device)
                confmap_gt = data_dict['anno']['confmaps'].to(self.device)

                self.optimizer.zero_grad()
                confmap_preds_1,confmap_preds_2 = self.net(data.float())
                loss_confmap_1, loss_confmap_2 = self.criterion(confmap_preds_1, confmap_gt.float()), self.criterion(confmap_preds_2, confmap_gt.float())
                total1loss += loss_confmap_1.item()
                total2loss += loss_confmap_2.item()
                totalloss += 0.5*loss_confmap_1.item() + 0.5*loss_confmap_2.item()
                loss_confmap_1.backward(inputs=list(self.net.encoder_base.parameters())+list(self.net.decoder_short.parameters()), retain_graph=True)
                loss_confmap_2.backward(inputs=list(self.net.decoder_long.parameters()))
                self.optimizer.step()
            self.scheduler.step()
            total1loss = total1loss/len(trainLoader)
            total2loss = total2loss/len(trainLoader)
            totalloss = totalloss/len(trainLoader)
            
            self.history["1"]["train"]["loss"].append(total1loss)
            self.history["2"]["train"]["loss"].append(total2loss)
            self.history["T"]["train"]["loss"].append(totalloss)
            
            print("epoch {} --> shortbranchloss: {:0.3f}, longbranchloss: {:0.3f}, totalloss: {:0.3f}"
              .format(epoch+1, total1loss, total2loss, totalloss))
            
            print("saving current epoch model ...")
            save_model_path = '%s/multi_epoch_%02d_final.pkl' % (model_dir, epoch + 1)
            torch.save(self.net.state_dict(), save_model_path)
        return self.net, self.history
    


class baseHandler():
    def __init__(self, net, criterion, optimizer, device, scheduler=None, num_epochs=1):
        self.net = net
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.NUM_EPOCHS = num_epochs
        self.device = device

        self.history = {
            "train": {"loss": []}
              }
        
    def train(self, trainLoader, model_dir= ''):

        for epoch in range(self.NUM_EPOCHS):
            totalloss = 0
            tic = time()
            self.net.train()
            for iter, data_dict in enumerate(trainLoader):
                data = data_dict['radar_data'].to(self.device)
                confmap_gt = data_dict['anno']['confmaps'].to(self.device)
                self.optimizer.zero_grad()
                confmap_preds = self.net(data.float())
                loss_confmap = self.criterion(confmap_preds, confmap_gt.float())
                totalloss += loss_confmap.item()
                loss_confmap.backward()
                self.optimizer.step()
            self.scheduler.step()
            totalloss = totalloss/len(trainLoader)
            toc = time()
            self.history["train"]["loss"].append(totalloss)
            print("epoch {} --> trainLoss: {:0.3f}".format(epoch+1, totalloss))
            print("saving current epoch model ...")
            save_model_path = '%s/base_epoch_%02d_final.pkl' % (model_dir, epoch + 1)
            torch.save(self.net.state_dict(), save_model_path)
        return self.net, self.history
