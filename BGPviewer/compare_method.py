from modules import (ConvLayer,
    FeatureAttentionLayer,
    TemporalAttentionLayer,
    TemporalAttentionLayer,
    GRULayer,
    Forecasting_Model,)
from torch import nn
# from training import Trainer
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import pickle
import numpy as np
from mydataset import SlidingWindowDataset
from mydataset import create_data_loaders
import logging
import torch
import torch.nn.functional as F
from collections import Counter
import os
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, precision_score, classification_report
import random

def early_perf(label, predict):
    predict = [1 if i != 0 else 0 for i in predict]
    label = [1 if i != 0 else 0 for i in label]
    first_anomaly = label.index(1)
    # print('first_anomaly:',first_anomaly)
    t = None
    # print(predict[first_anomaly: first_anomaly+20])
    if 1 in predict[first_anomaly: first_anomaly+20]:
        t = predict[first_anomaly: first_anomaly+20].index(1)
    return t


class MTAD_GAT(pl.LightningModule):
    def __init__(self, n_features, window_size, dropout, hid_dim, num_classes, gru_hid_dim, n_layers):
        super(MTAD_GAT, self).__init__()
        self.n_features = n_features
        self.window_size = window_size
        self.dropout = dropout
        self.hid_dim = hid_dim
        self.out_dim = num_classes

        self.gru_hid_dim = gru_hid_dim
        self.n_layers = n_layers
        self.norm0 = nn.BatchNorm1d(self.window_size)
        self.norm = nn.BatchNorm1d(self.window_size)
        self.conv = ConvLayer(self.n_features)
        self.featureAttentionLayer = FeatureAttentionLayer(self.n_features, self.window_size, self.dropout, alpha=0.5)
        self.temporal_gat = TemporalAttentionLayer(self.n_features, self.window_size, self.dropout, alpha=0.5,  use_gatv2=True)
        self.forecasting_model = Forecasting_Model(self.n_features*10, self.hid_dim, self.out_dim, self.n_layers, dropout)
        self.gru = GRULayer(self.n_features * 3, self.gru_hid_dim, n_layers, dropout)

        self.validation_step_outputs = []
        self.train_step_outputs = []

    def forward(self, x):
        # x = self.norm(x)
        x = self.norm0(x)
        x = self.conv(x)
        h_fea = self.featureAttentionLayer(x)
        t_fea = self.temporal_gat(x)
        # h_cat = torch.cat([h_fea, t_fea], dim=2)  # (b, n, 3k)        
        h_cat = torch.cat([h_fea], dim=2)  # (b, n, 3k)        
        
        h_cat = self.norm(h_cat)
        # h_cat = torch.cat([x], dim=2)  # (b, n, 3k)
        # _, h_end = self.gru(h_cat)
        # print("h_cat:", h_cat.size())
        h_end = h_cat.view(x.shape[0], -1)   # Hidden state for last timestamp
        # print("h_end:", h_end.size())
        predictions = self.forecasting_model(h_end)
        return predictions

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.long()
        pred_list = []
        x_out = self.forward(x)
        loss = F.cross_entropy(x_out, y, weight=torch.tensor(weight, dtype=torch.float).to(device))

        pred = x_out.argmax(-1)
        accuracy = (pred == y).sum() / pred.shape[0]
        y_list = y.tolist()
        pred_list = pred.tolist()
        self.train_step_outputs.append({"loss":loss, "accuracy": accuracy, 'pred_list':pred_list, 'y':y_list})
        return loss

    def on_train_epoch_end(self) -> None:
        train_loss = 0.0
        train_acc = 0.0
        idx = 0
        pred_list_s = []
        y_list_s = []
        for i in self.train_step_outputs:
            idx += 1
            train_acc += i['accuracy']
            train_loss += i['loss']
            pred_list_s.extend(i['pred_list'])
            y_list_s.extend(i['y'])
        
        train_loss = train_loss / idx
        train_acc = train_acc / idx
        logger_.info(f"Epoch: {self.current_epoch}: training_epoch_end--loss/train: loss {train_loss}, accuracy {train_acc}") 

        self.log("loss/tra", train_loss, batch_size=10)
        self.log("accuracy/tra", train_acc, batch_size=10)

    def on_validation_epoch_end(self):

        val_loss = 0.0
        num_correct = 0
        num_total = 0

        for output, pred, labels in self.validation_step_outputs:
            val_loss += F.cross_entropy(output, labels, reduction="sum")
            num_correct += (pred == labels).sum()
            num_total += len(pred)
            # print("The number of pred:", len(pred))
        
        val_accuracy = num_correct / num_total
        val_loss = val_loss / num_total
        logger_.info(f"Epoch: {self.current_epoch}: val_epoch_end--loss/train: loss {val_loss}, accuracy {val_accuracy}") 
        self.log("accuracy/val", val_accuracy, batch_size=10)
        self.log("loss/val", val_loss, batch_size=10)

    def validation_step(self, batch, batch_index):
        x, y = batch
        y = y.long()
        x_out = self.forward(x)
        loss = F.cross_entropy(x_out, y, weight=torch.tensor(weight, dtype=torch.float).to(device))
        pred = x_out.argmax(-1)
        self.validation_step_outputs.append([x_out, pred, y])
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)
        return ({"optimizer": optimizer, "lr_scheduler":scheduler})

    def predict_step(self, batch, batch_idx):
        
        x, y = batch
        # y = y.long()
        # self.starter.record()
        x_out = self.forward(x.float())
        pred = x_out.argmax(-1)
        return pred, y

checkpoint_callback = ModelCheckpoint(
    monitor='loss/val',
    filename='sample-mnist-{epoch:02d}-{val_loss:.2f}',
    save_top_k=1,
    mode='min',
    save_last=True
)

logger_ = logging.getLogger("pytorch_lightning")
logger = pl.loggers.TensorBoardLogger(save_dir='/home/whm/Code/BGP-Security/code/BGP-Baseline', version='my_name9', name='lightning_logs')


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(12345)
    
    # dataset creation 
    # file_paths = ["/home/wuzheng/Packet/CompareMethod/Multiview/dataset/S_Outage_1.pkl", "/home/wuzheng/Packet/CompareMethod/Multiview/dataset/S_Outage_2.pkl", "/home/wuzheng/Packet/CompareMethod/Multiview/dataset/S_Prefix_2.pkl", "/home/wuzheng/Packet/CompareMethod/Multiview/dataset/S_Rl_1.pkl", "/home/wuzheng/Packet/CompareMethod/Multiview/dataset/S_Rl_2.pkl"]
    # path='/home/wuzheng/Packet/CompareMethod/Multiview/balanced_dataset/'
    # files = os.listdir(path)
    # file_paths = [os.path.join(path, i) for i in files] # for balanced dataset
    
    # class_ = {'Rl': 1,'Prefix':2, 'Outage': 3, 'normal':0}
    lr_monitor = LearningRateMonitor(logging_interval='step')
    # datasets = []
    window_size = 10
    
    # for file_path in file_paths:
    #     with open(file_path,'rb') as fp:
    #         data = pickle.load(fp)
        
    #     data = data.astype(np.float32)
    #     x = data[:,:-1]
        
    #     y = data[:, -1]
    #     for c in class_:
    #         if c in file_path:
    #             y[y == 1] = class_[c]
    #             break
    #     datasets.append(SlidingWindowDataset(x, y=y, window=window_size))
    # train_dataset = torch.utils.data.ConcatDataset(datasets)
    
    # torch.save(train_dataset, '/home/wuzheng/Packet/CompareMethod/Multiview/data/balanced_dataset.pt')
    # train_dataset = torch.load('/home/wuzheng/Packet/CompareMethod/Multiview/data/open_world.pt')
    # train_dataset = torch.load('/home/wuzheng/Packet/CompareMethod/Multiview/data/balanced_dataset.pt')
    train_dataset = torch.load('/data/data/sampling_link_1.0/fea/bgpviewer.pt')
    # print(train_dataset)
    # exit()
    
    # ind_list = list(range(len(train_dataset)))
    # random.shuffle(ind_list)

    # train_dataset = train_dataset[3:4]
    # exit()

    # data_path = "/home/wuzheng/Packet/CompareMethod/Multiview/dataset/S_Outage_1.pkl"
    # with open(data_path, 'rb') as file:
    #     anomaly_event_data = pickle.load(file)
    # print(anomaly_event_data[:,-1])
    # exit()
    # anomaly_event_X = anomaly_event_data[:, :-1].astype(np.float64)
    # anomaly_y = anomaly_event_data[:, -1]

    # anomaly_y_mask = anomaly_y == "1"

    # anomaly_y[anomaly_y_mask] = 3
    # anomaly_y = anomaly_y.astype('int')

    # anomaly_dataset = SlidingWindowDataset(data=anomaly_event_X, y=anomaly_y, window=window_size)
    
    y_ratio = Counter(i for i in train_dataset.y)

    weight = np.array(list(y_ratio.values())) / sum(y_ratio.values())

    train_loader, val_loader, _ = create_data_loaders(train_dataset, batch_size=100, shuffle=True)
    # print("Output:",train_loader.dataset.y)
    # exit()
    # predict_loader = torch.utils.data.DataLoader(anomaly_dataset, batch_size=100, shuffle=False)
    # print(train_loader.size)
    # exit()

    lighning_model = MTAD_GAT(n_features=10, window_size=window_size, dropout=0.2, n_layers=3, gru_hid_dim=130, hid_dim=100, num_classes=2) 
    
    num_epochs = 300
    val_check_interval = 0.25
    
    trainer = pl.Trainer(max_epochs = num_epochs, val_check_interval=1.0, log_every_n_steps=3, accelerator="gpu", callbacks=[checkpoint_callback, lr_monitor], logger=logger, enable_progress_bar=1)
    trainer.fit(lighning_model, train_loader, val_loader)

    res = trainer.predict(lighning_model, val_loader)
    
    pre_ = []
    label_ = []

    for r in res:
        pre_ += r[0].tolist()
        label_ += r[1].tolist()

    print(early_perf(label_, pre_))
    print(classification_report(label_, pre_, ))
    print('Accuracy:', accuracy_score(label_, pre_, ))