import os
import numpy as np
import torch
from torch import nn
import pytorch_lightning as pl
from feature_extraction import SlidingWindowDataset, create_data_loaders
import logging
from pytorch_lightning.callbacks import ModelCheckpoint
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, precision_score, classification_report
import random

class GRULayer(nn.Module):
    """Gated Recurrent Unit (GRU) Layer
    :param in_dim: number of input features
    :param hid_dim: hidden size of the GRU
    :param n_layers: number of layers in GRU
    :param dropout: dropout rate
    """

    def __init__(self, in_dim, hid_dim, n_layers, dropout):
        super(GRULayer, self).__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.dropout = 0.0 if n_layers == 1 else dropout
        self.gru = nn.GRU(in_dim, hid_dim, num_layers=n_layers, batch_first=True, dropout=self.dropout)

    def forward(self, x):
        out, h = self.gru(x)
        # Extracting from last layer
        out, h = out[-1, :, :], h[-1, :, :]  
        return out, h

class forcastModel(nn.Module):
    
    def __init__(self, in_dim, hid_dim, out_dim, dropout, n_layers):
        super(forcastModel, self).__init__()
        layers = [nn.Linear(in_dim, hid_dim)]
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hid_dim, hid_dim))
        layers.append(nn.Linear(hid_dim, out_dim))
        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        for l in range(len(self.layers)-1):
            x = self.relu(self.layers[l](x))
            x = self.dropout(x)
        return self.layers[-1](x)

class mymodel(pl.LightningModule):
    def __init__(self, in_dim, hid_dim, n_layers, dropout, fore_out, for_n_layer, for_hid_dim):
        self.n_layers = n_layers
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.fore_out = fore_out
        self.dropout = dropout
        self.for_n_layer = for_n_layer
        self.for_hid_dim = for_hid_dim
        super(mymodel, self).__init__()

        # GRU层
        self.gru = GRULayer(self.in_dim, self.hid_dim, self.n_layers, self.dropout)

        # 预测模型
        self.fore = forcastModel(self.hid_dim, self.for_hid_dim, self.fore_out, self.dropout, self.n_layers)

        # 归一化层
        self.norm1 = nn.BatchNorm1d(self.in_dim, eps=1e-10)  # 输入特征归一化
        self.norm2 = nn.BatchNorm1d(self.hid_dim, eps=1e-10)  # GRU 输出特征归一化

        self.validation_step_outputs = []
        self.train_step_outputs = []
        
    def forward(self, x):
        print("Shape of x before norm1:", x.shape)  # 打印x的形状
        #[10,10,26]
        #x = torch.permute(x, (0,2,1)).float()  # permute 用于改变维度顺序
        #[10,26,10]
        print("Shape of x before norm1:", x.shape)  # 打印x的形状

        # BatchNorm1d应用于输入
        x = x.float()  # Convert to float before passing to norm1
        x = self.norm1(x)
        print("Shape of x after norm1:", x.shape)

        # GRU层
        x = torch.permute(x, (0, 2, 1))  # 转换回原始顺序
        _, h_end = self.gru(x)

        # GRU的输出h_end进行归一化
        h_end = self.norm2(h_end)

        # 经过全连接层
        h_end = h_end.view(x.shape[0], -1)  # 扁平化
        x = self.fore(h_end)
        return x

    def training_step(self, batch, batch_idx):
        
        x, y = batch
        y = y.long()
        
        x_out = self.forward(x)
        loss = F.cross_entropy(x_out, y)
        pred = x_out.argmax(-1)
        accuracy = (pred == y).sum() / pred.shape[0]
        self.train_step_outputs.append({"acc": accuracy.detach().cpu().item(), 'loss': loss.detach().cpu().item()})
        return loss

    def on_train_epoch_end(self) -> None:
        train_loss = 0.0
        train_acc = 0.0
        idx = 0
        for i in self.train_step_outputs:
            idx += 1
            train_acc += i['acc']
            train_loss += i['loss']
        train_loss = train_loss / idx
        train_acc = train_acc / idx
        logger_.info(f"Epoch: {self.current_epoch}: training_epoch_end--loss/train: loss {train_loss}, accuracy {train_acc}") 

    def on_validation_epoch_end(self):
        val_loss = 0.0
        num_correct = 0
        num_total = 0
        for i in self.validation_step_outputs:
            val_loss += i["loss"]
            num_correct += i["accuracy"]
            num_total += 1
        val_accuracy = num_correct / num_total
        val_loss = val_loss / num_total
        logger_.info(f"Epoch: {self.current_epoch}: val_epoch_end--loss/train: loss {val_loss}, accuracy {val_accuracy}") 
        self.log("accuracy/val", val_accuracy, batch_size=10)
        self.log("loss/val", val_loss, batch_size=10)

    def validation_step(self, batch, batch_index):
     x, y = batch
     y = y.long()
     
     # 打印调试信息
     print(f"Unique labels in batch: {torch.unique(y)}")
     x_out = self.forward(x)
     
     # 打印模型输出的形状和部分输出值
     print(f"x_out shape: {x_out.shape}")
     print(f"x_out (some values): {x_out[:5]}")  # 打印前5个输出
     
     # 检查输出维度是否正确
     if x_out.shape[1] != self.fore_out:  # 这是类别数量
         print(f"Warning: Model output has {x_out.shape[1]} classes, but expected {self.fore_out}.")
     
     pred = x_out.argmax(-1)
     
     # 打印预测标签与真实标签的对比
     print(f"Predicted labels: {pred[:5]}")
     print(f"True labels: {y[:5]}")
     
     accuracy_score = (pred == y).sum() / pred.shape[0]
     loss = F.cross_entropy(x_out, y)
     
     self.validation_step_outputs.append({'loss': loss.detach().cpu().item(), "accuracy": accuracy_score.detach().cpu().item()})
     return loss


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer    

    def predict_step(self, batch, batch_idx):
        x, y = batch
        x_out = self.forward(x)
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
logger = pl.loggers.TensorBoardLogger(save_dir='./', version='my_name12', name='lightning_logs')

if __name__ == "__main__":
    path = "/home/whm/Code/bgp/BGP-Security/BGP-Security/code/sendXiaohui/dataset/test/test.pt"  # pt路径
    
    train_dataset = torch.load(path)

    ind_list = list(range(len(train_dataset)))
    random.shuffle(ind_list)
    
    random_dataset = []
    for i in ind_list:
        random_dataset.append(train_dataset[i])
    
    #原来为10
    train_loader, val_loader, _ = create_data_loaders(random_dataset, batch_size=10)
    
    window_size = 10
    n_feature = 10
    num_class = 2

    my = mymodel(in_dim=n_feature, hid_dim=64, n_layers=4, dropout=0.2, fore_out=num_class, for_n_layer=2, for_hid_dim=100)
    num_epochs = 300
    val_check_interval = 0.25
    precision = 32
    trainer = pl.Trainer(precision=precision, max_epochs=num_epochs, val_check_interval=1.0, log_every_n_steps=3, accelerator="gpu", callbacks=[checkpoint_callback], logger=logger, enable_progress_bar=1, detect_anomaly=True)
    trainer.fit(my, train_loader, val_loader)
    res = trainer.predict(my, val_loader)

    pre_ = []
    label_ = []
    for r in res:
        pre_ += r[0].tolist()
        label_ += r[1].tolist()
    
    print(classification_report(label_, pre_))
    print('Accuracy:', accuracy_score(label_, pre_))
