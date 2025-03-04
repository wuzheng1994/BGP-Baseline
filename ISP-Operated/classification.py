import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import logging
from pytorch_lightning.callbacks import ModelCheckpoint
from dataset import SlidingWindowDataset
from dataset import create_data_loaders
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, precision_score, classification_report
from collections import Counter

def calculate_rates(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    false_positive_rate = fp / (fp + tn)
    false_negative_rate = fn / (tp + fn)
    return false_positive_rate, false_negative_rate

class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 1)
        )

    def forward(self, encoder_outputs):
        # (B, L, H) -> (B , L, 1)
        energy = self.projection(encoder_outputs)
        weights = F.softmax(energy.squeeze(-1), dim=1)
        # (B, L, H) * (B, L, 1) -> (B, H)
        outputs = (encoder_outputs * weights.unsqueeze(-1)).sum(dim=1)
        return outputs, weights

class SA_LSTM(nn.Module):
    def __init__(self,WINDOW_SIZE,INPUT_SIZE,Hidden_SIZE,LSTM_layer_NUM, Num_class):
        super(SA_LSTM, self).__init__()
        self.WINDOW_SIZE = WINDOW_SIZE
        self.INPUT_SIZE = INPUT_SIZE
        self.Hidden_SIZE = Hidden_SIZE
        self.LSTM_layer_NUM = LSTM_layer_NUM
        self.Num_class = Num_class
        
        self.BN = nn.BatchNorm1d(self.WINDOW_SIZE)
        self.lstm = nn.LSTM(input_size=INPUT_SIZE,
                            hidden_size=Hidden_SIZE,
                            num_layers=LSTM_layer_NUM,
                            batch_first=True,
                            )
        self.attention = SelfAttention(Hidden_SIZE)
        self.out = nn.Sequential(nn.Linear(Hidden_SIZE, self.Num_class), nn.Softmax())

    def forward(self, x):
        # print("The size of x:",x.size())
        x = self.BN(x)
        # x = torch.permute(x, (0,2,1))
        
        r_out, hidden = self.lstm(x, None)  # x(batch,time_step,input_size)
        r_out, attn_weights = self.attention(r_out)
        out = self.out(r_out)
        
        return out ,torch.mean(attn_weights, dim=-2)

class mymodel(pl.LightningModule):
    def __init__(self, WINDOW_SIZE, INPUT_SIZE, Hidden_SIZE, LSTM_layer_NUM, Num_class):
        super(mymodel, self).__init__()
        self.model = SA_LSTM(WINDOW_SIZE, INPUT_SIZE, Hidden_SIZE, LSTM_layer_NUM, Num_class)
        self.validation_step_outputs = []
        self.train_step_outputs = []
        
    def forward(self, x):
        out, _ = self.model(x)
        return out

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
        loss = F.cross_entropy(x_out, y)
        pred = x_out.argmax(-1)
        self.validation_step_outputs.append([x_out, pred, y])
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
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
logger = pl.loggers.TensorBoardLogger(save_dir='/home/wuzheng/Packet/SimFingerPrint/Method_code/baseline/Self-operated/lightning_logs/', version='my_name5', name='lightning_logs')

if __name__ == "__main__":
    # path = "/home/wuzheng/Packet/CompareMethod/Self-operated/data/balanced_data.pt"
    # path = '/home/wuzheng/Packet/CompareMethod/Multiview/data/open_world.pt'
    path = '/home/wuzheng/Packet/SimFingerPrint/Method_code/baseline/BGPviewer/dataset/sampling_outage1_1.0.pt'
    # path = '/home/wuzheng/Packet/SimFingerPrint/Method_code/Manual_dataset/Prefix/baseline/data.pt'
    
    train_dataset = torch.load(path)
    
    train_loader, val_loader, _ = create_data_loaders(train_dataset, batch_size=100)
    y_val = []
    for i in range(len(val_loader)):
        y_val += list(val_loader)[i][1].tolist()
    print(Counter(y_val))
    # exit()
    
    
    # window_size = 10
    window_size = 10
    n_feature = 27
    num_class = 2

    my = mymodel(WINDOW_SIZE=window_size, INPUT_SIZE=n_feature, Hidden_SIZE=100, LSTM_layer_NUM=2, Num_class=num_class)
    num_epochs = 300
    val_check_interval = 0.25
    trainer = pl.Trainer(max_epochs = num_epochs, val_check_interval=1.0, log_every_n_steps=3, accelerator="gpu", callbacks=[checkpoint_callback], logger=logger, enable_progress_bar=1, detect_anomaly=True, gradient_clip_val=0.5, gradient_clip_algorithm="value")
    

    trainer.fit(my, train_loader, val_loader)
    res = trainer.predict(my, val_loader)

    pre_ = []
    label_ = []

    for r in res:
        pre_ += r[0].tolist()
        label_ += r[1].tolist()
    
    print(classification_report(label_, pre_))
    print('Accuracy:', accuracy_score(label_, pre_))
        
    fpr, fnr = calculate_rates(label_, pre_)
    print('fpr:', fpr, 'fnr', fnr)