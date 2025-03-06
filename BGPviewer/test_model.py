import torch
import pytorch_lightning as pl
import sys
# sys.path.append("/home/wuzheng/Packet/Graph_network/")
# from Mydataset import MyOwnDataset
from collections import Counter
from compare_method import MTAD_GAT
from torch import nn
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, precision_score, classification_report
import time
from mydataset import SlidingWindowDataset
from mydataset import create_data_loaders

def calculate_rates(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    false_positive_rate = fp / (fp + tn)
    false_negative_rate = fn / (tp + fn)
    return false_positive_rate, false_negative_rate


def compute_missing_rate(y_predict, y_label):
    matrix = confusion_matrix(y_pred=y_predict, y_true=y_label, )
    tn, fp, fn, tp =  matrix.ravel()
    missing_rate = fn / (tp + fn)
    false_rate = fp / (fp + tn)
    return (missing_rate, false_rate)

class MyDetector(pl.LightningModule):
    def __init__(self, model_path):
        super(MyDetector, self).__init__()
        self.model_path = model_path
        self.model = MTAD_GAT.load_from_checkpoint(checkpoint_path=self.model_path, n_features=10, window_size=10, dropout=0.2, n_layers=3, gru_hid_dim=130, hid_dim=100, num_classes=2)
    
    def forward(self, batch):
        x = batch
        x_com = self.model(x)
        pred = x_com.argmax(-1)
        return pred


def hook(module, input, output):
    features.append(output.detach())
    return None

if __name__ == "__main__":
    
    features = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_path = "/home/wuzheng/Packet/SimFingerPrint/Method_code/baseline/BGPviewer/lightning_logs/my_name9/checkpoints/last.ckpt"
    # data_path = '/home/wuzheng/Packet/CompareMethod/Multiview/data/data.pt'
    test_data_path = "/data/data/sampling_link_1.0/fea/bgpviewer.pt"

    dataset = torch.load(test_data_path)

    print("Complete loading model!")
    torch.manual_seed(12345)
    clf_model = MyDetector(model_path=model_path)

    y_true_list = []
    y_pred_list = []
    ind_list = []

    model = clf_model.eval().to(device)

    starter = time.time()        
    for i in range(len(dataset)):
        X, y_true = dataset[i]
        X = torch.from_numpy(X).unsqueeze(0)
        pred = model(X.to(device))
        y_pred_list.append(int(pred))
        y_true_list.append(int(y_true))
    ender = time.time()

    print("y_pred_list:", Counter(y_pred_list))
    running_time= ender - starter
    # print(running_time/len(dataset))
    # print(classification_report(y_true_list, y_pred_list))
    
    
    # fpr, fnr = calculate_rates( y_true_list, y_pred_list)
    # print('fpr:', fpr, 'fnr:', fnr)
    print("f1-score:", f1_score(y_true_list, y_pred_list))
    print("Accuracy:", accuracy_score(y_true_list, y_pred_list))
    print('time:', running_time)
    
    fpr, fnr = calculate_rates( y_true_list, y_pred_list)
    print("fpr:", fpr, "fnr:", fnr)
    # print(y_true_list)
    # print(y_pred_list)
    
    # with open('/home/wuzheng/Packet/99code/plot/roc.dat', 'a') as fw:
    #     fw.write('\n')
    #     fw.write(','.join([str(i) for i in y_true_list]))
    #     fw.write('\n')
    #     fw.write(','.join([str(i) for i in y_pred_list]))