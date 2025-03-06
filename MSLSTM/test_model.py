import torch
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
import sys
# sys.path.append("/home/wuzheng/Packet/Graph_network/")
# from Mydataset import MyOwnDataset
from collections import Counter
# from pyg_test_418 import GCN
from classification import mymodel
from torch import nn
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, precision_score, classification_report
import time
from feature_extraction import SlidingWindowDataset
import pickle
import numpy as np

def calculate_rates(y_true, y_pred):
    com = confusion_matrix(y_true, y_pred).ravel()
    print(com)
    # tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    # false_positive_rate = fp / (fp + tn)
    # false_negative_rate = fn / (tp + fn)
    # return false_positive_rate, false_negative_rate

def early_perf(label, predict):
    predict = [1 if i != 0 else 0 for i in predict]
    first_anomaly = label.index(1)
    print('first_anomaly:',first_anomaly)
    t = None
    # print(predict[first_anomaly: first_anomaly+20])
    if 1 in predict[first_anomaly: first_anomaly+20]:
        t = predict[first_anomaly: first_anomaly+20].index(1)
    return t
 
class MyDetector(pl.LightningModule):
    def __init__(self, model_path):
        super(MyDetector, self).__init__()
        self.model_path = model_path
        self.model = mymodel.load_from_checkpoint(checkpoint_path=self.model_path, in_dim=10, hid_dim=64, n_layers=4, dropout=0.2, fore_out=2, for_n_layer=2, for_hid_dim=100)
    
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
    model_path = "/home/whm/Code/bgp/BGP-Security/BGP-Security/code/sendXiaohui/dataset/test/test.pt"
    # data_path = '/home/wuzheng/Packet/CompareMethod/mslstm/dataset/balanced_data.pt'

    torch.manual_seed(12345)
    clf_model = MyDetector(model_path=model_path)
    
    # print(f'Number of class:', dataset.num_classes)
    # print('The distribution is {}'.format(Counter([int(d.y[0]) for d in dataset])))

    # anomaly_list = [i for i, d in enumerate(dataset) if int(d.y[0]) != 0]
    # anomaly_dataset = dataset[anomaly_list]
    
    y_true_list = []
    y_pred_list = []
    ind_list = []
    
    model = clf_model.eval().to(device)
    
    # data_path = '/home/wuzheng/Packet/CompareMethod/Multiview/dataset/S_Prefix_2.pkl'
    # with open(data_path, 'rb') as file:
    #     anomaly_event_data = pickle.load(file)
    # # print(anomaly_event_data[:,-1])
    # # exit()
    # anomaly_event_X = anomaly_event_data[:, 1:-1].astype(np.float64)
    # anomaly_y = anomaly_event_data[:, -1]
    # # print(anomaly_y)
    # anomaly_y_mask = anomaly_y == "1"
    # # print(anomaly_y_mask)
    # # exit()
    # anomaly_y[anomaly_y_mask] = 1
    # anomaly_y = anomaly_y.astype('int')
    # # print(anomaly_event_data[:, :-1])
    # anomaly_dataset = SlidingWindowDataset(data=anomaly_event_X, y=anomaly_y, window=5)
    
    # sampling_rate
    # data_path = '/home/wuzheng/Packet/SimFingerPrint/Method_code/baseline/MSLSTM/dataset/sampling_outage1_0.2.pt'
    # test_data_path = "/data/data/sampling_link_0.8/fea/mslstm.pt"
    test_data_path = "/home/whm/Code/bgp/BGP-Security/BGP-Security/code/sendXiaohui/dataset/test/test.pt"
    # dataset = torch.load(data_path)
    # print("The number of dataset:", len(dataset))
    print("Complete loading model!")
    
    test_dataset = torch.load(test_data_path)
    
    starter = time.time()        
    for i in range(len(test_dataset)):
        X, y_true = test_dataset[i]
        X = torch.from_numpy(X).unsqueeze(0)
        pred = model(X.to(device))
        y_pred_list.append(int(pred))
        y_true_list.append(int(y_true))
    ender = time.time()
    
    print("y_pred:", Counter(y_pred_list))
    print("y_true:", Counter(y_true_list))
    # exit()

    # print(len(y_pred_list))
    # print(len(y_true_list))
    # print(y_true_list)
    # print(y_pred_list)
    
    # with open('/home/wuzheng/Packet/99code/plot/roc.dat', 'a') as fw:
    #     fw.write('\n')
    #     fw.write(','.join([str(i) for i in y_true_list]))
    #     fw.write('\n')
    #     fw.write(','.join([str(i) for i in y_pred_list]))
    # print(early_perf(y_true_list, y_pred_list))
    # ender = time.time()
    # print("y_true_list:", Counter(y_true_list))
    running_time = ender - starter

    print("The length of dataset:", len(y_true_list))
    fpr, fnr = calculate_rates(y_true_list, y_pred_list)
    f1 = f1_score(y_true_list, y_pred_list, average='macro')
    print("fpr:", fpr, "fnr:", fnr, "f1-score:", f1)

    print(running_time/1440)
    print("OA:", accuracy_score(y_true_list, y_pred_list))
   
    
