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
from dataset import SlidingWindowDataset
import pickle
import numpy as np

def calculate_rates(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    false_positive_rate = fp / (fp + tn)
    false_negative_rate = fn / (tp + fn)
    return false_positive_rate, false_negative_rate

def early_perf(label, predict):
    predict = [1 if i != 0 else 0 for i in predict]
    first_anomaly = label.index(1)
    # print('first_anomaly:',first_anomaly)
    t = None
    # print(predict[first_anomaly: first_anomaly+20])
    if 1 in predict[first_anomaly: first_anomaly+20]:
        t = predict[first_anomaly: first_anomaly+20].index(1)
    return t


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
        self.model = mymodel.load_from_checkpoint(checkpoint_path=self.model_path, WINDOW_SIZE=1, INPUT_SIZE=10, Hidden_SIZE=100, LSTM_layer_NUM=2, Num_class=2)
    
    def forward(self, batch):
        # print(batch.size())
        # exit()
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
    model_path = "/home/wuzheng/Packet/lightning_logs/outage/my_name6/checkpoints/last.ckpt"
    # data_path = '/home/wuzheng/Packet/CompareMethod/Self-operated/data/data.pt'
    # dataset = torch.load(data_path)

    # test_path = "/home/wuzheng/Packet/SimFingerPrint/Method_code/baseline/BGPviewer/dataset/sampling_outage1_1.0.pt"
    test_path = "/home/wuzheng/Packet/SimFingerPrint/Method_code/Manual_dataset/baseline/ISP-Operated/Outage/test_data.pt"
    test_data = torch.load(test_path)

    # data_path = '/home/wuzheng/Packet/CompareMethod/Multiview/dataset/S_Rl_1.pkl'
    # with open(data_path, 'rb') as file:
    #     anomaly_event_data = pickle.load(file)
    # # print(anomaly_event_data[:,-1])
    # # exit()
    # print(anomaly_event_data.shape)
    # anomaly_y = anomaly_event_data[:, -1]
    # # print(anomaly_y)
    # anomaly_y_mask = anomaly_y == "1"
    # # print(anomaly_y_mask)
    # # exit()
    # anomaly_y[anomaly_y_mask] = 3
    # anomaly_event_X = anomaly_event_data[:, 1:-1].astype(np.float)
    # anomaly_dataset = SlidingWindowDataset(data=anomaly_event_X, y=anomaly_y, window=10)
    
    torch.manual_seed(12345)
    clf_model = MyDetector(model_path=model_path)
    # print(f'Number of class:', dataset.num_classes)
    # print('The distribution is {}'.format(Counter([int(d.y[0]) for d in dataset])))

    # anomaly_list = [i for i, d in enumerate(dataset) if int(d.y[0]) != 0]
    # anomaly_dataset = dataset[anomaly_list]

    print("Complete loading model!")

    y_true_list = []
    y_pred_list = []
    ind_list = []
    
    model = clf_model.eval().to(device)
    
    # for child in model.modules():
    #     if isinstance(child, nn.Linear) and child.in_features == 384:
    #         child.register_forward_hook(hook=hook)
    
    starter = time.time()        
    for i in range(len(test_data)):
        X, y_true = test_data[i]
        X = torch.from_numpy(X).unsqueeze(0).float()
        
        pred = model(X.to(device))
        y_pred_list.append(int(pred))
        y_true_list.append(int(y_true))
    
    ender = time.time()
    # print("y_true_list:", Counter(y_true_list))
    running_time= ender - starter
    print(running_time/1440)
    # print('y_true_list:', y_true_list)
    # print('---------------')
    # print('y_pred_list:', y_pred_list)
    
    print(early_perf(y_true_list, y_pred_list))
    print(classification_report(y_true_list, y_pred_list))
    # print(y_pred_list)
    # print(y_true_list)
    # for i in range(len(y_true_list)):
    #     if y_pred_list[i] == 0 and y_true_list[i] != 0:
    #         print(ind_list[i])
    # print(ind_list)
    
    # a = sorted(zip(ind_list, range(len(ind_list))), key=lambda x: x[0])
    # ind_sorted = [i[1] for i in a]
    
    # '''
    # 提取输出层表征
    # '''
    # features = [i.squeeze().tolist() for i in features]
    
    # with open('/home/wuzheng/Packet/Graph_network/GCN/linear_output_outage.txt', 'w') as fw:
    #     for f in features:
    #         fw.write(','.join([str(i) for i in f]))
    #         fw.write('\n')
    
    #     fw.write(','.join([str(i) for i in y_pred_list]))
    #     fw.write('\n')
    #     fw.write(','.join([str(i) for i in ind_sorted]))

    # print("names: ", model.named_modules)
    # print(list(model.modules()))

    
    # print("Feature:", features)

    # print(sorted(zip(ind_list, y_pred_list), key=lambda x: x[0]))
    fpr, fnr = calculate_rates(y_true_list, y_pred_list)
    print('fpr:', fpr, 'fnr', fnr)
    accuracy = accuracy_score(y_true=y_true_list, y_pred=y_pred_list)
    # recall = recall_score(y_true=y_true_list, y_pred=y_pred_list, average='micro')
    # f1 = f1_score(y_true=y_true_list, y_pred=y_pred_list, average='micro')
    # miss_rate, false_rate = compute_missing_rate(y_pred_list, y_true_list)
    print('Overall accuracy', accuracy)
    
    # print(y_true_list)
    # print(y_pred_list)
    # with open('/home/wuzheng/Packet/99code/plot/roc.dat', 'a') as fw:
    #     fw.write('\n')
    #     fw.write(','.join([str(i) for i in y_true_list]))
    #     fw.write('\n')
    #     fw.write(','.join([str(i) for i in y_pred_list]))
    
    # print("Recall score:", recall)
    # print("f1_score:", f1)
    
    # print("miss rate:", miss_rate, "false rate:", false_rate)
    # print("precision:", precision_score(y_true=y_true_list, y_pred=y_pred_list, average='micro'))
    # print('running time:', running_time / len(y_pred_list))
    
