import torch
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
import pickle
import numpy as np
from collections import Counter

class SlidingWindowDataset(Dataset):
    def __init__(self, data, window, y, target_dim=None, horizon=0):
        super(SlidingWindowDataset, self).__init__()
        self.data = data
        self.window = window
        self.target_dim = target_dim
        self.horizon = horizon
        self.y = y

    def __getitem__(self, index):
        x = self.data[index : (index + self.window)]
        y = self.y[index + self.window + self.horizon]
        return x, y

    def __len__(self):
        return len(self.data) - self.window

def create_data_loaders(train_dataset, batch_size, val_split=0.2, shuffle=True, test_dataset=None):
    train_loader, val_loader, test_loader = None, None, None
    if val_split == 0.0:
        print(f"train_size: {len(train_dataset)}")
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    else:
        dataset_size = len(train_dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(val_split * dataset_size))
        if shuffle:
            np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]

        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
        val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=valid_sampler)

        print(f"train_size: {len(train_indices)}")
        print(f"validation_size: {len(val_indices)}")

    if test_dataset is not None:
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        print(f"test_size: {len(test_dataset)}")

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    
    # file_path = "/home/wuzheng/Packet/CompareMethod/Multiview/balanced_dataset/balanced_S_Outage1.pkl"
    # open_world = "/home/wuzheng/Packet/CompareMethod/Multiview/S_res.npy"
    # data = "/home/wuzheng/Packet/CompareMethod/Multiview/results/sampling_1.0.txt"
    # data = np.load(open_world)
    # print(data.shape)
    # exit()
    # sampling_file = "/home/wuzheng/Packet/CompareMethod/Multiview/results/sampling_0.8.txt"
    # sampling_data = []
    # sampling_y = []
    # with open(sampling_file, 'r') as f:
    #     for line in f:
    #         if line != ' ':
    #             line = line.rstrip().split(',')
    #             sampling_data.append(line[:-1])
    #             sampling_y.append(line[-1])
    #         else:
    #             pass
    anomaly = 'Outage'
    # sampling_file = f"/home/wuzheng/Packet/SimFingerPrint/Method_code/Manual_dataset/Prefix/feature/fea.json"
    sampling_file = f"/home/wuzheng/Packet/SimFingerPrint/Method_code/baseline/BGPviewer/G_outage1.json"
    
    sampling_data = []
    sampling_y = []
    # with open(sampling_file, 'r') as f:
    #     for line in f:
    #         if line != ' ':
    #             line = line.rstrip().split(',')
    #             sampling_data.append(line[:-1])
    #             sampling_y.append(line[-1])
    #         else:
    #             pass
    import json
    with open(sampling_file, 'r') as f:
        data = json.load(f)
    
    keys_name = list(data[0].keys())
    keys_name.remove('label')
    keys_name = keys_name + ['label']
    sampling_data = [[item[k_id] for k_id in keys_name] for item in data]

    y = np.array(sampling_data)[:,-1].astype(np.int16)
    data = np.array(sampling_data)[:,:-1].astype(np.float32)
    
    # data = np.array(sampling_data).astype(np.float32)
    # y = np.array(sampling_y).astype(np.int16)
    
    print("The shape of data:", data.shape)
    print("The shape of y:", y.shape)
    # print(y)
    # exit()
    # # 
    # data = np.load(open_world)
    
    # data = data.astype(np.float32)
    # x = data[:, :]
    # y = data[:, -1]

    train_dataset = SlidingWindowDataset(data=data, y=y, window=1)

    # torch.save(train_dataset, '/home/wuzheng/Packet/CompareMethod/Multiview/data/open_world.pt')
    # torch.save(train_dataset, f'/home/wuzheng/Packet/SimFingerPrint/Method_code/Manual_dataset/{anomaly}/baseline/data.pt')
    torch.save(train_dataset, f'/home/wuzheng/Packet/SimFingerPrint/Method_code/Manual_dataset/baseline/ISP-Operated/Outage/train_data.pt')

    train_loader, val_loader, _ = create_data_loaders(train_dataset, batch_size=10)
    print("The size of data:", train_loader.dataset.data.shape)