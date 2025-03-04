import os
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
import pickle
import numpy as np
import torch
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
        x = self.data[index : index + self.window]
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
    
    file_path = '/home/wuzheng/Packet/CompareMethod/Multiview/balanced_dataset/'
    files = [os.path.join(file_path, f) for f in os.listdir(file_path)]
    print("files:", files)
    datasets = []
    class_ = {'Rl':'1', 'Prefix': '2', 'Outage': '3'}
    for file in files[:]:
        name = os.path.basename(file)
        for c in class_:
            if c in name:
                label_ = int(class_[c])
                print(label_)
                break
        with open(file, 'rb') as pl:
            event = pickle.load(pl)
                
        a = np.array(event, dtype=np.float32)
        
        y = np.where(a[:, -1]==0, a[:, -1], label_)
        a = a[:, 1:-1]
        scaler = StandardScaler().fit(a)
        a = scaler.transform(a)
        # a = standarize(a[:, 1:-1])
        # print("a:", a.shape)
        # print("y:",Counter(y))
        # exit()
        # res_0 = Multi_Scale_Wavelet0(a[:, 1:-1], level=2)[0]
        # res_0 = [standarize(r) for r in res[0]]
        # print("res_0:", res_0[0][:3])
        # exit()
        # res_0 = np.concatenate(a, axis=1)
        datasets.append(SlidingWindowDataset(a, window=10, y=y))
    
    train_dataset = torch.utils.data.ConcatDataset(datasets)
    # print("y:", train_dataset.datasets)
    # train_loader, val_loader, _ = create_data_loaders(train_dataset, batch_size=10)
    torch.save(train_dataset, '/home/wuzheng/Packet/CompareMethod/Self-operated/data/balanced_data.pt')