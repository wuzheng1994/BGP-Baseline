import numpy as np
import os
import pickle
import pywt
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
import pickle
import time
from collections import Counter
import json


np.set_printoptions(threshold=np.inf)

def Multi_Scale_Wavelet(trainX, level, is_multi=True, wave_type='db1'):
    
    temp = [[] for i in range(level)]
    N = trainX.shape[0]
    if (is_multi == True) and (level > 1):
        for i in range(level):

            x = []
            for _feature in range(len(trainX[0])):
                coeffs = pywt.wavedec(trainX[:,_feature], wave_type, level=level)
                current_level = level  - i
                x.append(coeffs[i+1])

            temp[current_level - 1].extend(np.transpose(np.array(x)))

    else:
        for tab in range(level):
            current_level = level - tab
            temp[current_level - 1].extend(trainX)

    return  np.array(temp), trainX

def Multi_Scale_Wavelet0(trainX, level, is_multi=True, wave_type='db1'):
    temp = [[] for i in range(level)]
    N = trainX.shape[0]
    if (is_multi == True) and (level > 1):
        for i in range(level):
            x = []
            for _feature in range(len(trainX[0])):
                coeffs = pywt.wavedec(trainX[:,_feature], wave_type, level=level)
                current_level = level  - i
                for j in range(i+1,level+1):
                    coeffs[j] = None
                _rec = pywt.waverec(coeffs, wave_type)
                x.append(_rec[:N])

            temp[current_level - 1].extend(np.transpose(np.array(x)))

    else:
        for tab in range(level):
            current_level = level - tab
            temp[current_level - 1].extend(trainX)
    # print("ALA")
    print((np.array(temp)).shape)

    return  np.array(temp), trainX

class SlidingWindowDataset(Dataset):
    def __init__(self, data, window, y, target_dim=None, horizon=0):
        super(SlidingWindowDataset, self).__init__()
        self.data = data
        self.window = window
        self.target_dim = target_dim
        self.horizon = horizon
        self.y = y

    # def __getitem__(self, index):
    #     print(type(index), type(self.window))  # 检查类型
    #     #str int
    #     print(index)
    #     x = self.data[index : index + self.window]

    #     y = self.y[index + self.window + self.horizon]
    #     return x, y

  
    def __getitem__(self, index):
        # 打印类型，确认 index 和 self.window 的类型
        print(type(index), type(self.window))  # 应该是 str 和 int
        print(f"Index: {index}")

        # 跳过无效的 index
        if index == "pytorch-lightning_version":
            print(f"Skipping index: {index} (invalid index)")
            # 返回一个默认值，可以是任意类型，确保不会返回 None
            return torch.zeros(self.window), torch.zeros(1)

        # 确保 index 是一个整数，假设它是字符串类型时需要转化
        try:
            index = int(index)  # 将 index 转换为整数
        except ValueError:
            print(f"Error: index '{index}' cannot be converted to an integer.")
            return torch.zeros(self.window), torch.zeros(1)  # 返回默认值（空张量）

        # 获取数据切片
        x = self.data[index : index + self.window]

        # 获取目标值
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

def standarize(data):
    # print(data[:10])
    # print(data.shape)
    # exit()
    res = (data - data.mean(axis=0)) / data.std(axis=0) 
    # print("std:",data.std(axis=0))
    
    # res = data / np.sqrt(np.sum(data**2))
    # print("res:", res)
    return res 


#清洗数据，确保json文件中只用0 和 1 字符串，没有整型
def clean_data(sampling_data):
    cleaned_data = []
    cleaned_labels = []
    
    for item in sampling_data:
        # 转换标签为整数（假设标签位于最后一列）
        try:
            label = int(item['label'])
        except ValueError:
            print(f"Invalid label {item['label']} found, skipping this entry.")
            continue  # 跳过无效标签的数据

        # 清洗数据部分：这里假设所有特征列在除标签外的其他字段中
        features = [item[k] for k in item if k != 'label']
        
        # 检查是否有缺失值或无效值，如果有则跳过该行
        if any(np.isnan(feature) or np.isinf(feature) for feature in features):
            print(f"Invalid data found in features {features}, skipping this entry.")
            continue
        
        cleaned_data.append(features)
        cleaned_labels.append(label)
    
    return np.array(cleaned_data), np.array(cleaned_labels)



if __name__ == "__main__":
    sampling_file = "/home/whm/Code/bgp/BGP-Security/BGP-Security/code/sendXiaohui/fea.json"
    
    with open(sampling_file, 'r') as f:
        data = json.load(f)
    
    # 使用 clean_data 函数清洗数据
    data, y = clean_data(data)

    # 数据是一个二维数组，y 是标签
    print("Cleaned data shape:", data.shape)
    print("Cleaned labels shape:", y.shape)

    # 接下来的处理和转换操作
    start_time = time.time()
    (res_0), _ = Multi_Scale_Wavelet0(data[:,:], level=2)[0]
    end_time = time.time()
    print("Used time:", end_time - start_time)

    # 创建数据集
    dataset = SlidingWindowDataset(res_0, window=10, y=y)

    # 保存数据集
    torch.save(dataset, '/home/whm/Code/bgp/BGP-Security/BGP-Security/code/sendXiaohui/dataset/test/test.pt')
 