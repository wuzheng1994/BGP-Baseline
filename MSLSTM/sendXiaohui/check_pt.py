'''
import torch
from feature_extraction import SlidingWindowDataset

# 加载文件
file_path = '/home/whm/Code/bgp/BGP-Security/BGP-Security/code/sendXiaohui/dataset/sampling_1.0.pt'
data = torch.load(file_path)

# 打印文件中的内容
print(data)
'''
import json

## 假设 data 是你加载的 JSON 数据
with open("/home/whm/Code/bgp/BGP-Security/BGP-Security/code/sendXiaohui/fea.json", "r") as f:
    data = json.load(f)



# 提取标签字段
labels = [item['label'] for item in data]

# 检查标签的唯一值
print(set(labels))

# 将标签转换为整数类型
cleaned_labels = [int(label) if isinstance(label, str) else label for label in labels]

# 打印唯一标签
print(set(cleaned_labels)) 