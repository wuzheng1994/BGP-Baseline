from pathlib import Path
from datetime import datetime
import pandas as pd
import time
from utils import load_emb_distance

repo_dir = Path(__file__).resolve().parent.parent
route_change_dir = repo_dir/"routing_monitor"/"detection_result"/"2021"/"route_change"
beam_metric_dir = repo_dir/"routing_monitor"/"detection_result"/"2021"/"metric"  # 生成的度量文件存放在此
model_dir = repo_dir/"BEAM_engine"/"models"

beam_metric_dir.mkdir(exist_ok=True, parents=True)
# 加载嵌入距离模型
train_dir = model_dir/"20200901.as-rel.1000.10.128"
emb_d, dtw_d, path_d, emb, _, _ = load_emb_distance(train_dir, return_emb=True)

# 算仅存在于嵌入向量中的节点之间的动态时间弯曲（DTW）距离
def dtw_d_only_exist(s, t):
    return dtw_d([i for i in s if i in emb], [i for i in t if i in emb])

# 评估指定日期范围的路由变化
def evaluate_date_range(start_date, end_date):
    # 文件夹路径
    route_change_folder = route_change_dir  # 路由变化数据文件夹路径
    for i in route_change_folder.glob("*.csv"):  # 遍历当前文件夹中的所有CSV文件
        # 获取文件的日期部分，假设文件名的格式是 `YYYYMMDD.HHMM.csv`
        try:
            file_date = datetime.strptime(i.name.split('.')[0], '%Y%m%d')  # 解析 `YYYYMMDD` 格式
        except ValueError:
            print(f"Skipping file with invalid date format: {i.name}")
            continue  # 如果文件名不符合预期的格式，则跳过该文件
        
        if start_date <= file_date <= end_date:
            beam_metric_file = beam_metric_dir / f"{i.stem}.bm.csv"  # 生成的度量文件路径
            if beam_metric_file.exists():
                continue  # 如果度量文件已存在，则跳过

            df = pd.read_csv(i)
            start_time = time.time()
            print(start_time)
            path1 = [s.split(" ") for s in df["path1"].values]
            path2 = [t.split(" ") for t in df["path2"].values]

            metrics = pd.DataFrame.from_dict({
                "diff": [dtw_d(s,t) for s,t in zip(path1, path2)], 
                "diff_only_exist": [dtw_d_only_exist(s,t) for s,t in zip(path1, path2)], 
                "path_d1": [path_d(i) for i in path1],
                "path_d2": [path_d(i) for i in path2],
                "path_l1": [len(i) for i in path1],
                "path_l2": [len(i) for i in path2],
                "head_tail_d1": [emb_d(i[0], i[-1]) for i in path1],
                "head_tail_d2": [emb_d(i[0], i[-1]) for i in path2],
            })
            end_time = time.time()
            execution_time = end_time - start_time
            print(f"代码运行时间: {execution_time} 秒")
            # 保存度量结果到CSV文件
            metrics.to_csv(beam_metric_file, index=False)

# 设置日期范围并调用函数处理数据
start_date = datetime(2021, 10, 4)
end_date = datetime(2021, 10, 5)

# 单线程处理数据
evaluate_date_range(start_date, end_date)


'''from functools import lru_cache
from pathlib import Path
from datetime import datetime
import pandas as pd
from joblib import Parallel, delayed

from utils import load_emb_distance

repo_dir = Path(__file__).resolve().parent.parent
route_change_dir = repo_dir/"routing_monitor"/"detection_result"/"rrc00-24-12"/"route_change"
beam_metric_dir = repo_dir/"routing_monitor"/"detection_result"/"rrc00-24-12"/"BEAM_metric"
model_dir = repo_dir/"BEAM_engine"/"models"

beam_metric_dir.mkdir(exist_ok=True, parents=True)
#加载嵌入距离模型
train_dir = model_dir/"20240201.as-rel.500.10.128"
emb_d, dtw_d, path_d, emb, _, _ = load_emb_distance(train_dir, return_emb=True)
#算仅存在于嵌入向量中的节点之间的动态时间弯曲（DTW）距离
def dtw_d_only_exist(s, t):
    return dtw_d([i for i in s if i in emb], [i for i in t if i in emb])
#评估每个月的路由变化
def evaluate_monthly_for(ym):
    for i in route_change_dir.glob(f"{ym}*.csv"):
        beam_metric_file = beam_metric_dir/f"{i.stem}.bm.csv"
        if beam_metric_file.exists(): continue

        df = pd.read_csv(i)

        path1 = [s.split(" ") for s in df["path1"].values]
        path2 = [t.split(" ") for t in df["path2"].values]

        metrics = pd.DataFrame.from_dict({
            "diff": [dtw_d(s,t) for s,t in zip(path1, path2)], 
            "diff_only_exist": [dtw_d_only_exist(s,t) for s,t in zip(path1, path2)], 
            "path_d1": [path_d(i) for i in path1],
            "path_d2": [path_d(i) for i in path2],
            "path_l1": [len(i) for i in path1],
            "path_l2": [len(i) for i in path2],
            "head_tail_d1": [emb_d(i[0], i[-1]) for i in path1],
            "head_tail_d2": [emb_d(i[0], i[-1]) for i in path2],
        })
        
        metrics.to_csv(beam_metric_file, index=False)

# 并行处理2023年1月的每天数据
Parallel(backend="multiprocessing", n_jobs=7, verbose=10)(
    delayed(evaluate_monthly_for)(f"2023{m:02}") for m in range(1, 32)
)
'''