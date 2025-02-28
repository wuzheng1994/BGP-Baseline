from pathlib import Path
import pandas as pd
import numpy as np
import json
from utils import approx_knee_point, event_aggregate
from datetime import datetime
import time

repo_dir = Path(__file__).resolve().parent.parent
route_change_dir = repo_dir / "routing_monitor" / "detection_result" / "2018" / "route_change"
beam_metric_dir = repo_dir / "routing_monitor" / "detection_result" / "2018" / "metric"
reported_alarm_dir = repo_dir / "routing_monitor" / "detection_result" / "2018" / "reported_alarms"

def load_montly_data(ym, preprocessor=lambda df: df):
    route_change_files = sorted(route_change_dir.glob(f"{ym}*.csv"))
    beam_metric_files = sorted(beam_metric_dir.glob(f"{ym}*.csv"))
    datetimes = [i.stem.replace(".", "")[:-2] for i in route_change_files]
    bulk_datetimes, bulk_indices = np.unique(datetimes, return_index=True)
    bulk_ranges = zip(bulk_indices, bulk_indices[1:].tolist() + [len(datetimes)])
    
    # 加载并合并一个日期块的数据
    def load_one_bulk(i, j):
        rc_df = pd.concat(list(map(pd.read_csv, route_change_files[i:j])))
        bm_df = pd.concat(list(map(pd.read_csv, beam_metric_files[i:j])))
        return pd.concat([rc_df, bm_df], axis=1)

    # 不使用并行处理，直接逐个加载日期块的数据
    bulks = [preprocessor(load_one_bulk(i, j)) for i, j in bulk_ranges]

    return bulk_datetimes, bulks

def metric_threshold(df, metric_col):
    values = df[metric_col]
    mu = np.mean(values)
    sigma = np.std(values)
    metric_th = mu + 4 * sigma

    print("reference metric: ")
    print(values.describe())
    print(f"metric threshold: {metric_th}")

    return metric_th

def forwarder_threshold(df, event_key):
    route_changes = tuple(df.groupby(event_key))
    forwarder_num = [len(j["forwarder"].unique()) for _, j in route_changes]
    forwarder_th, cdf = approx_knee_point(forwarder_num)

    print("reference forwarder: ")
    print(pd.Series(forwarder_num).describe())
    print(f"forwarder threshold: {forwarder_th}")

    return forwarder_th

def window(df0, df1, metric="diff", event_key=["prefix1", "prefix2"], dedup_index=["prefix1", "prefix2", "forwarder", "path1", "path2"]):
    if dedup_index is not None:
        df0 = df0.drop_duplicates(dedup_index, keep="first", inplace=False, ignore_index=True)
    
    with pd.option_context("mode.use_inf_as_na", True):
        df0 = df0.dropna(how="any")
    
    metric_th = metric_threshold(df0, metric)
    forwarder_th = forwarder_threshold(df0, event_key)

    events = {}
    for key, ev in tuple(df1.groupby(event_key)):
        if len(ev["forwarder"].unique()) <= forwarder_th: continue
        ev_sig = ev.sort_values(metric, ascending=False).drop_duplicates("forwarder")
        ev_anomaly = ev_sig.loc[ev_sig[metric] > metric_th]
        if ev_anomaly.shape[0] <= forwarder_th: continue

        events[key] = ev_anomaly

    if events:
        start = time.time()
        _, df = event_aggregate(events)
        end = time.time()
        print("Time consumption:", end - start)
        print('---------------------------daole---------------------------')
        exit()
        n_alarms = len(df['group_id'].unique())
    else:
        df = None
        n_alarms = 0

    info = dict(
        metric=metric,
        event_key=event_key,
        metric_th=float(metric_th),
        forwarder_th=int(forwarder_th),
        n_raw_events=len(events),
        n_alarms=n_alarms,
    )

    return info, df

def report_alarm_monthly(ym, metric):
    def preprocessor(df):
        df["diff_balance"] = df["diff"] / (df["path_d1"] + df["path_d2"])
        return df

    save_dir = reported_alarm_dir / metric / ym
    save_dir.mkdir(parents=True, exist_ok=True)

    datetimes, bulks = load_montly_data(ym, preprocessor)
    indices = np.arange(len(bulks))
    infos = []

    for i, j in list(zip(indices[:-1], indices[1:])):
        info = dict(d0=datetimes[i], d1=datetimes[j])
        _info, df = window(bulks[i], bulks[j], metric=metric)
        info.update(**_info)

        if df is None:
            info.update(save_path=None)
        else:
            save_path = save_dir / f"{datetimes[i]}_{datetimes[j]}.alarms.csv"
            df.to_csv(save_path, index=False)
            info.update(save_path=str(save_path))

        infos.append(info)

    json.dump(infos, open(save_dir / f"info_{ym}.json", "w"), indent=2)


start_date = datetime(2018, 11, 12)
end_date = datetime(2018,11, 13)

for date in pd.date_range(start=start_date, end=end_date):
    ym = date.strftime("%Y%m")
    report_alarm_monthly(ym, "diff_balance")

'''#!/usr/bin/env python3
#-*- coding: utf-8 -*-

from pathlib import Path
from joblib import Parallel, delayed
from concurrent.futures import ThreadPoolExecutor
from utils import approx_knee_point, event_aggregate
import json
import pandas as pd
import numpy as np
import time

repo_dir = Path(__file__).resolve().parent.parent
route_change_dir = repo_dir/"routing_monitor"/"detection_result"/"rrc00-24-12"/"route_change"
beam_metric_dir = repo_dir/"routing_monitor"/"detection_result"/"rrc00-24-12"/"metric"
reported_alarm_dir = repo_dir/"routing_monitor"/"detection_result"/"rrc00-24-12"/"reported_alarms"

def load_montly_data(ym, preprocessor=lambda df: df):
    route_change_files = sorted(route_change_dir.glob(f"{ym}*.csv"))
    beam_metric_files = sorted(beam_metric_dir.glob(f"{ym}*.csv"))
    datetimes = [i.stem.replace(".","")[:-2] for i in route_change_files]
    #使用np.unique获取唯一的日期，并计算每个日期块的索引范围
    bulk_datetimes, bulk_indices = np.unique(datetimes, return_index=True)
    bulk_ranges = zip(bulk_indices, bulk_indices[1:].tolist()+[len(datetimes)])
    #加载并合并一个日期块的数据
    def load_one_bulk(i,j):
        rc_df = pd.concat(list(map(pd.read_csv, route_change_files[i:j])))
        bm_df = pd.concat(list(map(pd.read_csv, beam_metric_files[i:j])))
        return pd.concat([rc_df, bm_df], axis=1)
    #ThreadPoolExecutor并行加载所有日期块的数据
    with ThreadPoolExecutor(max_workers=4) as executor:
        bulks = list(executor.map(
                    lambda x: preprocessor(load_one_bulk(*x)), bulk_ranges))

    return bulk_datetimes, bulks
#定义一个函数metric_threshold，用于计算度量值的阈值
def metric_threshold(df, metric_col):
    values = df[metric_col]
    mu = np.mean(values)
    sigma = np.std(values)
    metric_th = mu+4*sigma

    print("reference metric: ")
    print(values.describe())
    print(f"metric threshold: {metric_th}")

    return metric_th
#函数forwarder_threshold，用于计算转发器数量的阈值
def forwarder_threshold(df, event_key):
    route_changes = tuple(df.groupby(event_key))
    forwarder_num = [len(j["forwarder"].unique()) for _, j in route_changes]
    forwarder_th, cdf = approx_knee_point(forwarder_num)

    print("reference forwarder: ")
    print(pd.Series(forwarder_num).describe())
    print(f"forwarder threshold: {forwarder_th}")

    return forwarder_th
#在两个数据帧之间检测异常
def window(df0, df1, # df0 for reference, df1 for detection
        metric="diff", event_key=["prefix1", "prefix2"],
        dedup_index=["prefix1", "prefix2", "forwarder", "path1", "path2"]):
    #如果指定了去重索引，去除df0中的重复行
    if dedup_index is not None:
        df0 = df0.drop_duplicates(dedup_index, keep="first", inplace=False, ignore_index=True)
    #将无穷大值视为缺失值，并去除包含缺失值的行
    with pd.option_context("mode.use_inf_as_na", True):
        df0 = df0.dropna(how="any")
    #计算度量值和转发器数量的阈值
    metric_th = metric_threshold(df0, metric)
    forwarder_th = forwarder_threshold(df0, event_key)
    #遍历df1中的事件，跳过转发器数量小于阈值的事件
    events = {}
    for key,ev in tuple(df1.groupby(event_key)):
        if len(ev["forwarder"].unique()) <= forwarder_th: continue
        #对每个事件进行排序和去重，然后检测度量值超过阈值的异常事件
        ev_sig = ev.sort_values(metric, ascending=False).drop_duplicates("forwarder")
        ev_anomaly = ev_sig.loc[ev_sig[metric]>metric_th]
        if ev_anomaly.shape[0] <= forwarder_th: continue

        events[key] = ev_anomaly
    #将检测到的异常事件存储在字典中
    if events:
        _, df = event_aggregate(events)
        n_alarms = len(df['group_id'].unique())
    else:
        df = None
        n_alarms = 0
    #构建一个包含检测信息的字典
    info = dict(
        metric=metric,
        event_key=event_key,
        metric_th=float(metric_th),
        forwarder_th=int(forwarder_th),
        n_raw_events=len(events),
        n_alarms=n_alarms,
    )

    return info, df
#定义一个函数report_alarm_monthly，用于报告每个月的警报
def report_alarm_monthly(ym, metric):
    def preprocessor(df):
        df["diff_balance"] = df["diff"]/(df["path_d1"]+df["path_d2"])
        # NOTE: add more metrics here if needed
        return df

    save_dir = reported_alarm_dir/metric/ym
    save_dir.mkdir(parents=True, exist_ok=True)

    datetimes, bulks = load_montly_data(ym, preprocessor)
    indices = np.arange(len(bulks))
    infos = []
    #对相邻的数据块进行窗口检测，并收集检测信息
    for i, j in list(zip(indices[:-1], indices[1:])):
        info = dict(d0=datetimes[i], d1=datetimes[j])
        _info, df = window(bulks[i], bulks[j], metric=metric)
        info.update(**_info)
        #如果没有检测到警报，设置保存路径为None；否则，保存警报数据到CSV文件
        if df is None:
            info.update(save_path=None)
        else: 
            save_path = save_dir/f"{datetimes[i]}_{datetimes[j]}.alarms.csv"
            df.to_csv(save_path, index=False)
            info.update(save_path=str(save_path))

        infos.append(info)

    json.dump(infos, open(save_dir/f"info_{ym}.json", "w"), indent=2)

# 并行处理2023年1月的每天数据
Parallel(backend="multiprocessing", n_jobs=7, verbose=10)(
    delayed(report_alarm_monthly)(f"202301", "diff_balance")
    for m in range(1, 32)  # 假设1月份有31天
)
'''