#!/usr/bin/env python3
#-*- coding: utf-8 -*-

import json
import numpy as np
import pandas as pd
from pathlib import Path

metric = "diff_balance" # NOTE

#reported_alarm_dir = Path(f"/opt/detection_result/reported_alarms/{metric}")
repo_dir = Path(__file__).resolve().parent.parent
reported_alarm_dir = repo_dir/"routing_monitor"/"detection_result"/"wide"/"reported_alarms"/metric/f"202301"
#info = json.load(open(reported_alarm_dir/"info.json", "r"))
info = json.load(open(reported_alarm_dir/f"info_202301.json", "r"))

print(f"load from: {reported_alarm_dir}")

n_alarms = pd.Series([i["n_alarms"] for i in info])
print(f"#alarms in each one-hour window:")
print(n_alarms.describe())

for i,j in zip(*np.unique(n_alarms, return_counts=True)):
    print(f"#alarm={i}: {j}")
#找出警报数量最多的前10个时间窗口，并打印总警报数量
windows_top10 = [info[idx]["d1"] for idx in np.argsort(n_alarms)[::-1][:10]]
print(f"windows with most alarms: {windows_top10}")
print(f"total alarms: {np.sum(n_alarms)}")
#将警报信息转换为字典，以时间窗口为键
info = {i["d1"]: i for i in info}
#函数inspect，用于打印指定时间窗口的警报信息
def inspect(ymdh):
    i = info[ymdh]
    print(json.dumps(i, indent=2))

    if i["save_path"] is None:
        print("no alarms reported in this window")
        return

    df = pd.read_csv(i["save_path"])
    #打印每个警报组的详细信息，并等待用户输入以查看下一个警报组
    for group_id, group in tuple(df.groupby("group_id")):
        print(f"alarm_id: {group_id}")
        for prefix_key, ev in tuple(group.groupby(["prefix1", "prefix2"])):
            print(f"* {' -> '.join(prefix_key)}")
            for _, row in ev.iterrows():
                print(f"  path1: {row['path1']}")
                print(f"  path2: {row['path2']}")
                print(f"  diff={row[metric]}")
                print(f"  culprit={row['culprit']}")
                print()
        input("..Enter to next")

while True:
    ymdh = input("Enter ymdh: ")
    try: inspect(ymdh)
    except KeyError as e:
        print(e)
        continue
