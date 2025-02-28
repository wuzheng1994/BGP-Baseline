#!/usr/bin/env python3
#-*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta
import pickle
from joblib import Parallel, delayed

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from data.routeviews.fetch_updates import load_updates_to_df, get_all_collectors, get_archive_list, download_data
from monitor import Monitor

SCRIPT_DIR = Path(__file__).resolve().parent

def detect(data, route_change_dir, snapshot_dir):
    mon = Monitor()

    for fpath in data:
        _, date, time = fpath.name.split(".")

        df = load_updates_to_df(fpath)
        df = df.sort_values(by="timestamp")
        #使用 consume 方法将 DataFrame 数据传递给 Monitor 实例 mon，并启用检测模式
        mon.consume(df, detect=True)
        #将记录的路由变化转换为 DataFrame
        route_change_df = pd.DataFrame.from_records(mon.route_changes)
        #清空 Monitor 实例的 route_changes 列表
        mon.route_changes = []

        route_change_df.to_csv(route_change_dir/f"{date}.{time}.csv", index=False)

        if time == "2345":
            pickle.dump(mon, open(snapshot_dir/f"{date}.end-of-the-day", "wb"))

def detect_monthly_for(collector, months, num_workers):
    result_dir = SCRIPT_DIR/"detection_result"/collector
    route_change_dir = result_dir/"route_change"
    snapshot_dir = result_dir/"snapshot"

    route_change_dir.mkdir(exist_ok=True, parents=True)
    snapshot_dir.mkdir(exist_ok=True, parents=True)

    collectors2url = get_all_collectors()

    def get_time_range(month):
        d1 = datetime(year=2023, month=month, day=1)
        d2 = (datetime(year=2023, month=month, day=28) + timedelta(days=4)
                ).replace(day=1) - timedelta(minutes=15)
        return d1, d2
    #获取每个月的数据文件列表
    monthly_data = [list(map(lambda url: download_data(url, collector),
                    get_archive_list(collector, collectors2url, d1, d2)))
                    for d1,d2 in map(get_time_range, months)]
    #使用并行计算处理每个月的数据
    Parallel(backend="multiprocessing", n_jobs=num_workers, verbose=10)(
            delayed(detect)(data, route_change_dir, snapshot_dir)
            for data in monthly_data)

detect_monthly_for("wide", range(1,2), 7)
