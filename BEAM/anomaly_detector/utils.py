import pandas as pd
import numpy as np
import json
from pathlib import Path
from functools import lru_cache
from scipy.special import softmax
from itertools import chain
from ipaddress import IPv4Network
import pickle
import time

def read_csv_empty(*args, **kwargs):
    try: return pd.read_csv(*args, **kwargs)
    except pd.errors.EmptyDataError: return pd.DataFrame()
#近似找到数据的拐点
def approx_knee_point(x):
    x, y = np.unique(x, return_counts=True)
    _x = (x-x.min())/(x.max()-x.min())
    _y = y.cumsum()/y.sum()
    idx = np.argmax(np.abs(_y-_x))
    return x[idx], _y[idx]
#加载嵌入距离模型
def load_emb_distance(train_dir, return_emb=False):
    train_dir = Path(train_dir)

    node_emb_path = train_dir / "node.emb"
    link_emb_path = train_dir / "link.emb"
    rela_emb_path = train_dir / "rela.emb"

    node_emb = pickle.load(open(node_emb_path, "rb"))
    link_emb = pickle.load(open(link_emb_path, "rb"))
    rela_emb = pickle.load(open(rela_emb_path, "rb"))
    rela = rela_emb["p2c"]
    link = link_emb["p2c"]
    link = softmax(link)

    @lru_cache(maxsize=100000)
    #计算两个节点嵌入之间的距离
    def _emb_distance(a, b): # could be cluster-like, e.g. '{123,456}'
        a = a.strip("{}").split(",")[0]
        b = b.strip("{}").split(",")[0]
        if a == b: return 0.
        if a not in node_emb or b not in node_emb:
            return np.inf
        xi = node_emb[a]
        xj = node_emb[b]
        return np.sum((xj-xi)**2*link) + np.abs(np.sum((xj-xi)*rela))
   #封装嵌入距离计算
    def emb_distance(a, b):
        return _emb_distance(str(a), str(b))
    #计算动态时间弯曲（DTW）距离
    @lru_cache(maxsize=100000)
    def _dtw_distance(s, t):
        s = [v for i,v in enumerate(s) if i == 0 or v != s[i-1]]
        t = [v for i,v in enumerate(t) if i == 0 or v != t[i-1]]
        ls, lt = len(s), len(t)
        DTW = np.full((ls+1, lt+1), np.inf)
        DTW[0,0] = 0.
        for i in range(ls):
            for j in range(lt):
                cost = emb_distance(s[i], t[j])
                DTW[i+1, j+1] = cost + min(DTW[i  , j+1],
                                           DTW[i+1, j  ],
                                           DTW[i  , j  ])
        return DTW[ls, lt]
    #封装DTW距离计算
    def dtw_distance(s, t):
        return _dtw_distance(tuple(s), tuple(t))
    #计算路径嵌入长度
    @lru_cache(maxsize=100000)
    def _path_emb_length(s):
        d = np.array([emb_distance(a,b) for a,b in zip(s[:-1], s[1:])])
        d = d[(d > 0) & (d < np.inf)]
        return np.nan if d.size == 0 else d.sum()
    #封装路径嵌入长度计算
    def path_emb_length(s):
        return _path_emb_length(tuple(s))

    if return_emb:
        return emb_distance, dtw_distance, path_emb_length, node_emb, link, rela

    return emb_distance, dtw_distance, path_emb_length

#用于定位两个路径集合差异的根因。它通过比较两个路径集合的差异来确定变化的自治系统（ASN）
def root_cause_localize_2set(df, th=0.95):
    set1_asn_cnt, set2_asn_cnt = {}, {}
    for i,j in df[["path1", "path2"]].values:
        set_i = set(i.split(" "))
        set_j = set(j.split(" "))
        set_ij = set_i - set_j
        set_ji = set_j - set_i
        for asn in set_ij:
            if asn not in set1_asn_cnt: set1_asn_cnt[asn] = 1
            else: set1_asn_cnt[asn] += 1
        for asn in set_ji:
            if asn not in set2_asn_cnt: set2_asn_cnt[asn] = 1
            else: set2_asn_cnt[asn] += 1
    #对第一个路径集合中变化的ASN进行排序
    set1, cnt1 = list(set1_asn_cnt.keys()), list(set1_asn_cnt.values())
    idx1 = np.argsort(cnt1)[::-1]
    set1 = np.array(set1)[idx1]
    cnt1 = np.array(cnt1)[idx1]
    #对第2个路径集合中变化的ASN进行排序
    set2, cnt2 = list(set2_asn_cnt.keys()), list(set2_asn_cnt.values())
    idx2 = np.argsort(cnt2)[::-1]
    set2 = np.array(set2)[idx2]
    cnt2 = np.array(cnt2)[idx2]
   #根据阈值th确定显著的变化ASN，并将它们添加到结果列表中
    rc_1, rc_2 = [], []
    for a,b in zip(set1, cnt1):
        if b/df.shape[0] > th: rc_1.append(a)
    for a,b in zip(set2, cnt2):
        if b/df.shape[0] > th: rc_2.append(a)
    #返回两个排序后的根因ASN列表
    return sorted(rc_1), sorted(rc_2)
#定位单个路径集合差异的根因
def root_cause_localize_1set(df, th=0.95):
    set_asn_cnt = {}
    for i,j in df[["path1", "path2"]].values:
        set_i = set(i.split(" "))
        set_j = set(j.split(" "))
        set_xor = set_i^set_j
        for asn in set_xor:
            if asn not in set_asn_cnt: set_asn_cnt[asn] = 1
            else: set_asn_cnt[asn] += 1

    set_asn, cnt = list(set_asn_cnt.keys()), list(set_asn_cnt.values())
    idx = np.argsort(cnt)[::-1]
    set_asn = np.array(set_asn)[idx]
    cnt = np.array(cnt)[idx]
    #根据阈值th确定显著的变化ASN
    rc = []
    for a,b in zip(set_asn, cnt):
        if b/df.shape[0] > th: rc.append(a)

    return sorted(rc)

#将具有相同根因的事件关联起来
def link_root_cause(culprit_to_df):
    rcs = list(culprit_to_df.keys())
    dfs = list(culprit_to_df.values())

    def rc_to_set(rc):
        culprit_type, culprit_tuple = rc
        assert culprit_type in ["Prefix", "AS"]
        if culprit_type == "AS":
            culprit_set = set(chain(*culprit_tuple))
        else: # must be "Prefix"
            culprit_set = {IPv4Network(p) for p in culprit_tuple}
        return culprit_type, culprit_set
    
    #检查两个根因是否相关
    def rc_set_related(rc1, rc2):
        t1, set1 = rc1
        t2, set2 = rc2
        if t1 != t2:
            return False
        if t1 == "AS":
            return set1&set2
        else: # t1 and t2 must be "Prefix"
            for i in set1:
                for j in set2:
                    if i.overlaps(j): # check if they overlap
                        return True
                    if i.prefixlen == j.prefixlen: # check if they're two consecutive prefixes
                        return abs((int(i[0])>>(32-i.prefixlen))
                                -(int(j[0])>>(32-j.prefixlen))) <= 1
            return False
    #将相关的根因分配到相同的组，并为每个组分配一个唯一的组ID
    pool = list(map(rc_to_set, rcs))
    group_id = [-1]*len(culprit_to_df)
    id_group = dict()
    next_id = 0
    for i in range(len(culprit_to_df)):
        if group_id[i] == -1: 
            group_id[i] = next_id
            next_id += 1
            id_group[group_id[i]] = [i]
        for j in range(i+1, len(culprit_to_df)):
            if group_id[j] == group_id[i]: continue
            if rc_set_related(pool[i], pool[j]):
                if group_id[j] == -1:
                    group_id[j] = group_id[i]
                    id_group[group_id[i]].append(j)
                else:
                    to_be_merged = id_group.pop(group_id[j])
                    id_group[group_id[i]] += to_be_merged
                    for k in to_be_merged: group_id[k] = group_id[i]
    #为每个事件分配组ID
    group_id_set = set(group_id)
    group_id_remapping = dict(zip(group_id_set, range(len(group_id_set))))
    for idx, df in enumerate(dfs):
        df["group_id"] = group_id_remapping[group_id[idx]]
    return id_group, pd.concat(dfs, ignore_index=True)
#聚合具有相同根因的事件
def event_aggregate(events):
    culprit2eventkey = {}
    eventkey2culprit = {}

    for k,v in events.items():
        print("start")
        start = time.time()
        rc_1, rc_2 = root_cause_localize_2set(v)
        rc_3 = root_cause_localize_1set(v)

        if rc_1 or rc_2:
            culprit = "AS", (tuple(rc_1), tuple(rc_2))
        elif rc_3:
            culprit = "AS", (tuple(rc_3),)
        else:
            culprit = "Prefix", k
        end = time.time()
        print('In time', end - start)
        culprit2eventkey.setdefault(culprit, set()).add(k)
        eventkey2culprit[k] = culprit
#根据事件的根因，将事件分配到不同的组
    culprit_to_df = {k: pd.concat([events[i] for i in v])
                                        for k, v in culprit2eventkey.items()}
    for k, v in culprit_to_df.items():
        _, culprit_tuple = k
        print('culprit_tuple', culprit_tuple)
        v["culprit"] = json.dumps(culprit_tuple)
    rc_groups, df = link_root_cause(culprit_to_df)

    return rc_groups, df
