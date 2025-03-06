import time
import os
import numpy as np
import editdistance
from collections import defaultdict
from math import log
import sys
sys.path.append('/home/wuzheng/Feature_extraction/')
from Routes_vp.Routes_vp import Routes
import json
from Data_generator import data_generator_wlabel
import pickle


# second time transfer to '%Y-%m-%d %H:%M:%S'.
def s2t(seconds:int) -> str:
    utcTime = time.gmtime(seconds)
    strTime = time.strftime("%Y-%m-%d %H:%M:%S",utcTime)
    return strTime

# str time transfer to second time.
def t2s(str_time:str) -> int:
    time_format = '%Y-%m-%d %H:%M:%S'
    time_int = int(time.mktime(time.strptime(str_time, time_format)))
    return time_int

def entropy(list1:list):
    set_ = set(list1)
    counts = [list1.count(i) for i in set_]
    total = len(list1)
    result = sum([-(i / total) * log(i / total, 2) for i in counts])
    return result
                        
class Feature:
    
    '''
    Extract the statistical features from updates slices;
    '''

    def __init__(self):
        self.features = {}
    
    def init(self):
        self.features = {
            'index': 0, # the no. of instances.
            'A_num': 0,  # the number of Announcement.
            'W_num': 0,  # the number of Withdraw.
            'num_new_A': 0,  # the number of new annountment.
            'max_A_AS': 0,  # the largest number of annountment by AS.
            'min_A_AS': 0,  # the least number of announcement by AS.
            'ave_A_AS': 0,  # the average number of announcement by AS.
            'max_A_prefix': 0,  # the largest number of annountment by prefix.
            'ave_A_prefix': 0,  # the average number of announcement by prefix.
            'min_A_prefix': 0,  # the min number of announcement by prefix.
            'max_length': 0,  # the max length of as path.
            'min_length': 0,  # the min length of as path.
            'ave_length': 0,  # the average length of as path.
            'num_A_prefix': 0,  # the number of prefix announcing A updates.
            'num_W_prefix':0,  # the number of prefix announcing withdraw.
            'num_longer': 0,  # the number of longer path of announcement.
            'num_shorter': 0,  # the number of shorter path of announcement.
            'num_ori_change': 0,  # ori_as changes.
            'label': None,  # label 0 represents normal event; label 1 represents anomaly events.
            'editDis_entropy': 0,  # the entropy of editDis.
            'length_entropy': 0,  # entropy of as path length.
            'max_editDis': 0,  # the maximum edit distance.
            'min_editDis': 0,  # the min edit distance.
            'num_dup_A': 0,  # the number of duplicate announcements.
            "num_dup_W": 0,  # the number of duplicate withdraw.
            "ave_arrival_interval": 0,  # the average of arrival interval.
            'num_new_A_afterW': 0,  # the number of new announcement after withdraw.
            'num_A_AS': 0
        }
        # print(self.features.keys())

    def extract_features(self, ind, updates_message, routes):
        
        updates, self.features['label'] = updates_message
        self.features['index'] = ind
        extraction_time = 0
        if len(updates) < 3:
            return self.features, extraction_time

        A_num = 0
        W_num = 0
        A_per_as = {}
        A_per_prefix = {}
        W_per_prefix = {}
        path_len = []
        
        edist_list = []
        interval_list = []
        first_flag = True

        starttime = time.time()
        for update in updates:
            if first_flag:
                first_flag = False
                if '.' in update[1]:
                    cur_time_ = int(float(update[1]))
                else:
                    cur_time_ = int(update[1])

            pre_time_ = cur_time_
            if '.' in update[1]:
                cur_time_ = int(float(update[1]))
            else:
                cur_time_ = int(update[1])
            
            op_ = update[2]
            prefix_ = update[5]
            peer_as_ = update[4]
            
            interval = cur_time_ - pre_time_
            interval_list.append(interval)

            # 如果为宣告消息
            if op_ == 'A':
                A_num += 1
                as_path = update[6].split(' ')
                ori_as_ = as_path[-1]
                if (not routes.get(prefix_)) or (routes[prefix_][peer_as_] == None):
                    self.features['num_new_A'] += 1
                elif routes[prefix_][peer_as_] == 'w' + str(ind):
                    self.features['num_new_A_afterW'] += 1
                    # print('the num new A after W:',self.features['num_new_A_afterW'])
                    self.features['num_new_A'] += 1
                else:
                    if routes[prefix_][peer_as_] != None:
                        as_path_prev = routes[prefix_][peer_as_].split(' ')
                        if as_path_prev == as_path:
                            self.features['num_dup_A'] += 1
                        if ori_as_ != as_path_prev[-1]:
                            self.features['num_ori_change'] += 1
                        if len(as_path_prev) < len(as_path):
                            self.features['num_longer'] += 1
                        if len(as_path_prev) > len(as_path):
                            self.features['num_shorter'] += 1
                        edist = editdistance.eval(as_path_prev, as_path)        
                        edist_list.append(edist)
                
                if ori_as_ not in A_per_as:
                    A_per_as[ori_as_] = 0

                A_per_as[ori_as_] += 1

                if prefix_ not in A_per_prefix:
                    A_per_prefix[prefix_] = 0
                A_per_prefix[prefix_] += 1
                path_len.append(len(as_path))
                routes[prefix_][peer_as_] = ' '.join(as_path)

            # 如果为撤销消息,提取特征                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
            elif op_ == 'W':
                W_num += 1
                if prefix_ not in W_per_prefix:
                    W_per_prefix[prefix_] = 0
                W_per_prefix[prefix_] += 1
                if routes[prefix_][peer_as_] == 'w' + str(ind):
                    self.features['num_dup_W'] += 1
                routes[prefix_][peer_as_] = 'w{}'.format(ind)


        A_per_as_values = list(A_per_as.values())
        self.features["ave_arrival_interval"] = np.average(interval_list)
        self.features['length_entropy'] = entropy(path_len)
        
        try:
            self.features['editDis_entropy'] = entropy(edist_list)
            self.features['min_editDis'] = min(edist_list)
            self.features['max_editDis'] = max(edist_list)
        except:
            if len(edist_list) < 2:
                self.features['editDis_entropy'] = 0
                self.features['min_editDis'] = 0
                self.features['max_editDis'] = 0
        if A_per_as_values == []:
            self.features['max_A_AS'] = 0
            self.features['min_A_AS'] = 0
            self.features['ave_A_AS'] = 0
            self.features['num_A_AS'] = 0
        else:
            self.features['max_A_AS'] = max(A_per_as_values)
            self.features['min_A_AS'] = min(A_per_as_values)
            self.features['ave_A_AS'] = np.average(A_per_as_values)
            self.features['num_A_AS'] = len(A_per_as_values)

        A_per_prefix_values = list(A_per_prefix.values())
        if A_per_prefix_values == []:
            
            self.features['max_A_prefix'] = 0
            self.features['min_A_prefix'] = 0
            self.features['ave_A_prefix'] = 0
            self.features['num_A_prefix'] = 0
        else:
            self.features['max_A_prefix'] = max(A_per_prefix_values)
            self.features['min_A_prefix'] = min(A_per_prefix_values)
            self.features['ave_A_prefix'] = np.average(A_per_prefix_values)
            self.features['num_A_prefix'] = len(A_per_as_values)
        if path_len == []:
            self.features['max_length'] = 0
            self.features['min_length'] = 0
            self.features['ave_length'] = 0
        else:
            self.features['max_length'] = max(path_len)
            self.features['min_length'] = min(path_len)
            self.features['ave_length'] = np.average(path_len)

        W_per_prefix_values = list(W_per_prefix.values())
        self.features['num_W_prefix'] = len(W_per_prefix_values)

        self.features['A_num'] = A_num
        self.features['W_num'] = W_num
        extract_time = time.time() - starttime
        print('The extraction time:', extract_time)

        return (self.features, extract_time)


if __name__ == "__main__":

    # # initialized info.   
    # collector = 'route-views.amsix'
    # path = "/data/wuzheng_data/anomaly_event/outage/event1/"
    # data_path = path + 'txt/' + collector
    # event_time = ("2021-10-04 00:00:00", "2021-10-05 00:00:00")
    # anomaly_time = ("2021-10-04 15:07:00", "2021-10-04 21:49:00")
    # save_path = '/home/wuzheng/Feature_extraction/Routes_vp/window_slice/'

    collector = 'route-views3'
    path = "/data/wuzheng_data/anomaly_event/prefix_hijack/event2/"
    data_path = path + 'txt/' + collector
    event_time = ("2020-7-30 00:00:00", "2020-7-31 00:00:00")
    anomaly_time = ("2020-7-30 00:55:00", "2020-7-30 02:35:00")
    save_path = '/home/wuzheng/Feature_extraction/Routes_vp/window_slice/prefix_hijack/'
    
    # collector = 'rrc01'
    # path = "/data/wuzheng_data/anomaly_event/route_leak/event1/"
    # data_path = path + 'txt/' + collector
    # event_time = ("2018-11-12 00:00:00", "2018-11-13 00:00:00")
    # anomaly_time = ("2018-11-12 21:12:00", "2018-11-12 22:32:00")
    # save_path = '/home/wuzheng/Feature_extraction/Routes_vp/window_slice/Rl1/'
    
    # collector = 'route-views.chile'
    # path = "/data/wuzheng_data/anomaly_event/outage/event2/"
    # data_path = path + 'txt/' + collector
    # event_time = ("2022-06-21 00:00:00", "2022-06-22 00:00:00")
    # anomaly_time = ("2022-06-21 06:27:00", "2022-06-21 07:42:00")
    # save_path = '/home/wuzheng/Feature_extraction/window_slice/outage2/'
    
    # collector = 'route-views.chicago'
    # path = "/data/wuzheng_data/anomaly_event/route_leak/event2/"
    # data_path = path + 'txt/' + collector
    # event_time = ("2019-06-24 00:00:00", "2019-06-25 00:00:00")
    # anomaly_time = ("2019-06-24 10:34:25", "2019-06-24 12:38:54")
    # save_path = '/home/wuzheng/Packet/CompareMethod/Multiview/dataset/Rl2/'
    
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    if t2s(event_time[0]) > t2s(event_time[1]):
        print('Error: Starting time is bigger than ending time.')
        raise TypeError

    if t2s(anomaly_time[0]) > t2s(anomaly_time[1]):
        print('Error: Starting time is bigger than ending time.')
        raise TypeError

    Period = [1]
    for p in Period:
        
        print("Period window slice:", p)
        updates_files = sorted([os.path.join(data_path, i) for i in os.listdir(data_path)])

        # construct instances from updates.
        a = data_generator_wlabel(updates_files, Period = p, start_time=event_time[0], end_time=event_time[1], anomaly_start_time=anomaly_time[0], anomaly_end_time=anomaly_time[1])

        # collect the routes info.
        priming_path = path + 'priming_data/txt/'
        r = Routes(priming_path)
        r.collect_routes()
        r1 = r.get_route
        
        # extract the feature combined with routes and updates.
        F1 = Feature()
        ind = 0
        start_time = time.time()
        features = []
        # extraction_time = []
        for u in a:
            F1.init()
            res = F1.extract_features(ind, u, r1)
            features.append(res[0])
            ind += 1
        # elapse_time = time.time() - start_time
        # print('Feature extraction average time consumes: {}.'.format(np.average(extraction_time)))

        fea_name = list(self.features.keys())
        fea_name = fea_name.pop("label") + ['label']
        fea_name = ['ind'] + fea_name.pop('ind')
        res = []
        for fea in features:
            res.append([fea[n] for n in fea_name])
        
        
        w_path = save_path + 'S_outage_1_{:.2f}.pkl'.format(p)
        res = np.array(res)
        with open(w_path, 'wb') as fw:
            pickle.dump(res, fw)
        
        del a
        del r
        del r1
        del F1






