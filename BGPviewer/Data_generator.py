import time


# second time transfer to '%Y-%m-%d %H:%M:%S'.
def s2t(seconds: int) -> str:
    utcTime = time.gmtime(seconds)
    strTime = time.strftime("%Y-%m-%d %H:%M:%S", utcTime)
    return strTime


# str time transfer to second time.
def t2s(str_time: str) -> int:
    time_format = '%Y-%m-%d %H:%M:%S'
    time_int = int(time.mktime(time.strptime(str_time, time_format)))
    return time_int


# generate the update traffic within the given windows with labels.
def data_generator_wlabel(files, Period, start_time: str, end_time: str, anomaly_start_time: str,
                          anomaly_end_time: str):
    updates_list = []
    interval = Period * 60  # 每个窗口大小，单位为秒
    left_time = t2s(start_time)  # 转换开始时间
    right_time = left_time + interval  # 计算右边界
    end_time = t2s(end_time)  # 转换结束时间

    anomaly_start_time = t2s(anomaly_start_time)  # 转换异常开始时间
    anomaly_end_time = t2s(anomaly_end_time)  # 转换异常结束时间

    count = 0
    for file in files:
        try:
            with open(file) as f:
                for l in f:
                    if l.strip() != '':
                        line = l.strip().split('|')
                        time_ = line[1]
                        prefix_ = line[5]

                        # Convert time to integer
                        if '.' in time_:
                            time_ = int(float(time_))
                        else:
                            time_ = int(time_)

                        # Debug output to check values
                        print(f"Processing line: {line}")
                        print(
                            f"Converted time: {time_}, Left time: {left_time}, Right time: {right_time}, Anomaly start: {anomaly_start_time}, Anomaly end: {anomaly_end_time}")

                        # Process only IPv4 addresses
                        if '.' in prefix_:
                            if time_ <= right_time:
                                updates_list.append(line)
                            elif time_ > right_time:
                                # When time exceeds right_time, yield the collected updates
                                if count % 100 == 0:
                                    print(
                                        f'No.{count}: the starting time {s2t(left_time).split(" ")[1]} and ending time {s2t(right_time).split(" ")[1]}')

                                # Check if the time window intersects with the anomaly period
                                if (right_time < anomaly_start_time or left_time > anomaly_end_time):
                                    yield (updates_list, '0')  # normal label
                                else:
                                    yield (updates_list, '1')  # anomaly label

                                # Update the time window for the next iteration
                                left_time = right_time
                                right_time = left_time + interval
                                updates_list = [line]  # Reset updates_list with the current line
                                count += 1
                            if time_ > end_time:
                                break
        except FileNotFoundError:
            print(f"File {file} not found.")  # Debug output
            return


# pri rib表
# 测试数据
files = [
    '/home/whm/Code/BGP-Security/code/BGP-Baseline/BGPviewer/data/rib/priming_data/txt/route-views.amsix/test.txt']  # 使用包含BGP数据的文本文件路径
Period = 1  # 将窗口大小设置为1分钟
start_time = '2021-10-04 00:00:00'  # 设置开始时间
end_time = '2021-10-05 00:00:00'  # 设置结束时间
anomaly_start_time = '2021-10-04 15:07:00'  # 设置异常开始时间
anomaly_end_time = '2021-10-04 21:49:00'  # 设置异常结束时间

# 使用有标签的生成器进行测试
generator = data_generator_wlabel(files, Period, start_time, end_time, anomaly_start_time, anomaly_end_time)

for updates, label in generator:
    print(f"Updates: {updates}")
    print(f"Label: {label}")
