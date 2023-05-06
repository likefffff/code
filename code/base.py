#%%
from random import random
import pandas as pd

class Model():
    def __init__(self) -> None:
        self.worker_num = 20
        self.taxi_trip_file = pd.read_csv("./april2016_data.csv")
        self.taxi_trip_file.dropna(axis=0, how='any', inplace=True)

    # 计算序号
    def cal_num(self, a):
        day = int(a[3:5])
        base = (day - 1) * 96
        if(int(a[0:2]) == 5):
            base = base + 2880
        if(a[20:22] == "PM"):
            base = base + 48
        return int(int(a[11:13]) * 4 + int(a[14:16])/15 + base)

    def generate_data(self, day_count):
        some_year_trip = self.taxi_trip_file
        pre_str = "04/"
        final_data = pd.DataFrame(columns=self.taxi_trip_file.columns)
        for i in range(day_count):
            date = ""
            if(i < 9):
                date = pre_str + "0" + str(i + 1)
            else:
                date = pre_str + str(i + 1)
        
            some_day_trip = some_year_trip[some_year_trip["Trip Start Timestamp"].str.slice(start = 0, stop = 5) == date]

            some_day_trip_am = some_day_trip[some_day_trip["Trip Start Timestamp"].str.slice(start = 20, stop = 22) == "AM"]
            # 1.抽样
            # some_day_trip_am = some_day_trip_am.sample(frac=750/some_day_trip_am.shape[0], random_state=None).sort_values(by = "Trip Start Timestamp")
            some_day_trip_am = some_day_trip_am.sample(frac=0.05, random_state=None)
            # 2.修正时间，12点改为00点
            some_day_trip_am['Trip Start Timestamp'] = some_day_trip_am.xs('Trip Start Timestamp', axis=1).str.replace(r' 12:', ' 00:', regex=True)
            some_day_trip_am['Trip End Timestamp'] = some_day_trip_am.xs('Trip End Timestamp', axis=1).str.replace(r' 12:', ' 00:', regex=True)
            some_day_trip_am = some_day_trip_am.sort_values(by = "Trip Start Timestamp")
            # 3.计算开始时间和结束时间的序列号
            some_day_trip_am['start_sn'] = some_day_trip_am.apply(lambda x: self.cal_num(x['Trip Start Timestamp']),axis = 1)
            some_day_trip_am['end_sn'] = some_day_trip_am.apply(lambda x: self.cal_num(x['Trip End Timestamp']),axis = 1)

            some_day_trip_pm = some_day_trip[some_day_trip["Trip Start Timestamp"].str.slice(start = 20, stop = 22) == "PM"]
            # 1.抽样
            # some_day_trip_pm = some_day_trip_pm.sample(frac=750/some_day_trip_pm.shape[0], random_state=None).sort_values(by = "Trip Start Timestamp")
            some_day_trip_pm = some_day_trip_pm.sample(frac=0.05, random_state=None)
            # 2.修正时间，12点改为00点
            some_day_trip_pm['Trip Start Timestamp'] = some_day_trip_pm.xs('Trip Start Timestamp', axis=1).str.replace(r' 12:', ' 00:', regex=True)
            some_day_trip_pm['Trip End Timestamp'] = some_day_trip_pm.xs('Trip End Timestamp', axis=1).str.replace(r' 12:', ' 00:', regex=True)
            some_day_trip_pm = some_day_trip_pm.sort_values(by = "Trip Start Timestamp")
            # 3.计算开始时间和结束时间的序列号
            some_day_trip_pm['start_sn'] = some_day_trip_pm.apply(lambda x: self.cal_num(x['Trip Start Timestamp']),axis = 1)
            some_day_trip_pm['end_sn'] = some_day_trip_pm.apply(lambda x: self.cal_num(x['Trip End Timestamp']),axis = 1)

            # 将上午和下午的数据结合
            final_data = pd.concat([final_data, some_day_trip_am, some_day_trip_pm], ignore_index= True)
            if((i+1)%10 == 0):
                print(date + " : " + str((some_day_trip_am.shape[0] + some_day_trip_pm.shape[0])) + "   |   " +  "final data : " + str(final_data.shape[0]))

        final_data["start_sn"] = final_data["start_sn"].astype(int)
        final_data["end_sn"] = final_data["end_sn"].astype(int)

        return final_data

# %%
