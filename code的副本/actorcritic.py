#%%
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from worker import Worker
from trip import Trip
import random

#%%
# worker列表
worker_num = 10
# 动作空间
actions = list(range(0, worker_num + 1))

# Actor使用策略梯度更新(接收状态，输出策略)，Critic使用价值函数更新(接收状态，输出价值)
actor_model = None
critic_model = None


#%%
taxi_trip_file = pd.read_csv("./april2016_data.csv")
taxi_trip_file.dropna(axis=0, how='any', inplace=True)

#%%
# taxi_trip_april = pd.DataFrame(columns=taxi_trip_file.columns)
# some_year_trip = taxi_trip_file[taxi_trip_file["Trip Start Timestamp"].str.slice(start = 6, stop = 10) == "2016"]
# pre_str = "04/"
# final_data = pd.DataFrame(columns=taxi_trip_file.columns)
# for i in range(30):
#     date = ""
#     if(i < 9):
#         date = pre_str + "0" + str(i + 1)
#     else:
#         date = pre_str + str(i + 1)
    
#     some_day_trip = some_year_trip[some_year_trip["Trip Start Timestamp"].str.slice(start = 0, stop = 5) == date]
#     taxi_trip_april = pd.concat([taxi_trip_april, some_day_trip], ignore_index= True)

# taxi_trip_april.to_csv("./april2016_data.csv")

#%%
def init_model():
    am = torch.nn.Sequential(torch.nn.Linear(3 * worker_num + 1, 128),
                                 torch.nn.ReLU(),
                                 torch.nn.Linear(128, worker_num),
                                 torch.nn.Softmax(dim=1))   
    cm = torch.nn.Sequential(torch.nn.Linear(3 * worker_num + 1, 128),
                                  torch.nn.ReLU(),
                                  torch.nn.Linear(128, 1))
    return am, cm

def cal_num(a):
    day = int(a[3:5])
    base = (day - 1) * 96
    if(int(a[0:2]) == 5):
        base = base + 2880
    if(a[20:22] == "PM"):
        base = base + 48
    return int(int(a[11:13]) * 4 + int(a[14:16])/15 + base)

def generate_data():
    some_year_trip = taxi_trip_file[taxi_trip_file["Trip Start Timestamp"].str.slice(start = 6, stop = 10) == "2016"]
    pre_str = "04/"
    final_data = pd.DataFrame(columns=taxi_trip_file.columns)
    for i in range(15):
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
        some_day_trip_am['start_sn'] = some_day_trip_am.apply(lambda x: cal_num(x['Trip Start Timestamp']),axis = 1)
        some_day_trip_am['end_sn'] = some_day_trip_am.apply(lambda x: cal_num(x['Trip End Timestamp']),axis = 1)

        some_day_trip_pm = some_day_trip[some_day_trip["Trip Start Timestamp"].str.slice(start = 20, stop = 22) == "PM"]
        # 1.抽样
        # some_day_trip_pm = some_day_trip_pm.sample(frac=750/some_day_trip_pm.shape[0], random_state=None).sort_values(by = "Trip Start Timestamp")
        some_day_trip_pm = some_day_trip_pm.sample(frac=0.05, random_state=None)
        # 2.修正时间，12点改为00点
        some_day_trip_pm['Trip Start Timestamp'] = some_day_trip_pm.xs('Trip Start Timestamp', axis=1).str.replace(r' 12:', ' 00:', regex=True)
        some_day_trip_pm['Trip End Timestamp'] = some_day_trip_pm.xs('Trip End Timestamp', axis=1).str.replace(r' 12:', ' 00:', regex=True)
        some_day_trip_pm = some_day_trip_pm.sort_values(by = "Trip Start Timestamp")
        # 3.计算开始时间和结束时间的序列号
        some_day_trip_pm['start_sn'] = some_day_trip_pm.apply(lambda x: cal_num(x['Trip Start Timestamp']),axis = 1)
        some_day_trip_pm['end_sn'] = some_day_trip_pm.apply(lambda x: cal_num(x['Trip End Timestamp']),axis = 1)

        # 将上午和下午的数据结合
        final_data = pd.concat([final_data, some_day_trip_am, some_day_trip_pm], ignore_index= True)
        if((i+1)%10 == 0):
            print(date + " : " + str((some_day_trip_am.shape[0] + some_day_trip_pm.shape[0])) + "   |   " +  "final data : " + str(final_data.shape[0]))

    final_data["start_sn"] = final_data["start_sn"].astype(int)
    final_data["end_sn"] = final_data["end_sn"].astype(int)

    return final_data


def get_action(state):
    state = torch.FloatTensor(state).reshape(1, 3 * worker_num + 1)
    prob = actor_model(state)
    action = random.choices(range(worker_num), weights=prob[0].tolist(), k=1)[0]
    return action

def get_data(curr_data):
    state = []
    states = []
    actions = []
    rewards = []
    next_states = []
    dones = []
    worker_list = [Worker() for n in range(worker_num)]

    total_reward = 0
    total_completed = 0
    completed_wait_time = 0
    total_trip = 0
    total_assigned = 0

    for worker in worker_list:
        state.append([worker.latitude, worker.longtude, int(worker.isFree) * 10])
    state.append([0]) # 添加当前时间片序列%96

    for n in range(960):
        processing_wait_time = 0
        newTrip = curr_data[curr_data["start_sn"] == n]
        state[len(state) - 1] = [n % 96]

        for worker in worker_list:
            new_reward, new_wait_time = worker.Update(n - 1)
            if(new_reward > 0):
                total_reward = total_reward + new_reward
                total_completed = total_completed + 1
                completed_wait_time = completed_wait_time + new_wait_time
            else:
                processing_wait_time = processing_wait_time + new_wait_time
        
        for row in zip(newTrip["Trip ID"], newTrip["start_sn"], 
                newTrip["end_sn"], newTrip["Pickup Centroid Latitude"], 
                newTrip["Pickup Centroid Longitude"], newTrip["Dropoff Centroid Latitude"], 
                newTrip["Dropoff Centroid Longitude"], newTrip["Trip Total"]):
            total_trip = total_trip + 1
            trip = Trip(row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7])
            state_flat = sum(state, [])
            action = get_action(state_flat)
            
            next_state = state
            if(action != worker_num):
                if(worker_list[action].isFree):
                    worker_list[action].setTrip(trip, n)
                    next_state[action] = worker_list[action].nextLocation(n)
                    distance = ((worker_list[action].latitude - trip.start_lat)**2 + (worker_list[action].longtude - trip.start_lon)**2)**0.5
                    rewards.append(trip.cost - distance * 2)
                    total_assigned = total_assigned + 1
                else:
                    rewards.append(0)
            else:
                rewards.append(0)

            actions.append(action)
            states.append(state_flat)
            next_state_flat = sum(next_state, [])
            next_states.append(next_state_flat)
            dones.append(False)
    
    dones[len(dones) - 1] = True
    states = torch.FloatTensor(states).reshape(-1, 3 * worker_num + 1)
    rewards = torch.FloatTensor(rewards).reshape(-1, 1)
    actions = torch.LongTensor(actions).reshape(-1, 1)
    next_states = torch.FloatTensor(next_states).reshape(-1, 3 * worker_num + 1)
    dones = torch.LongTensor(actions).reshape(-1, 1)
    
    return states, rewards, actions, next_states, dones, total_assigned/total_trip

def test():
    rewards_sum = 0
    curr_data = pd.read_csv("data.csv")
    states, rewards, actions, next_states, dones, RR = get_data(curr_data)

    for reward in rewards:
        rewards_sum += reward
    # while not done:
    #     action = get_action(state)
    #     state, reward, done, _ = env.step(action) 
    #     rewards_sum += reward
    return [rewards_sum, RR]

def train(f):
    curr_data = generate_data()
    # curr_data = pd.read_csv("data.csv")
    print("generate data finished")

    optimizer = torch.optim.Adam(actor_model.parameters(), lr=1e-1)
    optimizer_td = torch.optim.Adam(critic_model.parameters(), lr=1e-1)
    
    # 玩N局游戏，每局游戏训练一次
    for epoch in range(200):
        
        states, rewards, actions, next_states, dones, RR = get_data(curr_data)
        # 分batch优化
        current_values = critic_model(states)
        next_state_values = critic_model(next_states) * 0.98
        next_state_values *= (1 - dones)
        next_values = rewards + next_state_values
        # 时序差分误差.单纯使用值，不反向传播梯度. detach:阻断反向梯度传播
        delta = (next_values - current_values).detach()
        
        # actor重新评估动作计算得分
        probs = actor_model(states)
        probs = probs.gather(dim=1, index=actions)
        actor_loss = (-probs.log() * delta).mean()
        # 时序差分loss。均方误差
        critic_loss = torch.nn.MSELoss()(current_values, next_values.detach())
        
        optimizer.zero_grad()
        actor_loss.backward()
        optimizer.step()
        
        optimizer_td.zero_grad()
        critic_loss.backward()
        optimizer_td.step()
        
        if epoch % 10 == 0:
            result = test()
            print("epoch: {}   |   reward: {}   |   RR: {}".format(epoch, result[0].float(), result[1]), file = f)
# %%
actor_model, critic_model = init_model()
print("-----train start-----")
f = open("output_ct.txt",'w')
train(f)
f.close()
# print("-----test start-----")
# print(test())

# %%
