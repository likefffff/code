#%%
from signal import valid_signals
from base import Model
from worker import Worker
from trip import Trip
from agent import Agent
import numpy as np
import pandas as pd


#%%
curr_data = pd.read_csv("data.csv")
#%%
model = Model()
agent = Agent()

#%%

'''
    最短距离优先NNP
'''

worker_List = [Worker() for n in range(model.worker_num)]
unassigned = []

total_reward = 0
total_trip = 0
total_assigned = 0
total_completed = 0
total_wait_time = 0
total_distance = 0
total_value = 0

f = open("output_NNP_worker20.txt",'w')

for n in range(2880):
    processing_wait_time = 0
    
    # 上一个时间片用完后，更新worker状态
    for worker in worker_List:
        new_reward, new_wait_time, new_value = worker.Update(n-1)
        if(new_reward > 0):
            total_reward = total_reward + new_reward
            total_completed = total_completed + 1
            total_value += new_value
        else:
            processing_wait_time = processing_wait_time + new_wait_time

    newTrip = curr_data[curr_data["start_sn"] == n]
    for row in zip(newTrip["Trip ID"], newTrip["start_sn"], 
            newTrip["end_sn"], newTrip["Pickup Centroid Latitude"], 
            newTrip["Pickup Centroid Longitude"], newTrip["Dropoff Centroid Latitude"], 
            newTrip["Dropoff Centroid Longitude"], newTrip["Trip Total"]):
        trip = Trip(row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7])
        unassigned.append(trip)
        total_trip = total_trip + 1

    new_assigned, new_distance, new_assigned_time = agent.NNP(unassigned, worker_List, n)
    total_assigned += new_assigned
    total_distance += new_distance
    total_wait_time += new_assigned_time

    divi = 1
    if(total_completed > 0):
        divi = total_completed

    # 已分配，已完成，总收益，总等待时间，应答率
    s = "SN: {}:  assigned: {}  |  completed: {}  |   value: {}   |  reward: {}  |   distance: {}   |   wait_time: {}   |  RR: {}".format(n, str(total_assigned) + '/' + str(total_trip), str(total_completed) + '/' + str(total_trip), total_value/divi, total_reward/divi, total_distance/divi, total_wait_time/15/divi, total_assigned/total_trip)
    print(s, file = f)

f.close()


#%%

'''
    贪心算法GR
'''

worker_List = [Worker() for n in range(model.worker_num)]
unassigned = []

total_reward = 0
total_trip = 0
total_assigned = 0
total_completed = 0
total_wait_time = 0
total_distance = 0
total_value = 0

f = open("output_GR_worker20.txt",'w')

for n in range(2880):
    processing_wait_time = 0

    # 上一个时间片用完后，更新worker状态
    for worker in worker_List:
        new_reward, new_wait_time, new_value = worker.Update(n-1)
        if(new_reward > 0):
            total_reward = total_reward + new_reward
            total_completed = total_completed + 1
            total_value += new_value
        else:
            processing_wait_time = processing_wait_time + new_wait_time

    newTrip = curr_data[curr_data["start_sn"] == n]
    for row in zip(newTrip["Trip ID"], newTrip["start_sn"], 
            newTrip["end_sn"], newTrip["Pickup Centroid Latitude"], 
            newTrip["Pickup Centroid Longitude"], newTrip["Dropoff Centroid Latitude"], 
            newTrip["Dropoff Centroid Longitude"], newTrip["Trip Total"]):
        trip = Trip(row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7])
        unassigned.append(trip)
        total_trip = total_trip + 1

    new_assigned, new_distance, new_assigned_time = agent.GR(unassigned, worker_List, n)
    total_assigned += new_assigned
    total_distance += new_distance
    total_wait_time += new_assigned_time

    divi = 1
    if(total_completed > 0):
        divi = total_completed

    # 已分配，已完成，总收益，总等待时间，应答率
    s = "SN: {}:  assigned: {}  |  completed: {}  |   value: {}   |  reward: {}  |   distance: {}   |   wait_time: {}   |  RR: {}".format(n, str(total_assigned) + '/' + str(total_trip), str(total_completed) + '/' + str(total_trip), total_value/divi, total_reward/divi, total_distance/divi, total_wait_time/15/divi, total_assigned/total_trip)
    print(s, file = f)

f.close()


#%%
'''
    规划算法
'''

f = open("./train_DP.txt", 'w')

# train阶段
def train(curr_data, alpha, step, epoch_count):
    reward_list = []
    alpha_list = []
    distance_list = []
    value_list = []

    # while(last_reward <= total_reward):
    for _ in range(epoch_count):
        total_reward = 0
        unassigned = []
        worker_List = [Worker() for _ in range(model.worker_num)]
        total_trip = 0
        total_assigned = 0
        total_completed = 0
        total_wait_time = 0
        total_distance = 0
        total_value = 0

        for n in range(480):
            processing_wait_time = 0

            # 上一个时间片用完后，更新worker状态
            for worker in worker_List:
                new_reward, new_wait_time, new_value = worker.Update(n-1)
                if(new_reward > 0):
                    total_reward = total_reward + new_reward
                    total_completed = total_completed + 1
                    total_wait_time = total_wait_time + new_wait_time
                    total_value += new_value
                else:
                    processing_wait_time = processing_wait_time + new_wait_time

            newTrip = curr_data[curr_data["start_sn"] == n]
            for row in zip(newTrip["Trip ID"], newTrip["start_sn"], 
                    newTrip["end_sn"], newTrip["Pickup Centroid Latitude"], 
                    newTrip["Pickup Centroid Longitude"], newTrip["Dropoff Centroid Latitude"], 
                    newTrip["Dropoff Centroid Longitude"], newTrip["Trip Total"],
                    newTrip["Trip Seconds"]):
                trip = Trip(row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8])
                unassigned.append(trip)
                total_trip = total_trip + 1

            new_assigned, new_distance = agent.DP(unassigned, worker_List, n, alpha, 0.5, 0.3, 0.2)
            total_assigned += new_assigned
            total_distance += new_distance
        
        divi = 1
        if(total_completed > 0):
            divi = total_completed

        # 已分配，已完成，总收益，总等待时间，应答率
        s = "alpha: {}:  assigned: {}  |  completed: {}  |   value:{}   |  reward: {}  |   distance: {}   |   wait_time: {}   |  RR: {}".format(alpha, str(total_assigned) + '/' + str(total_trip), str(total_completed) + '/' + str(total_trip), total_value, total_reward, total_distance, total_wait_time/divi, total_assigned/total_trip)

        print(s, file = f)

        # if(total_distance == 0):
        #     total_distance = 1
        #     if(total_reward > 0):
        #         print("error")
        reward_list.append(total_reward)
        alpha_list.append(alpha)
        distance_list.append(total_distance)
        value_list.append(total_value)

        alpha += step
    return np.array(reward_list), np.array(alpha_list), np.array(distance_list), np.array(value_list)
    # f.close()

best_alpha_list = []
for n in range(5):
    # curr_data = model.generate_data(5)
    curr_data = pd.read_csv("data.csv")
    alpha = 1
    step = 1
    epoch_count = 28
    reward_list, alpha_list, distance_list, value_list = train(curr_data, alpha, step, epoch_count)
    max_index = int(np.mean(np.where(value_list == np.max(value_list)), axis=1)[0] + 0.5)
    best_alpha_list.append(alpha_list[max_index])

    print("alpha: {}   |   value:{}   |   reward: {}   |   distance: {}".format(alpha_list[max_index], value_list[max_index], reward_list[max_index], distance_list[max_index]), file = f)
    print("---------------------------------------------------------------------------", file = f)
    print("epoch {} done".format(n))

best_alpha = np.mean(np.array(best_alpha_list))
print("best alpha: {}".format(best_alpha), file = f)
f.close()

#%%
best_alpha

#%%
# test阶段
f = open("output_DP_worker20_train10.txt", 'w')

def test(curr_data, alpha):
    total_reward = 0
    unassigned = []
    worker_List = [Worker() for _ in range(model.worker_num)]
    total_trip = 0
    total_assigned = 0
    total_completed = 0
    total_wait_time = 0
    total_distance = 0
    total_value = 0
    unassigned_wait_time = 0

    for n in range(2880):
        processing_wait_time = 0

        # 上一个时间片用完后，更新worker状态
        for worker in worker_List:
            new_reward, new_wait_time, new_value = worker.Update(n-1)
            if(new_reward > 0):
                total_reward = total_reward + new_reward
                total_completed = total_completed + 1
                total_value += new_value
            else:
                processing_wait_time = processing_wait_time + new_wait_time

        newTrip = curr_data[curr_data["start_sn"] == n]
        for row in zip(newTrip["Trip ID"], newTrip["start_sn"], 
                newTrip["end_sn"], newTrip["Pickup Centroid Latitude"], 
                newTrip["Pickup Centroid Longitude"], newTrip["Dropoff Centroid Latitude"], 
                newTrip["Dropoff Centroid Longitude"], newTrip["Trip Total"],
                newTrip["Trip Seconds"]):
            trip = Trip(row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8])
            unassigned.append(trip)
            total_trip = total_trip + 1

        new_assigned, new_distance, new_assigned_time = agent.DP(unassigned, worker_List, n, alpha, 0.5, 0.3, 0.2)
        total_assigned += new_assigned
        total_distance += new_distance
        total_wait_time += new_assigned_time

        for task in unassigned:
            unassigned_wait_time += task.wait_time

        divi = 1
        if(total_completed > 0):
            divi = total_completed
        
        # 已分配，已完成，总收益，总等待时间，应答率
        s = "SN: {}:  assigned: {}  |  completed: {}  |   value: {}   |  reward: {}  |   distance: {}   |   wait_time: {}   |  RR: {}".format(n, str(total_assigned) + '/' + str(total_trip), str(total_completed) + '/' + str(total_trip), total_value/divi, total_reward/divi, total_distance/divi, (total_wait_time/15 )/divi, total_assigned/total_trip)
        print(s, file = f)

curr_data = pd.read_csv("data.csv")
test(curr_data, 26.4)
# trian10 26.4
# train5 27.8
f.close()


# %%
best_alpha
# %%
model.worker_num
# %%
