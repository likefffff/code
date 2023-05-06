#%%
from random import random
import pandas as pd
import numpy as np

from worker import Worker
from trip import Trip
from agent import Agent
from base import Model 

#%%
curr_data = pd.read_csv("data.csv")

#%%
agent = Agent()
model = Model()

#%%

# 训练theta1，theta2
f = open("train_ex2.txt", 'w')

def train(curr_data, step, epoch_count, theta1, theta2, theta3):
    reward_list = []
    theta1_list = []
    theta2_list = []
    distance_list = []
    value_list = []

    for _ in range(epoch_count):
        while(theta2 <= 1 - theta1):
            total_reward = 0
            unassigned = []
            worker_List = [Worker() for _ in range(model.worker_num)]
            total_trip = 0
            total_assigned = 0
            total_completed = 0
            completed_wait_time = 0
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
                        completed_wait_time = completed_wait_time + new_wait_time
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

                new_assigned, new_distance = agent.DP(unassigned, worker_List, n, 26.4, theta1, theta2, theta3)
                total_assigned += new_assigned
                total_distance += new_distance
            # 已分配，已完成，总收益，总等待时间，应答率
            s = "theta1: {}, theta2: {}, theta3: {} :  assigned: {}  |  completed: {}  |   value:{}   |  reward: {}  |   distance: {}   |  RR: {}".format(theta1, theta2, theta3, str(total_assigned) + '/' + str(total_trip), str(total_completed) + '/' + str(total_trip), total_value, total_reward, total_distance, total_assigned/total_trip)
            print(s, file = f)

            reward_list.append(total_reward)
            theta1_list.append(theta1)
            theta2_list.append(theta2)
            distance_list.append(total_distance)
            value_list.append(total_value)

            print(theta1,theta2,theta3)

            theta2 += step
            theta3 -= step

        theta1 += step
        theta2 = 0
        theta3 = 1 - theta1 - theta2
    return np.array(reward_list), np.array(theta1_list), np.array(theta2_list), np.array(distance_list), np.array(value_list)

best_theta1_list = []
best_theta2_list = []
for n in range(5):
    data = model.generate_data(5)
    # data = pd.read_csv("data.csv")
    theta1 = 0
    theta2 = 0
    step = 0.05
    epoch_count = 20
    reward_list, theta1_list, theta2_list, distance_list, value_list = train(data, step, epoch_count, 0, 0, 1)
    max_index = int(np.mean(np.where(value_list == np.max(value_list)), axis=1)[0] + 0.5)
    best_theta1_list.append(theta1_list[max_index])
    best_theta2_list.append(theta2_list[max_index])

    print("theta1 = {}, theta2 = {}, theta3 = {}   |   reward: {}   |   distance: {}".format(theta1_list[max_index], theta2_list[max_index], 1 - theta1_list[max_index] - theta2_list[max_index], reward_list[max_index], distance_list[max_index]), file = f)

    print("epoch {} done".format(n))

best_theta1 = np.mean(np.array(best_theta1_list))
best_theta2 = np.mean(np.array(best_theta2_list))
print("best theta1: {}   |   best theta2: {}   |   best theta3: {}".format(best_theta1, best_theta2, 1 - best_theta1 - best_theta2), file = f)

f.close()


# %%

# test阶段
f = open("output_ex2_worker10_train5.txt", 'w')

def test(curr_data, theta1, theta2):
    total_reward = 0
    unassigned = []
    worker_List = [Worker() for _ in range(model.worker_num)]
    total_trip = 0
    total_assigned = 0
    total_completed = 0
    total_wait_time = 0
    total_distance = 0
    total_value = 0

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

        new_assigned, new_distance, new_assigned_time = agent.DP(unassigned, worker_List, n, 26, theta1, theta2, 1 - theta1 - theta2)
        total_assigned += new_assigned
        total_distance += new_distance
        total_wait_time += new_assigned_time

        divi = 1
        if(total_completed > 0):
            divi = total_completed
        # 已分配，已完成，总收益，总等待时间，应答率
        s = "SN: {}:  assigned: {}  |  completed: {}  |   value: {}   |  reward: {}  |   distance: {}   |   wait_time: {}   |  RR: {}".format(n, str(total_assigned) + '/' + str(total_trip), str(total_completed) + '/' + str(total_trip), total_value/divi, total_reward/divi, total_distance/divi, total_wait_time/15/divi, total_assigned/total_trip)
        print(s, file = f)

curr_data = pd.read_csv("data.csv")
test(curr_data, 0.36999999999999994, 0.08)
# train10 0.35, 0.1
# train5 0.36999999999999994, 0.08
f.close()
# %%
print(best_theta1, best_theta2)
# %%
model.worker_num
# %%
