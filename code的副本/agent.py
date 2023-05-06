
#%%
import operator
import numpy as np

class Agent(object):
  """ Agent for dispatching and reposition """

  def __init__(self, **kwargs):
    """ Load your trained model and initialize the parameters """
    self.MAX_DIS = 10000

  ''' 最短距离优先NNP 
  :param 
        unassigned: 待分配的任务
        workerList: 工人列表
        sn: 当前时间片
  ''' 
  def NNP(self, unassigned, workerList, sn):
    assign_count = 0
    total_distance = 0
    new_assigned_time = 0

    for trip in unassigned:
      trip.wait_time = (sn - trip.start_sn) * 15

    free_worker_list = []
    for worker in workerList:
      if(worker.isFree):
        free_worker_list.append(worker)
    
    matrix = self.matrix_update(free_worker_list, unassigned, self.MAX_DIS, 1, 0, 0)
    
    array = np.array(matrix)
    while len(array) > 0 and len(array[0]) > 0 and np.min(array) < self.MAX_DIS:
      min_val = np.min(array)
      row, col = np.where(array == min_val)
      free_worker_list[row[0]].setTrip(unassigned[col[0]], sn)
      new_diatance = ((free_worker_list[row[0]].latitude - unassigned[col[0]].start_lat)**2 + (free_worker_list[row[0]].longtude - unassigned[col[0]].start_lon)**2)**0.5
      total_distance += new_diatance
      new_assigned_time += unassigned[col[0]].wait_time
      unassigned[col[0]].wait_time += min_val * 2
      assign_count += 1
      unassigned.remove(unassigned[col[0]])
      free_worker_list.remove(free_worker_list[row[0]])
      array = np.delete(array, row[0], axis = 0)
      array = np.delete(array, col[0], axis = 1)
    
    return assign_count, total_distance, new_assigned_time
  
  '''贪心算法'''
  def GR(self, unassigned, workerList, sn):
    assign_count = 0
    unassigned_wait_time = 0
    total_distance = 0
    new_assigned_time = 0

    for trip in unassigned:
      trip.wait_time = (sn - trip.start_sn) * 15
      unassigned_wait_time = unassigned_wait_time + trip.wait_time

    for worker in workerList:
      if(worker.isFree):
        earnest_trip = None
        for trip in unassigned:
          if(earnest_trip == None):
            earnest_trip = trip
          else:
            if(earnest_trip.cost < trip.cost):
              earnest_trip = trip
        
        if(earnest_trip != None):
          new_distance = ((worker.latitude - earnest_trip.start_lat)**2 + (worker.longtude - earnest_trip.start_lon)**2)**0.5
          new_assigned_time += earnest_trip.wait_time
          earnest_trip.wait_time += new_distance * 2
          worker.setTrip(earnest_trip, sn)
          total_distance += new_distance
          assign_count = assign_count + 1
          unassigned.remove(earnest_trip)
    
    return assign_count, total_distance, new_assigned_time
   
  def matrix_update(self, free_worker_list, unassigned, alpha, theta1, theta2, theta3):
    matrix = []
    for worker in free_worker_list:
      val = []
      for trip in unassigned:
        distance = ((worker.latitude - trip.start_lat)**2 + (worker.longtude - trip.start_lon)**2)**0.5
        # print("distance: {}   |   cost_performance: {}".format(distance, trip.cost_performance))
        if(distance <= alpha):
          val.append(((worker.latitude - trip.start_lat)**2 + (worker.longtude - trip.start_lon)**2)**0.5 * theta1 + trip.cost_performance/3 * theta2 + trip.wait_time * theta3)
        else:
          val.append(self.MAX_DIS)
      matrix.append(val)

    return matrix

  '''规划算法'''
  def DP(self, unassigned, workerList, sn, alpha, theta1, theta2, theta3):
    assign_count = 0
    unassigned_wait_time = 0
    total_distance = 0
    new_assigned_time = 0

    for trip in unassigned:
      trip.wait_time = (sn - trip.start_sn) * 15
      unassigned_wait_time = unassigned_wait_time + trip.wait_time

    free_worker_list = []
    for worker in workerList:
      if(worker.isFree):
        free_worker_list.append(worker)
    
    matrix = self.matrix_update(free_worker_list, unassigned, alpha, theta1, theta2, theta3)
    
    array = np.array(matrix)
    while len(array) > 0 and len(array[0]) > 0 and np.min(array) < self.MAX_DIS:
      min_val = np.min(array)
      row, col = np.where(array == min_val)
      free_worker_list[row[0]].setTrip(unassigned[col[0]], sn)
      new_distance = ((free_worker_list[row[0]].latitude - unassigned[col[0]].start_lat)**2 + (free_worker_list[row[0]].longtude - unassigned[col[0]].start_lon)**2)**0.5
      total_distance += new_distance
      new_assigned_time += unassigned[col[0]].wait_time
      unassigned[col[0]].wait_time += new_distance * 2
      assign_count += 1
      unassigned.remove(unassigned[col[0]])
      free_worker_list.remove(free_worker_list[row[0]])
      array = np.delete(array, row[0], axis = 0)
      array = np.delete(array, col[0], axis = 1)
    
    return assign_count, total_distance, new_assigned_time

# %%
