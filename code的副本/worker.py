#%%
import random
import string

class Worker(object):
    def __init__(self):
        self.MIN_LAT = 41.785998518
        self.MAX_LAT = 42.021223593
        self.MIN_LON = -87.90303966100001
        self.MAX_LON = -87.58236570
        self.workerID = ''.join([random.choice(string.ascii_letters
            + string.digits) for n in range(32)])
        self.latitude = (random.uniform(self.MIN_LAT, self.MAX_LAT) - self.MIN_LAT)*100
        self.longtude = (random.uniform(self.MIN_LON, self.MAX_LON) - self.MIN_LON)*100
        self.trip = None
        self.trip_list = []
        self.isFree = True

    ''' 更新在时间片sn的时候worker的状态
    :param
        sn: 当前执行到的时间片
    :return
        返回获取到的收益，如果当前时间片内有trip完成，则获取对应收益，否则返回0
    '''
    def Update(self, sn):
        if(self.isFree == False):
            if(self.trip.end_sn== sn):
                wait_time = self.trip.wait_time
                cost = self.trip.cost
                value = self.trip.cost * self.getSatisfaction(self.trip.wait_time)
                if(len(self.trip_list) == 0):
                    self.latitude = self.trip.end_lat
                    self.longtude = self.trip.end_lon
                    self.isFree = True
                    self.trip = None
                else:
                    for trip in self.trip_list:
                        trip.wait_time = sn + 1 - trip.start_sn
                    self.trip = self.trip_list.pop(0)
                
                return cost, wait_time, value
            else:
                return 0,self.trip.wait_time, 0
        return 0, 0, 0

    def setTrip(self, trip, sn):
        self.trip = trip
        self.isFree = False
        self.trip.end_sn = sn + self.trip.end_sn - self.trip.start_sn
    
    def assignTrip(self, trip):
        if(self.trip == None):
            self.trip = trip
            self.isFree = False
        else:
            self.trip_list.append(trip)
    
    def nextLocation(self, sn):
        if(self.trip.end_sn + self.trip.wait_time == sn):
            return [self.trip.end_lat, self.trip.end_lon, 1]
        else:
            return [self.latitude, self.longtude, 0]
    
    def getSatisfaction(self, wait_time):
        if(0 <= wait_time <= 4):
            return 1
        elif(5 < wait_time <= 8):
            return 0.8
        elif(8 < wait_time <= 12):
            return 0.5
        else:
            return 0.3
# %%
