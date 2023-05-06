#%%
class Trip(object):
    def __init__(self, tripID, start_sn, end_sn, start_lat, 
                start_lon, end_lat, end_lon, cost, trip_sec = 1) -> None:
        MIN_LAT = 41.785998518
        MAX_LAT = 42.021223593
        MIN_LON = -87.90303966100001
        MAX_LON = -87.582365702

        self.tripID = tripID
        self.start_sn = start_sn
        self.end_sn = end_sn
        self.start_lat = (start_lat - MIN_LAT) * 100
        self.end_lat = (end_lat - MIN_LAT) * 100
        self.start_lon = (start_lon - MIN_LON) * 100
        self.end_lon = (end_lon - MIN_LON) * 100
        self.cost = cost
        self.wait_time = 0
        self.trip_sec = trip_sec
        if(cost == 0.0):
             self.cost_performance = 100000 # 越小越好
        else:
             self.cost_performance = trip_sec/cost
        self.distance_loss = 0
        
# %%
