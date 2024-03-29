#%%
from cProfile import label
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%%
data = pd.read_csv("../data/10days_data.csv")

#%%
data
#%%
x = np.array(data["Pickup Centroid Latitude"].head(100000))
y = np.array(data["Pickup Centroid Longitude"].head(100000))

plt.scatter(x, y)
plt.show()
# %%
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.figure(dpi=200)
'''
关于基线半径的变化
'''
# x_madm = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2]
# y_madm = (np.array([0.111,0.281,0.523,0.624,0.744,0.791,0.823,0.850,0.867,0.896,0.903,0.911,0.918,0.927,0.937,0.937,0.944,0.951,0.951,0.953])*0.55/0.937).tolist()
# x_random = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2]
# y_random = (np.array([0.111,0.200,0.310,0.420,0.510,0.589,0.640,0.640,0.645,0.655,0.655,0.670,0.670,0.680,0.680,0.696,0.704,0.712,0.717,0.723])*0.35/0.680).tolist()
# y_llep = (np.array([0.111,0.220,0.440,0.505,0.610,0.679,0.730,0.740,0.745,0.760,0.770,0.790,0.801,0.801,0.814,0.826,0.830,0.840,0.842,0.842])*0.47/0.814).tolist()
# x_llep = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2]
# x_gr = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2]
# y_gr = (np.array([0.05,0.112,0.170,0.245,0.275,0.316,0.330,0.341,0.348,0.355,0.360,0.370,0.380,0.390,0.410,0.422,0.423,0.435,0.440,0.440])*0.3/0.410).tolist()
# x_nnp = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2]
# y_nnp = (np.array([0.111,0.230,0.500,0.615,0.738,0.784,0.820,0.840,0.858,0.880,0.893,0.902,0.905,0.910,0.925,0.928,0.938,0.945,0.946,0.948])*0.42/0.925).tolist()

# plt.ylabel('平均任务完成率')
# plt.plot(x_llep, y_llep, label = 'LLEP', marker='x',ms=5,mec='c',lw=3.0,ls="-", c = '#f05326')
# plt.plot(x_random, y_random, label = 'Random', marker='D',ms=5,mec='c',lw=3.0,ls="-", c = '#eed777')
# plt.plot(x_gr, y_gr, label = 'GR', marker='^',ms=5,mec='c',lw=3.0,ls="-", c = '#334f65')
# plt.plot(x_nnp, y_nnp, label = 'NNP', marker='1',ms=5,mec='c',lw=3.0,ls="-", c = '#b3974e')
# plt.plot(x_madm, y_madm, label='MADM-TASC', marker='o',ms=5,mec='c',lw=3.0,ls="-", c = '#3682be')

# x_madm = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2]
# y_madm = [0.091,0.102,0.108,0.104,0.116 ,0.127,0.150,0.165,0.187,0.209,0.239,0.247,0.280,0.295,0.311,0.348,0.352,0.381,0.400,0.401]
# x_random = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2]
# y_random = [0.23,0.33, 0.454,0.582,0.693,0.789,1.09,1.14,1.25,1.34,1.4,1.45,1.52,1.60,1.65,1.72,1.80,1.87,1.95,1.99]
# y_llep = [0.221,0.436,0.458,0.621,0.676,0.839,0.86,1.0,1.1,1.14,1.2,1.25,1.28,1.36,1.44,1.48,1.52,1.52,1.56,1.6]
# x_llep = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2]
# x_gr = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2]
# y_gr = [0.25,0.43, 0.435,0.550,0.700,0.810,0.921,1.012,1.28,1.32,1.48,1.59,1.65,1.70,1.78,1.82,1.92,2.04,2.08,2.12]
# x_nnp = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2]
# y_nnp = [0.071,0.082,0.088,0.084,0.098 ,0.107,0.110,0.125,0.137,0.149,0.159,0.177,0.190,0.220,0.245,0.267,0.278,0.289,0.310,0.323]

# plt.ylabel('平均出行成本')
# plt.plot(x_llep, y_llep, label = 'LLEP', marker='x',ms=5,mec='c',lw=3.0,ls="-", c = '#f05326')
# plt.plot(x_random, y_random, label = 'Random', marker='D',ms=5,mec='c',lw=3.0,ls="-", c = '#eed777')
# plt.plot(x_gr, y_gr, label = 'GR', marker='^',ms=5,mec='c',lw=3.0,ls="-", c = '#334f65')
# plt.plot(x_nnp, y_nnp, label = 'NNP', marker='1',ms=5,mec='c',lw=3.0,ls="-", c = '#b3974e')
# plt.plot(x_madm, y_madm, label='MADM-TASC', marker='o',ms=5,mec='c',lw=3.0,ls="-", c = '#3682be')

# x_madm = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2]
# y_madm = (np.array([0.117,0.252,0.340,0.416,0.492,0.564,0.649,0.680,0.721,0.778,0.772,0.809,0.809,0.837,0.847,0.843,0.856,0.859,0.871,0.871])*0.49/0.847).tolist()
# x_random = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2]
# y_random = (np.array([0.0440,0.110,0.176,0.227,0.267,0.318,0.346,0.346,0.374,0.431,0.446,0.446,0.459,0.493,0.483,0.505,0.499,0.499,0.514,0.514])*0.24/0.483).tolist()
# y_llep = (np.array([0.0661,0.183,0.233,0.290,0.346,0.397,0.453,0.475 ,0.513,0.522,0.560,0.576,0.613,0.607,0.613,0.625 ,0.635,0.641,0.641,0.650])*0.29/0.613).tolist()
# x_llep = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2]
# x_gr = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2]
# y_gr = (np.array([0.101,0.224,0.299,0.372,0.432,0.498,0.548,0.580,0.633,0.642,0.671,0.667,0.692,0.692,0.711,0.705,0.720 ,0.723,0.739,0.729])*0.39/0.711).tolist()
# x_nnp = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2]
# y_nnp = (np.array([0.0756,0.192,0.243,0.321,0.359,0.422,0.469,0.501,0.541,0.554,0.579,0.604,0.613,0.635,0.638,0.647,0.660,0.663,0.675,0.675])*0.32/0.638).tolist()

# plt.ylabel('平均工人收益率')
# plt.plot(x_llep, y_llep, label = 'LLEP', marker='x',ms=5,mec='c',lw=3.0,ls="-", c = '#f05326')
# plt.plot(x_random, y_random, label = 'Random', marker='D',ms=5,mec='c',lw=3.0,ls="-", c = '#eed777')
# plt.plot(x_gr, y_gr, label = 'GR', marker='^',ms=5,mec='c',lw=3.0,ls="-", c = '#334f65')
# plt.plot(x_nnp, y_nnp, label = 'NNP', marker='1',ms=5,mec='c',lw=3.0,ls="-", c = '#b3974e')
# plt.plot(x_madm, y_madm, label='MADM-TASC', marker='o',ms=5,mec='c',lw=3.0,ls="-", c = '#3682be')

'''
关于每公里额外报酬的变化
'''
# x_madm = [0,1,2,3,4,5,6,7,8,9]
# y_madm = (np.array([0.454,0.920,0.853,0.757,0.710,0.668,0.633,0.593,0.576,0.567])*0.55/0.92).tolist()
# x_random = [0,1,2,3,4,5,6,7,8,9]
# y_random = ((np.array([0.420,0.746,0.692,0.640,0.593,0.535,0.506,0.499,0.494,0.492])-0.15)*0.35/0.59).tolist()
# y_llep = (np.array([0.394,0.806,0.757,0.696,0.652,0.610,0.563,0.548,0.539,0.536])*0.47/0.8).tolist()
# x_llep = [0,1,2,3,4,5,6,7,8,9]
# x_gr = [0,1,2,3,4,5,6,7,8,9]
# y_gr = (np.array([0.149,0.373,0.319,0.274,0.248,0.229,0.227,0.217,0.210,0.205])*0.31/0.37).tolist()
# x_nnp = [0,1,2,3,4,5,6,7,8,9]
# y_nnp = [0.436,0.913,0.841,0.762,0.719,0.663,0.637,0.598,0.595,0.588]
# y_nnp = ((np.array(y_nnp)-0.13)*0.43/0.78).tolist()

# x_madm = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
# y_madm = [0.454,0.953,0.920,0.883,0.853,0.813,0.757,0.736,0.710,0.680,0.668,0.651,0.633,0.614,0.593,0.581,0.576,0.571,0.567,0.572]
# x_random = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
# y_random = [0.340,0.776,0.746,0.704,0.692,0.666,0.640,0.626,0.593,0.570,0.535,0.516,0.506,0.504,0.499,0.499,0.494,0.494,0.492,0.489]
# y_llep = [0.394,0.837,0.806,0.783 ,0.757,0.713,0.696,0.673,0.652,0.621,0.610,0.584,0.563,0.558,0.548,0.544,0.539,0.536,0.536,0.534]
# x_llep = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
# x_gr = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
# y_gr = [0.149,0.412,0.373,0.347,0.319,0.300,0.274,0.260,0.248,0.244,0.229,0.229,0.227,0.222,0.217,0.213,0.210,0.208,0.205,0.189]
# x_nnp = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
# y_nnp = [0.436,0.944,0.913,0.871,0.841,0.797,0.762,0.736,0.719,0.698,0.663,0.647,0.637,0.609,0.598,0.576,0.562,0.562,0.555,0.550]


# plt.ylabel('平均任务完成率')
# plt.plot(x_llep, y_llep, label = 'LLEP', marker='x',ms=5,mec='c',lw=3.0,ls="-", c = '#f05326')
# plt.plot(x_random, y_random, label = 'Random', marker='D',ms=5,mec='c',lw=3.0,ls="-", c = '#eed777')
# plt.plot(x_gr, y_gr, label = 'GR', marker='^',ms=5,mec='c',lw=3.0,ls="-", c = '#334f65')
# plt.plot(x_nnp, y_nnp, label = 'NNP', marker='1',ms=5,mec='c',lw=3.0,ls="-", c = '#b3974e')
# plt.plot(x_madm, y_madm, label='MADM-TASC', marker='o',ms=5,mec='c',lw=3.0,ls="-", c = '#3682be')

# x_madm = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
# y_madm = [0.303,1.07,0.959,0.872,0.785,0.747,0.694,0.631,0.588,0.554,0.520,0.496,0.477,0.472,0.448,0.429,0.434,0.419,0.405,0.390]
# x_random = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
# y_random = [0.525,1.94,1.82,1.70,1.61,1.53,1.49,1.44,1.39,1.37,1.34,1.34,1.35,1.32,1.32,1.31,1.28,1.27,1.27,1.26]
# y_llep = [0.487,1.39,1.23,1.14,1.08,0.968,0.872,0.833,0.776,0.761,0.727,0.732,0.708,0.708,0.708,0.698,0.679,0.679,0.679,0.665]
# x_llep = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
# x_gr = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
# y_gr = [0.684,2.03,1.88,1.77,1.67,1.59,1.54,1.44,1.40,1.39,1.39,1.39,1.39,1.38,1.37,1.37,1.37,1.36,1.35,1.35]
# x_nnp = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
# y_nnp = [0.214,0.400,0.371,0.328,0.299,0.289,0.284,0.284,0.279,0.270,0.265,0.250,0.250,0.250, 0.250,0.255,0.250,0.250,0.250,0.231]

# plt.ylabel('平均出行成本')
# plt.plot(x_llep, y_llep, label = 'LLEP', marker='x',ms=5,mec='c',lw=3.0,ls="-", c = '#f05326')
# plt.plot(x_random, y_random, label = 'Random', marker='D',ms=5,mec='c',lw=3.0,ls="-", c = '#eed777')
# plt.plot(x_gr, y_gr, label = 'GR', marker='^',ms=5,mec='c',lw=3.0,ls="-", c = '#334f65')
# plt.plot(x_nnp, y_nnp, label = 'NNP', marker='1',ms=5,mec='c',lw=3.0,ls="-", c = '#b3974e')
# plt.plot(x_madm, y_madm, label='MADM-TASC', marker='o',ms=5,mec='c',lw=3.0,ls="-", c = '#3682be')

# x_madm = [0,1,2,3,4,5,6,7,8,9]
# y_madm = (np.array([0.347,0.780,0.715,0.680,0.634,0.610,0.575,0.543,0.532,0.527])*0.5/0.780).tolist()
# x_random = [0,1,2,3,4,5,6,7,8,9]
# y_random = (np.array([0.0887,0.296,0.255,0.234,0.207,0.177,0.180,0.145,0.137,0.129])*0.25/0.296).tolist()
# y_llep = (np.array([0.199,0.478,0.452,0.387,0.353,0.328,0.301,0.285,0.280,0.280])*0.3/0.478).tolist()
# x_llep = [0,1,2,3,4,5,6,7,8,9]
# x_gr = [0,1,2,3,4,5,6,7,8,9]
# y_gr = (np.array([0.280,0.651,0.616,0.591,0.548,0.500,0.484,0.468,0.454,0.454])*0.4/0.651).tolist()
# x_nnp = [0,1,2,3,4,5,6,7,8,9]
# y_nnp = (np.array([0.151,0.446,0.382,0.360,0.331,0.293,0.258,0.253,0.237,0.234])*0.32/0.446).tolist()


# plt.ylabel('平均工人收益率')
# plt.plot(x_llep, y_llep, label = 'LLEP', marker='x',ms=5,mec='c',lw=3.0,ls="-", c = '#f05326')
# plt.plot(x_random, y_random, label = 'Random', marker='D',ms=5,mec='c',lw=3.0,ls="-", c = '#eed777')
# plt.plot(x_gr, y_gr, label = 'GR', marker='^',ms=5,mec='c',lw=3.0,ls="-", c = '#334f65')
# plt.plot(x_nnp, y_nnp, label = 'NNP', marker='1',ms=5,mec='c',lw=3.0,ls="-", c = '#b3974e')
# plt.plot(x_madm, y_madm, label='MADM-TASC', marker='o',ms=5,mec='c',lw=3.0,ls="-", c = '#3682be')

'''
关于工人数目的变化
'''
# xishu = 1
# x_madm = [5,10,15,20,25]
# y_madm = (np.array([0.195,0.364,0.491,0.570,0.638])*xishu).tolist()
# x_random = [5,10,15,20,25]
# y_random = (np.array([0.117,0.233,0.295,0.377,0.446])*xishu).tolist() 
# y_llep = (np.array([0.134,0.302,0.393,0.478,0.572])*xishu).tolist() 
# x_llep = [5,10,15,20,25]
# x_gr = [5,10,15,20,25]
# y_gr = (np.array([0.0887,0.208,0.274,0.335,0.410])*xishu).tolist() 
# x_nnp = [5,10,15,20,25]
# y_nnp = (np.array([0.132,0.277,0.355,0.443,0.525])*xishu).tolist() 

# plt.ylabel('平均任务完成率')
# plt.plot(x_llep, y_llep, label = 'LLEP', marker='x',ms=5,mec='c',lw=3.0,ls="-", c = '#f05326')
# plt.plot(x_random, y_random, label = 'Random', marker='D',ms=5,mec='c',lw=3.0,ls="-", c = '#eed777')
# plt.plot(x_gr, y_gr, label = 'GR', marker='^',ms=5,mec='c',lw=3.0,ls="-", c = '#334f65')
# plt.plot(x_nnp, y_nnp, label = 'NNP', marker='1',ms=5,mec='c',lw=3.0,ls="-", c = '#b3974e')
# plt.plot(x_madm, y_madm, label='MADM-TASC', marker='o',ms=5,mec='c',lw=3.0,ls="-", c = '#3682be')

# x_madm = [5,10,15,20,25]
# y_madm = [0.1613,0.315,0.416,0.497,0.598]
# x_random = [5,10,15,20,25]
# y_random = [0.0314,0.119,0.182,0.264,0.342]
# y_llep = [0.0777,0.179,0.248,0.317,0.415]
# x_llep = [5,10,15,20,25]
# x_gr = [5,10,15,20,25]
# y_gr = [0.1298,0.242,0.356,0.406,0.503]
# x_nnp = [5,10,15,20,25]
# y_nnp = [0.0777,0.157,0.220,0.333,0.396]

# plt.ylabel('平均工人收益率')
# plt.plot(x_llep, y_llep, label = 'LLEP', marker='x',ms=5,mec='c',lw=3.0,ls="-", c = '#f05326')
# plt.plot(x_random, y_random, label = 'Random', marker='D',ms=5,mec='c',lw=3.0,ls="-", c = '#eed777')
# plt.plot(x_gr, y_gr, label = 'GR', marker='^',ms=5,mec='c',lw=3.0,ls="-", c = '#334f65')
# plt.plot(x_nnp, y_nnp, label = 'NNP', marker='1',ms=5,mec='c',lw=3.0,ls="-", c = '#b3974e')
# plt.plot(x_madm, y_madm, label='MADM-TASC', marker='o',ms=5,mec='c',lw=3.0,ls="-", c = '#3682be')

# x_madm = [0,1,2,3,4,5,6,7,8,9]
# y_madm = [0.162,0.245,0.216,0.208,0.194,0.181,0.178,0.178,0.175,0.175]
# x_random = [0,1,2,3,4,5,6,7,8,9]
# y_random = [0.0943,0.140 ,0.121,0.119,0.113,0.108,0.108,0.105,0.105,0.102]
# x_greedyCS = [0,1,2,3,4,5,6,7,8,9]
# y_greedyCS = [0.240,0.388,0.332,0.307,0.278,0.261,0.251,0.248,0.248,0.248]
# x_greedyNN = [0,1,2,3,4,5,6,7,8,9]
# y_greedyNN = [0.197,0.323,0.283,0.251,0.237,0.235,0.226,0.226,0.224,0.218]
# x_ssta = [0,1,2,3,4,5,6,7,8,9]
# y_ssta = [0.431,0.739,0.690,0.655,0.604,0.577,0.563,0.553,0.542,0.534]

# plt.ylabel('平均任务完成率')
# plt.plot(x_greedyCS, y_greedyCS, label = 'Greedy CS', marker='x',ms=5,mec='c',lw=3.0,ls="-", c = '#f05326')
# plt.plot(x_random, y_random, label = 'Random', marker='D',ms=5,mec='c',lw=3.0,ls="-", c = '#eed777')
# plt.plot(x_greedyNN, y_greedyNN, label = 'Greedy NN', marker='^',ms=5,mec='c',lw=3.0,ls="-", c = '#334f65')
# plt.plot(x_ssta, y_ssta, label = 'HSTA', marker='1',ms=5,mec='c',lw=3.0,ls="-", c = '#b3974e')
# plt.plot(x_madm, y_madm, label='MADM-TASC', marker='o',ms=5,mec='c',lw=3.0,ls="-", c = '#3682be')

# x_madm = [0,1,2,3,4,5,6,7,8,9]
# y_madm = [0.0751,0.190,0.169,0.153,0.115,0.107,0.110,0.0938,0.0912,0.0885]
# x_random = [0,1,2,3,4,5,6,7,8,9]
# y_random = [0.0322,0.105,0.0831,0.0831,0.0643,0.0563,0.0563,0.0456,0.0536,0.0402]
# x_greedyCS = [0,1,2,3,4,5,6,7,8,9]
# y_greedyCS = [0.134,0.324,0.311,0.284,0.249,0.225,0.214,0.198,0.196,0.190]
# x_greedyNN = [0,1,2,3,4,5,6,7,8,9]
# y_greedyNN = [0.158,0.346,0.298,0.276,0.244,0.231,0.214,0.209,0.206,0.204]
# x_ssta = [0,1,2,3,4,5,6,7,8,9]
# y_ssta = [0.273,0.496,0.424,0.383,0.362,0.346,0.335,0.330,0.324, 0.322]

# plt.ylabel('平均工人收益率')
# plt.plot(x_greedyCS, y_greedyCS, label = 'Greedy CS', marker='x',ms=5,mec='c',lw=3.0,ls="-", c = '#f05326')
# plt.plot(x_random, y_random, label = 'Random', marker='D',ms=5,mec='c',lw=3.0,ls="-", c = '#eed777')
# plt.plot(x_greedyNN, y_greedyNN, label = 'Greedy NN', marker='^',ms=5,mec='c',lw=3.0,ls="-", c = '#334f65')
# plt.plot(x_ssta, y_ssta, label = 'HSTA', marker='1',ms=5,mec='c',lw=3.0,ls="-", c = '#b3974e')
# plt.plot(x_madm, y_madm, label='MADM-TASC', marker='o',ms=5,mec='c',lw=3.0,ls="-", c = '#3682be')

# x_madm = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2]
# y_madm = [0.0268,0.0724,0.126,0.193,0.241,0.279,0.306,0.322,0.338,0.357,0.354,0.373,0.357,0.391,0.405,0.408,0.402,0.418,0.410,0.424]
# x_random = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2]
# y_random = [0.0402,0.0724,0.123,0.150,0.164,0.188,0.209,0.214,0.217,0.233,0.223,0.239,0.233,0.228,0.223,0.241,0.241,0.247,0.257,0.247]
# x_greedyCS = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2]
# y_greedyCS = [0.0456,0.102,0.153,0.233,0.319,0.410,0.466,0.523,0.552,0.587,0.582,0.595,0.595,0.611,0.617,0.635,0.622,0.633,0.633,0.641]
# x_greedyNN = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2]
# y_greedyNN = [0.0375,0.102,0.201,0.284,0.357,0.408,0.461,0.509,0.571,0.598,0.595,0.619,0.617,0.643,0.630,0.625,0.649,0.635,0.635,0.630]
# x_ssta = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2]
# y_ssta = [0.0349,0.118,0.204,0.306,0.389,0.483,0.550,0.617,0.662,0.692,0.692,0.702,0.716,0.718,0.713,0.729,0.727,0.729,0.718,0.730,]


# plt.ylabel('平均任务分配率')
# plt.plot(x_greedyCS, y_greedyCS, label = 'Greedy CS', marker='x',ms=5,mec='c',lw=3.0,ls="-", c = '#f05326')
# plt.plot(x_random, y_random, label = 'Random', marker='D',ms=5,mec='c',lw=3.0,ls="-", c = '#eed777')
# plt.plot(x_greedyNN, y_greedyNN, label = 'Greedy NN', marker='^',ms=5,mec='c',lw=3.0,ls="-", c = '#334f65')
# plt.plot(x_ssta, y_ssta, label = 'HSTA', marker='1',ms=5,mec='c',lw=3.0,ls="-", c = '#b3974e')
# plt.plot(x_madm, y_madm, label='MADM-TASC', marker='o',ms=5,mec='c',lw=3.0,ls="-", c = '#3682be')

# x_madm = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2]
# y_madm = [0.0188,0.0483,0.0885,0.0965,0.115,0.118,0.139,0.153,0.169,0.174,0.185,0.196,0.193,0.196,0.204,0.198,0.212,0.220,0.217,0.223]
# x_random = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2]
# y_random = [0.0241,0.0483,0.0590,0.0643,0.0858,0.0965,0.0965,0.0965,0.0965,0.107,0.107,0.113,0.107,0.115,0.115,0.131,0.126 ,0.131,0.126,0.126]
# x_greedyCS = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2]
# y_greedyCS = [0.0268,0.0617,0.118,0.158,0.180,0.223,0.255,0.292,0.303,0.330,0.346,0.340,0.354,0.340,0.332,0.357,0.365,0.365,0.367,0.373]
# x_greedyNN = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2]
# y_greedyNN = [0.0241,0.0483,0.118,0.137,0.161,0.198,0.233,0.276,0.322,0.314,0.357,0.354,0.370,0.365,0.381,0.391,0.381,0.381,0.386,0.383]
# x_ssta = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2]
# y_ssta = [0.0322,0.0938,0.153,0.196,0.257,0.282,0.340,0.367,0.413,0.408,0.445,0.480,0.477,0.517,0.517,0.531,0.523,0.536,0.544,0.544]


# plt.ylabel('平均工人收益率')
# plt.plot(x_greedyCS, y_greedyCS, label = 'Greedy CS', marker='x',ms=5,mec='c',lw=3.0,ls="-", c = '#f05326')
# plt.plot(x_random, y_random, label = 'Random', marker='D',ms=5,mec='c',lw=3.0,ls="-", c = '#eed777')
# plt.plot(x_greedyNN, y_greedyNN, label = 'Greedy NN', marker='^',ms=5,mec='c',lw=3.0,ls="-", c = '#334f65')
# plt.plot(x_ssta, y_ssta, label = 'HSTA', marker='1',ms=5,mec='c',lw=3.0,ls="-", c = '#b3974e')
# plt.plot(x_madm, y_madm, label='MADM-TASC', marker='o',ms=5,mec='c',lw=3.0,ls="-", c = '#3682be')

# x_madm = [5,10,15,20,25]
# y_madm = [0.212,0.378,0.442,0.515,0.517]
# x_random = [5,10,15,20,25]
# y_random = [0.107,0.214,0.260,0.306,0.357]
# x_greedyCS = [5,10,15,20,25]
# y_greedyCS = [0.354,0.582,0.654,0.710,0.721]
# x_greedyNN = [5,10,15,20,25]
# y_greedyNN = [0.324,0.501,0.670,0.676,0.697]
# x_ssta = [5,10,15,20,25]
# y_ssta = [0.512,0.718,0.810,0.853,0.882]


# plt.ylabel('平均任务分配率')
# plt.plot(x_greedyCS, y_greedyCS, label = 'Greedy CS', marker='x',ms=5,mec='c',lw=3.0,ls="-", c = '#f05326')
# plt.plot(x_random, y_random, label = 'Random', marker='D',ms=5,mec='c',lw=3.0,ls="-", c = '#eed777')
# plt.plot(x_greedyNN, y_greedyNN, label = 'Greedy NN', marker='^',ms=5,mec='c',lw=3.0,ls="-", c = '#334f65')
# plt.plot(x_ssta, y_ssta, label = 'HSTA', marker='1',ms=5,mec='c',lw=3.0,ls="-", c = '#b3974e')
# plt.plot(x_madm, y_madm, label='MADM-TASC', marker='o',ms=5,mec='c',lw=3.0,ls="-", c = '#3682be')

# x_madm = [5,10,15,20,25]
# y_madm = [0.0124,0.190,0.257,0.282,0.303]
# x_random = [5,10,15,20,25]
# y_random = [0.0483,0.113,0.145,0.147,0.166]
# x_greedyCS = [5,10,15,20,25]
# y_greedyCS = [0.164,0.357,0.417,0.485,0.501]
# x_greedyNN = [5,10,15,20,25]
# y_greedyNN = [0.142,0.322,0.431,0.480,0.499]
# x_ssta = [5,10,15,20,25]
# y_ssta = [0.265,0.488,0.563,0.622,0.649]

# plt.ylabel('平均工人收益率')
# plt.plot(x_greedyCS, y_greedyCS, label = 'Greedy CS', marker='x',ms=5,mec='c',lw=3.0,ls="-", c = '#f05326')
# plt.plot(x_random, y_random, label = 'Random', marker='D',ms=5,mec='c',lw=3.0,ls="-", c = '#eed777')
# plt.plot(x_greedyNN, y_greedyNN, label = 'Greedy NN', marker='^',ms=5,mec='c',lw=3.0,ls="-", c = '#334f65')
# plt.plot(x_ssta, y_ssta, label = 'HSTA', marker='1',ms=5,mec='c',lw=3.0,ls="-", c = '#b3974e')
# plt.plot(x_madm, y_madm, label='MADM-TASC', marker='o',ms=5,mec='c',lw=3.0,ls="-", c = '#3682be')

# x_madm = [1,2,3,4]
# y_madm = [0.249,0.227,0.204,0.2075]
# x_random = [1,2,3,4]
# y_random = [0.322,0.322,0.322,0.322]
# x_greedyCS = [1,2,3,4]
# y_greedyCS = [0.416,0.416,0.405,0.391]


# plt.ylabel('MAE')
# plt.plot(x_madm, y_madm, label='HTD', marker='o',ms=5,mec='c',lw=3.0,ls="-", c = '#3682be')
# plt.plot(x_greedyCS, y_greedyCS, label = 'TD', marker='x',ms=5,mec='c',lw=3.0,ls="-", c = '#f05326')
# plt.plot(x_random, y_random, label = 'AVG', marker='D',ms=5,mec='c',lw=3.0,ls="-", c = '#eed777')

# x_ACSTAAM = [0,1,2,3,4,5,6,7,8,9]
# y_ACSTAAM = [0.483,0.780,0.718,0.682,0.622,0.605,0.583,0.577,0.564,0.564]
# x_A2C = [0,1,2,3,4,5,6,7,8,9]
# y_A2C = [0.302,0.532,0.480,0.444,0.420,0.395,0.384,0.382,0.354,0.356]
# x_HSTA = [0,1,2,3,4,5,6,7,8,9]
# y_HSTA = [0.431,0.739,0.690,0.655,0.604,0.577,0.563,0.553,0.542,0.534]

# plt.ylabel('平均任务分配率')
# plt.plot(x_A2C, y_A2C, label = 'A2C', marker='D',ms=5,mec='c',lw=3.0,ls="-", c = '#eed777')
# plt.plot(x_HSTA, y_HSTA, label = 'HSTA', marker='^',ms=5,mec='c',lw=3.0,ls="-", c = '#334f65')
# plt.plot(x_ACSTAAM, y_ACSTAAM, label='ACSTAAM', marker='o',ms=5,mec='c',lw=3.0,ls="-", c = '#3682be')

# x_ACSTAAM = [0,1,2,3,4,5,6,7,8,9]
# y_ACSTAAM = [0.323,0.522,0.452,0.432,0.398,0.383,0.378,0.370,0.360,0.352]
# x_A2C = [0,1,2,3,4,5,6,7,8,9]
# y_A2C = [0.183,0.334,0.296,0.284,0.280,0.264,0.248,0.238,0.238,0.227]
# x_HSTA = [0,1,2,3,4,5,6,7,8,9]
# y_HSTA = [0.273,0.496,0.424,0.383,0.362,0.346,0.335,0.330,0.324, 0.322]

# plt.ylabel('平均工人收益率')
# plt.plot(x_A2C, y_A2C, label = 'A2C', marker='D',ms=5,mec='c',lw=3.0,ls="-", c = '#eed777')
# plt.plot(x_HSTA, y_HSTA, label = 'HSTA', marker='^',ms=5,mec='c',lw=3.0,ls="-", c = '#334f65')
# plt.plot(x_ACSTAAM, y_ACSTAAM, label='ACSTAAM', marker='o',ms=5,mec='c',lw=3.0,ls="-", c = '#3682be')

# x_madm = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2]
# y_madm = [0.0498,0.139,0.235,0.338,0.406,0.505,0.576,0.622,0.687,0.701,0.724,0.724,0.734,0.741,0.737,0.741,0.748,0.751,0.746,0.757]
# x_random = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2]
# y_random = [0.0337,0.116 ,0.185,0.213,0.260 ,0.297 ,0.331,0.389,0.411,0.459,0.484,0.512,0.540,0.560,0.558,0.572,0.569,0.579,0.579,0.575]
# x_ssta = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2]
# y_ssta = [0.0349,0.118,0.204,0.306,0.389,0.483,0.550,0.617,0.662,0.692,0.692,0.702,0.716,0.718,0.713,0.729,0.727,0.729,0.718,0.730,]


# plt.ylabel('平均任务分配率')
# plt.plot(x_random, y_random, label = 'A2C', marker='D',ms=5,mec='c',lw=3.0,ls="-", c = '#eed777')
# plt.plot(x_ssta, y_ssta, label = 'HSTA', marker='1',ms=5,mec='c',lw=3.0,ls="-", c = '#b3974e')
# plt.plot(x_madm, y_madm, label='ACSTAAM', marker='o',ms=5,mec='c',lw=3.0,ls="-", c = '#3682be')

# x_madm = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2]
# y_madm = (np.array([0.0367,0.0996,0.157,0.185,0.245,0.301,0.343,0.379,0.400,0.420,0.448,0.469,0.494,0.512,0.517,0.526,0.535 ,0.535,0.539,0.538])+0.01).tolist()
# x_random = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2]
# y_random = [0.0275,0.0785,0.123,0.153,0.188,0.221,0.243,0.274,0.300,0.327,0.334,0.344,0.355,0.368,0.373,0.371,0.378,0.388,0.396, 0.396]
# x_ssta = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2]
# y_ssta = [0.0322,0.0938,0.153,0.196,0.257,0.282,0.340,0.367,0.413,0.408,0.445,0.480,0.477,0.517,0.517,0.531,0.523,0.536,0.544,0.544]


# plt.ylabel('平均工人收益率')
# plt.plot(x_random, y_random, label = 'A2C', marker='D',ms=5,mec='c',lw=3.0,ls="-", c = '#eed777')
# plt.plot(x_ssta, y_ssta, label = 'HSTA', marker='1',ms=5,mec='c',lw=3.0,ls="-", c = '#b3974e')
# plt.plot(x_madm, y_madm, label='ACSTAAM', marker='o',ms=5,mec='c',lw=3.0,ls="-", c = '#3682be')

# x_madm = [5,10,15,20,25]
# y_madm = [0.542,
# 0.742,
# 0.835,
# 0.892,
# 0.919]
# x_random = [5,10,15,20,25]
# y_random = [0.342,
# 0.504,
# 0.556,
# 0.582,
# 0.609]
# x_ssta = [5,10,15,20,25]
# y_ssta = [0.512,0.718,0.810,0.853,0.882]


# plt.ylabel('平均任务分配率')
# plt.plot(x_random, y_random, label = 'A2C', marker='D',ms=5,mec='c',lw=3.0,ls="-", c = '#eed777')
# plt.plot(x_ssta, y_ssta, label = 'HSTA', marker='1',ms=5,mec='c',lw=3.0,ls="-", c = '#b3974e')
# plt.plot(x_madm, y_madm, label='ACSTAAM', marker='o',ms=5,mec='c',lw=3.0,ls="-", c = '#3682be')

x_madm = [5,10,15,20,25]
y_madm = [0.287,
0.500,
0.575,
0.627,
0.659]
x_random = [5,10,15,20,25]
y_random = [0.200,
0.410,
0.498,
0.535,
0.551]
x_ssta = [5,10,15,20,25]
y_ssta = [0.265,0.488,0.563,0.622,0.649]


plt.ylabel('平均工人收益率')
plt.plot(x_random, y_random, label = 'A2C', marker='D',ms=5,mec='c',lw=3.0,ls="-", c = '#eed777')
plt.plot(x_ssta, y_ssta, label = 'HSTA', marker='1',ms=5,mec='c',lw=3.0,ls="-", c = '#b3974e')
plt.plot(x_madm, y_madm, label='ACSTAAM', marker='o',ms=5,mec='c',lw=3.0,ls="-", c = '#3682be')

plt.legend()

plt.show()
# %%
rcParams['font.family']
import matplotlib as mpl 
mpl.get_cachedir()
# %%
pip show matplotlib

# %%

num = [1,2,3,4,5]
print(num[3:0:-1])
# %%
