
import mmwave.radarconfig as cfg

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mmwave.signalprocfun_np import frame2pointcloud
from mmwave.processconfig import PointCloudProcConfig
from mmwave.adcreader import RawDataReader

import time

pointCloudProcConfig = PointCloudProcConfig()

dataPath = "adc_fb.bin"
reader = RawDataReader(dataPath)
time1 = time.time()
for i in range(600):
    frame = reader.getNextFrame()
    pointCloud = frame2pointcloud(frame,pointCloudProcConfig)
print(pointCloud)
time2 = time.time()
print(time2-time1)


fig = plt.figure("new")
plt.ion()
elev = 50
azim = 40
# 清空图像
fig.clf()
# 设定标题
fig.suptitle("static removed")

gs = fig.add_gridspec(3, 4)
# 生成画布
# add_subplot(子图 总行数，总列数，此图位置)
pointCloudSubplot = fig.add_subplot(gs[0:3,0:3], projection="3d")
pointCloudSubplot.view_init(elev, azim)

# 画三维散点图
color = pointCloud[pointCloudProcConfig.velocityDim]
pointCloud[pointCloudProcConfig.SNRRangeDim]
pointCloud[pointCloudProcConfig.rangeDim]

scale = 4

x = pointCloud[0]
y = pointCloud[1]
z = pointCloud[2]

pointCloudSubplot.scatter(x, y, z, s=scale, c=color, marker=".")

# 设置坐标轴图标
pointCloudSubplot.set_xlabel("X Label")
pointCloudSubplot.set_ylabel("Y Label")
pointCloudSubplot.set_zlabel("Z Label")

# 设置坐标轴范围
xlimmax = 200
ylimmax = 150
zlimmax = 200
if pointCloudProcConfig.outputInMeter:
    xlimmax = cfg.RANGE_RESOLUTION*cfg.NUM_RANGE_BINS
    ylimmax = cfg.RANGE_RESOLUTION*cfg.NUM_RANGE_BINS
    zlimmax = cfg.RANGE_RESOLUTION*cfg.NUM_RANGE_BINS


pointCloudSubplot.set_xlim(-xlimmax, xlimmax)
pointCloudSubplot.set_ylim(0, ylimmax)
pointCloudSubplot.set_zlim(-zlimmax, zlimmax)


XZplot = fig.add_subplot(gs[0,-1])
XZplot.scatter(x, z, s=scale, c=color, marker=".")
XZplot.set_xlabel("X Label")
XZplot.set_ylabel("Z Label")
XZplot.set_xlim(-xlimmax, xlimmax)            
XZplot.set_ylim(-zlimmax, zlimmax)

YZplot = fig.add_subplot(gs[1,-1])
YZplot.scatter(y, z, s=scale, c=color, marker=".")
YZplot.set_xlabel("Y Label")
YZplot.set_ylabel("Z Label")
YZplot.set_xlim(0, ylimmax)            
YZplot.set_ylim(-zlimmax, zlimmax)

XYplot = fig.add_subplot(gs[2,-1])
XYplot.scatter(x, y, s=scale, c=color, marker=".")
XYplot.set_xlabel("X Label")
XYplot.set_ylabel("Y Label")
XYplot.set_xlim(-xlimmax, xlimmax)            
XYplot.set_ylim(0, ylimmax)  

# 暂停
plt.pause(0.1)
azim=pointCloudSubplot.azim
elev=pointCloudSubplot.elev

# 关闭交互模式
plt.ioff()

# 图形显示
plt.show()