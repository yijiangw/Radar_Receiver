'''
@Author: your name
@Date: 2020-06-08 19:21:47
@LastEditTime: 2020-06-10 21:20:14
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /mmwave/receiver_test.py
'''
from realtime_capture import adcCapThread
import numpy as np
import threading
import time

a = adcCapThread(1,"adc",receiverType="frame")
a.start()
counter = 0
nlist = []
lostPacketList =[]
ItemNumList =[]
t = time.time()
print(t)
time_min = 0

f = open("adc"+str(t).split(".")[0]+".bin", "wb")
# f = open("adc.bin", "wb")

while True:
# for i in range(1000):
    readItem,ItemNum,lostPacketListFlag=a.getFrame()
    if ItemNum>0:
        lostPacketList.append(lostPacketListFlag)
        ItemNumList.append(ItemNum)
        # print(lostPacketListFlag)
        counter+=1
        f.write(readItem.tobytes())
        if counter == 600:
            # np.save(str(n),nlist)
            counter = 0
            time_min+=1        
        
    elif ItemNum==-1:
        print(readItem)
    elif ItemNum==-2:       
        # print(readItem)
        time.sleep(0.04)
    if time_min>=23:    
       a.whileSign = False 
       print(lostPacketList)
       f.close()
       break
