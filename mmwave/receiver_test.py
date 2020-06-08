from mmwave.realtime_capture import adcCapThread
import numpy as np
import threading
import time

a = adcCapThread(1,"adc",receiverType="frame")
a.start()
counter = 0
nlist = []
lostPacketList =[]
ItemNumList =[]
n = 0
f = open("adc.bin", "wb")
while True:
# for i in range(1000):
    readItem,ItemNum,lostPacketListFlag=a.getFrame()
    if ItemNum>0:
        lostPacketList.append(lostPacketListFlag)
        ItemNumList.append(ItemNum)
        print(lostPacketListFlag)
        counter+=1
        f.write(readItem.tobytes())
        if counter == 150:
            # np.save(str(n),nlist)
            counter = 0
            n+=1        
        
    elif ItemNum==-1:
        print(readItem)
    elif ItemNum==-2:       
        print(readItem)
        time.sleep(0.04)
    if n>10:    
       a.whileSign = False 
       print(lostPacketList)
       f.close()
       break