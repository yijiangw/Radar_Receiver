from realtime_capture import adcCapThread
import numpy as np
import threading
import time

a = adcCapThread(1,"adc")
a.start()
time.sleep(1)
counter = 0
nlist = []
lostPacket =[]
FrameNuml =[]
n = 0
f = open("adc.bin", "wb")
while True:
# for i in range(1000):
    readframe,FrameNum,lostPacketFlag=a.getFrame()
    if FrameNum>0:
        lostPacket.append(lostPacketFlag)
        FrameNuml.append(FrameNum)
        # print(lostPacketFlag)
        counter+=1
        f.write(readframe.tobytes())
        if counter == 150:
            # np.save(str(n),nlist)
            counter = 0
            n+=1        
        
    elif FrameNum==-1:
        print(readframe)
    elif FrameNum==-2:       
        # print(readframe)
        time.sleep(0.04)
    if n>10:    
       a.whileSign = False 
       print(lostPacket)
       f.close()
       break