from mmwave.realtime_capture import adcCapThread
import threading
import time

a = adcCapThread(1,"adc",receiverType="direct_save")
a.start()
time.sleep(1)
while True:
    stop = input()
    if stop == "s":
        a.whileSign = False
        break