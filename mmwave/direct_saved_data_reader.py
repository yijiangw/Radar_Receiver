import sys
import struct
import numpy as np


ADCBinFile=open("adc_direct_save.bin",'rb')
last_packet_num = 0
while True:

    data = ADCBinFile.read(1466)
    # print(data)
    # input()
    packet_num = struct.unpack('<1l', data[:4])[0]
    if last_packet_num < packet_num -1:
        print("\nlost packet*****\n")
        print("packet num:", packet_num)
        print("packet num:",last_packet_num)
    # print("packet num:", packet_num)
    byte_count = struct.unpack('>Q', b'\x00\x00' + data[4:10][::-1])[0]
    packet_data = np.frombuffer(data[10:1456], dtype=np.uint16)
    last_packet_num = packet_num
    
