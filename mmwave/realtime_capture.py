import socket
import struct
import threading
import time
import array as arr
import numpy as np

ADC_PARAMS = {'chirps': 50,  # 32
              'rx': 4,
              'tx': 3,
              'samples': 300,
              'IQ': 2,
              'bytes': 2}
# STATIC
MAX_PACKET_SIZE = 4096
BYTES_IN_PACKET = 1456

# DYNAMIC
BYTES_IN_FRAME = (ADC_PARAMS['chirps'] * ADC_PARAMS['rx'] * ADC_PARAMS['tx'] *
                  ADC_PARAMS['IQ'] * ADC_PARAMS['samples'] * ADC_PARAMS['bytes'])
BYTES_IN_FRAME_CLIPPED = (BYTES_IN_FRAME // BYTES_IN_PACKET) * BYTES_IN_PACKET
PACKETS_IN_FRAME = BYTES_IN_FRAME / BYTES_IN_PACKET
PACKETS_IN_FRAME_CLIPPED = BYTES_IN_FRAME // BYTES_IN_PACKET
UINT16_IN_PACKET = BYTES_IN_PACKET // 2
UINT16_IN_FRAME = BYTES_IN_FRAME // 2

class adcCapThread (threading.Thread):
    def __init__(self, threadID, name, static_ip='192.168.33.30', adc_ip='192.168.33.180',
                 data_port=4098, config_port=4096, receiverType="packet",bufferSize = 1500):
        threading.Thread.__init__(self)
        self.whileSign = True
        self.threadID = threadID
        self.name = name
        self.resentCapNum = 0
        self.latestReadNum = 0
        self.nextReadBufferPosition = 0
        self.nextCapBufferPosition = 0
        self.bufferOverWritten = True
        self.bufferSize = bufferSize
        
        # find the 2bytes length item_type 
        shortarr = arr.array("h",[1])        
        intarr = arr.array("i",[1])
        if shortarr.itemsize==2:
            bufferType = "h"
        elif intarr.itemsize==2:
            bufferType = "i"


    
        # Create configuration and data destinations
        self.cfg_dest = (adc_ip, config_port)
        self.cfg_recv = (static_ip, config_port)
        self.data_recv = (static_ip, data_port)

        # Create sockets
        self.config_socket = socket.socket(socket.AF_INET,
                                           socket.SOCK_DGRAM,
                                           socket.IPPROTO_UDP)
        self.data_socket = socket.socket(socket.AF_INET,
                                         socket.SOCK_DGRAM,
                                         socket.IPPROTO_UDP)

        # Bind data socket to fpga
        self.data_socket.bind(self.data_recv)

        # Bind config socket to fpga
        self.config_socket.bind(self.cfg_recv)
        if receiverType == "frame":
            # self.bufferArray = arr.array("h",np.ones([bufferSize*BYTES_IN_FRAME//2], dtype = np.int16))
            # self.itemNumArray = arr.array("l",np.ones([bufferSize], dtype = np.int16))
            # self.lostPackeFlagtArray = arr.array("h",np.zeros([bufferSize], dtype = np.int16))

            self.bufferArray = np.zeros((self.bufferSize,BYTES_IN_FRAME//2), dtype = np.int16)
            self.itemNumArray = np.zeros(self.bufferSize, dtype = np.int32)
            self.lostPackeFlagtArray = np.zeros(self.bufferSize,  dtype = bool)
            self.receiver == "frame"

        elif receiverType == "packet": 
            self.bufferSize *= 500
            self.bufferArray = np.zeros((self.bufferSize,BYTES_IN_PACKET//2), dtype = np.int16)
            self.itemNumArray = np.zeros(self.bufferSize, dtype = np.int32)
            self.lostPackeFlagtArray = np.zeros(self.bufferSize,  dtype = bool)
            self.receiver == "packet"


    def run(self):

        if self.receiver == "packet":
            self._packet_receiver(self)
        elif self.receiver == "frame":
            self._frame_receiver(self)
      
    def _packet_receiver(self):
        # first capture -- find the beginning of a Frame
        zero_packet = np.zeros(BYTES_IN_PACKET//2,dtype = np.int16)
        self.data_socket.settimeout(1)
        packet_num, byte_count, packet_data = self._read_data_packet()
        self.resentCapNum = packet_num
        while self.whileSign:
            packet_num, byte_count, packet_data = self._read_data_packet()
            # fix up the lost packets
            if self.resentCapNum < packet_num-1:
                while self.resentCapNum < packet_num:
                    self.resentCapNum+=1                    
                    self.lostPackeFlagtArray[self.nextCapBufferPosition] = True
                    self._store_frame(zero_packet)
            self.resentCapNum = packet_num
            self.lostPackeFlagtArray[self.nextCapBufferPosition] = False
            self._store_frame(packet_data)

    def _frame_receiver(self):
        # first capture -- find the beginning of a Frame
        self.data_socket.settimeout(1)
        lost_packets = False
        recentframe = np.zeros(UINT16_IN_FRAME, dtype=np.int16)
        while self.whileSign:
            packet_num, byte_count, packet_data = self._read_data_packet()
            after_packet_count = (byte_count+BYTES_IN_PACKET)% BYTES_IN_FRAME
            
            
            # the recent Frame begin at the middle of this packet
            if after_packet_count < BYTES_IN_PACKET :
                recentframe[0:after_packet_count//2] = packet_data[(BYTES_IN_PACKET-after_packet_count)//2:]
                self.resentCapNum = (byte_count+BYTES_IN_PACKET)//BYTES_IN_FRAME
                recentframe_collect_count = after_packet_count
                last_packet_num = packet_num
                break
                # packets_read = 1
                # store the cpature part of recentframe to the np array
                
            last_packet_num = packet_num
            
        print("aaaaa")
        print(self.nextCapBufferPosition)
        # input()
        while self.whileSign:
            packet_num, byte_count, packet_data = self._read_data_packet()
            # fix up the lost packets
            if last_packet_num < packet_num-1:                
                lost_packets = True
                recentframe_collect_count += BYTES_IN_PACKET*(packet_num-last_packet_num-1)
                # record lost packet
                self.lostPackeFlagtArray[self.nextCapBufferPosition] = True
                # packets_read += packet_num-last_packet_num-1
                # if the frame finished after this packet collected
                if recentframe_collect_count >= BYTES_IN_FRAME:
                    # store recent frame in the bufferArray
                    self._store_frame(recentframe)
                    # # store process can be improved
                    # self.bufferArray[self.nextCapBufferPosition] = recentframe
                    # self.itemNumArray[self.nextCapBufferPosition] = self.resentCapNum 
                    # # if the new frame over write the buffer at the location where the old frame even has not been read 
                    # if(self.nextReadBufferPosition == self.nextCapBufferPosition):
                    #     self.bufferOverWritten = True
                    # # after store
                    # self.nextCapBufferPosition += 1
                    # self.nextCapBufferPosition %= self.bufferSize

                    # refresh the recentFrame & put the rest of packet(beginning of next frame) into the recentFrame  
                    # record lost packet
                    self.lostPackeFlagtArray[self.nextCapBufferPosition] = True      
                    self.resentCapNum = byte_count//BYTES_IN_FRAME            
                    recentframe = np.zeros(UINT16_IN_FRAME, dtype=np.int16)
                    recentframe_collect_count %= BYTES_IN_FRAME

            # begin to process the recent packet
            # if the frame finished when this packet collected
            if recentframe_collect_count + BYTES_IN_PACKET >= BYTES_IN_FRAME:                
                recentframe[recentframe_collect_count//2:]=packet_data[:(BYTES_IN_FRAME-recentframe_collect_count)//2]
                self._store_frame(recentframe)                
                self.lostPackeFlagtArray[self.nextCapBufferPosition] = False
                # self.bufferArray[self.nextCapBufferPosition] = recentframe                    
                # self.itemNumArray[self.nextCapBufferPosition] = self.resentCapNum
                # # if the new frame over write the buffer at the location where the old frame even has not been read 
                # if(self.nextReadBufferPosition == self.nextCapBufferPosition):
                #     self.bufferOverWritten = True
                # self.nextCapBufferPosition += 1
                # self.nextCapBufferPosition %= self.bufferSize

                # refresh the recentFrame & put the rest of packet(beginning of next frame) into the recentFrame                  
                self.resentCapNum = (byte_count + BYTES_IN_PACKET)//BYTES_IN_FRAME
                recentframe = np.zeros(UINT16_IN_FRAME, dtype=np.int16)
                after_packet_count = (recentframe_collect_count + BYTES_IN_PACKET)%BYTES_IN_FRAME
                recentframe[0:after_packet_count//2] = packet_data[(BYTES_IN_PACKET-after_packet_count)//2:]
                recentframe_collect_count = after_packet_count
                lost_packets = False
            # if the frame not finished -- the full packet will be put into recentframe
            else:
                after_packet_count = (recentframe_collect_count + BYTES_IN_PACKET)%BYTES_IN_FRAME
                recentframe[recentframe_collect_count//2:after_packet_count//2]=packet_data
                recentframe_collect_count = after_packet_count                
            # this packet finish
            last_packet_num = packet_num
    
    def getFrame(self):
        if self.latestReadNum != 0:
            if self.bufferOverWritten == True:
                return "bufferOverWritten",-1,False
        else: 
            self.bufferOverWritten = False
        nextReadPosition = (self.nextReadBufferPosition+1)%self.bufferSize 
        if nextReadPosition == self.nextCapBufferPosition:
            return "wait new frame",-2,False
        else:
            readframe = self.bufferArray[self.nextReadBufferPosition]
            self.latestReadNum = self.itemNumArray[self.nextReadBufferPosition]            
            lostPacketFlag = self.lostPackeFlagtArray[self.nextReadBufferPosition]
            self.nextReadBufferPosition = nextReadPosition
        return readframe,self.latestReadNum,lostPacketFlag
    
    def getPacket(self):
        if self.latestReadNum != 0:
            if self.bufferOverWritten == True:
                return "bufferOverWritten",-1,False
        else: 
            self.bufferOverWritten = False
        nextReadPosition = (self.nextReadBufferPosition+1)%self.bufferSize 
        if nextReadPosition == self.nextCapBufferPosition:
            return "wait new packet",-2,False
        else:
            readframe = self.bufferArray[self.nextReadBufferPosition]
            self.latestReadNum = self.itemNumArray[self.nextReadBufferPosition]            
            lostPacketFlag = self.lostPackeFlagtArray[self.nextReadBufferPosition]
            self.nextReadBufferPosition = nextReadPosition
        return readframe,self.latestReadNum,lostPacketFlag

    def _store_frame(self,recentframe):
        self.bufferArray[self.nextCapBufferPosition] = recentframe                    
        self.itemNumArray[self.nextCapBufferPosition] = self.resentCapNum
        # if the new frame over write the buffer at the location where the old frame even has not been read 
        if(self.nextReadBufferPosition == self.nextCapBufferPosition):
            # print("ttttt")
            self.bufferOverWritten = True
        self.nextCapBufferPosition += 1
        self.nextCapBufferPosition %= self.bufferSize
        # print(self.nextCapBufferPosition)

    def _read_data_packet(self):
        """Helper function to read in a single ADC packet via UDP

        Returns:
            int: Current packet number, byte count of data that has already been read, raw ADC data in current packet

        """
        data, addr = self.data_socket.recvfrom(MAX_PACKET_SIZE)
        packet_num = struct.unpack('<1l', data[:4])[0]

        # print("packet num:", packet_num)

        byte_count = struct.unpack('>Q', b'\x00\x00' + data[4:10][::-1])[0]
        packet_data = np.frombuffer(data[10:], dtype=np.uint16)
        return packet_num, byte_count, packet_data

