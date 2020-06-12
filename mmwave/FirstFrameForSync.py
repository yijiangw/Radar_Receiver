'''
@Author: Yijiang Wang
@Date: 2020-06-11 19:17:49
@LastEditTime: 2020-06-11 23:04:08
@LastEditors: Yijiang Wang
@FilePath: /mmwave/FirstFrameForSync.py
@Description: Use to find out the mmwave raw adc record sync pattern performed by experimenter
'''

import configuration
import numpy as np
from util import pm 

class FrameConfig:
    def __init__(self):
         # one chirp take 300 samples, sample rate 3000ksps
        self.numADCSamples=configuration.ADC_SAMPLES

        self.numRxAntennas=configuration.NUM_RX
        # tx order tx0,tx2,tx1  face to the board (left,right,upper) 
        self.numTxAntennas=configuration.NUM_TX

        # num of chirp loop, one loop has three chirps
        self.numLoopsPerFrame=configuration.CHIRP_LOOPS
        # num of frame, frame rate 25hz
        self.numFrames=600


        self.numChirpsPerFrame = self.numTxAntennas * self.numLoopsPerFrame

        self.numRangeBins = self.numADCSamples
        self.numDopplerBins = self.numLoopsPerFrame
        self.numAngleBins = 64
      
        # calculate size of one chirp in short.
        self.chirpSize = self.numRxAntennas * self.numADCSamples 
        # calculate size of one chirp loop in short. 3Tx has three chirps in one loop for TDM.
        self.chirpLoopSize = self.chirpSize * self.numTxAntennas
        # calculate size of one frame in short.
        self.frameSize = self.chirpLoopSize * self.numLoopsPerFrame

class RawDataReader:
    def __init__(self,path="adc1591833088.1103013.bin"):
        self.path = path
        self.ADCBinFile=open(path,'rb')
        self.nextFrameNum = 0
    
    def finish(self):
        self.ADCBinFile.close()
       
    def getNextFrame(self,frameconfig):
        # NextRawFrame = arr.array('h',FilePoint.read(frameSize*4))
        # numpyFrame = np.array(NextRawFrame)
        numpyFrame = np.frombuffer(self.ADCBinFile.read(frameconfig.frameSize*4),dtype=np.int16)
        numpyCompFrame=np.zeros(shape=(len(numpyFrame)//2), dtype=np.complex_)
        numpyCompFrame[0::2] = numpyFrame[0::4]+1j*numpyFrame[2::4]
        numpyCompFrame[1::2] = numpyFrame[1::4]+1j*numpyFrame[3::4]
        NextFrame = numpyCompFrame
        self.nextFrameNum += 1
        # print(numpyCompFrame[:8])
        return NextFrame 

def rangeFFT(frame,frameConfig):    
    """ 1D FFT for range
    """
    windowedBins1D = frameReshape(frame,frameConfig)*np.hamming(frameConfig.numADCSamples)
    rangeFFTResult=np.fft.fft(windowedBins1D)
    return rangeFFTResult
    
def frameReshape(frame,frameConfig):
    frameWithChirp = np.reshape(frame,(frameConfig.numLoopsPerFrame,frameConfig.numTxAntennas,frameConfig.numRxAntennas,-1))
    # tx rx chirp simple
    return frameWithChirp.transpose(1,2,0,3)

def findSyncFrame(path ,syncType = "vanish",rangeInBin = (5,8)):
    # sync action happens on where
    syncEdge = 0
    adc = RawDataReader(path)
    syncRecord = open(path.split(".bin")[0]+".sync",'w')
    
    fConfig = FrameConfig()
    lastFrameAmp = np.sum(np.abs(rangeFFT(adc.getNextFrame(fConfig),fConfig)[:,:,:,rangeInBin[0]:rangeInBin[1]]))
    syncEdge+=1
    avgAmp = lastFrameAmp
    for i in range(10):
        syncEdge+=1
        lastFrameAmp = np.sum(np.abs(rangeFFT(adc.getNextFrame(fConfig),fConfig)[:,:,:,rangeInBin[0]:rangeInBin[1]]))
        avgAmp = avgAmp*0.9 + lastFrameAmp*0.1
       
    lastFrameAmp = np.sum(np.abs(rangeFFT(adc.getNextFrame(fConfig),fConfig)[:,:,:,rangeInBin[0]:rangeInBin[1]]))
    avgAmp = avgAmp = avgAmp*0.9 + lastFrameAmp*0.1
    
    for i in range(fConfig.frameSize):
        syncEdge+=1
        lastFrameAmp = np.sum(np.abs(rangeFFT(adc.getNextFrame(fConfig),fConfig)[:,:,:,rangeInBin[0]:rangeInBin[1]]))        
        if(syncType == "arise"):
            if(lastFrameAmp>2*avgAmp):
                # found an Edge, but still check next 10 frame for sure
                check = True
                for j in range(10):
                    nextFrameAmp = np.sum(np.abs(rangeFFT(adc.getNextFrame(fConfig),fConfig)[:,:,:,rangeInBin[0]:rangeInBin[1]]))
                    if(nextFrameAmp<3*avgAmp):
                        syncEdge+=j+1
                        i+=j+1
                        check = False
                        break
                if(check):
                    print(lastFrameAmp,avgAmp)
                    print(syncEdge)
                    syncRecord.writelines(str(syncEdge))
                    syncRecord.close()
                    adc.finish()
                    return syncEdge

        if(syncType == "vanish"):
            if(2*lastFrameAmp<avgAmp):
                # found an Edge, but still check next frame for sure
                check = True
                for j in range(10):
                    nextFrameAmp = np.sum(np.abs(rangeFFT(adc.getNextFrame(fConfig),fConfig)[:,:,:,rangeInBin[0]:rangeInBin[1]]))
                    if(3*nextFrameAmp>avgAmp):
                        syncEdge+=j+1
                        i+=j+1
                        check = False
                        break
                if(check):
                    print(lastFrameAmp,avgAmp)
                    print(syncEdge)
                    syncRecord.writelines(str(syncEdge))
                    syncRecord.close()
                    adc.finish()                    
                    return syncEdge     
        avgAmp = avgAmp = avgAmp*0.9 + lastFrameAmp*0.1
    
if __name__ == '__main__':
    path = "adc1591838635.bin"
    # can create sync record file and also return the sync edge number
    edge = findSyncFrame(path,syncType = "arise")
    rader = RawDataReader(path)
    frameConfig = FrameConfig()
    
    counter = 0
    while True:
        frame = rader.getNextFrame(frameConfig)

        rangeResult = rangeFFT(frame,frameConfig)

        # pm from 10 frame ahead of the edge
        if(counter>edge-10):
            pm(abs(rangeResult[1][1]))
        print("sum"+str(counter),np.sum(abs(rangeResult[1][1][:,5:8])))        
        counter+=1
