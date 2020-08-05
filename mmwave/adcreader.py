import mmwave.radarconfig as cfg
# import cupy as cp
import numpy as np

class RawDataReader:
    def __init__(self,path="adc_fb.bin",frameCount = 0):
        self.path = path
        self.ADCBinFile=open(path,'rb')
        self.frameCount=frameCount
        self.frameNum=0
        self.frameSize = cfg.FRAME_SIZE
       
    def getNextFrame(self):
        if self.frameCount==0 or self.frameNum < self.frameCount :
            numpyFrame = np.frombuffer(self.ADCBinFile.read(self.frameSize),dtype=np.int16)
            numpyCompFrame=np.zeros(shape=(len(numpyFrame)//2), dtype=np.complex_)
            numpyCompFrame[0::2] = numpyFrame[0::4]+1j*numpyFrame[2::4]
            numpyCompFrame[1::2] = numpyFrame[1::4]+1j*numpyFrame[3::4]
            NextFrame = RawDataReader._frameReshape(numpyCompFrame)
            self.frameNum+=1
            return NextFrame 

    def close(self):
        self.ADCBinFile.close()
    
    def _frameReshape(frame):
        frameWithChirp = np.reshape(np.array(frame),(cfg.CHIRP_LOOPS,cfg.NUM_TX,cfg.NUM_RX,-1))
        # tx rx chirp simple
        return frameWithChirp.transpose(1,2,0,3)