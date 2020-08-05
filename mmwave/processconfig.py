import mmwave.radarconfig as cfg
# import cupy as cp
import numpy as np

class PointCloudProcConfig:
    
    def _dopplerCompensationTable(dopplerBins,num_tx):
        # doppler 补偿 先预计算对应多普勒bin处应该增加的补偿值
        table = np.exp(-1j*2 * np.pi * (np.arange(dopplerBins) - dopplerBins/2)/dopplerBins/num_tx).reshape(1,-1)
        # 如果是三发天线 再附加计算2倍补偿值 
        if(num_tx == 3):
            table = np.concatenate((np.ones(dopplerBins,dtype = "complex").reshape(1,-1),table,table*table))
#             table = np.append(table,table*table,axis=0)
#             table = np.append(np.ones(dopplerBins,dtype = "complex").reshape(1,-1),table,axis=0)
            table = np.array([table for i in range(4)]).transpose((1,0,2)).reshape(12,-1)
        return table
    
    def __init__(self):
        self.rangeBins=cfg.NUM_RANGE_BINS
        self.dopplerBins=cfg.NUM_DOPPLER_BINS
        self.AngleBins = 64
        
        self.enableCouplingSignatureRemoval = False
        self.enableStaticClutterRemoval = True
        self.enableDopplerCompensation = False
        self.reFFTBeforeAOA = False
        self.CFARresultFilterTop = True
        
        self.outputVelocity = True
        self.outputSNR = True
        self.outputRange = True
        self.outputInMeter = True
        
        # 0,1,2 for x,y,z additional data begin from 3
        dim = 3
        if self.outputVelocity:
            self.velocityDim = dim
            dim+=1
        if self.outputSNR:
            self.SNRRangeDim = dim
            dim+=1
        if self.outputSNR:
            self.SNRDopplerDim = dim
            dim+=1
        if self.outputRange:
            self.rangeDim = dim
            dim+=1
        self.outputDim = dim
        
        # coupling Signature collect
        self.couplingSignatureArray = np.array([])
        self.couplingSignatureBinFrontIdx = 5
        self.couplingSignatureBinRearIdx  = 4
        self.couplingSignatureFrameCount  = 20
        self.processedSignatureFrameCount = 0
        self.sumCouplingSignatureArray = np.zeros((cfg.NUM_TX,cfg.NUM_RX,
                                                   self.couplingSignatureBinFrontIdx+self.couplingSignatureBinRearIdx),dtype = np.complex)
        
        # pre-calculate doppler Compensation Table        
        self.dopplerCompensationTable = None
        if self.enableDopplerCompensation:
            self.dopplerCompensationTable = PointCloudProcConfig._dopplerCompensationTable(self.dopplerBins,cfg.NUM_TX)
                
        # CFAR result Filter from Top
        self.rangeCFARTopNum = 300
        self.dopplerCFARTopNum = 150
        self.rangeCFARTopScale = 1-self.rangeCFARTopNum/self.dopplerBins/self.rangeBins
        self.dopplerCFARTopScale = 1-self.dopplerCFARTopNum/self.dopplerBins/self.rangeBins
        
    
    
    def calculateCouplingSignatureArray(self,rawBinPath):
        reader = RawDataReader(rawBinPath)
        couplingSignatureArray = ncpp.zeros((cfg.NUM_TX,cfg.NUM_RX,
                                           self.couplingSignatureBinFrontIdx+self.couplingSignatureBinRearIdx),dtype = np.complex)
        for i in range(self.couplingSignatureFrameCount):
            frame = reader.getNextFrame()
            couplingSignatureArray+=np.sum(np.concatenate((frame[...,:self.couplingSignatureBinFrontIdx],
                                                           frame[...,-1*self.couplingSignatureBinRearIdx:]),
                                                           axis=3),
                                                           axis=2)/frame.shape[2]
        reader.close()
        shape=couplingSignatureArray.shape[0:2]+(1,couplingSignatureArray.shape[2])
        self.couplingSignatureArray = np.reshape(couplingSignatureArray/self.couplingSignatureFrameCount,shape)
    
    def realtimeCalculateCouplingSignatureArray(self,frame):
        if self.self.processedSignatureFrameCount < self.couplingSignatureFrameCount:
            self.sumCouplingSignatureArray+=np.sum(np.concatenate((frame[...,:self.couplingSignatureBinFrontIdx],
                                                                   frame[...,-1*self.couplingSignatureBinRearIdx:]),
                                                                   axis=3),
                                                                   axis=2)/frame.shape[2]
            self.processedSignatureFrameCount += 1
            if self.couplingSignatureFrameCount == self.couplingSignatureFrameCount:                
                shape=couplingSignatureArray.shape[0:2]+(1,couplingSignatureArray.shape[2])
                self.couplingSignatureArray=np.reshape(self.sumCouplingSignatureArray/self.couplingSignatureFrameCount,shape)