# coding=UTF-8

import sys
import struct
import numpy as np
import scipy as sp
import array as arr

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from mmwave.util import pm 
from scipy.ndimage import convolve1d

import mmwave.configuration as cfg

# folder=sys.argv[1]

# read 8 byte bin data and turn it to 4 short
def read8byte(x):
	return struct.unpack('<hhhh', x)
class FrameConfig:
    def __init__(self):
         # one chirp take 300 samples, sample rate 3000ksps
        self.numADCSamples=300
        self.numRxAntennas=4
        # tx order tx0,tx2,tx1  face to the board (left,right,upper) 
        self.numTxAntennas=3
        # num of chirp loop, one loop has three chirps
        self.numLoopsPerFrame=50
        # num of frame, frame rate 25hz
        self.numFrames=2500
        
        #  config with configuration.py
        self.numADCSamples = cfg.ADC_SAMPLES
        self.numTxAntennas = cfg.NUM_TX
        self.numRxAntennas = cfg.NUM_RX
        self.numLoopsPerFrame = cfg.CHIRP_LOOPS

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

class PointCloudProcessCFG:    
    def __init__(self):
        self.frameConfig = FrameConfig()
        self.enableCouplingSignatureRemoval = True
        self.enableStaticClutterRemoval = True
        self.enableDopplerCompensation = True
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
            self.SNRDim = dim
            dim+=1
        if self.outputRange:
            self.rangeDim = dim
            dim+=1
        self.couplingSignatureArray = np.array([])
        self.couplingSignatureBinFrontIdx = 5
        self.couplingSignatureBinRearIdx  = 4
        self.couplingSignatureFrameCount  = 20
        self.processedSignatureFrameCount = 0
        self.dopplerCompensationTable = None
        if self.enableDopplerCompensation:
            self.dopplerCompensationTable = dopplerCompensationTable(self.frameConfig.numDopplerBins,self.frameConfig.numTxAntennas)
        self.sumCouplingSignatureArray = np.zeros((self.frameConfig.numTxAntennas,self.frameConfig.numRxAntennas,self.couplingSignatureBinFrontIdx+self.couplingSignatureBinRearIdx),dtype = np.complex)

    def calculateCouplingSignatureArray(self,rawBinPath):
        reader = RawDataReader(rawBinPath)
        couplingSignatureArray = np.zeros((self.frameConfig.numTxAntennas,self.frameConfig.numRxAntennas,self.couplingSignatureBinFrontIdx+self.couplingSignatureBinRearIdx),dtype = np.complex)
        for i in range(self.couplingSignatureFrameCount):
            frame = reader.getNextFrame(self.frameConfig)
            reshapedFrame = frameReshape(frame,self.frameConfig)
            couplingSignatureArray+=np.sum(np.concatenate((reshapedFrame[...,:self.couplingSignatureBinFrontIdx],
                                                           reshapedFrame[...,-1*self.couplingSignatureBinRearIdx:]),
                                                           axis=3),
                                           axis=2)/reshapedFrame.shape[2]
        reader.close()
        shape=couplingSignatureArray.shape[0:2]+(1,couplingSignatureArray.shape[2])
        self.couplingSignatureArray = np.reshape(couplingSignatureArray/self.couplingSignatureFrameCount,shape)
    
    def realtimeCalculateCouplingSignatureArray(self,reshapedFrame):
        if self.self.processedSignatureFrameCount < self.couplingSignatureFrameCount:
            self.sumCouplingSignatureArray+=np.sum(np.concatenate((reshapedFrame[...,:self.couplingSignatureBinFrontIdx],
                                                                   reshapedFrame[...,-1*self.couplingSignatureBinRearIdx:]),
                                                                   axis=3),
                                                   axis=2)/reshapedFrame.shape[2]
            self.self.processedSignatureFrameCount += 1
            if self.couplingSignatureFrameCount == self.couplingSignatureFrameCount:                
                shape=shape=couplingSignatureArray.shape[0:2]+(1,couplingSignatureArray.shape[2])
                self.couplingSignatureArray=np.reshape(self.sumCouplingSignatureArray/self.couplingSignatureFrameCount,shape)

class RawDataReader:
    def __init__(self,path="C:\\workspace\\adc_dataLR.bin"):
        self.path = path
        self.ADCBinFile=open(path,'rb')
        
       
    def getNextFrame(self,frameconfig):
        # NextRawFrame = arr.array('h',FilePoint.read(frameSize*4))
        # numpyFrame = np.array(NextRawFrame)
        numpyFrame = np.frombuffer(self.ADCBinFile.read(frameconfig.frameSize*4),dtype=np.int16)
        numpyCompFrame=np.zeros(shape=(len(numpyFrame)//2), dtype=np.complex_)
        numpyCompFrame[0::2] = numpyFrame[0::4]+1j*numpyFrame[2::4]
        numpyCompFrame[1::2] = numpyFrame[1::4]+1j*numpyFrame[3::4]
        NextFrame = numpyCompFrame
        # print(numpyCompFrame[:8])
        return NextFrame 

    def close(self):
        self.ADCBinFile.close()

def frameReshape(frame,frameConfig):
    frameWithChirp = np.reshape(frame,(frameConfig.numLoopsPerFrame,frameConfig.numTxAntennas,frameConfig.numRxAntennas,-1))
    # tx rx chirp simple
    return frameWithChirp.transpose(1,2,0,3)

def rangeFFT(reshapedFrame,frameConfig):    
    """ 1D FFT for range
    """
    windowedBins1D = reshapedFrame*np.hamming(frameConfig.numADCSamples)
    rangeFFTResult=np.fft.fft(windowedBins1D)
    return rangeFFTResult

def couplingSignatureRemoval(rangeFFTResult, pointCloudProcessCFG):
    rangeFFTResult[:,:,:,:pointCloudProcessCFG.couplingSignatureBinFrontIdx+1] -= pointCloudProcessCFG.couplingSignatureArray[:,:,:,:pointCloudProcessCFG.couplingSignatureBinFrontIdx+1]
    rangeFFTResult[:,:,:,int(-1*pointCloudProcessCFG.couplingSignatureBinRearIdx):] -= pointCloudProcessCFG.couplingSignatureArray[:,:,:,int(-1*pointCloudProcessCFG.couplingSignatureBinRearIdx):]

    return rangeFFTResult

def clutter_removal(input_val, axis=0):
    """Perform basic static clutter removal by removing the mean from the input_val on the specified doppler axis.
    Args:
        input_val (ndarray): Array to perform static clutter removal on. Usually applied before performing doppler FFT.
            e.g. [num_chirps, num_vx_antennas, num_samples], it is applied along the first axis.
        axis (int): Axis to calculate mean of pre-doppler.
    Returns:
        ndarray: Array with static clutter removed.
    """
    # Reorder the axes
    reordering = np.arange(len(input_val.shape))
    reordering[0] = axis
    reordering[axis] = 0
    input_val = input_val.transpose(reordering)

    # Apply static clutter removal
    mean = input_val.mean(0)
    output_val = input_val - mean

    return output_val.transpose(reordering)
    
def rangeAndDopplerFFT(reshapedFrame,frameConfig):
    """ perform Range and Doppler FFT together
    """
    windowedBins2D = reshapedFrame*np.reshape(np.hamming(frameConfig.numLoopsPerFrame),(-1,1))*np.hamming(frameConfig.numADCSamples)
    dopplerFFTResult=np.fft.fft2(windowedBins2D)
    return dopplerFFTResult

def dopplerFFT(rangeResult,frameConfig):    
    """ input result from rangeFFT and do 2D FFT for velocity
    """
    windowedBins2D = rangeResult*np.reshape(np.hamming(frameConfig.numLoopsPerFrame),(1,1,-1,1))
    dopplerFFTResult=np.fft.fft(windowedBins2D,axis=2)
    dopplerFFTResult=np.fft.fftshift(dopplerFFTResult,axes=2)
    return dopplerFFTResult

def dopplerCompensationTable(numdopplerBins,numTxAntennas):
    # doppler 补偿 先预计算对应多普勒bin处应该增加的补偿值
    table = np.exp(-1j*2 * np.pi * (np.arange(numdopplerBins) - numdopplerBins/2)/numdopplerBins/numTxAntennas).reshape(1,-1)
    # 如果是三发天线 再附加计算2倍补偿值 
    if(numTxAntennas == 3):
        table = np.append(table,table*table,axis=0)
        table = np.append(np.ones(numdopplerBins,dtype = "complex").reshape(1,-1),table,axis=0)
        table = np.array([table for i in range(4)]).transpose((1,0,2)).reshape(12,-1)
    return table


def ca(x, *argv, **kwargs):
    """Detects peaks in signal using Cell-Averaging CFAR (CA-CFAR).

    Args:
        x (~numpy.ndarray): Signal.
        *argv: See mmwave.dsp.cfar.ca\_
        **kwargs: See mmwave.dsp.cfar.ca\_

    Returns:
        ~numpy.ndarray: Boolean array of detected peaks in x.

    Examples:
        >>> signal = np.random.randint(100, size=10)
        >>> signal
            array([41, 76, 95, 28, 25, 53, 10, 93, 54, 85])
        >>> det = mm.dsp.ca(signal, l_bound=20, guard_len=1, noise_len=3)
        >>> det
            array([False, False,  True, False, False, False, False,  True, False,
                    True])

        Perform a non-wrapping CFAR

        >>> signal = np.random.randint(100, size=10)
        >>> signal
            array([41, 76, 95, 28, 25, 53, 10, 93, 54, 85])
        >>> det =  mm.dsp.ca(signal, l_bound=20, guard_len=1, noise_len=3, mode='constant')
        >>> det
            array([False,  True,  True, False, False, False, False,  True,  True,
                    True])

    """
    if isinstance(x, list):
        x = np.array(x)
    threshold, _ = ca_(x, *argv, **kwargs)
    ret = (x > threshold)
    return ret


def ca_(x, guard_len=4, noise_len=8, mode='wrap', l_bound=4000):
    """Uses Cell-Averaging CFAR (CA-CFAR) to calculate a threshold that can be used to calculate peaks in a signal.

    Args:
        x (~numpy.ndarray): Signal.
        guard_len (int): Number of samples adjacent to the CUT that are ignored.
        noise_len (int): Number of samples adjacent to the guard padding that are factored into the calculation.
        mode (str): Specify how to deal with edge cells. Examples include 'wrap' and 'constant'.
        l_bound (float or int): Additive lower bound while calculating peak threshold.

    Returns:
        Tuple [ndarray, ndarray]
            1. (ndarray): Upper bound of noise threshold.
            #. (ndarray): Raw noise strength.

    Examples:
        >>> signal = np.random.randint(100, size=10)
        >>> signal
            array([41, 76, 95, 28, 25, 53, 10, 93, 54, 85])
        >>> threshold = mm.dsp.ca_(signal, l_bound=20, guard_len=1, noise_len=3)
        >>> threshold
            (array([70, 76, 64, 79, 81, 91, 74, 71, 70, 79]), array([50, 56, 44, 59, 61, 71, 54, 51, 50, 59]))

        Perform a non-wrapping CFAR thresholding

        >>> signal = np.random.randint(100, size=10)
        >>> signal
            array([41, 76, 95, 28, 25, 53, 10, 93, 54, 85])
        >>> threshold = mm.dsp.ca_(signal, l_bound=20, guard_len=1, noise_len=3, mode='constant')
        >>> threshold
            (array([44, 37, 41, 65, 81, 91, 67, 51, 34, 46]), array([24, 17, 21, 45, 61, 71, 47, 31, 14, 26]))

    """
    # print("xshape:",x.shape)
    if isinstance(x, list):
        x = np.array(x)
    assert type(x) == np.ndarray

    kernel = np.ones(1 + (2 * guard_len) + (2 * noise_len), dtype=x.dtype) / (2 * noise_len)
    kernel[noise_len:noise_len + (2 * guard_len) + 1] = 0

    noise_floor = convolve1d(x, kernel, mode=mode)
    threshold = noise_floor + l_bound

    return threshold, noise_floor

def naive_xyz(virtual_ant, num_tx=3, num_rx=4, fft_size=64):
    """ Estimate the phase introduced from the elevation of the elevation antennas

    Args:
        virtual_ant: Signal received by the rx antennas, shape = [#angleBins, #detectedObjs], zero-pad #virtualAnts to #angleBins
        num_tx: Number of transmitter antennas used
        num_rx: Number of receiver antennas used
        fft_size: Size of the fft performed on the signals

    Returns:
        x_vector (float): Estimated x axis coordinate in meters (m)
        y_vector (float): Estimated y axis coordinate in meters (m)
        z_vector (float): Estimated z axis coordinate in meters (m)

    """
    assert num_tx > 2, "need a config for more than 2 TXs"
    num_detected_obj = virtual_ant.shape[1]
    # print("virtual_ant",virtual_ant.shape)
    # Zero pad azimuth
    azimuth_ant = virtual_ant[:2 * num_rx, :]
    azimuth_ant_padded = np.zeros(shape=(fft_size, num_detected_obj), dtype=np.complex_)
    # print("azimuth_ant_padded",azimuth_ant_padded.shape)
    azimuth_ant_padded[:2 * num_rx, :] = azimuth_ant

    # Process azimuth information
    azimuth_fft = np.fft.fft(azimuth_ant_padded, axis=0)
    k_max = np.argmax(np.abs(azimuth_fft), axis=0)  # shape = (num_detected_obj, )
    # peak_1 = azimuth_fft[k_max]
    peak_1 = np.zeros_like(k_max, dtype=np.complex_)
    for i in range(len(k_max)):
        peak_1[i] = azimuth_fft[k_max[i], i]

    k_max[k_max > (fft_size // 2) - 1] = k_max[k_max > (fft_size // 2) - 1] - fft_size
    wx = 2 * np.pi / fft_size * k_max  # shape = (num_detected_obj, )
    x_vector = wx / np.pi

    # Zero pad elevation
    elevation_ant = virtual_ant[2 * num_rx:, :]
    elevation_ant_padded = np.zeros(shape=(fft_size, num_detected_obj), dtype=np.complex_)
    # elevation_ant_padded[:len(elevation_ant)] = elevation_ant
    elevation_ant_padded[:num_rx, :] = elevation_ant

    # Process elevation information
    elevation_fft = np.fft.fft(elevation_ant, axis=0)
    elevation_max = np.argmax(np.log2(np.abs(elevation_fft)), axis=0)  # shape = (num_detected_obj, )
    peak_2 = np.zeros_like(elevation_max, dtype=np.complex_)
    # peak_2 = elevation_fft[np.argmax(np.log2(np.abs(elevation_fft)))]
    for i in range(len(elevation_max)):
        peak_2[i] = elevation_fft[elevation_max[i], i]

    # Calculate elevation phase shift
    wz = np.angle(peak_1 * peak_2.conj() * np.exp(1j * 2 * wx))
    z_vector = wz / np.pi
    # print("x_vector",x_vector)
    # print("y_vector",z_vector)
    ypossible = 1 - x_vector ** 2 - z_vector ** 2
    y_vector=ypossible
    y_vector[ ypossible<0 ] = 0
    x_vector[ ypossible<0 ] = 0
    z_vector[ ypossible<0 ] = 0
    y_vector = np.sqrt(y_vector)
    return x_vector, y_vector, z_vector

def frame2pointcloud(frame,pointCloudProcessCFG):
    frameConfig = pointCloudProcessCFG.frameConfig

    reshapedFrame = frameReshape(frame,frameConfig)

    rangeResult = rangeFFT(reshapedFrame,frameConfig)
    
    if pointCloudProcessCFG.enableCouplingSignatureRemoval and pointCloudProcessCFG.couplingSignatureArray.any():
        rangeResult = couplingSignatureRemoval(rangeResult,pointCloudProcessCFG)

    if pointCloudProcessCFG.enableStaticClutterRemoval:
        rangeResult = clutter_removal(rangeResult,axis=2)
    
    dopplerResult = dopplerFFT(rangeResult,frameConfig)
    # sum all antenna to get a range*doppler array for cfar use
    dopplerResultSumAllAntenna = np.sum(dopplerResult, axis=(0,1))
    # transform the complex value array to DB value array  
    dopplerResultInDB = 20*np.log10(np.absolute(dopplerResultSumAllAntenna))
    # another method to get log2 value
    # dopplerResultInDB = 10*np.log2(np.absolute(dopplerResult))

    cfarRangeResult = np.apply_along_axis(func1d=ca_,
                                            axis=1,
                                            arr=dopplerResultInDB,
                                            l_bound=25,
                                            guard_len=4,
                                            noise_len=8)
    # print(cfarRangeResult.shape)
    thresholdRange, noiseFloorRange = cfarRangeResult[...,0,:],cfarRangeResult[...,1,:]
    rangeCFAR = np.zeros(thresholdRange.shape, bool)
    rangeCFAR[dopplerResultInDB>thresholdRange] = True
    # input()
    cfarDopplerResult = np.apply_along_axis(func1d=ca_,
                                            axis=0,
                                            arr=dopplerResultInDB,
                                            l_bound=25,
                                            guard_len=4,
                                            noise_len=8)
    # print(cfarDopplerResult.shape)
    thresholdDoppler, noiseFloorDoppler = cfarDopplerResult[...,0,:,:],cfarDopplerResult[...,1,:,:]
    
    dopplerCFAR = np.zeros(thresholdDoppler.shape, bool)
    dopplerCFAR[dopplerResultInDB>thresholdDoppler] = True

    cfarResult = rangeCFAR|dopplerCFAR
    
    det_peaks_indices = np.argwhere(cfarResult == True)
    # record range and velocity of detected points
    R = det_peaks_indices[:,1].astype(np.float64)
    V = (det_peaks_indices[:,0]-frameConfig.numDopplerBins//2).astype(np.float64)
    if pointCloudProcessCFG.outputInMeter:
        R *= cfg.RANGE_RESOLUTION
        V *= cfg.DOPPLER_RESOLUTION
    print(R.shape)
    # record rangeCFAR SNR of detected points  
    SNR = dopplerResultInDB - noiseFloorRange
    SNR = SNR[cfarResult==True]
    print("SNR shape",SNR.shape)

    AOAInput = dopplerResult[:,:,cfarResult==True]
    AOAInput = AOAInput.reshape(12,-1)
    if pointCloudProcessCFG.enableDopplerCompensation:
        AOAInput *= pointCloudProcessCFG.dopplerCompensationTable[:,det_peaks_indices[:,0]]

    print("AOAInput:",AOAInput.shape,len(AOAInput))
    if AOAInput.shape[1]==0:
        print("no cfar det point")
        return np.array([]).reshape(6,0)
    x_vec, y_vec, z_vec = naive_xyz(AOAInput)   
    
    # print(S.shape)
    # print(R)
    # print(S)
    x,y,z = x_vec*R, y_vec*R, z_vec*R
    pointCloud=np.concatenate((x,y,z,V,SNR,R))
    print("pointCloud",pointCloud.shape)
    pointCloud = np.reshape(pointCloud,(6,-1))
    print(pointCloud.shape)
    pointCloud = pointCloud[:,y_vec!=0]      
    print(pointCloud.shape) 
    return pointCloud

def compareframe2pointcloud(frame,pointCloudProcessCFG):
    frameConfig = pointCloudProcessCFG.frameConfig

    reshapedFrame = frameReshape(frame,frameConfig)

    rangeResult = rangeFFT(reshapedFrame,frameConfig)
    
    if pointCloudProcessCFG.enableCouplingSignatureRemoval and pointCloudProcessCFG.couplingSignatureArray.any():
        rangeResult = couplingSignatureRemoval(rangeResult,pointCloudProcessCFG)

    if pointCloudProcessCFG.enableStaticClutterRemoval:
        rangeResult = clutter_removal(rangeResult,axis=2)
    
    dopplerResult = dopplerFFT(rangeResult,frameConfig)
    # sum all antenna to get a range*doppler array for cfar use
    dopplerResultSumAllAntenna = np.sum(dopplerResult, axis=(0,1))
    # transform the complex value array to DB value array  
    dopplerResultInDB = 20*np.log10(np.absolute(dopplerResultSumAllAntenna))
    # another method to get log2 value
    # dopplerResultInDB = 10*np.log2(np.absolute(dopplerResult))

    cfarRangeResult = np.apply_along_axis(func1d=ca_,
                                            axis=1,
                                            arr=dopplerResultInDB,
                                            l_bound=25,
                                            guard_len=4,
                                            noise_len=8)
    # print(cfarRangeResult.shape)
    thresholdRange, noiseFloorRange = cfarRangeResult[...,0,:],cfarRangeResult[...,1,:]
    rangeCFAR = np.zeros(thresholdRange.shape, bool)
    rangeCFAR[dopplerResultInDB>thresholdRange] = True
    # input()
    cfarDopplerResult = np.apply_along_axis(func1d=ca_,
                                            axis=0,
                                            arr=dopplerResultInDB,
                                            l_bound=25,
                                            guard_len=4,
                                            noise_len=8)
    # print(cfarDopplerResult.shape)
    thresholdDoppler, noiseFloorDoppler = cfarDopplerResult[...,0,:,:],cfarDopplerResult[...,1,:,:]
    
    dopplerCFAR = np.zeros(thresholdDoppler.shape, bool)
    dopplerCFAR[dopplerResultInDB>thresholdDoppler] = True

    cfarResult1 = rangeCFAR|dopplerCFAR 

    rangeResult = rangeFFT(reshapedFrame,frameConfig)
   
    
    dopplerResult = dopplerFFT(rangeResult,frameConfig)
    # sum all antenna to get a range*doppler array for cfar use
    dopplerResultSumAllAntenna = np.sum(dopplerResult, axis=(0,1))
    # transform the complex value array to DB value array  
    dopplerResultInDB = 20*np.log10(np.absolute(dopplerResultSumAllAntenna))
    # another method to get log2 value
    # dopplerResultInDB = 10*np.log2(np.absolute(dopplerResult))

    cfarRangeResult = np.apply_along_axis(func1d=ca_,
                                            axis=1,
                                            arr=dopplerResultInDB,
                                            l_bound=25,
                                            guard_len=4,
                                            noise_len=8)
    # print(cfarRangeResult.shape)
    thresholdRange, noiseFloorRange = cfarRangeResult[...,0,:],cfarRangeResult[...,1,:]
    rangeCFAR = np.zeros(thresholdRange.shape, bool)
    rangeCFAR[dopplerResultInDB>thresholdRange] = True
    # input()
    cfarDopplerResult = np.apply_along_axis(func1d=ca_,
                                            axis=0,
                                            arr=dopplerResultInDB,
                                            l_bound=25,
                                            guard_len=4,
                                            noise_len=8)
    # print(cfarDopplerResult.shape)
    thresholdDoppler, noiseFloorDoppler = cfarDopplerResult[...,0,:,:],cfarDopplerResult[...,1,:,:]
    
    dopplerCFAR = np.zeros(thresholdDoppler.shape, bool)
    dopplerCFAR[dopplerResultInDB>thresholdDoppler] = True
    cfarResult = ~cfarResult1
    cfarResult2 = rangeCFAR|dopplerCFAR
    cfarResult = cfarResult&cfarResult2

    AOAInput = dopplerResult[:,:,cfarResult==True]
    AOAInput = AOAInput.reshape(12,-1)
    print("AOAInput:",AOAInput.shape,len(AOAInput))
    if AOAInput.shape[1]==0:
        return np.array([]).reshape(6,0)
    x_vec, y_vec, z_vec = naive_xyz(AOAInput)
    det_peaks_indices = np.argwhere(cfarResult == True)
    SNR = dopplerResultInDB - noiseFloorRange
    SNR = SNR[cfarResult==True]
    print("SNR shape",SNR.shape)

    R = det_peaks_indices[:,1].astype(np.float64)
    V = (det_peaks_indices[:,0]-frameConfig.numDopplerBins//2).astype(np.float64)
    if pointCloudProcessCFG.outputInMeter:
        R *= cfg.RANGE_RESOLUTION
        V *= cfg.DOPPLER_RESOLUTION
    print(R.shape)
    # print(S.shape)
    # print(R)
    # print(S)
    x,y,z = x_vec*R, y_vec*R, z_vec*R
    pointCloud=np.concatenate((x,y,z,V,SNR,R))
    print("pointCloud",pointCloud.shape)
    pointCloud = np.reshape(pointCloud,(6,-1))
    print(pointCloud.shape)
    pointCloud = pointCloud[:,y_vec!=0]      
    print(pointCloud.shape) 

    return cfarResult2,cfarResult1,pointCloud
    
if __name__ == '__main__':
    compare = False

    pointCloudProcessCFG = PointCloudProcessCFG()
    originalfig = plt.figure("orgin")
    if compare == True:
        originalpointCloudProcessCFG = PointCloudProcessCFG()
        originalpointCloudProcessCFG.enableCouplingSignatureRemoval = False
        originalpointCloudProcessCFG.enableStaticClutterRemoval = False        
        comparefig = plt.figure("diff")

    frameConfig = pointCloudProcessCFG.frameConfig
    dataPath = "adc_data_Raw_0.bin"
    reader = RawDataReader(dataPath)
    pointCloudProcessCFG.calculateCouplingSignatureArray(dataPath)

    print3Dfig = True

    if print3Dfig == True:
        fig = plt.figure("new")
        plt.ion()
        elev = 0
        azim = 0
    while True:
        frame = reader.getNextFrame(frameConfig)
        pointCloud = frame2pointcloud(frame,pointCloudProcessCFG)

        if compare == True:
            originalpointCloud = frame2pointcloud(frame,originalpointCloudProcessCFG)
            originalcfarResult,newcfarResult,comparepointCloud = compareframe2pointcloud(frame,pointCloudProcessCFG)            
        
        if print3Dfig == True:
            if compare == True:
                comparefig.clf()
                gs = comparefig.add_gridspec(3, 4)
                pointCloudSubplot = comparefig.add_subplot(gs[0:3,0:3],projection="3d")
                pointCloudSubplot.view_init(elev, azim)
                color = comparepointCloud[pointCloudProcessCFG.velocityDim]
                scale = 4
                x = comparepointCloud[0]
                y = comparepointCloud[1]
                z = comparepointCloud[2]
                pointCloudSubplot.scatter(x, y, z, s=scale, c=color, marker=".")

                # 设置坐标轴图标
                pointCloudSubplot.set_xlabel("X Label")
                pointCloudSubplot.set_ylabel("Y Label")
                pointCloudSubplot.set_zlabel("Z Label")

                # 设置坐标轴范围
                xlimmax = 200
                ylimmax = 150
                zlimmax = 200
                if pointCloudProcessCFG.outputInMeter:
                    xlimmax = cfg.RANGE_RESOLUTION*cfg.NUM_RANGE_BINS
                    ylimmax = cfg.RANGE_RESOLUTION*cfg.NUM_RANGE_BINS
                    zlimmax = cfg.RANGE_RESOLUTION*cfg.NUM_RANGE_BINS                

                pointCloudSubplot.set_xlim(-xlimmax, xlimmax)
                pointCloudSubplot.set_ylim(0, ylimmax)
                pointCloudSubplot.set_zlim(-zlimmax, zlimmax)

                XZplot = comparefig.add_subplot(gs[0,-1])
                XZplot.scatter(x, z, s=scale, c=color, marker=".")
                XZplot.set_xlabel("X Label")
                XZplot.set_ylabel("Z Label")
                XZplot.set_xlim(-xlimmax, xlimmax)            
                XZplot.set_ylim(-zlimmax, zlimmax)

                YZplot = comparefig.add_subplot(gs[1,-1])
                YZplot.scatter(y, z, s=scale, c=color, marker=".")
                YZplot.set_xlabel("Y Label")
                YZplot.set_ylabel("Z Label")
                YZplot.set_xlim(0, ylimmax)            
                YZplot.set_ylim(-zlimmax, zlimmax)

                XYplot = comparefig.add_subplot(gs[2,-1])
                XYplot.scatter(x, y, s=scale, c=color, marker=".")
                XYplot.set_xlabel("X Label")
                XYplot.set_ylabel("Y Label")
                XYplot.set_xlim(-xlimmax, xlimmax)            
                XYplot.set_ylim(0, ylimmax)  

                
                gs = originalfig.add_gridspec(4, 4)
                originalfig.clf()
                originalfig.suptitle("original")
                pointCloudSubplot = originalfig.add_subplot(gs[0:3,0:3], projection="3d")
                pointCloudSubplot.view_init(elev, azim)
                color = originalpointCloud[pointCloudProcessCFG.velocityDim]

                scale = 4

                x = originalpointCloud[0]
                y = originalpointCloud[1]
                z = originalpointCloud[2]
                
                pointCloudSubplot.scatter(x, y, z, s=scale, c=color, marker=".")

                # 设置坐标轴图标
                pointCloudSubplot.set_xlabel("X Label")
                pointCloudSubplot.set_ylabel("Y Label")
                pointCloudSubplot.set_zlabel("Z Label")

                # 设置坐标轴范围
                xlimmax = 200
                ylimmax = 150
                zlimmax = 200
                if pointCloudProcessCFG.outputInMeter:
                    xlimmax = cfg.RANGE_RESOLUTION*cfg.NUM_RANGE_BINS
                    ylimmax = cfg.RANGE_RESOLUTION*cfg.NUM_RANGE_BINS
                    zlimmax = cfg.RANGE_RESOLUTION*cfg.NUM_RANGE_BINS
                

                pointCloudSubplot.set_xlim(-xlimmax, xlimmax)
                pointCloudSubplot.set_ylim(0, ylimmax)
                pointCloudSubplot.set_zlim(-zlimmax, zlimmax)
                
                
                XZplot = originalfig.add_subplot(gs[0,-1])
                XZplot.scatter(x, z, s=scale, c=color, marker=".")
                XZplot.set_xlabel("X Label")
                XZplot.set_ylabel("Z Label")
                XZplot.set_xlim(-xlimmax, xlimmax)            
                XZplot.set_ylim(-zlimmax, zlimmax)

                YZplot = originalfig.add_subplot(gs[1,-1])
                YZplot.scatter(y, z, s=scale, c=color, marker=".")
                YZplot.set_xlabel("Y Label")
                YZplot.set_ylabel("Z Label")
                YZplot.set_xlim(0, ylimmax)            
                YZplot.set_ylim(-zlimmax, zlimmax)

                XYplot = originalfig.add_subplot(gs[2,-1])
                XYplot.scatter(x, y, s=scale, c=color, marker=".")
                XYplot.set_xlabel("X Label")
                XYplot.set_ylabel("Y Label")
                XYplot.set_xlim(-xlimmax, xlimmax)            
                XYplot.set_ylim(0, ylimmax)  

                originalcfarplot = originalfig.add_subplot(gs[3,:2])
                originalcfarplot.matshow(originalcfarResult)
                cfarplot = originalfig.add_subplot(gs[3,2:])
                cfarplot.matshow(newcfarResult)
                # 暂停
                azim=pointCloudSubplot.azim
                elev=pointCloudSubplot.elev

            # 清空图像
            fig.clf()
            # 设定标题
            fig.suptitle("static removed")

            gs = originalfig.add_gridspec(3, 4)
            # 生成画布
            # add_subplot(子图 总行数，总列数，此图位置)
            pointCloudSubplot = fig.add_subplot(gs[0:3,0:3], projection="3d")
            pointCloudSubplot.view_init(elev, azim)

            # 画三维散点图
            color = pointCloud[pointCloudProcessCFG.velocityDim]
            pointCloud[pointCloudProcessCFG.SNRDim]
            pointCloud[pointCloudProcessCFG.rangeDim]

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
            if pointCloudProcessCFG.outputInMeter:
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
