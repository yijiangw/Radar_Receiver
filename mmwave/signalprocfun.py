import cupy as cp
import numpy as np
import mmwave.radarconfig as cfg

from scipy.ndimage import convolve

import time
# from cupyx.scipy.ndimage import convolve
# from scipy.signal import convolve


def rangeFFT(frame,pointCloudProcConfig):    
    """ 1D FFT for range
    """
    windowedBins1D = cp.array(frame*np.blackman(pointCloudProcConfig.rangeBins))
    rangeFFTResult = cp.asnumpy(cp.fft.fft(windowedBins1D))

    # windowedBins1D = frame*np.blackman(pointCloudProcConfig.rangeBins)
    # rangeFFTResult=np.fft.fft(windowedBins1D)
    return rangeFFTResult

def rangeAndDopplerFFT(frame,pointCloudProcConfig):
    """ perform Range and Doppler FFT together
    """
    windowedBins2D = frame*np.reshape(np.blackman(pointCloudProcConfig.dopplerBins),(-1,1))*np.blackman(pointCloudProcConfig.rangeBins)
    dopplerFFTResult=np.fft.fft2(windowedBins2D)
   
    return dopplerFFTResult

def dopplerFFT(rangeResult,pointCloudProcConfig):    
    """ input result from rangeFFT and do 2D FFT for velocity
    """
    windowedBins2D = cp.array(rangeResult*np.reshape(np.blackman(pointCloudProcConfig.dopplerBins),(1,1,-1,1)))
    dopplerFFTResult=cp.fft.fft(windowedBins2D,axis=2)
    dopplerFFTResult=cp.asnumpy(cp.fft.fftshift(dopplerFFTResult,axes=2))
    
    # windowedBins2D = rangeResult*np.reshape(np.blackman(pointCloudProcConfig.dopplerBins),(1,1,-1,1))
    # dopplerFFTResult=np.fft.fft(windowedBins2D,axis=2)
    # dopplerFFTResult=np.fft.fftshift(dopplerFFTResult,axes=2)
    return dopplerFFTResult

# def AOAFFT(dopplerResult,pointCloudProcConfig): #num_tx=3, num_rx=4, fft_size=64):
#     dopplerResult.reshape()
#     azimuth_ant = dopplerResult[:2,...].reshape(2*cfg.NUM_RX,cfg.NUM_DOPPLER_BINS,cfg.NUM_RANGE_BINS)   
#     azimuth_ant_padded = np.zeros(shape=(pointCloudProcConfig.AngleBins,cfg.NUM_DOPPLER_BINS,cfg.NUM_RANGE_BINS), dtype=np.complex64)    
#     azimuth_ant_padded[:2 * cfg.NUM_RX, ...] = azimuth_ant

#     elevation_ant = dopplerResult[2,...]
#     elevation_ant_padded = np.zeros(shape=(pointCloudProcConfig.AngleBins, num_detected_obj), dtype=np.complex64)
#     elevation_ant_padded[:cfg.NUM_RX, :] = elevation_ant

def couplingSignatureRemoval(rangeFFTResult, pointCloudProcConfig):
    rangeFFTResult[:,:,:,:pointCloudProcConfig.couplingSignatureBinFrontIdx+1] -= pointCloudProcConfig.couplingSignatureArray[:,:,:,:pointCloudProcConfig.couplingSignatureBinFrontIdx+1]
    rangeFFTResult[:,:,:,int(-1*pointCloudProcConfig.couplingSignatureBinRearIdx):] -= pointCloudProcConfig.couplingSignatureArray[:,:,:,int(-1*pointCloudProcConfig.couplingSignatureBinRearIdx):]
    return rangeFFTResult

def clutter_removal(input_val, axis=0):
    # Reorder the axes
    reordering = np.arange(len(input_val.shape))
    reordering[0] = axis
    reordering[axis] = 0
    input_val = input_val.transpose(reordering)

    # Apply static clutter removal
    mean = input_val.mean(0)
    output_val = input_val - mean

    return output_val.transpose(reordering)

def cfar(x, guard_len=4, noise_len=8, mode='wrap'):
    kernel = np.ones(1 + (2 * guard_len) + (2 * noise_len), dtype=x.dtype) / (2 * noise_len)
    kernel[noise_len:noise_len + (2 * guard_len) + 1] = 0
    kernel = np.concatenate((np.zeros(kernel.shape[0],dtype=np.int32),kernel)).reshape(2,-1)
    noise_floor = convolve(x, kernel, mode=mode)

    return noise_floor

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
    # Zero pad azimuth
    azimuth_ant = virtual_ant[:2 * num_rx, :]
    azimuth_ant_padded = np.zeros(shape=(fft_size, num_detected_obj), dtype=np.complex_)
    azimuth_ant_padded[:2 * num_rx, :] = azimuth_ant

    # Process azimuth information
    azimuth_fft = np.fft.fft(azimuth_ant_padded, axis=0)
    azimuth_ffts = np.fft.fftshift(azimuth_fft,axes=0)
    azimuth_max = np.argmax(np.abs(azimuth_ffts), axis=0)

    peak_1=azimuth_ffts[azimuth_max,np.arange(num_detected_obj)]
    azimuth_max -= fft_size//2

    wx = 2 * np.pi / fft_size * azimuth_max  
    x_vector = wx / np.pi

    # Zero pad elevation
    elevation_ant = virtual_ant[2 * num_rx:, :]
    elevation_ant_padded = np.zeros(shape=(fft_size, num_detected_obj), dtype=np.complex_)
    elevation_ant_padded[:num_rx, :] = elevation_ant

    # Process elevation information
    elevation_fft = np.fft.fft(elevation_ant, axis=0)
    elevation_max = np.argmax(np.abs(elevation_fft), axis=0)    
    peak_2 =elevation_fft[elevation_max,np.arange(num_detected_obj)]    

    # Calculate elevation phase shift
    wz = np.angle(peak_1 * peak_2.conj() * np.exp(1j * 2 * wx))
    z_vector = wz / np.pi
    ypossible = 1 - x_vector ** 2 - z_vector ** 2
    y_vector=ypossible
    y_vector[ ypossible<0 ] = 0
    x_vector[ ypossible<0 ] = 0
    z_vector[ ypossible<0 ] = 0
    y_vector = np.sqrt(y_vector)
    return x_vector, y_vector, z_vector

def frame2pointcloud(frame,pointCloudProcConfig):    
    # time0 = time.time()
    rangeResult = rangeFFT(frame,pointCloudProcConfig)

    if pointCloudProcConfig.reFFTBeforeAOA == True:
        rangeResultForReFFT = rangeResult

    if pointCloudProcConfig.enableCouplingSignatureRemoval and pointCloudProcConfig.couplingSignatureArray.any():
        rangeResult = couplingSignatureRemoval(rangeResult,pointCloudProcConfig)

    if pointCloudProcConfig.enableStaticClutterRemoval:
        rangeResult = clutter_removal(rangeResult,axis=2)
    
    dopplerResult = dopplerFFT(rangeResult,pointCloudProcConfig)
    # sum all antenna to get a range*doppler array for cfar use
    dopplerResultSumAllAntenna = np.sum(dopplerResult, axis=(0,1))
    # transform the complex value array to DB value array  
    dopplerResultInDB = 20*np.log10(np.absolute(dopplerResultSumAllAntenna))
    # another method to get log2 value
    # dopplerResultInDB = 10*np.log2(np.absolute(dopplerResult))
    # time1 = time.time()

    # cfarDopplerResult = np.apply_along_axis(func1d=cfar_ca,
    #                                         axis=0,
    #                                         arr=dopplerResultInDB,
    #                                         guard_len=4,
    #                                         noise_len=8)
    cfarDopplerResult = cfar(dopplerResultInDB.transpose(),guard_len=4,noise_len=8).transpose()
    
    if pointCloudProcConfig.reFFTBeforeAOA == True:
        dopplerResult = dopplerFFT(rangeResultForReFFT,pointCloudProcConfig)
        dopplerResultSumAllAntenna = np.sum(dopplerResult, axis=(0,1))
        dopplerResultInDB = 20*np.log10(np.absolute(dopplerResultSumAllAntenna))
        
    # cfarRangeResult = np.apply_along_axis(func1d=cfar_ca,
    #                                         axis=1,
    #                                         arr=dopplerResultInDB,
    #                                         guard_len=4,
    #                                         noise_len=8)

    cfarRangeResult = cfar(dopplerResultInDB,guard_len=4,noise_len=8)
    # print(cfarDopplerResult.shape)
    # print(cfarRangeResult.shape)
    
    # l_bound = 25
    # thresholdRange = cfarRangeResult+l_bound 
    # thresholdDoppler = cfarDopplerResult+l_bound
    
    rangeCFAR = np.zeros(dopplerResultInDB.shape, bool)
    dopplerCFAR = np.zeros(dopplerResultInDB.shape, bool)
    rangeCFARElite = np.zeros(dopplerResultInDB.shape, bool)
    dopplerCFARlite= np.zeros(dopplerResultInDB.shape, bool)
    SNRRange = (dopplerResultInDB - cfarRangeResult)
    SNRDoppler = (dopplerResultInDB - cfarDopplerResult)
    # time2 = time.time()
    if (pointCloudProcConfig.CFARresultFilterTop):

        thresholdRange = np.percentile(SNRRange,pointCloudProcConfig.rangeCFARTopScale*100,interpolation='lower')
        thresholdDoppler = np.percentile(SNRDoppler,pointCloudProcConfig.dopplerCFARTopScale*100,interpolation='lower')
        thresholdRangeElite = np.percentile(SNRRange,(1-0.4*(1-pointCloudProcConfig.rangeCFARTopScale))*100,interpolation='lower')
        thresholdDopplerElite = np.percentile(SNRRange,(1-0.2*(1-pointCloudProcConfig.dopplerCFARTopScale))*100,interpolation='lower')
        rangeCFARElite[SNRRange>thresholdRangeElite] =True
        dopplerCFARlite[SNRDoppler>thresholdDopplerElite] =True
        rangeCFAR[SNRRange>thresholdRange] = True
        dopplerCFAR[SNRDoppler>thresholdDoppler] = True
    else:
        rangeCFAR[dopplerResultInDB>thresholdRange] = True
        dopplerCFAR[dopplerResultInDB>thresholdDoppler] = True
    # time3 = time.time()
    cfarResult = (rangeCFAR&dopplerCFAR)|rangeCFARElite|dopplerCFARlite
    det_peaks_indices = np.argwhere(cfarResult == True)
    # record range and velocity of detected points
    R = det_peaks_indices[:,1].astype(np.float64)
    V = (det_peaks_indices[:,0]-pointCloudProcConfig.dopplerBins//2).astype(np.float64)
    if pointCloudProcConfig.outputInMeter:
        R *= cfg.RANGE_RESOLUTION
        V *= cfg.DOPPLER_RESOLUTION
    # print("cfarshape",R.shape)
    # record rangeCFAR SNR of detected points  
    SNR_R = SNRRange[cfarResult==True]
    SNR_D = SNRDoppler[cfarResult==True]
    # print("SNR shape",SNR.shape)
    if pointCloudProcConfig.reFFTBeforeAOA == True:
        dopplerResult = dopplerFFT(rangeResultForReFFT,pointCloudProcConfig)

    AOAInput = dopplerResult[:,:,cfarResult==True]
    AOAInput = AOAInput.reshape(12,-1)
    if pointCloudProcConfig.enableDopplerCompensation:
        AOAInput *= pointCloudProcConfig.dopplerCompensationTable[:,det_peaks_indices[:,0]]

    # print("AOAInput:",AOAInput.shape,len(AOAInput))
    if AOAInput.shape[1]==0:
        # print("no cfar det point")
        pm(dopplerResult[0,0,...])
        pm(dopplerResultSumAllAntenna)
        pm(dopplerResultInDB)
        pm(noiseFloorDoppler)
        return np.array([]).reshape(pointCloudProcConfig.outputDim,0)
        
    x_vec, y_vec, z_vec = naive_xyz(AOAInput)   
    # time4 = time.time()
    # print(S.shape)
    # print(R)
    # print(S)
    x,y,z = x_vec*R, y_vec*R, z_vec*R
    pointCloud=np.concatenate((x,y,z,V,SNR_R,SNR_D,R))
    pointCloud = np.reshape(pointCloud,(pointCloudProcConfig.outputDim,-1))
    # print(pointCloud.shape)
    pointCloud = pointCloud[:,y_vec!=0]      
    # print(pointCloud.shape) 
    # print("pointCloud",pointCloud.shape)
    # time5 = time.time()

    # print("1:", time1-time0)    
    # print("2:", time2-time1)    
    # print("3:", time3-time2)    
    # print("4:", time4-time3)    
    # print("5:", time5-time4)
    return pointCloud


