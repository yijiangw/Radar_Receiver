
NUM_TX = 3
NUM_RX = 4


START_FREQ = 77 
ADC_START_TIME = 6 
FREQ_SLOPE = 29.982 
ADC_SAMPLES = 300
SAMPLE_RATE = 3000
RX_GAIN = 30 

IDLE_TIME = 50
RAMP_END_TIME = 107
#  2 for 1843
NUM_FRAMES = 0 
#  Set this to 0 to continuously stream data
CHIRP_LOOPS = 50
PERIODICITY = 100 
# time for one chirp in ms  100ms == 10FPS
NUM_DOPPLER_BINS = CHIRP_LOOPS
NUM_RANGE_BINS = ADC_SAMPLES
RANGE_RESOLUTION = (3e8 * SAMPLE_RATE * 1e3) / (2 * FREQ_SLOPE * 1e12 * ADC_SAMPLES)
MAX_RANGE = (300 * 0.9 * SAMPLE_RATE) / (2 * FREQ_SLOPE * 1e3)
DOPPLER_RESOLUTION = 3e8 / (2 * START_FREQ * 1e9 * (IDLE_TIME + RAMP_END_TIME) * 1e-6 * NUM_DOPPLER_BINS * NUM_TX)
MAX_DOPPLER = 3e8 / (4 * START_FREQ * 1e9 * (IDLE_TIME + RAMP_END_TIME) * 1e-6 * NUM_TX)