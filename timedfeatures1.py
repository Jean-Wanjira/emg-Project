import serial
import numpy as np
from scipy.signal import butter, filtfilt
import time
import csv
from datetime import datetime

# Bandpass filter
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

# Normalize signal
def normalize_signal(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

# Feature extraction functions
def mean_absolute_value(data):
    return np.mean(np.abs(data))

def root_mean_square(data):
    return np.sqrt(np.mean(data**2))

def variance(data):
    return np.var(data)

def waveform_length(data):
    return np.sum(np.abs(np.diff(data)))

def slope_sign_changes(data, threshold=0.01):
    return np.sum(np.diff(np.sign(np.diff(data))) != 0)

def integrated_emg(data):
    return np.sum(np.abs(data))

def mean_frequency(data, fs):
    f = np.fft.fftfreq(len(data), 1/fs)
    Y = np.fft.fft(data)
    power_spectrum = np.abs(Y)**2
    return np.sum(f * power_spectrum) / np.sum(power_spectrum)

def median_frequency(data, fs):
    f = np.fft.fftfreq(len(data), 1/fs)
    Y = np.fft.fft(data)
    power_spectrum = np.abs(Y)**2
    cumulative_power = np.cumsum(power_spectrum)
    return f[np.where(cumulative_power >= cumulative_power[-1] / 2)[0][0]]

# Setup serial connection
try:
    ser = serial.Serial('COM6', 9600)  # Update COM port as needed
    time.sleep(2)  # Wait for the serial connection to initialize
except serial.SerialException as e:
    print(f"Error opening serial port: {e}")
    exit()

# Parameters
fs = 1000  # Sampling frequency (adjust based on your setup)
lowcut = 20.0
highcut = 450.0
window_size = 100  # Number of samples per window for processing
flex_threshold = 0.5  # Threshold to determine muscle flexed or relaxed

# Real-time data processing
emg_data_window = []
state = 'Relaxed'
state_start_time = datetime.now()

# Open CSV file for writing
try:
    csv_file = open('emg_features_with_state_timing1.csv', mode='w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Timestamp', 'State', 'State Start Time', 'State Duration',
                         'MAV', 'RMS', 'Variance', 'Waveform Length', 'SSC', 'IEMG', 'Mean Frequency', 'Median Frequency'])
except IOError as e:
    print(f"Error opening CSV file: {e}")
    ser.close()
    exit()

print("Flex your muscle now or relax to test the system...")

while True:
    if ser.in_waiting > 0:
        try:
            emg_value = int(ser.readline().decode('latin-1').strip())
            emg_data_window.append(emg_value)

            if len(emg_data_window) >= window_size:
                emg_data = np.array(emg_data_window)
                emg_data_baseline_removed = emg_data - np.mean(emg_data)
                emg_data_filtered = apply_bandpass_filter(emg_data_baseline_removed, lowcut, highcut, fs)
                emg_data_normalized = normalize_signal(emg_data_filtered)

                # Determine the current state based on the average normalized value
                current_muscle_activity = np.mean(emg_data_normalized)
                new_state = 'Flexed' if current_muscle_activity > flex_threshold else 'Relaxed'

                # Check if the state has changed
                if new_state != state:
                    # Record state change
                    state_end_time = datetime.now()
                    state_duration = (state_end_time - state_start_time).total_seconds()
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                    csv_writer.writerow([timestamp, new_state, state_start_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                                         state_duration, mean_absolute_value(emg_data_normalized), root_mean_square(emg_data_normalized),
                                         variance(emg_data_normalized), waveform_length(emg_data_normalized),
                                         slope_sign_changes(emg_data_normalized), integrated_emg(emg_data_normalized),
                                         mean_frequency(emg_data_normalized, fs), median_frequency(emg_data_normalized, fs)])
                    
                    # Update state and timestamp
                    state = new_state
                    state_start_time = state_end_time

                # Clear the window for the next set of samples
                emg_data_window = []
        except ValueError:
            print("Error decoding or converting data. Skipping this sample.")
        except UnicodeDecodeError:
            print("Unicode decode error. Skipping this sample.")

ser.close()
csv_file.close()
