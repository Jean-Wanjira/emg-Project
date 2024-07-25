import serial
import numpy as np
from scipy.signal import butter, filtfilt
import time
import csv

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
    data_min = np.min(data)
    data_max = np.max(data)
    if data_max != data_min:
        return (data - data_min) / (data_max - data_min)
    else:
        return data  # or return np.zeros_like(data) if you prefer

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
    if len(cumulative_power) > 0:
        return f[np.where(cumulative_power >= cumulative_power[-1] / 2)[0][0]]
    else:
        return 0  # or another appropriate value

# Setup serial connection
ser = serial.Serial('COM6', 9600)  # Update COM port as needed
time.sleep(2)  # Wait for the serial connection to initialize

# Parameters
fs = 1000  # Sampling frequency (adjust based on your setup)
lowcut = 20.0
highcut = 450.0
window_size = 100  # Number of samples per window for processing

# Real-time data processing
emg_data_window = []

# Open CSV file for writing
csv_file = open('emg_features_with_gestures3.csv', mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Raw Data', 'Baseline Removed Data', 'Filtered Data', 'Normalized Data', 
                     'MAV', 'RMS', 'Variance', 'Waveform Length', 'SSC', 'IEMG', 'Mean Frequency', 'Median Frequency', 'Gesture'])

# Define the gesture labels here
gestures = ["Relaxed", "Flexing", "Holding Flex", "Extending"]
current_gesture = "Relaxed"  # Initial gesture, update as needed during your experiment

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

                # Extract features
                mav = mean_absolute_value(emg_data_normalized)
                rms = root_mean_square(emg_data_normalized)
                var = variance(emg_data_normalized)
                wl = waveform_length(emg_data_normalized)
                ssc = slope_sign_changes(emg_data_normalized)
                iemg = integrated_emg(emg_data_normalized)
                mnf = mean_frequency(emg_data_normalized, fs)
                mdf = median_frequency(emg_data_normalized, fs)

                # Append data to CSV file
                for i in range(len(emg_data)):
                    csv_writer.writerow([emg_data[i], emg_data_baseline_removed[i], emg_data_filtered[i], emg_data_normalized[i],
                                         mav, rms, var, wl, ssc, iemg, mnf, mdf, current_gesture])

                # Clear the window for next set of samples
                emg_data_window = []
        except ValueError:
            print("Error decoding or converting data. Skipping this sample.")
        except UnicodeDecodeError:
            print("Unicode decode error. Skipping this sample.")

ser.close()
csv_file.close()
