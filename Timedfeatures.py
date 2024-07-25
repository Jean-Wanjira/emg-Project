import serial
import numpy as np
import matplotlib.pyplot as plt
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

# Function to collect data from Arduino with labeling
def collect_data_with_labels(ser, duration_per_state, fs):
    data = []
    labels = []
    
    # Collect relaxed data
    print("Collecting relaxed data...")
    start_time = time.time()
    while time.time() - start_time < duration_per_state:
        if ser.in_waiting > 0:
            try:
                emg_value = int(ser.readline().decode('latin-1').strip())
                data.append(emg_value)
                labels.append('Relaxed')
            except ValueError:
                continue
            except UnicodeDecodeError:
                continue
    
    # Collect flexed data
    print("Collecting flexed data...")
    start_time = time.time()
    while time.time() - start_time < duration_per_state:
        if ser.in_waiting > 0:
            try:
                emg_value = int(ser.readline().decode('latin-1').strip())
                data.append(emg_value)
                labels.append('Flexed')
            except ValueError:
                continue
            except UnicodeDecodeError:
                continue
    
    return np.array(data), labels

# Parameters
fs = 1000  # Sampling frequency
lowcut = 20.0
highcut = 450.0
duration_per_state = 10  # Duration to collect data for each state in seconds

# Setup serial connection
ser = serial.Serial('COM6', 9600)  # Update COM port as needed
time.sleep(2)  # Wait for the serial connection to initialize

# Collect data with labels
data, labels = collect_data_with_labels(ser, duration_per_state, fs)

# Apply bandpass filter
data_filtered = apply_bandpass_filter(data, lowcut, highcut, fs)

# Save data to CSV with labels
csv_file = open('emg_labeled_data7.csv', mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Raw Data', 'Filtered Data', 'Label'])

for i in range(len(data)):
    csv_writer.writerow([data[i], data_filtered[i], labels[i]])

csv_file.close()
ser.close()

print("Data collection and saving complete.")

# Compute statistical measures
relaxed_data_filtered = data_filtered[np.array(labels) == 'Relaxed']
flexed_data_filtered = data_filtered[np.array(labels) == 'Flexed']

relaxed_mean = np.mean(relaxed_data_filtered)
relaxed_std = np.std(relaxed_data_filtered)
flexed_mean = np.mean(flexed_data_filtered)
flexed_std = np.std(flexed_data_filtered)

# Determine threshold
threshold = relaxed_mean + 3 * relaxed_std

print(f"Relaxed Mean: {relaxed_mean}, Relaxed Std: {relaxed_std}")
print(f"Flexed Mean: {flexed_mean}, Flexed Std: {flexed_std}")
print(f"Determined Threshold: {threshold}")

# Plot data
plt.figure()
plt.plot(data_filtered[np.array(labels) == 'Relaxed'], label='Relaxed')
plt.plot(data_filtered[np.array(labels) == 'Flexed'], label='Flexed')
plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
plt.legend()
plt.title('EMG Data - Relaxed vs Flexed')
plt.xlabel('Samples')
plt.ylabel('Amplitude')
plt.show()
