from sklearn.model_selection import train_test_split
import os
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from scipy import signal
import numpy
def get_labels(files):
    label_list = []
    for x in files:
        substrings = x.split('_')
        label_list.append(substrings[1])
    return label_list

def get_frequency(file1, file2, file3, file4):
    all_labels = file1 + file2 + file3 + file4
    label_counts = pd.Series(all_labels).value_counts()
    df = pd.DataFrame({'label': label_counts.index, 'frequency': label_counts.values})
    return df


#Step 1: Split training and testing sets
angry_path = 'angry'
angry_files = os.listdir(angry_path)
angry_train, angry_test = train_test_split(angry_files, test_size=.3, random_state=42)

fear_path = 'fear'
fear_files = os.listdir(fear_path)
fear_train, fear_test = train_test_split(fear_files, test_size=.3, random_state=42)

happy_path = 'happy'
happy_files = os.listdir(happy_path)
happy_train, happy_test = train_test_split(happy_files, test_size=.3, random_state=42)

sad_path = 'sad'
sad_files = os.listdir(sad_path)
sad_train, sad_test = train_test_split(sad_files, test_size=.3, random_state=42)

#Step 2: Exploratory Data Analysis

#Label distribution
angry_labels = get_labels(angry_files)
fear_labels = get_labels(fear_files)
sad_labels = get_labels(sad_files)
happy_labels = get_labels(happy_files)

#get the labels, create a map with label and frequency
label_frequency = get_frequency(angry_labels, fear_labels, sad_labels, happy_labels)
print(label_frequency)

#plot in time
sample_rate, audio_data = wav.read('angry/YAF_merge_angry.wav')
plt.figure(figsize=(15, 5))
plt.plot(audio_data)
plt.title('Audio Waveform')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.show()
