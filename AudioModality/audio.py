from sklearn.model_selection import train_test_split
import os
import librosa
from librosa import feature
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def load_directory(name):
    file = os.listdir(name)
    train, test = train_test_split(file, test_size=.3, random_state=42)
    return train, test


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


def extract_audio_features(file):
    signal, sample_rate = librosa.load(file)
    # audio feature extraction: loudness
    df_loudness = pd.DataFrame()
    S, phase = librosa.magphase(librosa.stft(signal))
    rms = librosa.feature.rms(S=S)
    print(rms[0])
    df_loudness['Loudness'] = rms[0]
    print(df_loudness.head(5))
    plt.figure(4)
    times = librosa.times_like(rms)
    plt.plot(times, rms[0])
    plt.xlabel("Time / second")
    plt.ylabel("Amplitude")
    plt.show()
    #
    # audio feature extraction: mel-frequency cepstral coefficients
    df_mfccs = pd.DataFrame()
    mfccs = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=12)
    for n_mfcc in range(len(mfccs)):
        df_mfccs['MFCC_%d' % (n_mfcc + 1)] = mfccs.T[n_mfcc]
    print(df_mfccs.head(5))
    plt.figure(5)
    librosa.display.specshow(mfccs, sr=sample_rate, x_axis='time', y_axis='log')
    plt.show()

    # audio feature extraction: zero crossing rate
    df_zero_crossing_rate = pd.DataFrame()
    zcr = librosa.feature.zero_crossing_rate(y=signal)
    df_zero_crossing_rate['ZCR'] = zcr[0]
    print(df_zero_crossing_rate.head(5))
    plt.figure(6)
    times = librosa.times_like(zcr)
    plt.plot(times, zcr[0])
    plt.xlabel("Time / second")
    plt.ylabel("Zero Crossing Rate")
    plt.show()

    # audio feature extraction: chroma
    df_chroma = pd.DataFrame()
    chromagram = librosa.feature.chroma_stft(y=signal, sr=sample_rate)
    for n_chroma in range(len(chromagram)):
        df_chroma['Chroma_%d' % (n_chroma + 1)] = chromagram.T[n_chroma]
    print(df_chroma.head(5))
    plt.figure(7)
    librosa.display.specshow(chromagram, sr=sample_rate, x_axis='time', y_axis='log')
    plt.show()

    # audio feature extraction: mel spectrogram
    df_mel_spectrogram = pd.DataFrame()
    mel_spectrogram = librosa.feature.melspectrogram(y=signal, sr=sample_rate, n_mels=12)
    for n_mel in range(len(mel_spectrogram)):
        df_mel_spectrogram['Mel_Spectrogram_%d' % (n_mel + 1)] = mel_spectrogram.T[n_mel]
    print(df_mel_spectrogram.head(5))
    plt.figure(8)
    librosa.display.specshow(mel_spectrogram, sr=sample_rate, x_axis='time', y_axis='log')
    plt.show()

    # combine all features
    feature_matrix = pd.concat([df_loudness, df_mfccs, df_zero_crossing_rate, df_chroma, df_mel_spectrogram], axis=1)
    print(feature_matrix.head(5))
    feature_matrix.to_csv('feature_matrix.csv')


def plot_time_domain(file):
    signal, sample_rate = librosa.load(file)
    plt.figure(1)
    librosa.display.waveshow(y=signal, sr=sample_rate)
    plt.xlabel('Time / second')
    plt.ylabel('Amplitude')
    plt.show()


def plot_frequency_domain(file):
    signal, sample_rate = librosa.load(file)
    k = np.arange(len(signal))
    T = len(signal) / sample_rate
    freq = k / T

    DATA_0 = np.fft.fft(signal)
    abs_DATA_0 = abs(DATA_0)
    plt.figure(2)
    plt.plot(freq, abs_DATA_0)
    plt.xlabel("Frequency / Hz")
    plt.ylabel("Amplitude / dB")
    plt.xlim([0, 1000])
    plt.show()


def plot_time_frequency_variation(file):
    signal, sample_rate = librosa.load(file)
    D = librosa.stft(signal)  # STFT of y
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    plt.figure(3)
    librosa.display.specshow(S_db, x_axis='time', y_axis='log')
    plt.colorbar()
    plt.show()



#Step 1: Split training and testing sets
angry_train, angry_test = load_directory('angry')
fear_train, fear_test = load_directory('fear')
happy_train, happy_test = load_directory('happy')
sad_train, sad_test = load_directory('sad')

#need to assemble into test and train files
test = angry_test + fear_test + happy_test + sad_test
train = angry_train + fear_train + happy_train + sad_train

print (test)


#Step 2: Exploratory Data Analysis
#Label distribution
# angry_labels = get_labels(angry_files)
# fear_labels = get_labels(fear_files)
# sad_labels = get_labels(sad_files)
# happy_labels = get_labels(happy_files)

#get the labels, create a map with label and frequency
# label_frequency = get_frequency(angry_labels, fear_labels, sad_labels, happy_labels)
# print(label_frequency)