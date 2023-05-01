import numpy
from sklearn.model_selection import train_test_split
import os
import librosa
from librosa import feature
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.mixture import GaussianMixture
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

def load_directory(name):
    file = os.listdir(name)
    train, test = train_test_split(file, test_size=.3, random_state=42)
    new_train = []
    for x in train:
        new_name = name + '/' + x
        x = new_name
        new_train.append(x)

    new_test = []
    for x in test:
        new_name = name + '/' + x
        x = new_name
        new_test.append(x)

    return new_train, new_test


def get_labels(files):
    label_list = []
    for x in files:
        substrings = x.split('_')
        label_list.append(substrings[1])
    return label_list


def get_frequency(file1, file2):
    all_labels = file1 + file2
    label_counts = pd.Series(all_labels).value_counts()
    df = pd.DataFrame({'label': label_counts.index, 'frequency': label_counts.values})
    return df


def plot_time_domain(file, title):
    signal, sample_rate = librosa.load(file)
    plt.figure(1)
    librosa.display.waveshow(y=signal, sr=sample_rate)
    plt.title(title)
    plt.xlabel('Time / second')
    plt.ylabel('Amplitude')
    plt.show()


def plot_frequency_domain(file, title):
    signal, sample_rate = librosa.load(file)
    k = np.arange(len(signal))
    T = len(signal) / sample_rate
    freq = k / T

    DATA_0 = np.fft.fft(signal)
    abs_DATA_0 = abs(DATA_0)
    plt.figure(2)
    plt.plot(freq, abs_DATA_0)
    plt.title(title)
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


def extract_audio_features(file):
    signal, sample_rate = librosa.load(file)

    # audio feature extraction: loudness
    df_loudness = pd.DataFrame()
    S, phase = librosa.magphase(librosa.stft(signal))
    rms = librosa.feature.rms(S=S)
    df_loudness['Loudness'] = rms[0]

    # audio feature extraction: mel-frequency cepstral coefficients
    df_mfccs = pd.DataFrame()
    mfccs = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=12)
    for n_mfcc in range(len(mfccs)):
        df_mfccs['MFCC_%d' % (n_mfcc + 1)] = mfccs.T[n_mfcc]

    # audio feature extraction: zero crossing rate
    df_zero_crossing_rate = pd.DataFrame()
    zcr = librosa.feature.zero_crossing_rate(y=signal)
    df_zero_crossing_rate['ZCR'] = zcr[0]

    # audio feature extraction: chroma
    df_chroma = pd.DataFrame()
    chromagram = librosa.feature.chroma_stft(y=signal, sr=sample_rate)
    for n_chroma in range(len(chromagram)):
        df_chroma['Chroma_%d' % (n_chroma + 1)] = chromagram.T[n_chroma]

    # audio feature extraction: mel spectrogram
    df_mel_spectrogram = pd.DataFrame()
    mel_spectrogram = librosa.feature.melspectrogram(y=signal, sr=sample_rate, n_mels=12)
    for n_mel in range(len(mel_spectrogram)):
        df_mel_spectrogram['Mel_Spectrogram_%d' % (n_mel + 1)] = mel_spectrogram.T[n_mel]

    # combine all features
    feature_matrix = pd.concat([df_loudness, df_mfccs, df_zero_crossing_rate, df_chroma, df_mel_spectrogram], axis=1)
    return feature_matrix

def extract_y (file):
    string = ""
    if file.find("angry"):
        string = "angry"
    elif file.find("fear"):
        string = "fear"
    elif file.find("happy"):
        string = "happy"
    elif file.find("sad"):
        string = "sad"
    return string


#Step 1: Split training and testing sets
angry_train, angry_test = load_directory('angry')
fear_train, fear_test = load_directory('fear')
happy_train, happy_test = load_directory('happy')
sad_train, sad_test = load_directory('sad')

#need to assemble into test and train files
test = angry_test + fear_test + happy_test + sad_test
train = angry_train + fear_train + happy_train + sad_train
test_y = []
for x in range(0, 30):
    test_y.append('angry')
for x in range(0, 30):
    test_y.append('fear')
for x in range(0, 30):
    test_y.append('happy')
for x in range(0, 30):
    test_y.append('sad')

train_y = []
for x in range(0, 70):
    train_y.append('angry')
for x in range(0, 70):
    train_y.append('fear')
for x in range(0, 70):
    train_y.append('happy')
for x in range(0, 70):
    train_y.append('sad')


#Step 2: Exploratory Data Analysis
#Label distribution
test_labels = get_labels(test)
train_labels = get_labels(train)

#get the labels, create a map with label and frequency
label_frequency = get_frequency(test_labels, train_labels)
print(label_frequency)

#compare each emotion for the word "vine"
angry_vine = 'angry/YAF_vine_angry.wav'
fear_vine = 'fear/YAF_vine_fear.wav'
happy_vine = 'happy/YAF_vine_happy.wav'
sad_vine = 'sad/YAF_vine_sad.wav'

plot_time_domain(angry_vine, 'Angry')
plot_time_domain(fear_vine, 'Fear')
plot_time_domain(happy_vine, 'Happy')
plot_time_domain(sad_vine, 'Sad')

plot_frequency_domain(angry_vine, 'Angry')
plot_frequency_domain(fear_vine, 'Fear')
plot_frequency_domain(happy_vine, 'Happy')
plot_frequency_domain(sad_vine, 'Sad')

#Step 3: Acoustic Feature Extraction
list_matrix_train = []

for x in train:
    new_matrix = extract_audio_features(x)
    list_matrix_train.append(new_matrix)

list_matrix_test = []
for x in test:
    new_matrix = extract_audio_features(x)
    list_matrix_test.append(extract_audio_features(x))

#Step 4: Feature Post-Processing
scaler = MinMaxScaler(feature_range=(-1, 1))

list_normalized_test = []
list_features_test_avg = []
for matrix in list_matrix_test:
    df = matrix
    feature_vector = []
    for column in df.columns:
        df[column] = df[column] / df[column].abs().max()
        average = df[column].mean()
        feature_vector.append(average)
    list_normalized_test.append(df)
    list_features_test_avg.append(feature_vector)

list_normalized_train = []
list_features_train_avg = []
for matrix in list_matrix_train:
    df = matrix
    feature_vector = []
    for column in df.columns:
        df[column] = df[column] / df[column].abs().max()
        average = df[column].mean()
        feature_vector.append(average)
    list_normalized_train.append(df)
    list_features_train_avg.append(feature_vector)

test_x = list_features_test_avg
train_x = list_features_train_avg

#Step 5: Build Audio Emotion Recognition Model and Step 6:Model Evaluation
#output is either angry, fear, happy, sad

#x_test is the list of vectors
#y_test is the labels assigned to the values

#two different classifiers using same acoustic features
#random forest
rf = RandomForestClassifier(n_estimators=100, random_state=0)
rf.fit(train_x, train_y)

y_pred = rf.predict(test_x)

print(classification_report(test_y, y_pred))

#SVC
svm = SVC(kernel='linear', C=1, random_state=0)
svm.fit(train_x, train_y)

y_pred = svm.predict(test_x)
print(classification_report(test_y, y_pred))

#same classifier using different acoustic features