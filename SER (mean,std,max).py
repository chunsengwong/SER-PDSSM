import librosa
from librosa import feature
import soundfile
import os, glob, pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

#extract features [mean,std,max]
def extractFeature(fileName, mfcc, chroma, mel):
    with soundfile.SoundFile(fileName) as soundFile:
        X = soundFile.read(dtype="float32")
        sampleRate = soundFile.samplerate
        if chroma:
            stft = np.abs(librosa.stft(X))
        result = np.array([])
        if mfcc:
            mfccs = librosa.feature.mfcc(y=X, sr=sampleRate, n_mfcc=40).T
            result = np.hstack((result, np.mean(mfccs, axis=0), np.std(mfccs, axis=0), np.max(mfccs, axis=0)))

        if chroma:
            chroma = librosa.feature.chroma_stft(S=stft, sr=sampleRate).T
            result = np.hstack((result, np.mean(chroma, axis=0), np.std(chroma, axis=0), np.max(chroma, axis=0)))

        if mel:
            mel = librosa.feature.melspectrogram(y=X, sr=sampleRate).T
            result = np.hstack((result, np.mean(mel, axis=0), np.std(mel, axis=0), np.max(mel, axis=0)))

    return result

emotions={
    '01':'neutral',
    '02':'calm',
    '03':'happy',
    '04':'sad',
    '05':'angry',
    '06':'fearful',
    '07':'disgust',
    '08':'surprised'
}

#observe the emotion
observedEmotions=['sad', 'fearful', 'disgust', 'happy', 'angry']


#Load data and extract features
def loadData(test_size=0.1):
    x,y=[],[]
    for file in glob.glob("archive/Actor_*/*.wav"):
        fileName=os.path.basename(file)
        emotion=emotions[fileName.split("-")[2]]
        #print(fileName)
        if emotion not in observedEmotions:
            continue
        feature=extractFeature(file,mfcc=True, chroma=True, mel=True)
        x.append(feature)
        y.append(int(fileName.split("-")[2]))
    return train_test_split(np.array(x), y, test_size=test_size, random_state=9)


#split data
x_train,x_test,y_train,y_test=loadData(test_size=0.25)

#declare mlp model
model=MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300,),learning_rate='adaptive', max_iter=500)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
model.fit(x_train, y_train)
print((x_train.shape[0], x_test.shape[0]))
print(f'Features extracted:{x_train.shape[1]}')

y_pred=model.predict(x_test)


accuracy= accuracy_score(y_true=y_test, y_pred=y_pred)

#display accuracy report
print("Accurancy: {:.2f}%".format(accuracy*100))
report = classification_report(y_test, y_pred, target_names=observedEmotions, output_dict=True)
weighted_precision = report['weighted avg']['precision']
weighted_recall = report['weighted avg']['recall']
weighted_f1_score = report['weighted avg']['f1-score']

print("Weighted Precision: {:.2f}%".format(weighted_precision*100))
print("Weighted Recall: {:.2f}%".format(weighted_recall*100))
print("Weighted F1-score: {:.2f}%".format(weighted_f1_score*100))

'''
import joblib
# Save the model to a file
joblib.dump(model, 'modelRadvess.joblib')
'''
