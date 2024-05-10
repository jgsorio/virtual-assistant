import pyttsx3
import speech_recognition as sr
from playsound import playsound
import random
from datetime import datetime
import webbrowser as wb
import tensorflow as tf
import numpy as np
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

hour = datetime.now().strftime("%H:%M")
date = datetime.now().strftime("%d/%B/%Y")
date = date.split('/')

from modules import load_agenda, comands

comands_assistant = comands.comands
answare_assistant = comands.all_answares

assistant_name = 'Ana'

chrome_path = "C:\Program Files\Google\Chrome\Application\chrome.exe"
wb.register('chrome', None, wb.BackgroundBrowser(chrome_path))

def search(phrase):
    wb.get('chrome').open('https://www.google.com/search?q=' + phrase)


MODEL_TYPES = ["EMOÇÃO"]

def load_model_by_name(model_type):
    if model_type == MODEL_TYPES[0]:
        model = tf.keras.models.load_model('./models/speech_emotion_recognition.hdf5')
        model_dict = sorted(list(['calma', 'feliz', 'neutra', 'triste', 'nervosa', 'medo', 'nojo', 'surpreso']))
        SAMPLE_RATE = 48000
    return model, model_dict, SAMPLE_RATE


model_type = "EMOÇÃO"

loaded_model = load_model_by_name(model_type)

def predict_sound(AUDIO, SAMPLE_RATE, plot = True):
    results = []
    wav_data, sample_rate = librosa.load(AUDIO, sr = SAMPLE_RATE)
    clip, _ = librosa.effects.trim(wav_data, top_db=60, frame_length=512, hop_length=64)
    splitted_audio_data = tf.signal.frame(clip, sample_rate, sample_rate, pad_end=True, pad_value=0)
    for i, data in enumerate(splitted_audio_data.numpy()):
        mfccs_features = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=40)
        mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
        mfccs_scaled_features = mfccs_scaled_features.reshape(1, -1)
        mfccs_scaled_features = mfccs_scaled_features[:,:,np.newaxis]
        predictions = loaded_model[0].predict(mfccs_scaled_features, batch_size=32)
        if plot:
            plt.figure(figsize=(len(splitted_audio_data), 5))
            plt.bar(loaded_model[1], predictions[0])
            plt.show()

        predictions = predictions.argmax(axis=1)
        prediction = predictions.astype(int).flatten()
        prediction = loaded_model[1][prediction[0]]
        results.append(prediction)
    
    count_results = [[results.count(x), x] for x in set(results)]
    return max(count_results)[1]


def speak(audio):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 1)
    engine.say(audio)
    engine.runAndWait()


def listen_microphone():
    microphone = sr.Recognizer()
    with sr.Microphone() as source:
        microphone.adjust_for_ambient_noise(source, duration=0.8)
        print('Ouvindo:')
        audio = microphone.listen(source)
        with open('./recordings/speech.wav', 'wb') as f:
            f.write(audio.get_wav_data())
    try:
        phrase = microphone.recognize_google(audio, language='pt-BR')
        print('Voce disse: ' + phrase)
    except sr.UnknownValueError:
        phrase = ''
        print('Não entendi')
    return phrase


mode_control = False
print('[INFO] - Pronto para começar!')
playsound('./sounds/n1.mp3')


while True:
    phrase = listen_microphone()
    if assistant_name in phrase:
        phrase = phrase.replace(assistant_name, '').strip()

        if phrase in comands_assistant[0]:
            playsound('./sounds/n2.mp3')
            speak(answare_assistant[0])
            continue

        if phrase in comands_assistant[1]:
            playsound('./sounds/n2.mp3')
            speak('O que você deseja anotar?')
            phrase = listen_microphone()
            while True:
                phrase = phrase.replace(assistant_name, '').strip()
                with open('./recordings/notes.txt', 'a', encoding='utf-8') as f:
                    f.write(phrase + '\n')
                speak('Anotado!')
                speak('O que mais você deseja?')
                phrase = listen_microphone()
                if phrase == 'só isso':
                    break
            speak(random.sample(answare_assistant[1], k=1))
            continue

        if phrase in comands_assistant[2]:
            playsound('./sounds/n2.mp3')
            speak(random.sample(answare_assistant[2], k=1))
            phrase = listen_microphone()
            speak('Pesquisando...' + phrase)
            search(phrase)
            continue

        if phrase in ['minhas notas', 'minhas anotações', 'notas', 'anotações']:
            playsound('./sounds/n2.mp3')
            speak('As suas anotações sao:')
            with open('./recordings/notes.txt', 'r', encoding='utf-8') as f:
                for line in f:
                    speak(line)
            continue

        if phrase in comands_assistant[3]:
            playsound('./sounds/n2.mp3')
            speak('Agora são: ' + str(hour))
            continue
        
        if phrase in comands_assistant[4]:
            playsound('./sounds/n2.mp3')
            speak('Hoje é: ' + str(date[0]) + ' de ' + str(date[1]) + ' de ' + str(date[2]))
        
        if phrase == 'parar':
            playsound('./sounds/n2.mp3')
            speak(random.sample(answare_assistant[4], k=1))
            break
    else:
        playsound('./sounds/n3.mp3')
