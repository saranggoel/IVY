import subprocess

record = "arecord -D hw:tegrasndt210ref,0 -r 48000 -f S32_LE -c 1 -d 5 recording.wav"
p = subprocess.Popen(record, shell = True)

from time import sleep
sleep(7)

import speech_recognition as sr
r = sr.Recognizer()

hellow=sr.AudioFile('recording.wav')
with hellow as source:
    audio = r.record(source)
try:
    s = r.recognize_google(audio)
    print("Text: "+s)
except Exception as e:
    print("Exception: "+str(e))

from gtts import gTTS
import os
from pydub import AudioSegment
from pydub.playback import play

tts = gTTS(s)
tts.save("texttospeech.mp3")
song = AudioSegment.from_mp3("texttospeech.mp3")
play(song)
