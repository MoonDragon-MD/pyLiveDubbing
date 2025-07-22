# pyLiveDubbing
Live dubbing for video and audio completely offline with Vosk for transcription, translation is handled by translateLocally or LibreTranslate, the TTS engines are three ‘sherpa-onnx’/'pyttsx3 (espeak)‘/'mbrola (espeak-ng + mbrola)’. Currently only for Linux.


### Dependencies

Instructions for Debiane derivatives such as Ubuntu

```
pip3 install pyaudio numpy vosk sherpa-onnx scipy soundfile playsound pyttsx3
sudo apt install espeak-ng mbrola mbrola-it3 mbrola-it4 alsa-utils
mkdir models
mkdir vosk
cd vosk
wget https://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip
unzip vosk-model-en-us-0.22.zip
cd ..
cd models
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-it_IT-paola-medium.tar.bz2
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-it_IT-riccardo-x_low.tar.bz2
tar -xvf vits-piper-it_IT-paola-medium.tar.bz2
tar -xvf vits-piper-it_IT-riccardo-x_low.tar.bz2
cd ..
```
- LibreTranslate (Argos) set to localhost:5000 [LibreTranslate on Docker](https://hub.docker.com/r/libretranslate/libretranslate)
- [translateLocally]( https://github.com/XapaJIaMnu/translateLocally)  if you want the Italian language look at this: [ITA-models-translateLocally]( https://github.com/MoonDragon-MD/ITA-models-translateLocally-)
  
You can find other [models for Vosk here](https://alphacephei.com/vosk/models)

You can find other [models for onnx here](https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models).

Not all dependencies are necessary; you can choose what you prefer.

### Usage

```
python3 pyLiveDubbing.py
```

### Info

At the moment, I have set two 3-second blocks to get a better translation, so the delay will be about 6 seconds.

You can change the parameters, but I cannot guarantee optimal performance.

I would also like to point out a flaw in Vosk: if you use the minimal model, it may invent sentences when there is no audio. This may also happen with the medium model, but very rarely.

At the moment, it is designed to have dubbing from English to Italian. If you use other languages, remember to check the inside of the script.
