#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import pyaudio
import numpy as np
from vosk import Model, KaldiRecognizer, SetLogLevel
import multiprocessing as mp
import queue
import json
import subprocess
import logging
import time
import soundfile as sf
import io
from typing import Dict, List, Optional, Tuple
from sherpa_onnx import OfflineTts, OfflineTtsConfig, OfflineTtsModelConfig, OfflineTtsVitsModelConfig
import sounddevice as sd
import re
import pyttsx3
import tempfile
import requests # For LibreTranslate

# pyLiveDubbing V.1.0 rev.39 by MoonDragon
# https://github.com/MoonDragon-MD/pyLiveDubbing

# Configure Logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Settings for Alsa
os.environ["ALSA_CARD"] = "pulse"
os.environ['PA_ALSA_PLUGHW'] = '1'
os.environ['JACK_NO_START_SERVER'] = '1'

# --- Utility functions ---
def split_text_into_chunks(text: str, max_length: int = 15) -> List[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_length:
            current_chunk += sentence + " "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return [chunk for chunk in chunks if chunk]

def estimate_tts_duration(text: str) -> float:
    """Estimates the duration of the TTS in seconds based on the length of the text."""
    words = len(text.split())
    return max(0.25 * words, 0.5)  # Increased to 0.25s per word, minimum 0.5s for Mbrola

def select_audio_devices(p):
    print("Audio input devices available (for pyaudio):")
    input_devices = []
    for i in range(p.get_device_count()):
        dev = p.get_device_info_by_index(i)
        if dev['maxInputChannels'] > 0:
            input_devices.append((i, dev['name']))
            print(f"{i}: {dev['name']}")
    input_device_index = int(input("Select the input device index: "))

    print("\nAudio output devices available (for sounddevice):")
    output_devices = []
    for i, dev in enumerate(sd.query_devices()):
        if dev['max_output_channels'] > 0:
            output_devices.append((i, dev['name']))
            print(f"{i}: {dev['name']}")
    output_device_index = int(input("Select the output device index: "))

    return input_device_index, output_device_index

def select_tts_engine():
    print("Select the TTS engine:")
    print("1: sherpa-onnx (vits-piper)")
    print("2: pyttsx3 (espeak)")
    print("3: mbrola (espeak-ng + mbrola)")
    choice = input("Enter the number (1, 2 o 3): ")
    while choice not in ['1', '2', '3']:
        print("Not valid choice. Enter 1, 2 o 3.")
        choice = input("Enter the number (1, 2 o 3): ")
    return {'1': 'sherpa-onnx', '2': 'pyttsx3', '3': 'mbrola'}[choice]

def get_sherpa_models(models_dir: str = "models") -> List[str]:
    try:
        models = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d)) and d.startswith("vits-piper-")]
        valid_models = []
        for model in models:
            model_name = model.replace("vits-piper-", "")
            model_path = os.path.join(models_dir, model, f"{model_name}.onnx")
            tokens_path = os.path.join(models_dir, model, "tokens.txt")
            data_dir = os.path.join(models_dir, model, "espeak-ng-data")
            lexicon_path = os.path.join(models_dir, model, "lexicon.txt")
            if os.path.exists(model_path) and os.path.exists(tokens_path) and os.path.exists(data_dir):
                valid_models.append(model)
            else:
                missing_files = []
                if not os.path.exists(model_path):
                    missing_files.append(f"{model_name}.onnx")
                if not os.path.exists(tokens_path):
                    missing_files.append("tokens.txt")
                if not os.path.exists(data_dir):
                    missing_files.append("espeak-ng-data")
                logging.error(f"Modello {model} incompleto: mancano {', '.join(missing_files)}")
        return valid_models
    except Exception as e:
        logging.error(f"Error in scanning the Models Director: {e}")
        return []

def select_sherpa_model(models: List[str]):
    if not models:
        raise ValueError("No Sherpa-Onnx model valid found.")
    print("Modelli sherpa-onnx disponibili:")
    for i, model in enumerate(models, 1):
        print(f"{i}: {model}")
    choice = input(f"Select the model (1-{len(models)}): ")
    while not choice.isdigit() or int(choice) < 1 or int(choice) > len(models):
        print(f"Not valid choice. Enter a number between 1 and {len(models)}.")
        choice = input(f"Select the model (1-{len(models)}): ")
    return models[int(choice) - 1]

def select_tts_language(available_languages: List[str], tts_engine: str):
    print(f"Lingue disponibili per {tts_engine}:")
    if tts_engine == 'sherpa-onnx':
        print("- Italian (only model available)")
        return "Italian"
    elif tts_engine == 'mbrola':
        available = list(MBROLA_VOICE_PATHS.keys())
        for i, lang in enumerate(available, 1):
            print(f"{i}: {lang}")
        choice = input(f"Select the TTS language (1-{len(available)}): ")
        while not choice.isdigit() or int(choice) < 1 or int(choice) > len(available):
            print(f"Not valid choice. Choose between: {', '.join(f'{i}: {lang}' for i, lang in enumerate(available, 1))}")
            choice = input(f"Select the TTS language (1-{len(available)}): ")
        return available[int(choice) - 1]
    else:
        for lang in available_languages:
            print(f"- {lang}")
        language = input(f"Select the TTS language (default: Italian): ") or "Italian"
        while language not in available_languages:
            print(f"Non -valid language. Choose between: {', '.join(available_languages)}")
            language = input(f"Select the TTS language (default: Italian): ") or "Italian"
        return language

def get_locally_languages(translate_locally_path: str = '') -> List[str]:
    try:
        cmd = [translate_locally_path or 'translateLocally', '-l']
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        language_map = {
            "afrikaans": "af", "albanian": "sq", "arabic": "ar", "basque": "eu",
            "bulgarian": "bg", "catalan": "ca", "czech": "cs", "english": "en",
            "estonian": "et", "french": "fr", "galician": "gl", "german": "de",
            "greek": "el", "hebrew": "he", "hindi": "hi", "icelandic": "is",
            "italian": "it", "japanese": "ja", "korean": "ko", "macedonian": "mk",
            "malay": "ml", "maltese": "mt", "norwegian": "no", "polish": "pl",
            "serbo-croatian": "hbs", "sinhala": "si", "slovak": "sk", "slovene": "sl",
            "spanish": "es", "swahili": "sw", "thai": "th", "turkish": "tr",
            "ukrainian": "uk", "vietnamese": "vi"
        }
        full_language_map = {v: k.capitalize() for k, v in language_map.items()}
        languages = set()
        for line in result.stdout.splitlines():
            if "To invoke do -m" in line and any(model in line.lower() for model in ["tiny", "base", "transformer-tiny11", "full"]):
                start_idx = line.find("do -m") + 6
                if start_idx != -1 and start_idx < len(line):
                    translation_spec = line[start_idx:].strip().split("-")
                    if len(translation_spec) >= 3:
                        source_lang = translation_spec[0].lower()
                        target_lang = translation_spec[1].lower()
                        source_code = next((code for full, code in language_map.items() if full in source_lang), source_lang)
                        target_code = next((code for full, code in language_map.items() if full in target_lang), target_lang)
                        languages.add(full_language_map.get(source_code, source_code.capitalize()))
                        languages.add(full_language_map.get(target_code, target_code.capitalize()))
        return sorted(languages) or ['English', 'Italian']
    except Exception as e:
        logging.error(f"Language recovery error: {e}")
        return ['English', 'Italian']

def check_translate_locally_availability(custom_path: Optional[str] = None) -> bool:
    global TRANSLATE_LOCALLY_AVAILABLE, TRANSLATE_LOCALLY_MODELS, TRANSLATE_LOCALLY_PATH
    if TRANSLATE_LOCALLY_AVAILABLE is not None:
        return TRANSLATE_LOCALLY_AVAILABLE
    TRANSLATE_LOCALLY_PATH = custom_path or "translateLocally"
    try:
        result_list = subprocess.run(
            [TRANSLATE_LOCALLY_PATH, "-l"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        language_models: Dict[str, List[Tuple[str, str]]] = {}
        language_map = {
            "afrikaans": "af", "arabic": "ar", "bulgarian": "bg", "catalan": "ca", "czech": "cs",
            "german": "de", "greek": "el", "english": "en", "spanish": "es", "estonian": "et",
            "basque": "eu", "french": "fr", "galician": "gl", "serbo-croatian": "hbs", "hebrew": "he",
            "hindi": "hi", "icelandic": "is", "italian": "it", "japanese": "ja", "korean": "ko",
            "macedonian": "mk", "malay": "ml", "maltese": "mt", "norwegian": "no", "polish": "pl",
            "sinhala": "si", "slovak": "sk", "slovene": "sl", "albanian": "sq", "swahili": "sw",
            "thai": "th", "turkish": "tr", "ukrainian": "uk", "vietnamese": "vi"
        }
        for line in result_list.stdout.strip().splitlines():
            if "To invoke do -m" in line and any(model in line.lower() for model in ["tiny", "base", "transformer-tiny11", "full"]):
                start_idx = line.find("do -m") + 6
                if start_idx != -1 and start_idx < len(line):
                    translation_spec = line[start_idx:].strip().split("-")
                    if len(translation_spec) >= 3:
                        source_lang = translation_spec[0].lower()
                        target_lang = translation_spec[1].lower()
                        model = translation_spec[2].lower()
                        normalized_source = next((code for full, code in language_map.items() if full in source_lang), source_lang)
                        normalized_target = next((code for full, code in language_map.items() if full in target_lang), target_lang)
                        if normalized_source not in language_models:
                            language_models[normalized_source] = []
                        language_models[normalized_source].append((normalized_target, model))
        TRANSLATE_LOCALLY_MODELS = language_models
        TRANSLATE_LOCALLY_AVAILABLE = True
        logging.info(f"translateLocally models available: {language_models}")
        return True
    except Exception as e:
        logging.error(f"Error in checking translateLocally: {e}")
        TRANSLATE_LOCALLY_AVAILABLE = False
        return False

def translate_locally(text: str, source: str, target: str) -> str:
    if not TRANSLATE_LOCALLY_AVAILABLE:
        raise ValueError("translateLocally not available.")
    language_models = TRANSLATE_LOCALLY_MODELS
    if source.lower() not in language_models:
        raise ValueError(f"No model for the language of origin: {source}")
    direct_translation = next((model for target_lang, model in language_models[source.lower()] if target_lang == target.lower()), None)
    if direct_translation:
        max_chunk_size = 500
        chunks = [text[i:i + max_chunk_size] for i in range(0, len(text), max_chunk_size)]
        translated_chunks = []
        for chunk in chunks:
            if not chunk.strip():
                translated_chunks.append("")
                continue
            command = [TRANSLATE_LOCALLY_PATH, "-m", f"{source.lower()}-{target.lower()}-{direct_translation}"]
            logging.debug(f"Run: {' '.join(command)}")
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdout, stderr = process.communicate(input=chunk)
            if process.returncode == 0:
                translated_chunk = stdout.strip()
                translated_chunk = "\n".join(line for line in translated_chunk.splitlines() if not line.startswith("QVariant"))
                translated_chunks.append(translated_chunk or "")
            else:
                logging.error(f"Error in translation: {stderr}")
                raise ValueError(f"Error in translation: {stderr}")
        return " ".join(translated_chunks)
    intermediate_lang = "en"
    if intermediate_lang not in language_models:
        raise ValueError(f"Intermediate language {intermediate_lang} not supported")
    en_model = next((model for t, model in language_models[source.lower()] if t == intermediate_lang), None)
    if not en_model:
        raise ValueError(f"Nessun modello per {source.lower()} -> {intermediate_lang}")
    command = [TRANSLATE_LOCALLY_PATH, "-m", f"{source.lower()}-{intermediate_lang}-{en_model}"]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate(input=text)
    if process.returncode != 0:
        raise ValueError(f"Error in the first translation: {stderr}")
    intermediate_text = "\n".join(line for line in stdout.strip().splitlines() if not line.startswith("QVariant"))
    target_model = next((model for t, model in language_models[intermediate_lang] if t == target.lower()), None)
    if not target_model:
        raise ValueError(f"No model for {intermediate_lang} -> {target.lower()}")
    command = [TRANSLATE_LOCALLY_PATH, "-m", f"{intermediate_lang}-{target.lower()}-{target_model}"]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate(input=intermediate_text)
    if process.returncode != 0:
        raise ValueError(f"Error in the second translation: {stderr}")
    translated_text = "\n".join(line for line in stdout.strip().splitlines() if not line.startswith("QVariant"))
    return translated_text

# Main configuration
TRANSLATOR = None  # Will ask if used translateLocally or LibreTranslate
LIBRETRANSLATE_URL = "http://localhost:5000"  # URL of server LibreTranslate, modificabile se necessario
SAMPLE_RATE = 16000
BLOCK_SIZE = 48000 # 3 seconds of audio to be sent to Vosk for transcription (1 sec = 16000)
BUFFER_BLOCKS = 2  # Accumulate 2 blocks (6 total seconds) before transcription
VOSK_MODEL_PATH = "vosk/vosk-model-en-us-0.22"
TRANSLATE_LOCALLY_PATH = ""
SOURCE_LANG = "en"
MBROLA_VOICE_PATHS = {
    "English (British, male, en1)": "/usr/share/mbrola/en1/en1",
    "Italian (male, it3)": "/usr/share/mbrola/it3/it3",
    "Italian (female, it4)": "/usr/share/mbrola/it4/it4",
    "English (American, male, us3)": "/usr/share/mbrola/us3/us3",
}
TTS_PAUSE = 0.5  # Increased for Mbrola
ENERGY_THRESHOLD = 0.001
MIN_CONFIDENCE = 0.5
AUDIO_BLOCKS_COUNTER = mp.Value('i', 0)  # Meter for audio blocks received
PROCESSED_BLOCKS_COUNTER = mp.Value('i', 0)  # Processed audio block counter
PYTTSX3_SPEED = 220
PYTTSX3_VOLUME = 0.9
ESPEAK_SPEED = 200
TTS_TIMEOUT = 5
MBROLA_MAX_TEXT_LENGTH = 15

# Language map
LANGUAGE_MAP = {
    "af": "Afrikaans", "sq": "Albanian", "ar": "Arabic", "eu": "Basque",
    "bg": "Bulgarian", "ca": "Catalan", "cs": "Czech", "en": "English",
    "et": "Estonian", "fr": "French", "gl": "Galician", "de": "German",
    "el": "Greek", "he": "Hebrew", "hi": "Hindi", "is": "Icelandic",
    "it": "Italian", "ja": "Japanese", "ko": "Korean", "mk": "Macedonian",
    "ml": "Malay", "mt": "Maltese", "no": "Norwegian", "pl": "Polish",
    "hbs": "Serbo-Croatian", "si": "Sinhala", "sk": "Slovak", "sl": "Slovene",
    "es": "Spanish", "sw": "Swahili", "th": "Thai", "tr": "Turkish",
    "uk": "Ukrainian", "vi": "Vietnamese"
}
REVERSE_LANGUAGE_MAP = {v.lower(): k for k, v in LANGUAGE_MAP.items()}

# Shared queues
audio_queue = mp.Queue(maxsize=20)
text_queue = mp.Queue(maxsize=5)
translated_queue = mp.Queue(maxsize=3)
tts_play_queue = mp.Queue(maxsize=3)  # Queue for TTS

# Global variables
RUNNING = mp.Value('b', True)
TRANSLATE_LOCALLY_AVAILABLE = None
TRANSLATE_LOCALLY_MODELS = None
TRANSLATE_LOCALLY_PATH = None
TTS = None
TTS_MODEL_PATH = None
TTS_TOKENS_PATH = None
TTS_DATA_DIR = None
TTS_LEXICON_PATH = None
tts_lock = mp.Lock()
OUTPUT_DEVICE_INDEX = None  # Output device index

# Initialize translateLocally
AVAILABLE_LANGUAGES = get_locally_languages(TRANSLATE_LOCALLY_PATH)
check_translate_locally_availability(TRANSLATE_LOCALLY_PATH)

# Select TTS and language engine
TTS_ENGINE = select_tts_engine()
TTS_LANGUAGE = select_tts_language(AVAILABLE_LANGUAGES, TTS_ENGINE)
TARGET_LANG = REVERSE_LANGUAGE_MAP.get(TTS_LANGUAGE.split(' (')[0].lower(), 'it')

# Upload models
logging.info("Upload...")
SetLogLevel(-1)

# Charge Model Vosk
start_time = time.time()
VOSK_MODEL = Model(VOSK_MODEL_PATH)
logging.info(f"Loading Vosk completed: {time.time() - start_time:.2f} secondi")

# Pre-initiate Mbrola
if TTS_ENGINE == 'mbrola':
    start_time = time.time()
    subprocess.run(['espeak-ng', '--version'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    subprocess.run(['mbrola', '-h'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if TTS_LANGUAGE not in MBROLA_VOICE_PATHS:
        raise ValueError(f"No voice mbola available for the language {TTS_LANGUAGE}")
    if not os.path.exists(MBROLA_VOICE_PATHS[TTS_LANGUAGE]):
        raise FileNotFoundError(f"MBROLA voice not found: {MBROLA_VOICE_PATHS[TTS_LANGUAGE]}")
    command = f'echo "Test" | espeak-ng -v {TTS_LANGUAGE.split(" (")[0]} -s {ESPEAK_SPEED} --pho | mbrola -e -t 0.8 {MBROLA_VOICE_PATHS[TTS_LANGUAGE]} - -'
    try:
        subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5)
        logging.info(f"MBROLA pre-initiating completed: {time.time() - start_time:.2f} secondi")
    except Exception as e:
        logging.error(f"Mbrola pre-initialization error: {e}")
        TTS_ENGINE = 'pyttsx3'

# Loading TTS
if TTS_ENGINE == 'sherpa-onnx':
    start_time = time.time()
    sherpa_models = get_sherpa_models()
    if not sherpa_models:
        raise ValueError("No Sherpa-Onnx model valid found.")
    selected_model = select_sherpa_model(sherpa_models)
    model_name = selected_model.replace("vits-piper-", "")
    TTS_MODEL_PATH = f"models/{selected_model}/{model_name}.onnx"
    TTS_TOKENS_PATH = f"models/{selected_model}/tokens.txt"
    TTS_DATA_DIR = f"models/{selected_model}/espeak-ng-data"
    TTS_LEXICON_PATH = f"models/{selected_model}/lexicon.txt"
    try:
        if not os.path.exists(TTS_MODEL_PATH):
            raise FileNotFoundError(f"Model not found: {TTS_MODEL_PATH}")
        if not os.path.exists(TTS_TOKENS_PATH):
            raise FileNotFoundError(f"Tokens not found: {TTS_TOKENS_PATH}")
        if not os.path.exists(TTS_DATA_DIR):
            raise FileNotFoundError(f"Data dir not found: {TTS_DATA_DIR}")
        VITS_CONFIG = OfflineTtsVitsModelConfig(
            model=TTS_MODEL_PATH,
            tokens=TTS_TOKENS_PATH,
            data_dir=TTS_DATA_DIR,
            lexicon=TTS_LEXICON_PATH if os.path.exists(TTS_LEXICON_PATH) else "",
            noise_scale=0.667,
            noise_scale_w=0.8,
            length_scale=1.0
        )
        TTS_CONFIG = OfflineTtsConfig(model=OfflineTtsModelConfig(vits=VITS_CONFIG))
        TTS = OfflineTts(TTS_CONFIG)
        # Model test with a simple input
        test_audio = TTS.generate("Ciao")
        if not test_audio or len(test_audio.samples) == 0:
            raise ValueError("Error: the Sherpa-onnx model does not generate audio valid for the test")
        logging.info(f"Loading Sherpa-ONNX ({selected_model}) completed: {time.time() - start_time:.2f} secondi")
    except Exception as e:
        logging.error(f"Sherpa-Onnx loading error: {e}")
        raise
        
# Function to check the availability of LibreTranslate
def is_libretranslate_available() -> bool:
    try:
        response = requests.get(f"{LIBRETRANSLATE_URL}/languages", timeout=2)
        return response.status_code == 200
    except requests.RequestException as e:
        logging.warning(f"Server LibreTranslate not available: {e}")
        return False

# Function to choose the translator
def select_translator():
    print("Select the translator:")
    print("1: translateLocally")
    libretranslate_available = is_libretranslate_available()
    if libretranslate_available:
        print(f"2: LibreTranslate ({LIBRETRANSLATE_URL})")
    choice = input("Enter the number (1" + (" o 2" if libretranslate_available else "") + "): ")
    while choice not in ['1'] + (['2'] if libretranslate_available else []):
        print(f"Not valid choice. Enter 1" + (" o 2" if libretranslate_available else "") + ".")
        choice = input("Enter the number (1" + (" o 2" if libretranslate_available else "") + "): ")
    return 'LibreTranslate' if choice == '2' else 'translateLocally'

# Audio reproduction functions
def init_pyttsx3():
    try:
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        selected_voice = None
        for voice in voices:
            logging.info(f"Voce: {voice.name}, ID: {voice.id}, Lingue: {voice.languages}")
            if "mbrola-it3" in voice.id.lower() or "mbrola-it3" in voice.name.lower():
                selected_voice = voice.id
                break
            elif "italian" in voice.name.lower() or "it" in voice.languages:
                selected_voice = voice.id
        if selected_voice:
            engine.setProperty('voice', selected_voice)
            logging.info(f"Item selected for pyttsx3: {selected_voice}")
        else:
            logging.warning("No Italian item found, I use the default item")
            engine.setProperty('voice', 'default')
        engine.setProperty('rate', PYTTSX3_SPEED)
        engine.setProperty('volume', PYTTSX3_VOLUME)
        return engine
    except Exception as e:
        logging.error(f"Pyttsx3 initialization error: {e}")
        return None

def play_sherpa_audio(audio, sample_rate):
    try:
        with tts_lock:
            logging.info("TTS (IT): Direct reproduction Sherpa-Onnx with aplay")
            start_time = time.time()
            if len(audio.samples) > 0:
                # Create a temporary WAV file
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                    sf.write(tmp_file.name, audio.samples, sample_rate)
                    # Reproduce with aplay using the default device
                    subprocess.run(["aplay", "-D", "default", tmp_file.name], check=True)
                    # Remove the temporary file
                    os.remove(tmp_file.name)
            else:
                logging.warning("Audio vuoto, ignorato")
            elapsed_time = time.time() - start_time
            logging.info(f"TTS (IT): Riproduzione completata in {elapsed_time:.2f} secondi")
    except subprocess.CalledProcessError as e:
        logging.error(f"Sherpa-onnx reproduction error with aplay: {e}")
    except Exception as e:
        logging.error(f"Generic error Sherpa-Onnx reproduction: {e}")

def pyttsx3_say(engine, text, chunk_index=None):
    if engine is None:
        logging.error(f"pyttsx3 engine not initialized for chunk {chunk_index}: {text}")
        return
    try:
        with tts_lock:
            logging.info(f"TTS (IT): Reproduced chunk {chunk_index if chunk_index is not None else 'completo'}: {text}")
            engine.say(text)
            engine.runAndWait()
            time.sleep(estimate_tts_duration(text))
            logging.info(f"TTS (IT): Completed Chunk {chunk_index if chunk_index is not None else 'completo'}: {text}")
    except Exception as e:
        logging.error(f"Error pyttsx3 for '{text}': {e}")

def mbrola_say(text, chunk_index=None):
    global OUTPUT_DEVICE_INDEX
    try:
        with tts_lock:
            logging.info(f"TTS (IT): Reproduced chunk {chunk_index if chunk_index is not None else 'completo'}: {text}")
            start_time = time.time()
            command = (
                f'echo "{text}" | espeak-ng -v {TTS_LANGUAGE.split(" (")[0]} -s {ESPEAK_SPEED} --pho | '
                f'mbrola -e -t 0.8 {MBROLA_VOICE_PATHS[TTS_LANGUAGE]} - -'
            )
            process = subprocess.Popen(
                command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=False
            )
            audio_data, stderr = process.communicate(timeout=TTS_TIMEOUT)
            if process.returncode != 0:
                logging.error(f"Error mbrola for chunk {chunk_index}: {stderr.decode()}")
                return
            # Converte i dati raw in campioni PCM a 16 bit
            sample_rate = 16000  # Take 16000 Hz, check for your item
            samples = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            if samples.size > 0:
                sd.play(samples, sample_rate, device=OUTPUT_DEVICE_INDEX)
                sd.wait()
                time.sleep(TTS_PAUSE)
            else:
                logging.warning(f"Empty audio mbrola segment for chunk {chunk_index}: {text}")
            elapsed_time = time.time() - start_time
            logging.info(f"TTS (IT): Completed Chunk {chunk_index if chunk_index is not None else 'completo'} in {elapsed_time:.2f} secondi")
    except subprocess.TimeoutExpired:
        logging.error(f"Timeout mbrola for chunk {chunk_index}: {text}")
    except Exception as e:
        logging.error(f"Errore mbrola per '{text}': {e}")
        # Fallback with default device
        try:
            logging.info("Attempt with default output device")
            sample_rate = 16000
            samples = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            if samples.size > 0:
                sd.play(samples, sample_rate, device=None)
                sd.wait()
                time.sleep(TTS_PAUSE)
            elapsed_time = time.time() - start_time
            logging.info(f"TTS (IT): Completed Chunk (fallback) in {elapsed_time:.2f} secondi")
        except Exception as fallback_e:
            logging.error(f"Errore mbrola (fallback): {fallback_e}")

# Worker per i processi
def audio_callback(in_data, frame_count, time_info, status, audio_queue):
    if status:
        logging.error(f"Error callback audio: {status}")
    if RUNNING.value:
        try:
            audio_data = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
            rms = np.sqrt(np.mean(audio_data**2))
            logging.debug(f"RMS calculated: {rms:.4f}, Theshold: {ENERGY_THRESHOLD}")
            if rms > ENERGY_THRESHOLD:
                try:
                    audio_queue.put(audio_data, timeout=0.1)
                    with AUDIO_BLOCKS_COUNTER.get_lock():
                        AUDIO_BLOCKS_COUNTER.value += 1
                    logging.info(f"Audio received, RMS: {rms:.4f}, Block #{AUDIO_BLOCKS_COUNTER.value}")
                except queue.Full:
                    logging.warning(f"audio_queue full, discarded audio block (#{AUDIO_BLOCKS_COUNTER.value})")
            else:
                logging.debug(f"Ignored audio, too low RMS: {rms:.4f}")
        except Exception as e:
            logging.error(f"Error in audio_callback: {e}")
    return (None, pyaudio.paContinue)

# Manage Vosk transcription with bufferization
def vosk_worker(audio_queue, text_queue):
    recognizer = KaldiRecognizer(VOSK_MODEL, SAMPLE_RATE)
    recognizer.SetMaxAlternatives(0)
    recognizer.SetWords(True)
    recognizer.SetPartialWords(False)  # Disable partial results
    buffer = bytearray()  # Buffer to accumulate audio blocks
    max_buffer_size = BLOCK_SIZE * BUFFER_BLOCKS * 2  # BLOCK_SIZE * 2 block * 2 bytes per sample
    while RUNNING.value:
        try:
            start_time = time.time()
            data = audio_queue.get(timeout=0.5)
            data = (data * 32767).astype(np.int16).tobytes()
            buffer.extend(data)
            with PROCESSED_BLOCKS_COUNTER.get_lock():
                PROCESSED_BLOCKS_COUNTER.value += 1
            if len(buffer) >= max_buffer_size:  # Processed when the buffer reaches 6 seconds
                # Converti bytearray in bytes
                buffer_bytes = bytes(buffer)
                recognizer.AcceptWaveform(buffer_bytes)
                result = json.loads(recognizer.FinalResult())
                text = result.get("text", "")
                confidence = 0.0
                if "result" in result and result["result"]:
                    confidences = [word.get("conf", 0.0) for word in result["result"]]
                    confidence = sum(confidences) / len(confidences) if confidences else 0.0
                if text.strip() and confidence >= MIN_CONFIDENCE:
                    logging.info(f"Transcription (EN): {text} (confidence: {confidence:.2f}, Blocco #{PROCESSED_BLOCKS_COUNTER.value})")
                    try:
                        text_queue.put(text, timeout=0.1)
                    except queue.Full:
                        logging.debug("text_queue full, old text waste")
                        text_queue.get_nowait()
                        text_queue.put(text, timeout=0.1)
                else:
                    logging.debug(f"Ignored transcription: testo='{text}', confidenza={confidence:.2f}, Blocco #{PROCESSED_BLOCKS_COUNTER.value}")
                buffer = bytearray()  # Reset the buffer
            logging.debug(f"Vosk elaborate {len(data)} byte in {time.time() - start_time:.3f} secondi, Blocco #{PROCESSED_BLOCKS_COUNTER.value}")
        except queue.Empty:
            continue
        except Exception as e:
            logging.error(f"Errore Vosk: {e}")


# Funzione translate_text per LibreTranslate
def translate_text(text, source_lang, target_lang):
    """Function to perform the translation of the text through the API of LibreTranslate, with management of special characters and accents."""
    url = f"{LIBRETRANSLATE_URL}/translate"
    headers = {
        "Content-Type": "application/json; charset=UTF-8"
    }
    data = {
        "q": text,
        "source": source_lang,
        "target": target_lang
    }
    try:
        response = requests.post(url, headers=headers, json=data)
        response.encoding = 'utf-8'
        response.raise_for_status()
        translated_text = response.json().get('translatedText', '').strip()
        return translated_text
    except requests.exceptions.RequestException as e:
        raise Exception(f"Error in the request to Libretranslate: {e}")
    except ValueError as e:
        raise Exception(f"Error in decoding Jsons response: {e}")
        
def translation_worker(text_queue, translated_queue):
    global TRANSLATOR
    while RUNNING.value:
        try:
            start_time = time.time()
            text = text_queue.get(timeout=0.5)
            if not text.strip() or len(text.strip().split()) < 2:
                logging.debug(f"Testo ignorato per traduzione: {text}")
                continue
            if TRANSLATOR == 'LibreTranslate':
                translated_text = translate_text(text, SOURCE_LANG, TARGET_LANG)
            else:
                translated_text = translate_locally(text, SOURCE_LANG, TARGET_LANG)
            if translated_text.strip():
                logging.info(f"Traduzione (IT): {translated_text}")
                try:
                    translated_queue.put(translated_text, timeout=0.1)
                except queue.Full:
                    logging.debug("translated_queue full, old text waste")
                    translated_queue.get_nowait()
                    translated_queue.put(translated_text, timeout=0.1)
            logging.debug(f"Translation completed in {time.time() - start_time:.2f} seconds")
        except queue.Empty:
            continue
        except Exception as e:
            logging.error(f"Translation error ({TRANSLATOR}): {e}")
        
def tts_worker(translated_queue, tts_play_queue):
    global TTS, TTS_ENGINE
    pyttsx3_engine = None
    if TTS_ENGINE == 'pyttsx3' or TTS_ENGINE == 'mbrola':
        pyttsx3_engine = init_pyttsx3()
    while RUNNING.value:
        try:
            start_time = time.time()
            translated_text = translated_queue.get(timeout=0.5)
            if not translated_text.strip() or len(translated_text.strip().split()) < 2:  # Correction: cambiato 'text' in 'translated_text'
                logging.debug(f"Text ignored for TTS: {translated_text}")
                continue
            logging.info(f"TTS (IT): {translated_text}")
            text_chunks = split_text_into_chunks(translated_text, MBROLA_MAX_TEXT_LENGTH)
            for i, chunk in enumerate(text_chunks):
                chunk = re.sub(r'[^\w\s.,!?]', '', chunk)
                if not chunk.strip():
                    logging.debug(f"Ignored empty chunk: {chunk}")
                    continue
                logging.debug(f"Chunk processing {i+1}: {chunk}")
                # Empty the queue if full to avoid accumulation
                while tts_play_queue.full():
                    logging.debug("tts_play_queue Full, I empty old chunk")
                    tts_play_queue.get_nowait()
                tts_play_queue.put((chunk, i + 1), timeout=0.1)
            logging.debug(f"TTS completed in {time.time() - start_time:.3f} secondi")
        except queue.Empty:
            continue
        except Exception as e:
            logging.error(f"Error TTS: {e}")

def tts_play_worker(tts_play_queue):
    global TTS, TTS_ENGINE
    pyttsx3_engine = None
    if TTS_ENGINE == 'pyttsx3' or TTS_ENGINE == 'mbrola':
        pyttsx3_engine = init_pyttsx3()
    while RUNNING.value:
        try:
            chunk, chunk_index = tts_play_queue.get(timeout=0.5)
            if TTS_ENGINE == 'sherpa-onnx':
                try:
                    audio = TTS.generate(chunk)
                    if not audio or len(audio.samples) == 0:
                        logging.error(f"Error sherpa-onnx: No audio generated for '{chunk}'")
                        continue
                    play_sherpa_audio(audio, audio.sample_rate)
                except Exception as e:
                    logging.error(f"Sherpa-onnx generation error for '{chunk}': {e}")
                    continue
            elif TTS_ENGINE == 'pyttsx3' and pyttsx3_engine:
                pyttsx3_say(pyttsx3_engine, chunk, chunk_index)
            elif TTS_ENGINE == 'mbrola':
                mbrola_say(chunk, chunk_index)
        except queue.Empty:
            continue
        except Exception as e:
            logging.error(f"TTS reproduction error: {e}")

# Monitoraggio coda
def queue_monitor():
    while RUNNING.value:
        logging.info(f"Code - audio: {audio_queue.qsize()}, text: {text_queue.qsize()}, translated: {translated_queue.qsize()}, tts_play: {tts_play_queue.qsize()}, Blocks received: {AUDIO_BLOCKS_COUNTER.value}, Processed blocks: {PROCESSED_BLOCKS_COUNTER.value}")
        time.sleep(1)

def start_stream():
    global RUNNING, SAMPLE_RATE, OUTPUT_DEVICE_INDEX
    p = pyaudio.PyAudio()
    stream = None
    vosk_process = None
    translation_process = None
    tts_process = None
    tts_play_process = None
    monitor_process = None
    try:
        input_device_index, OUTPUT_DEVICE_INDEX = select_audio_devices(p)
        sample_rates = [16000, 44100, 48000]
        selected_sample_rate = SAMPLE_RATE
        for sr in sample_rates:
            try:
                logging.debug(f"Test sample rate: {sr}")
                stream = p.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=sr,
                    input=True,
                    frames_per_buffer=BLOCK_SIZE,  # Usa il nuovo BLOCK_SIZE
                    input_device_index=input_device_index,
                    stream_callback=lambda in_data, frame_count, time_info, status: audio_callback(in_data, frame_count, time_info, status, audio_queue)
                )
                selected_sample_rate = sr
                logging.info(f"Sample rate {sr} accepted")
                break
            except Exception as e:
                logging.debug(f"Sample rate {sr} failed: {e}")
                if stream is not None:
                    stream.close()
                    stream = None
                continue

        if stream is None:
            raise ValueError(f"No valid sample rates found for the device {input_device_index}")

        SAMPLE_RATE = selected_sample_rate

        vosk_process = mp.Process(target=vosk_worker, args=(audio_queue, text_queue))
        translation_process = mp.Process(target=translation_worker, args=(text_queue, translated_queue))
        tts_process = mp.Process(target=tts_worker, args=(translated_queue, tts_play_queue))
        tts_play_process = mp.Process(target=tts_play_worker, args=(tts_play_queue,))
        monitor_process = mp.Process(target=queue_monitor)

        vosk_process.start()
        translation_process.start()
        tts_process.start()
        tts_play_process.start()
        monitor_process.start()

        stream.start_stream()
        logging.info(f"Starting audio capture a {selected_sample_rate} Hz completata")
        while stream.is_active() and RUNNING.value:
            time.sleep(0.1)
    except Exception as e:
        logging.error(f"Flow start -up error: {e}")
        RUNNING.value = False
    finally:
        RUNNING.value = False
        if stream is not None:
            stream.stop_stream()
            stream.close()
        p.terminate()
        for process in [vosk_process, translation_process, tts_process, tts_play_process, monitor_process]:
            if process is not None and process.is_alive():
                process.terminate()

if __name__ == "__main__":
    try:
        # Inizializza translateLocally
        AVAILABLE_LANGUAGES = get_locally_languages(TRANSLATE_LOCALLY_PATH)
        check_translate_locally_availability(TRANSLATE_LOCALLY_PATH)
        
        # Select the translator
        TRANSLATOR = select_translator()
        
        # Seleziona motore TTS e lingua
        #TTS_ENGINE = select_tts_engine()
        #TTS_LANGUAGE = select_tts_language(AVAILABLE_LANGUAGES, TTS_ENGINE)
        #TARGET_LANG = REVERSE_LANGUAGE_MAP.get(TTS_LANGUAGE.split(' (')[0].lower(), 'it')
        
        start_stream()
    except KeyboardInterrupt:
        logging.info("Manually interrupted")
    except Exception as e:
        logging.error(f"Main error: {e}")
