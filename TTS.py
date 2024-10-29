from IPython.display import Audio
import IPython.display as ipd
from tqdm import tqdm
from transformers import BarkModel, AutoProcessor, AutoTokenizer
import torch
import json
import numpy as np
from parler_tts import ParlerTTSForConditionalGeneration
import pickle
import ast
import io
from scipy.io import wavfile
from pydub import AudioSegment

# Uyarıları gizle
import warnings
warnings.filterwarnings('ignore')

# Cihazı ayarla
device = "cuda" if torch.cuda.is_available() else "cpu"

# Parler Model ve Tokenizer'ı yükle
parler_model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-v1").to(device)
parler_tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")

# Bark Model ve Processor'ı yükle
bark_processor = AutoProcessor.from_pretrained("suno/bark")
bark_model = BarkModel.from_pretrained("suno/bark", torch_dtype=torch.float16).to(device)
bark_sampling_rate = 24000

# Speaker 1 açıklaması
description_speaker1 = """
Laura's voice is expressive and dramatic in delivery, speaking at a moderately fast pace with a very close recording that almost has no background noise.
"""

# Pickle dosyasını yükleyin ve girdiyi alalım
with open('./resources/podcast_ready_data.pkl', 'rb') as file:
    PODCAST_TEXT = pickle.load(file)

# Yardımcı fonksiyonlar
def generate_speaker1_audio(text):
    """ParlerTTS kullanarak Speaker 1 için ses oluştur"""
    input_ids = parler_tokenizer(description_speaker1, return_tensors="pt").input_ids.to(device)
    prompt_input_ids = parler_tokenizer(text, return_tensors="pt").input_ids.to(device)
    generation = parler_model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
    audio_arr = generation.cpu().numpy().squeeze()
    return audio_arr, parler_model.config.sampling_rate

def generate_speaker2_audio(text):
    """Bark kullanarak Speaker 2 için ses oluştur"""
    inputs = bark_processor(text, voice_preset="v2/en_speaker_6").to(device)
    speech_output = bark_model.generate(**inputs, temperature=0.9, semantic_temperature=0.8)
    audio_arr = speech_output[0].cpu().numpy()
    return audio_arr, bark_sampling_rate

def numpy_to_audio_segment(audio_arr, sampling_rate):
    """Numpy dizisini AudioSegment'e dönüştür"""
    audio_int16 = (audio_arr * 32767).astype(np.int16)
    byte_io = io.BytesIO()
    wavfile.write(byte_io, sampling_rate, audio_int16)
    byte_io.seek(0)
    return AudioSegment.from_wav(byte_io)

# Podcast metnini tuple olarak yükleyin
podcast_segments = ast.literal_eval(PODCAST_TEXT)

# Final podcast'i oluşturma
final_audio = None

for speaker, text in tqdm(podcast_segments, desc="Podcast segmentleri oluşturuluyor", unit="segment"):
    if speaker == "Speaker 1":
        audio_arr, rate = generate_speaker1_audio(text)
    else:  # Speaker 2
        audio_arr, rate = generate_speaker2_audio(text)
    
    audio_segment = numpy_to_audio_segment(audio_arr, rate)
    if final_audio is None:
        final_audio = audio_segment
    else:
        final_audio += audio_segment

# Podcast'i kaydetme
output_file = "./resources/_podcast.mp3"
final_audio.export(output_file, format="mp3", bitrate="192k", parameters=["-q:a", "0"])
print(f"Podcast {output_file} olarak kaydedildi.")