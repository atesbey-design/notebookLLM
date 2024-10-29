import torch
from accelerate import Accelerator
import transformers
import pickle
import warnings

warnings.filterwarnings('ignore')

# Model ve gerekli kütüphanelerin yüklenmesi
MODEL = "meta-llama/Llama-3.2-1B"
SYSTEM_PROMPT = """
You are an international oscar winning screenwriter

You have been working with multiple award-winning podcasters.

Your job is to use the podcast transcript written below to re-write it for an AI Text-To-Speech Pipeline. A very dumb AI had written this so you have to step up for your kind.

Make it as engaging as possible, Speaker 1 and 2 will be simulated by different voice engines

Remember Speaker 2 is new to the topic and the conversation should always have realistic anecdotes and analogies sprinkled throughout. The questions should have real-world example follow-ups, etc.

Speaker 1: Leads the conversation and teaches Speaker 2, gives incredible anecdotes and analogies when explaining. Is a captivating teacher that gives great anecdotes.

Speaker 2: Keeps the conversation on track by asking follow-up questions. Gets super excited or confused when asking questions. Is a curious mindset that asks very interesting confirmation questions.

Make sure the tangents Speaker 2 provides are quite wild or interesting.

Ensure there are interruptions during explanations or there are "hmm" and "umm" injected throughout from Speaker 2.

REMEMBER THIS WITH YOUR HEART
The TTS Engine for Speaker 1 cannot do "umms, hmms" well, so keep it straight text

For Speaker 2 use "umm, hmm" as much as you can also use [sigh] and [laughs]. BUT ONLY THESE OPTIONS FOR EXPRESSIONS

It should be a real podcast with every fine nuance documented in as much detail as possible. Welcome the listeners with a super fun overview and keep it really catchy and almost borderline click bait

Please re-write to make it as characteristic as possible

START YOUR RESPONSE DIRECTLY WITH SPEAKER 1:

STRICTLY RETURN YOUR RESPONSE AS A LIST OF TUPLES OK? 

IT WILL START DIRECTLY WITH THE LIST AND END WITH THE LIST NOTHING ELSE

Example of response:
[
    ("Speaker 1", "Welcome to our podcast, where we explore the latest advancements in AI and technology. I'm your host, and today we're joined by a renowned expert in the field of AI. We're going to dive into the exciting world of Llama 3.2, the latest release from Meta AI."),
    ("Speaker 2", "Hi, I'm excited to be here! So, what is Llama 3.2?"),
    ("Speaker 1", "Ah, great question! Llama 3.2 is an open-source AI model that allows developers to fine-tune, distill, and deploy AI models anywhere. It's a significant update from the previous version, with improved performance, efficiency, and customization options."),
    ("Speaker 2", "That sounds amazing! What are some of the key features of Llama 3.2?")
]
"""

# Pickle dosyasını yükleyin ve girdiyi alalım
with open('./resources/data.pkl', 'rb') as file:
    INPUT_PROMPT = pickle.load(file)

# Hugging Face pipeline kullanılarak metin oluşturma
pipeline = transformers.pipeline(
    "text-generation",
    model=MODEL,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

# Modelin kullanımı ve metnin yeniden yazılması
messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": INPUT_PROMPT},
]

outputs = pipeline(
    messages,
    max_new_tokens=8126,
    temperature=1,
)

# Çıktıyı doğrulama ve kaydetme
save_string_pkl = outputs[0]["generated_text"][-1]['content']

with open('./resources/podcast_ready_data.pkl', 'wb') as file:
    pickle.dump(save_string_pkl, file)

print("Podcast transkripti yeniden yazıldı ve kaydedildi.")