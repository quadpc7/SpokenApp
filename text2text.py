# app.py

import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import io
import numpy as np

# Initialize FastAPI app
app = FastAPI()

# Load Wav2Vec2 model and processor
model_name = "facebook/wav2vec2-large-960h-lv60-self"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)

# Set up Jinja2 templates for HTML rendering
templates = Jinja2Templates(directory="templates")

# Load and resample audio to 16000 Hz
def load_audio(file: bytes):
    waveform, sample_rate = torchaudio.load(io.BytesIO(file))
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
    return waveform.squeeze().numpy()

# Transcribe audio with confidence scoring
def transcribe_audio_with_confidence(file: bytes):
    speech = load_audio(file)
    inputs = processor(speech, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(inputs.input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])

    # Calculate confidence for each word
    confidences = torch.softmax(logits, dim=-1).max(dim=-1).values[0].cpu().numpy()
    word_confidences = [confidences[i].mean() for i in range(len(predicted_ids[0]))]

    return transcription, word_confidences

# Scoring functions for fluency and pronunciation clarity
def calculate_fluency_percentage(transcription):
    words = transcription.split()
    num_words = len(words)
    return min(1.0, num_words / 20) * 100

def estimate_pronunciation_clarity_percentage(transcription):
    clarity_score = len(transcription.split()) / (len(transcription) + 1)
    return min(clarity_score * 100, 100)

def detect_unclear_words(transcription, confidences, threshold=0.5):
    words = transcription.split()
    unclear_words = [word for i, word in enumerate(words) if confidences[i] < threshold]
    return unclear_words

# Route for main HTML page
@app.get("/", response_class=HTMLResponse)
async def main(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# API endpoint for file upload and processing
@app.post("/analyze/")
async def analyze_audio(file: UploadFile = File(...)):
    audio_data = await file.read()
    transcription, confidences = transcribe_audio_with_confidence(audio_data)
    fluency_score = calculate_fluency_percentage(transcription)
    clarity_score = estimate_pronunciation_clarity_percentage(transcription)
    unclear_words = detect_unclear_words(transcription, confidences)

    return {
        "transcription": transcription,
        "fluency_score": fluency_score,
        "pronunciation_score": clarity_score,
        "unclear_words": unclear_words,
    }
