from openai import OpenAI
import os
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from pathlib import Path
import base64

# Initialize the client
client = OpenAI()

def transcribe_audio(audio_file):
    """Transcribe audio using OpenAI Whisper"""
    try:
        # Save temporarily
        temp_path = default_storage.save('temp_audio.webm', ContentFile(audio_file.read()))
        full_path = default_storage.path(temp_path)
        
        with open(full_path, 'rb') as f:
            transcript = client.audio.transcriptions.create(
                file=f,
                model="whisper-1"
            )
        
        # Clean up
        default_storage.delete(temp_path)
        
        return transcript.text
    except Exception as e:
        print(f"Transcription error: {e}")
        return None

def text_to_speech(text, voice="alloy"):
    """Convert text to speech using OpenAI TTS"""
    try:
        response = client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=text
        )
        return base64.b64encode(response.content).decode('utf-8')
    except Exception as e:
        print(f"TTS error: {e}")
        return None
    
def read_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()
