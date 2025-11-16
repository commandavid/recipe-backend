from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import subprocess, uuid, os
from pydub import AudioSegment
from openai import OpenAI

app = FastAPI()
client = OpenAI()

class VideoURL(BaseModel):
    url: str

def download_video(url, output_path):
    cmd = ["yt-dlp", "-o", output_path, url]
    subprocess.run(cmd, check=True)

def extract_audio(video_path, audio_path):
    audio = AudioSegment.from_file(video_path)
    audio.export(audio_path, format="wav")

def transcribe(audio_path):
    with open(audio_path, "rb") as f:
        transcript = client.audio.transcriptions.create(
            model="gpt-4o-transcribe",
            file=f
        )
    return transcript.text

def format_recipe(transcription):
    prompt = f"""
Turn this cooking video transcription into a structured recipe.

Include sections:
- Title
- Ingredients
- Steps
- Notes

Transcription:
{transcription}
"""

    result = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )

    return result.output_text

@app.post("/recipe")
async def process_recipe(data: VideoURL):
    try:
        # Create unique filenames
        file_id = str(uuid.uuid4())
        video_path = f"/tmp/{file_id}.mp4"
        audio_path = f"/tmp/{file_id}.wav"

        # 1 — Download
        download_video(data.url, video_path)

        # 2 — Extract audio
        extract_audio(video_path, audio_path)

        # 3 — Transcribe audio
        transcription = transcribe(audio_path)

        # 4 — Convert transcription → recipe
        recipe = format_recipe(transcription)

        # Clean up
        os.remove(video_path)
        os.remove(audio_path)

        return {"recipe": recipe}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
