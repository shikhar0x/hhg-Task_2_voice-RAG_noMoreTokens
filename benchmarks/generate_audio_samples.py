import os
import json
import subprocess
from gtts import gTTS
from dotenv import load_dotenv

load_dotenv(override=True)

AUDIO_DIR = os.path.join("benchmarks", "audio_samples")
os.makedirs(AUDIO_DIR, exist_ok=True)

SAMPLE_QUERIES = [
    {
        "query_id": "1102432",
        "query": "what is a corporation?",
        "type": "MSMARCO-XI Grounded",
        "audio_filename": "query_1102432.wav"
    },
    {
        "query_id": "1102431",
        "query": "why did rachel carson write an obligation to endure",
        "type": "MSMARCO-XI Grounded",
        "audio_filename": "query_1102431.wav"
    },
    {
        "query_id": "205107",
        "query": "honesty or integrity definition",
        "type": "MSMARCO-XI Grounded",
        "audio_filename": "query_205107.wav"
    },
    {
        "query_id": "55665",
        "query": "bottom front of a cargo ship",
        "type": "MSMARCO-XI Grounded",
        "audio_filename": "query_55665.wav"
    },
    {
        "query_id": "168868",
        "query": "does medical marijuana help with ptsd",
        "type": "MSMARCO-XI Grounded",
        "audio_filename": "query_168868.wav"
    },
    {
        "query_id": "refusal_cake",
        "query": "What is the recipe for baking a chocolate lava cake?",
        "type": "Out-of-Domain Refusal",
        "audio_filename": "query_refusal_cake.wav"
    },
    {
        "query_id": "refusal_quantum",
        "query": "How do quantum computers factor 2048-bit RSA keys using Shor's algorithm?",
        "type": "Out-of-Domain Refusal",
        "audio_filename": "query_refusal_quantum.wav"
    }
]

def synthesize_gtts_wav(text: str, output_wav_path: str) -> bool:
    temp_mp3 = output_wav_path.replace(".wav", ".mp3")
    try:
        tts = gTTS(text=text, lang="en")
        tts.save(temp_mp3)
        # Convert MP3 to 16kHz WAV using ffmpeg
        res = subprocess.run(
            ["ffmpeg", "-y", "-i", temp_mp3, "-ar", "16000", "-ac", "1", output_wav_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        if os.path.exists(temp_mp3):
            os.remove(temp_mp3)
        if res.returncode == 0 and os.path.exists(output_wav_path):
            print(f"✅ Generated high-quality WAV via gTTS + ffmpeg: {output_wav_path}")
            return True
        else:
            print(f"ffmpeg conversion failed: {res.stderr.decode()}")
    except Exception as e:
        print(f"gTTS note: {e}")
    return False

def main():
    manifest_items = []
    for item in SAMPLE_QUERIES:
        out_file = os.path.join(AUDIO_DIR, item["audio_filename"])
        print(f"Synthesizing audio for '{item['query']}'...")
        success = synthesize_gtts_wav(item["query"], out_file)
        
        if success:
            manifest_items.append({
                "query_id": item["query_id"],
                "query": item["query"],
                "type": item["type"],
                "audio_path": out_file
            })
        else:
            print(f"❌ Failed to synthesize audio for query: {item['query']}")

    manifest_path = os.path.join("benchmarks", "audio_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_items, f, indent=2)
    print(f"\nManifest saved to {manifest_path} with {len(manifest_items)} audio samples.")

if __name__ == "__main__":
    main()
