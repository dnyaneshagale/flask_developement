import os
import uuid
from flask import Flask, request, jsonify
import numpy as np
import torch
import soundfile as sf
import librosa
from df.enhance import enhance, init_df
from google.cloud import speech_v1p1beta1 as speech
from pyannote.audio import Pipeline

# Initialize Flask app
app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PROCESSED_FOLDER'] = 'processed'
app.config['GOOGLE_CREDENTIALS'] = "vocal-gist-450614-f9-381e907dce73.json"
app.config['HF_TOKEN'] = "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"

# Create directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

# Initialize models and pipelines once at startup
print("Initializing models...")

# Initialize DeepFilterNet components
df_model, df_state, _ = init_df()

# Initialize Diarization pipeline
diarization_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=app.config['HF_TOKEN']
)

# Initialize Google Cloud client
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = app.config['GOOGLE_CREDENTIALS']
speech_client = speech.SpeechClient()


def enhance_audio(input_path, output_path):
    """Enhanced audio processing with DeepFilterNet"""
    try:
        audio, orig_sr = librosa.load(input_path, sr=df_state.sr())
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
        enhanced = enhance(df_model, df_state, audio_tensor)
        enhanced_audio = enhanced.squeeze().numpy()

        if df_state.sr() != 16000:
            enhanced_audio = librosa.resample(enhanced_audio,
                                              orig_sr=df_state.sr(),
                                              target_sr=16000)

        sf.write(output_path, enhanced_audio, 16000, subtype='PCM_16')
        return True
    except Exception as e:
        print(f"Enhancement error: {str(e)}")
        return False


def process_audio(filename):
    """Process enhanced audio through diarization and STT"""
    try:
        # Diarization
        diarization = diarization_pipeline(filename)

        # Speech-to-Text
        with open(filename, "rb") as f:
            audio_content = f.read()

        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="en-US",
            enable_word_time_offsets=True,
        )

        response = speech_client.recognize(
            config=config,
            audio=speech.RecognitionAudio(content=audio_content)
        )

        # Collect words with timestamps
        stt_words = []
        for result in response.results:
            if result.alternatives:
                for word_info in result.alternatives[0].words:
                    stt_words.append({
                        "word": word_info.word,
                        "start": word_info.start_time.total_seconds(),
                        "end": word_info.end_time.total_seconds()
                    })

        # Align results
        # Corrected indentation in the diarization loop
        aligned_results = []
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            seg_start = segment.start
            seg_end = segment.end

            # Find words that overlap with this segment (even partially)
            segment_words = [
                w["word"] for w in stt_words
                if not (w["end"] <= seg_start or w["start"] >= seg_end)
            ]

            if segment_words:
                aligned_results.append({
                    "speaker": speaker,
                    "start": seg_start,
                    "end": seg_end,
                    "text": " ".join(segment_words)
                })

        return aligned_results
    except Exception as e:
        print(f"Processing error: {str(e)}")
        return None


@app.route('/process-audio', methods=['POST'])
def handle_audio():
    """Endpoint for audio processing"""
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    # Generate unique filenames
    file_id = uuid.uuid4().hex
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{file_id}_input.wav")
    enhanced_path = os.path.join(app.config['PROCESSED_FOLDER'], f"{file_id}_enhanced.wav")

    try:
        # Save uploaded file
        file.save(input_path)

        # Process audio
        if not enhance_audio(input_path, enhanced_path):
            return jsonify({"error": "Audio enhancement failed"}), 500

        results = process_audio(enhanced_path)
        if not results:
            return jsonify({"error": "Audio processing failed"}), 500

        return jsonify({
            "status": "success",
            "results": results
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        # Cleanup files
        if os.path.exists(input_path):
            os.remove(input_path)
        if os.path.exists(enhanced_path):
            os.remove(enhanced_path)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
