import os
from pathlib import Path
import torch
import torchaudio
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
import io
from datetime import datetime
import re
from itertools import islice
from typing import List, Dict
import traceback  # Add this at the top with other imports

from zonos.model import Zonos
from zonos.speaker_cloning import SpeakerEmbeddingLDA
from zonos.conditioning import make_cond_dict

app = Flask(__name__)

# Constants
SAVED_VOICES_DIR = Path("saved_voices")
TEMP_DIR = Path("temp")
ALLOWED_AUDIO_EXTENSIONS = {'.mp3', '.wav'}
MODEL_TYPE = "Zyphra/Zonos-v0.1-hybrid"  # Default model, same as gradio
CHUNK_SIZE = 4096  # Match client's chunk size

# Ensure directories exist
SAVED_VOICES_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)

# Initialize models
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize model exactly like gradio does
print(f"Loading {MODEL_TYPE} model...")
model = Zonos.from_pretrained(MODEL_TYPE, device=device)
model.requires_grad_(False).eval()

speaker_model = SpeakerEmbeddingLDA(device=device)

class AudioBuffer:
    def __init__(self):
        self.sample_rate = None
        self.chunks: List[torch.Tensor] = []
        self.current_chunk = 0

    def add_chunk(self, audio: torch.Tensor):
        self.chunks.append(audio)

    def get_next_chunk(self) -> torch.Tensor:
        if self.current_chunk < len(self.chunks):
            chunk = self.chunks[self.current_chunk]
            self.current_chunk += 1
            return chunk
        return None

    def reset(self):
        self.chunks = []
        self.current_chunk = 0

audio_buffer = AudioBuffer()

def load_voice_embedding(voice_name: str) -> torch.Tensor:
    """Load a saved voice embedding."""
    voice_path = SAVED_VOICES_DIR / f"{voice_name}.pt"
    if not voice_path.exists():
        raise ValueError(f"Voice '{voice_name}' not found")
    return torch.load(voice_path)

def save_voice_embedding(name: str, embedding: torch.Tensor):
    """Save a voice embedding."""
    voice_path = SAVED_VOICES_DIR / f"{name}.pt"
    torch.save(embedding, voice_path)

def process_audio_file(file_path: str) -> torch.Tensor:
    """Process an audio file and return speaker embedding."""
    waveform, sample_rate = torchaudio.load(file_path)
    
    # Convert stereo to mono by averaging channels if needed
    if waveform.dim() == 2 and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    # Add resampling if needed
    if sample_rate != 16000:  # Speaker model expects 16kHz
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
        sample_rate = 16000
    
    # Proper audio normalization
    waveform = waveform - waveform.mean()
    waveform = waveform / (torch.abs(waveform).max() + 1e-8)
    waveform = waveform.to(device)
    
    print(f"Processing voice - Waveform shape: {waveform.shape}, Sample rate: {sample_rate}")
    print(f"Audio stats - Min: {waveform.min():.3f}, Max: {waveform.max():.3f}, Mean: {waveform.mean():.3f}")
    
    with torch.inference_mode():
        # Use the proper speaker embedding model
        _, embedding = speaker_model(waveform, sample_rate)
        embedding = embedding.to(dtype=torch.bfloat16)
    
    return embedding

def generate_output_filename(voice_name: str) -> str:
    """Generate a unique filename for the output audio."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"output_{voice_name}_{timestamp}.wav"

def get_emotion_dict(emotion_temps: dict) -> list[float]:
    """Convert emotion temperatures to the expected format."""
    # Default emotion values
    emotions = [0.3077, 0.0256, 0.0256, 0.0256, 0.0256, 0.0256, 0.2564, 0.3077]
    
    if emotion_temps:
        # Update with provided values
        for idx, value in emotion_temps.items():
            idx = int(idx)
            if 0 <= idx < len(emotions):
                emotions[idx] = float(value)
    
    return emotions

def get_default_settings():
    """Get default settings from config"""
    return {
        "cfg_scale": 2.0,
        "min_p": 0.05,
        "temperature": 1.0,
        "repetition_penalty": 3.0,
        "repetition_penalty_window": 2,
        "emotion": [0.3077, 0.0256, 0.0256, 0.0256, 0.0256, 0.0256, 0.2564, 0.3077],
        "fmax": 24000,
        "pitch_std": 45.0,
        "speaking_rate": 15.0,
        "dnsmos_ovrl": 4.0,
        "vqscore_8": [0.78] * 8,
        "speaker_noised": False
    }

def generate_audio(text: str, voice_embedding: torch.Tensor, settings: dict = None) -> tuple[torch.Tensor, int]:
    """Generate audio using Zonos model with config settings."""
    try:
        # Use the same default settings as gradio
        generation_settings = {
            "cfg_scale": 2.0,
            "min_p": 0.15,  # Note: gradio uses 0.15, not 0.05
            "emotion": [1.0, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.2],  # Match gradio defaults
            "fmax": 24000,
            "pitch_std": 45.0,
            "speaking_rate": 15.0,
            "dnsmos_ovrl": 4.0,
            "vqscore_8": [0.78] * 8,
            "speaker_noised": False,
            "unconditional_keys": ["emotion"]  # Match gradio's default
        }
        if settings:
            generation_settings.update(settings)

        # Create conditioning dictionary exactly like gradio does
        conditioning = make_cond_dict(
            text=text,
            speaker=voice_embedding,
            emotion=generation_settings["emotion"],
            language="en-us",
            device=device,
            fmax=generation_settings["fmax"],
            pitch_std=generation_settings["pitch_std"],
            speaking_rate=generation_settings["speaking_rate"],
            dnsmos_ovrl=generation_settings["dnsmos_ovrl"],
            vqscore_8=generation_settings["vqscore_8"],
            speaker_noised=generation_settings["speaker_noised"],
            unconditional_keys=generation_settings["unconditional_keys"]
        )
        
        # Prepare conditioning using the model's method
        prefix_conditioning = model.prepare_conditioning(conditioning)
        
        # Generate using same parameters as gradio
        with torch.inference_mode():
            codes = model.generate(
                prefix_conditioning=prefix_conditioning,
                max_new_tokens=86 * 30,
                cfg_scale=generation_settings["cfg_scale"],
                batch_size=1,
                sampling_params={"min_p": generation_settings["min_p"]},
                progress_bar=True
            )
            
            audio = model.autoencoder.decode(codes)[0]
        
        return audio, model.autoencoder.sampling_rate
        
    except Exception as e:
        print(f"Error in generate_audio: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise


def chunk_text(text: str) -> list[dict]:
    """Split text into semantic chunks with proper sentence boundaries."""
    sentences = []
    current = []
    
    for line in text.split('\n'):
        words = line.strip().split()
        if not words:
            if current:
                sentences.append(' '.join(current))
                current = []
            continue
            
        current.extend(words)
        if words[-1].endswith(('.', '!', '?')):
            sentences.append(' '.join(current))
            current = []
    
    if current:
        sentences.append(' '.join(current))
    
    # Group sentences into properly sized chunks
    chunks = []
    current_chunk = []
    current_word_count = 0
    max_words = 20  # Conservative chunk size
    
    for sentence in sentences:
        words = sentence.split()
        word_count = len(words)
        
        if word_count > max_words:
            # Handle long sentence by itself
            if current_chunk:
                chunks.append({
                    'text': ' '.join(current_chunk),
                    'word_count': current_word_count
                })
            chunks.append({
                'text': sentence,
                'word_count': word_count
            })
            current_chunk = []
            current_word_count = 0
        elif current_word_count + word_count > max_words:
            # Save current chunk and start new one
            chunks.append({
                'text': ' '.join(current_chunk),
                'word_count': current_word_count
            })
            current_chunk = [sentence]
            current_word_count = word_count
        else:
            current_chunk.append(sentence)
            current_word_count += word_count
    
    if current_chunk:
        chunks.append({
            'text': ' '.join(current_chunk),
            'word_count': current_word_count
        })
    
    return chunks

def split_into_chunks(text: str) -> list[dict]:
    """Split text into chunks that respect sentence boundaries."""
    sentences = []
    current = []
    
    for line in text.split('\n'):
        words = line.strip().split()
        if not words:
            continue
            
        current.extend(words)
        if words[-1].endswith(('.', '!', '?')):
            if current:
                sentences.append(' '.join(current))
                current = []
    
    if current:
        sentences.append(' '.join(current))
    
    # Group into chunks
    chunks = []
    current_chunk = []
    current_word_count = 0
    max_words = 15  # Smaller chunks for better continuity
    
    for sentence in sentences:
        words = sentence.split()
        word_count = len(words)
        
        if current_word_count + word_count > max_words:
            if current_chunk:
                chunks.append({
                    'text': ' '.join(current_chunk),
                    'word_count': current_word_count
                })
            current_chunk = [sentence]
            current_word_count = word_count
        else:
            current_chunk.append(sentence)
            current_word_count += word_count
    
    if current_chunk:
        chunks.append({
            'text': ' '.join(current_chunk),
            'word_count': current_word_count
        })
    
    return chunks

# Add debug print at startup
print("\n=== Service Startup ===")
print(f"Using device: {device}")
print(f"Model loaded successfully")
print("===================\n")

@app.route('/voices', methods=['GET'])
def list_voices():
    """List all available voices."""
    voices = [f.stem for f in SAVED_VOICES_DIR.glob("*.pt")]
    return jsonify(voices)

@app.route('/audio-to-voice', methods=['POST'])
def audio_to_voice():
    """Create a new voice from audio file."""
    print("Request headers:", dict(request.headers))  # Additional debug info
    print("Files in request:", request.files)
    print("Form data:", request.form)
    print("Request data:", request.get_data())  # Raw request data
    
    # First check if we have any files at all
    if not request.files:
        return jsonify({
            "error": "No files in request",
            "help": "Make sure to send the file with key 'file' in form-data",
            "request_info": {
                "content_type": request.content_type,
                "method": request.method,
                "headers": dict(request.headers)
            }
        }), 400

    # Check for file in different possible keys
    file = None
    for key in ['file', 'files', 'audio', 'voice']:
        if key in request.files:
            file = request.files[key]
            break
    
    if not file or not file.filename:
        return jsonify({
            "error": "No valid file provided",
            "available_files": list(request.files.keys()),
            "content_type": request.content_type,
            "help": "Send the file with key 'file' and include a filename"
        }), 400

    name = request.form.get('name')
    if not name:
        return jsonify({"error": "No name provided"}), 400
    
    name = secure_filename(name)
    if not file.filename:
        return jsonify({"error": "No file selected"}), 400
    
    # Validate file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_AUDIO_EXTENSIONS:
        return jsonify({
            "error": f"Invalid file type. Allowed types: {ALLOWED_AUDIO_EXTENSIONS}"
        }), 400
    
    # Save temporary file
    temp_path = TEMP_DIR / f"temp_{name}{file_ext}"
    file.save(temp_path)
    
    try:
        print(f"Starting voice processing for {name}")
        start_time = datetime.now()
        
        embedding = process_audio_file(str(temp_path))
        
        print(f"Voice embedding generated in {datetime.now() - start_time}")
        print(f"Embedding shape: {embedding.shape}, dtype: {embedding.dtype}")
        
        save_voice_embedding(name, embedding)
        return jsonify({
            "message": f"Voice '{name}' created successfully",
            "name": name,
            "processing_time": str(datetime.now() - start_time)
        })
    except Exception as e:
        print(f"Error processing file: {str(e)}")  # Debug logging
        return jsonify({"error": str(e)}), 500
    finally:
        temp_path.unlink(missing_ok=True)

@app.route('/text-to-voice', methods=['POST'])
def text_to_voice():
    """Convert text to speech using specified voice and settings."""
    data = request.json
    
    if not data or 'text' not in data:
        return jsonify({"error": "No text provided"}), 400
    
    text = data['text']
    voice_name = data.get('voice', None)
    
    if not voice_name:
        return jsonify({"error": "No voice specified"}), 400
    
    try:
        voice_embedding = load_voice_embedding(voice_name)
        
        # Get generation settings from request or use defaults
        settings = data.get('settings', {})
        
        audio, sample_rate = generate_audio(text, voice_embedding, settings)
        
        # Ensure audio is in the right format for saving
        audio = audio.view(1, -1).cpu()  # Reshape to [channels, samples]
        
        buffer = io.BytesIO()
        torchaudio.save(buffer, audio, sample_rate, format="wav")
        buffer.seek(0)
        
        output_filename = generate_output_filename(voice_name)
        return send_file(
            buffer,
            mimetype="audio/wav",
            as_attachment=True,
            download_name=output_filename
        )
        
    except Exception as e:
        print(f"Error generating audio: {str(e)}")  # Debug logging
        print("Audio shape at error:", getattr(audio, 'shape', None))  # Additional debug info
        import traceback
        print(traceback.format_exc())  # Add detailed error traceback
        return jsonify({"error": str(e)}), 500

def generate_combined_audio(text: str, voice_embedding: torch.Tensor, max_words_per_chunk: int = 20):
    """Generate audio with proper chunking and continuation, following gradio example."""
    try:
        # Split text into chunks
        text_chunks = chunk_text(text)
        print(f"Split text into {len(text_chunks)} chunks")
        
        all_audio_segments = []
        previous_codes = None
        audio_prefix_codes = None
        
        for chunk_idx, chunk_info in enumerate(text_chunks):
            chunk_text = chunk_info['text']
            print(f"Processing chunk {chunk_idx + 1}/{len(text_chunks)}: {chunk_text}")
            
            # Create conditioning dict
            conditioning = make_cond_dict(
                text=chunk_text,
                speaker=voice_embedding,
                emotion=[1.0, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.2],
                language="en-us",
                device=device,
                fmax=24000,
                pitch_std=45.0,
                speaking_rate=15.0,
                dnsmos_ovrl=4.0,
                vqscore_8=[0.78] * 8,
                speaker_noised=False,
                unconditional_keys=["emotion"]
            )
            
            prefix_conditioning = model.prepare_conditioning(conditioning)
            
            # Generate with continuation from previous chunk
            with torch.inference_mode():
                codes = model.generate(
                    prefix_conditioning=prefix_conditioning,
                    audio_prefix_codes=previous_codes if chunk_idx > 0 else None,
                    max_new_tokens=86 * max(10, chunk_info['word_count']),
                    cfg_scale=3.0,
                    batch_size=1,
                    sampling_params={"min_p": 0.15},
                    progress_bar=True
                )
                
                # Save last portion for next chunk's continuation
                previous_codes = codes[:, :, -512:].detach().clone()
                
                # Decode to audio
                audio = model.autoencoder.decode(codes)[0]
                
                # Convert to PCM16
                if audio.abs().max() > 1:
                    audio = audio / audio.abs().max()
                audio = (audio * 32767).clamp(-32768, 32767).to(torch.int16)
                
                # Add to segments
                all_audio_segments.append(audio)
                
                # Ensure GPU sync
                torch.cuda.synchronize()
        
        # Concatenate all segments
        final_audio = torch.cat(all_audio_segments, dim=-1)
        return final_audio
        
    except Exception as e:
        print(f"Error generating audio: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise

def generate_combined_audio(text: str, voice_embedding: torch.Tensor):
    """Generate audio with proper chunking and continuation."""
    try:
        # Split text into chunks
        chunks = split_into_chunks(text)
        print(f"Split text into {len(chunks)} chunks")
        
        all_audio_segments = []
        previous_codes = None
        
        for chunk_idx, chunk_info in enumerate(chunks):
            chunk_text = chunk_info['text']
            print(f"Processing chunk {chunk_idx + 1}/{len(chunks)}: {chunk_text}")
            
            # Create conditioning
            conditioning = make_cond_dict(
                text=chunk_text,
                speaker=voice_embedding,
                emotion=[1.0, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.2],
                language="en-us",
                device=device,
                fmax=24000,
                pitch_std=45.0,
                speaking_rate=15.0,
                dnsmos_ovrl=4.0,
                vqscore_8=[0.78] * 8,
                speaker_noised=False,
                unconditional_keys=["emotion"]
            )
            
            prefix_conditioning = model.prepare_conditioning(conditioning)
            
            with torch.inference_mode():
                codes = model.generate(
                    prefix_conditioning=prefix_conditioning,
                    audio_prefix_codes=previous_codes if chunk_idx > 0 else None,
                    max_new_tokens=86 * max(10, chunk_info['word_count']),
                    cfg_scale=3.0,
                    batch_size=1,
                    sampling_params={"min_p": 0.15},
                    progress_bar=True
                )
                
                previous_codes = codes[:, :, -512:].detach().clone()
                audio = model.autoencoder.decode(codes)[0]
                
                if audio.abs().max() > 1:
                    audio = audio / audio.abs().max()
                audio = (audio * 32767).clamp(-32768, 32767).to(torch.int16)
                
                all_audio_segments.append(audio)
                torch.cuda.synchronize()
        
        final_audio = torch.cat(all_audio_segments, dim=-1)
        return final_audio
        
    except Exception as e:
        print(f"Error generating audio: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise

def sequential_chunk_text(text: str) -> list[dict]:
    """Split text into sequential chunks without overlapping content."""
    sentences = []
    current = []
    
    # First, split into complete sentences
    for line in text.split('\n'):
        words = line.strip().split()
        if not words:
            continue
            
        current.extend(words)
        if words[-1].endswith(('.', '!', '?')):
            if current:
                sentences.append(' '.join(current))
                current = []
    
    if current:
        sentences.append(' '.join(current))
    
    # Then group sentences into chunks with strict boundaries
    chunks = []
    current_chunk = []
    word_count = 0
    max_words = 25  # Balanced chunk size
    
    for sentence in sentences:
        sentence_words = len(sentence.split())
        
        # If adding this sentence would exceed limit, save current chunk
        if word_count + sentence_words > max_words and current_chunk:
            chunks.append({
                'text': ' '.join(current_chunk),
                'word_count': word_count
            })
            current_chunk = []
            word_count = 0
        
        # Add sentence to current chunk
        current_chunk.append(sentence)
        word_count += sentence_words
    
    # Add final chunk if any
    if current_chunk:
        chunks.append({
            'text': ' '.join(current_chunk),
            'word_count': word_count
        })
    
    return chunks

def generate_audio_chunk(text: str, voice_embedding: torch.Tensor, prev_codes=None, is_first_chunk=False, emotions=None):
    """Generate a single chunk of audio with proper continuation."""
    try:
        # Use provided emotions or default
        emotion_values = emotions if emotions is not None else [1.0, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.2]
        
        # Create conditioning
        conditioning = make_cond_dict(
            text=text,
            speaker=voice_embedding,
            emotion=emotion_values,  # Use the emotion values here
            language="en-us",
            device=device,
            fmax=24000,
            pitch_std=45.0,
            speaking_rate=15.0,
            dnsmos_ovrl=4.0,
            vqscore_8=[0.78] * 8,
            speaker_noised=False,
            unconditional_keys=["emotion"]
        )
        
        prefix_conditioning = model.prepare_conditioning(conditioning)
        
        print(f"\nServer Debug: Generating audio for text: {text}")
        if prev_codes is not None:
            print(f"Server Debug: Previous codes shape: {prev_codes.shape}")
            # Reset the last bit of previous codes to avoid repeating
            prev_codes = prev_codes[:, :, :-86].detach().clone()
            print(f"Server Debug: Trimmed previous codes shape: {prev_codes.shape}")
        
        with torch.inference_mode():
            # Adjust token count based on text length
            words = len(text.split())
            max_tokens = 86 * max(30, words * 2)  # Increased token count
            
            codes = model.generate(
                prefix_conditioning=prefix_conditioning,
                audio_prefix_codes=prev_codes,
                max_new_tokens=max_tokens,
                cfg_scale=3.0,
                batch_size=1,
                sampling_params={"min_p": 0.15},
                progress_bar=True
            )
            
            print(f"Server Debug: Generated codes shape: {codes.shape}")
            
            # Take more context but not the very end for continuation
            if codes.size(-1) >= 86 * 6:
                continuation_codes = codes[:, :, -86 * 6:-86].detach().clone()
            else:
                continuation_codes = None
            
            # Decode to audio
            audio = model.autoencoder.decode(codes)[0]
            
            print(f"Server Debug: Raw audio shape: {audio.shape}, min: {audio.min():.2f}, max: {audio.max():.2f}")
            
            if audio.numel() == 0 or torch.all(audio == 0):
                print(f"Server Warning: Empty or zero audio generated")
                return None, continuation_codes
            
            # Skip some initial audio if using continuation to avoid overlap
            if prev_codes is not None and audio.shape[-1] > 0:
                skip_samples = int(0.2 * model.autoencoder.sampling_rate)
                if audio.shape[-1] > skip_samples:
                    audio = audio[skip_samples:]
            
            # Safe normalization with dimension specified
            if audio.numel() > 0:
                abs_audio = torch.abs(audio)
                if abs_audio.numel() > 0:
                    max_val = abs_audio.max(dim=-1, keepdim=True)[0]
                    if max_val > 0:
                        audio = audio / max_val
                    
                # Convert to PCM16 after normalization check
                audio = (audio * 32767).clamp(-32768, 32767).to(torch.int16)
                
                print(f"Server Debug: Final audio shape: {audio.shape}, non-zero elements: {torch.count_nonzero(audio)}")
                return audio, continuation_codes
            else:
                print("Server Warning: Empty audio tensor generated")
                return None, continuation_codes
            
    except Exception as e:
        print(f"Server Error in generate_audio_chunk: {str(e)}")
        traceback.print_exc()
        return None, None

def sentence_chunk_text(text: str, max_words_per_chunk: int = 15) -> list[dict]:
    """
    Split text into chunks by complete sentences, with a soft word limit per chunk.
    
    Args:
        text: Input text to split
        max_words_per_chunk: Soft limit for words per chunk (will not break sentences)
        
    Returns:
        List of dicts containing text chunks and their word counts
    """
    # Clean and normalize text
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    
    # Split into sentences using basic punctuation
    sentence_pattern = r'[^.!?]+[.!?]+'
    sentences = re.findall(sentence_pattern, text)
    
    # Handle any remaining text that didn't end with punctuation
    remaining = re.sub(r'.*[.!?]+\s*', '', text).strip()
    if remaining:
        sentences.append(remaining + '.')
    
    chunks = []
    current_chunk = []
    current_word_count = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        sentence_words = len(sentence.split())
        
        # If adding this sentence would exceed limit and we have existing content,
        # save current chunk and start new one
        if current_chunk and (current_word_count + sentence_words > max_words_per_chunk):
            chunks.append({
                'text': ' '.join(current_chunk),
                'word_count': current_word_count
            })
            current_chunk = []
            current_word_count = 0
        
        # Add sentence to current chunk
        current_chunk.append(sentence)
        current_word_count += sentence_words
    
    # Add final chunk if any
    if current_chunk:
        chunks.append({
            'text': ' '.join(current_chunk),
            'word_count': current_word_count
        })
    
    return chunks

@app.route('/text-to-voice-speed', methods=['POST'])
def text_to_voice_speed():
    """Stream audio generation using sentence-based text chunking and audio continuation."""
    data = request.json
    if not data or 'text' not in data:
        return jsonify({"error": "No text provided"}), 400
    
    text = data['text']
    voice_name = data.get('voice', None)
    emotions_list = data.get('emotions', None)  # Get emotions array if provided
    
    if not voice_name:
        return jsonify({"error": "No voice specified"}), 400
        
    try:
        voice_embedding = load_voice_embedding(voice_name)
        chunks = sentence_chunk_text(text)
        print(f"\nServer: Starting generation of {len(chunks)} chunks")
        
        def check_client_connected():
            try:
                # The request is always connected initially
                if not hasattr(request, 'environ'):
                    return True
                # Check if client is still connected
                return request.environ.get('wsgi.input_terminated', True)
            except Exception:
                # If we can't check, assume connected
                return True

        def generate_chunks():
            prev_codes = None
            first_chunk = True
            
            for idx, chunk in enumerate(chunks):
                try:
                    # Only check connection after first chunk
                    if not first_chunk and not check_client_connected():
                        print("\nServer: Client disconnected, stopping generation")
                        return
                    
                    # Rest of chunk generation code
                    current_emotion = None
                    if emotions_list:
                        emotion_idx = min(idx, len(emotions_list) - 1)
                        current_emotion = emotions_list[emotion_idx]
                    
                    chunk_text = chunk['text']
                    # ...existing chunk processing code...
                    
                    success = False
                    max_attempts = 2  # Reduced from 3 to 2 attempts
                    
                    for attempt in range(max_attempts):
                        try:
                            # Only use continuation on first attempt of non-first chunks
                            use_prev_codes = prev_codes if (attempt == 0 and idx > 0) else None
                            
                            audio, continuation = generate_audio_chunk(
                                chunk_text, 
                                voice_embedding,
                                use_prev_codes,
                                is_first_chunk=(idx == 0),
                                emotions=current_emotion  # Pass the emotion values
                            )
                            
                            if audio is not None and audio.numel() > 0:
                                success = True
                                prev_codes = continuation
                                break
                                
                            print(f"Server: Attempt {attempt + 1} produced invalid audio, "
                                  f"{'retrying' if attempt < max_attempts-1 else 'skipping chunk'}")
                            
                        except Exception as e:
                            print(f"Server: Error in attempt {attempt + 1}: {str(e)}")
                            if attempt < max_attempts-1:
                                print("Server: Retrying without continuation...")
                    
                    if not success:
                        print(f"Server: Failed to generate audio for chunk {idx + 1}")
                        continue
                    
                    chunk_data = audio.cpu().numpy().tobytes()
                    print(f"Server: Generated {len(chunk_data)} bytes for chunk {idx + 1}")
                    print(f"Server: Audio Emotion/s: {current_emotion}")
                    # Check connection again before yielding
                    if check_client_connected():
                        yield chunk_data
                    else:
                        print("\nServer: Client disconnected before yielding chunk")
                        return
                    
                    first_chunk = False  # Mark first chunk as complete
                    torch.cuda.synchronize()
                    print(f"Server: Finished chunk {idx + 1}")
                
                except GeneratorExit:
                    print("\nServer: Stream closed by client")
                    return
                except Exception as e:
                    print(f"Server: Error in chunk {idx + 1}: {str(e)}")
                    continue

        response = app.response_class(
            generate_chunks(),
            mimetype='audio/pcm',
            headers={
                'Content-Type': 'audio/pcm',
                'Sample-Rate': str(model.autoencoder.sampling_rate),
                'Channels': '1',
                'Transfer-Encoding': 'chunked'
            },
            direct_passthrough=True
        )
        
        @response.call_on_close
        def on_close():
            print("\nServer: Stream closed, cleaning up")
            torch.cuda.empty_cache()
        
        return response
        
    except Exception as e:
        print(f"Server Error: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)