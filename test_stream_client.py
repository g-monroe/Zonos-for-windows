import requests
import pyaudio
import numpy as np
import time
from queue import Queue, Empty
from threading import Thread, Event

# Audio settings
CHUNK_SIZE = 4096
SAMPLE_RATE = 44100
CHANNELS = 1
FORMAT = pyaudio.paInt16

class AudioPlayer:
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.audio_queue = Queue()
        self.stop_event = Event()
        self.player_thread = None
        self.bytes_played = 0
        
    def start(self, sample_rate):
        if self.stream:
            self.stop()
            
        self.stream = self.p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=sample_rate,
            output=True,
            frames_per_buffer=CHUNK_SIZE
        )
        
        # Start playback thread
        self.stop_event.clear()
        self.player_thread = Thread(target=self._play_from_queue)
        self.player_thread.start()
        
    def _play_from_queue(self):
        while not self.stop_event.is_set():
            try:
                data = self.audio_queue.get(timeout=0.1)
                if data and self.stream:
                    self.stream.write(data)
                    self.bytes_played += len(data)
                    print(f"Client: Played {len(data)} bytes (total: {self.bytes_played})")
            except Empty:
                continue
            except Exception as e:
                print(f"Client: Playback error: {e}")
                break
                
    def write(self, data):
        if data and len(data) > 0:
            print(f"Client: Queueing {len(data)} bytes")
            self.audio_queue.put(data)
                
    def stop(self):
        self.stop_event.set()
        if self.player_thread:
            self.player_thread.join()
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
            
    def close(self):
        self.stop()
        self.p.terminate()

def stream_audio_speed(text, voice_name="morg"):
    """Test the speed streaming endpoint with threaded playback."""
    url = "http://localhost:5000/text-to-voice-speed"
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'audio/pcm'
    }
    data = {
        'text': text,
        'voice': voice_name,
        "emotions": [
                        [1.0, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.2],
                        [0.2, 0.8, 0.05, 0.05, 0.05, 0.05, 0.1, 0.2]
        ],
        "audioPrefix": True
    }

    player = None

    try:
        print("\nClient: Connecting to server...")
        with requests.post(url, headers=headers, json=data, stream=True) as response:
            if response.status_code == 200:
                print("Client: Connected, starting playback...")
                
                # Get sample rate from headers
                sample_rate = int(response.headers.get('Sample-Rate', '44100'))
                player = AudioPlayer()
                player.start(sample_rate)
                
                # Process incoming chunks
                buffer = bytearray()
                total_received = 0
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk and len(chunk) > 0:
                        buffer.extend(chunk)
                        total_received += len(chunk)
                        print(f"Client: Received {len(chunk)} bytes (total: {total_received})")
                        
                        # When we have enough data, send to player
                        while len(buffer) >= CHUNK_SIZE * 2:
                            audio_chunk = bytes(buffer[:CHUNK_SIZE * 2])
                            print(f"Client: Queuing {len(audio_chunk)} bytes for playback")
                            player.write(audio_chunk)
                            buffer = buffer[CHUNK_SIZE * 2:]
                
                # Play any remaining data
                if len(buffer) > 0:
                    print(f"Client: Processing final {len(buffer)} bytes")
                    player.write(bytes(buffer))
                    
                print("Client: Waiting for playback to complete...")
                # Wait for queue to empty
                while not player.audio_queue.empty():
                    time.sleep(0.1)
                    
                # Give a moment for final audio to play
                time.sleep(0.5)
                    
                print(f"Client: Finished playing {player.bytes_played} total bytes")
            else:
                print(f"Client Error: {response.status_code}")
                print(response.text)
                
    except Exception as e:
        print(f"Client Error: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        if player:
            player.close()
        print("Client: Stream finished")

if __name__ == "__main__":
    test_text = """This is a test of the streaming synthesis using text chunking. 
                   Each chunk should play as soon as it's generated. Like Wow! This stuff can be done in many different ways. In fact, 
                   there are many different ways to do this. 
                   But the most important thing is that it works! Let's see how it goes. See you later!"""
    print("Starting speed streaming synthesis...")
    stream_audio_speed(test_text)
    print("Streaming complete!")
