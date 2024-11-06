import datetime
import sys
import wave

sys.path.insert(0, '/app/ZEGGS')
sys.path.insert(0, 'ZEGGS')

import asyncio
import io
import json
import logging
import time
import traceback
from pathlib import Path
from threading import Lock
from typing import Dict

import numpy as np
import websockets
from websockets.server import WebSocketServerProtocol

from ZEGGS.generate import generate_gesture

# Set up logging to output to both file and console
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app.log')
    ]
)

logger = logging.getLogger(__name__)

class AudioBuffer:
    """Manages streaming audio data with a circular buffer and validation"""
    def __init__(self, buffer_size_seconds: float = 3.0, sample_rate: int = 48000,
                 min_sequence_length: int = 240, 
                 debug: bool = False, debug_path: Path = Path("debug")):
        self.sample_rate = sample_rate
        self.buffer_size = int(buffer_size_seconds * sample_rate)
        self.min_sequence_length = min_sequence_length
        self.buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self.write_pos = 0
        self.is_full = False
        self.lock = Lock()
        self.first_chunk = True
        
        # Calculate minimum buffer size needed for sequence
        self.min_buffer_samples = int((min_sequence_length / 60) * sample_rate)  # Convert frames to samples

        # Ensure buffer size is at least as large as minimum required
        requested_buffer_size = int(buffer_size_seconds * sample_rate)
        self.buffer_size = max(requested_buffer_size, self.min_buffer_samples)
        
        self.buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self.write_pos = 0
        self.is_full = False
        self.lock = Lock()
        self.first_chunk = True

        # Debugging related attributes
        self.debug = debug
        if self.debug:
            self.debug_path = Path(debug_path)
            self.debug_path.parents[0].mkdir(exist_ok=True)
            self.debug_path.mkdir(exist_ok=True)
            self.received_chunks = []
            self.chunk_count = 0
            self.total_samples = 0
            self.session_start = datetime.datetime.now()
            self.logger = logging.getLogger(__name__)

            self.logger.info(
                f"AudioBuffer initialized:"
                f"\n  Sample rate: {sample_rate}Hz"
                f"\n  Buffer size: {self.buffer_size} samples ({self.buffer_size/sample_rate:.2f}s)"
                f"\n  Min sequence length: {min_sequence_length} frames"
                f"\n  Min buffer samples: {self.min_buffer_samples} ({self.min_buffer_samples/sample_rate:.2f}s)"
            )

    def has_minimum_data(self) -> bool:
        """Check if buffer has minimum required data for processing"""
        with self.lock:
            if self.is_full:
                buffer_samples = self.buffer_size
            else:
                buffer_samples = self.write_pos
                
            has_enough = buffer_samples >= self.min_buffer_samples
            
            if self.debug:
                logger.debug(
                    f"Minimum data check:"
                    f"\n  Buffer size: {self.buffer_size}"
                    f"\n  Write position: {self.write_pos}"
                    f"\n  Is full: {self.is_full}"
                    f"\n  Available samples: {buffer_samples}"
                    f"\n  Required samples: {self.min_buffer_samples}"
                    f"\n  Result: {'ENOUGH DATA' if has_enough else 'NEED MORE DATA'}"
                )
            
            return has_enough

    def validate_audio_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """Validate and normalize audio chunk"""
        if self.debug:
            # Log raw chunk stats
            self.logger.debug(f"Raw chunk stats - size: {len(chunk)}, "
                            f"min: {np.min(chunk):.6f}, max: {np.max(chunk):.6f}, "
                            f"mean: {np.mean(chunk):.6f}, std: {np.std(chunk):.6f}")
            
            # Check for invalid values
            if np.any(np.isnan(chunk)) or np.any(np.isinf(chunk)):
                self.logger.error("Found NaN or Inf values in chunk!")
                chunk = np.nan_to_num(chunk, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Check amplitude range
            if np.max(np.abs(chunk)) > 1.0:
                self.logger.warning("Chunk contains values outside [-1, 1] range")
                chunk = np.clip(chunk, -1.0, 1.0)

        return chunk

    def add_samples(self, samples: np.ndarray) -> bool:
        """Add new samples to the buffer, returns True if buffer has minimum required data"""
        try:
            # Validate incoming samples
            samples = self.validate_audio_chunk(samples)
            
            if self.debug:
                self.chunk_count += 1
                self.received_chunks.append(samples.copy())
                self.total_samples += len(samples)
                logger.debug(
                    f"Processing chunk {self.chunk_count}:"
                    f"\n  Size: {len(samples)} samples"
                    f"\n  Buffer size: {self.buffer_size}"
                    f"\n  Min required: {self.min_buffer_samples}"
                    f"\n  Current position: {self.write_pos}"
                    f"\n  Total samples: {self.total_samples}"
                    f"\n  Is full: {self.is_full}"
                )

            with self.lock:
                # If incoming chunk is larger than buffer, resize buffer
                if len(samples) > self.buffer_size:
                    new_size = len(samples) * 2  # Double size to avoid frequent resizing
                    if self.debug:
                        logger.info(f"Resizing buffer from {self.buffer_size} to {new_size} samples")
                    new_buffer = np.zeros(new_size, dtype=np.float32)
                    if self.is_full:
                        new_buffer[:self.buffer_size] = self.get_buffer()
                    else:
                        new_buffer[:self.write_pos] = self.buffer[:self.write_pos]
                    self.buffer = new_buffer
                    self.buffer_size = new_size

                # Add new samples
                samples_length = len(samples)
                if self.write_pos + samples_length <= self.buffer_size:
                    # Simple case: enough space in buffer
                    self.buffer[self.write_pos:self.write_pos + samples_length] = samples
                    self.write_pos += samples_length
                    if self.write_pos >= self.buffer_size:
                        self.write_pos = 0
                        self.is_full = True
                        logger.debug("Buffer is full. Resetting write position to 0.")
                        logger.debug(f"Buffer state after fill:"
                                f"\n  Is full: {self.is_full}"
                                f"\n  Write position: {self.write_pos}"
                                f"\n  Total samples: {self.total_samples}")
                else:
                    # Split case: wrap around buffer
                    first_part = self.buffer_size - self.write_pos
                    second_part = samples_length - first_part
                    self.buffer[self.write_pos:] = samples[:first_part]
                    self.buffer[:second_part] = samples[first_part:]
                    self.write_pos = second_part
                    self.is_full = True
                    logger.debug("Buffer wrapped around during fill.")
                    logger.debug(f"Buffer state after wrap:"
                            f"\n  Is full: {self.is_full}"
                            f"\n  Write position: {self.write_pos}"
                            f"\n  Total samples: {self.total_samples}")
                
                has_enough = self.has_minimum_data()
                if self.debug:
                    logger.debug(
                        f"After processing chunk:"
                        f"\n  Write position: {self.write_pos}"
                        f"\n  Is full: {self.is_full}"
                        f"\n  Has minimum data: {has_enough}"
                        f"\n  Total samples received: {self.total_samples}"
                        f"\n  Available samples: {len(self.get_buffer())}"
                    )
                
                return has_enough

        except Exception as e:
            if self.debug:
                self.logger.error(f"Error processing audio chunk: {e}")
                self.save_debug_audio(reason=f"error_chunk_{self.chunk_count}")
            raise

    def save_debug_audio(self, client_id: str = "default", reason: str = "interval"):
        """Save accumulated audio chunks in PCM format"""
        if not self.debug or not self.received_chunks:
            return
            
        try:
            # Combine all chunks
            combined_audio = np.concatenate(self.received_chunks)
            
            # Convert to PCM
            pcm_samples = (combined_audio * 32767).astype(np.int16)
            
            # Generate filename
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = self.debug_path / f"audio_debug_{client_id}_{reason}_{timestamp}.wav"
            
            # Save as 16-bit PCM WAV
            with wave.open(str(filename), 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(pcm_samples.tobytes())
            
            # Save debug info
            debug_info = {
                "timestamp": timestamp,
                "reason": reason,
                "total_chunks": self.chunk_count,
                "total_samples": self.total_samples,
                "duration": len(combined_audio) / self.sample_rate,
                "max_amplitude": float(np.max(np.abs(combined_audio))),
                "min_amplitude": float(np.min(combined_audio)),
                "rms": float(np.sqrt(np.mean(combined_audio**2))),
                "sample_rate": self.sample_rate,
                "pcm_max": float(np.max(np.abs(pcm_samples))),
            }
            
            with open(str(filename).replace('.wav', '_info.json'), 'w') as f:
                json.dump(debug_info, f, indent=2)
                
            self.logger.info(f"Audio debug saved to {filename}")
            self.logger.info(f"Audio statistics:")
            self.logger.info(f"  Duration: {debug_info['duration']:.2f} seconds")
            self.logger.info(f"  Chunks: {debug_info['total_chunks']}")
            self.logger.info(f"  Samples: {debug_info['total_samples']}")
            self.logger.info(f"  PCM max: {debug_info['pcm_max']}")
            
        except Exception as e:
            self.logger.error(f"Error saving audio debug file: {e}")

    def get_buffer(self) -> np.ndarray:
        """Get the current buffer contents as float32 [-1,1] range"""
        with self.lock:
            if not self.is_full and self.write_pos == 0:
                return np.zeros(0, dtype=np.float32)
                
            if self.is_full:
                # Return buffer ordered from write_pos (oldest) to write_pos-1 (newest)
                return np.concatenate([
                    self.buffer[self.write_pos:],
                    self.buffer[:self.write_pos]
                ])
            else:
                # Return just the written portion
                return self.buffer[:self.write_pos].copy()

    def get_buffer_pcm(self) -> np.ndarray:
        """Get the current buffer contents converted to PCM format"""
        with self.lock:
            if not self.is_full and self.write_pos == 0:
                return np.zeros(0, dtype=np.int16)
                
            if self.is_full:
                # Return buffer ordered from write_pos (oldest) to write_pos-1 (newest)
                audio = np.concatenate([
                    self.buffer[self.write_pos:],
                    self.buffer[:self.write_pos]
                ])
            else:
                # Return just the written portion
                audio = self.buffer[:self.write_pos].copy()
            
            # Convert float32 [-1,1] to int16 [-32768,32767]
            return (audio * 32767).astype(np.int16)
    
    def reset_debug(self):
        """Clear debug storage"""
        if self.debug:
            self.received_chunks = []
            self.chunk_count = 0
            self.session_start = datetime.datetime.now()
            
    def reset(self):
        """Reset the buffer to initial state"""
        with self.lock:
            self.buffer.fill(0)
            self.write_pos = 0
            self.is_full = False
            self.first_chunk = True
            self.reset_debug()

class GestureWebSocketServer:
    def __init__(self, host: str = "localhost", port: int = 8000):
        self.host = host
        self.port = port
        self.active_connections: Dict[WebSocketServerProtocol, Dict] = {}
        self.min_sequence_length = 240  # Minimum sequence length for gesture generation
        
        # Create base debug directory at server start
        self.debug_base_path = Path("debug")
        try:
            self.debug_base_path.mkdir(exist_ok=True)
            logger.info(f"Debug directory initialized at {self.debug_base_path}")
        except Exception as e:
            logger.error(f"Failed to create debug directory: {e}")
            self.buffer_debug = False

        logger.info(f"Initializing server on {host}:{port}")

        # Load configuration once - removed relative path
        try:
            self.options, self.data_dir = self.load_options("options.json")
            self.setup_paths()
            logger.info("Successfully loaded options and setup paths")
        except Exception as e:
            logger.error(f"Failed to initialize server: {e}")
            raise
    
        # Processing parameters
        self.chunk_size = 4096  # Number of samples per chunk
        self.buffer_size_seconds = 6.0  # Size of sliding window
        self.sample_rate = 48000  # Expected sample rate
        self.buffer_debug = True
        self.process_interval = 0.5     # Process every 0.5 seconds
        self.first_chunk_processed = False
        
    def setup_paths(self):
        """Setup all necessary paths from options"""
        try:
            paths = self.options["paths"]
            self.base_path = self.data_dir
            
            # Get relative paths from options and make them absolute
            self.data_path = self.base_path / paths["path_processed_data"]
            
            # For models_dir, use the v2 directory structure
            self.network_path = self.data_dir / "outputs" / "v2" / "saved_models"
            
            # Set output and results paths
            self.output_path = self.data_dir / "outputs" / "v2"
            self.results_path = self.output_path / "results"
            
            # Create results directory if it doesn't exist
            self.results_path.mkdir(parents=True, exist_ok=True)
            
            # Log all paths for debugging
            logger.info(f"Base path: {self.base_path}")
            logger.info(f"Data path: {self.data_path}")
            logger.info(f"Network path: {self.network_path}")
            logger.info(f"Output path: {self.output_path}")
            logger.info(f"Results path: {self.results_path}")
        
        except Exception as e:
            logger.error(f"Error setting up paths: {e}")
            raise

    async def handle_client(self, websocket: WebSocketServerProtocol):
        """Handle individual client connections"""
        try:
            client_id = str(id(websocket))

            # Initialize client state with debugging enabled
            debug_path = self.debug_base_path / client_id if self.buffer_debug else None
            client_state = {
                'audio_buffer': AudioBuffer(
                    self.buffer_size_seconds, 
                    self.sample_rate,
                    min_sequence_length=self.min_sequence_length,
                    debug=self.buffer_debug,
                    debug_path=debug_path
                ),
                'processing_task': None,
                'last_process_time': 0,
                'has_sent_header': False  # Track if BVH header has been sent
            }
            self.active_connections[websocket] = client_state
            
            
            async for message in websocket:
                if isinstance(message, bytes):
                    audio_samples = np.frombuffer(message, dtype=np.float32)
                    logger.debug(f"Received audio chunk: {len(audio_samples)} samples")

                    has_enough_data = client_state['audio_buffer'].add_samples(audio_samples)
                    current_time = time.time()

                    # Periodic debug save if enabled
                    if self.buffer_debug and current_time % 30 < 1:
                        client_state['audio_buffer'].save_debug_audio(client_id)

                    logger.debug(
                        f"Checking processing conditions:"
                        f"\n  Has enough data: {has_enough_data}"
                        f"\n  Header sent: {client_state['has_sent_header']}"
                        f"\n  Time since last: {current_time - client_state['last_process_time']:.2f}s"
                    )
                            
                    # Simplify initial processing condition
                    if not client_state['has_sent_header'] and has_enough_data:
                        logger.debug("Starting initial processing with header")
                        # Start new processing task
                        client_state['processing_task'] = asyncio.create_task(
                            self.process_audio_buffer(
                                websocket, 
                                client_state['audio_buffer'],
                                send_header=True
                            )
                        )
                        client_state['last_process_time'] = current_time
                        client_state['has_sent_header'] = True
                        
                    # Regular processing condition
                    elif has_enough_data and current_time - client_state['last_process_time'] >= self.process_interval:
                        logger.debug("Starting regular processing")
                        # Cancel previous processing if still running
                        if client_state['processing_task'] is not None:
                            client_state['processing_task'].cancel()
                            
                        # Start new processing task
                        client_state['processing_task'] = asyncio.create_task(
                            self.process_audio_buffer(
                                websocket, 
                                client_state['audio_buffer'],
                                send_header=False
                            )
                        )
                        client_state['last_process_time'] = current_time
                    else:
                        logger.debug("Skipping processing - conditions not met")
                        
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client {client_id} connection closed")
        except Exception as e:
            logger.error(f"Error handling client: {str(e)}\n{traceback.format_exc()}")
            try:
                await websocket.close(1011, f"Internal error: {str(e)}")
            except Exception:
                pass
        finally:
            # Cleanup client state
            if websocket in self.active_connections:
                del self.active_connections[websocket]
            logger.info(f"Client disconnected. Remaining connections: {len(self.active_connections)}")


    async def process_audio_buffer(self, websocket: WebSocketServerProtocol, 
                                 audio_buffer: AudioBuffer, send_header: bool = False):
        """Process the current audio buffer and generate BVH data"""
        try:
            logger.debug(
                f"Starting process_audio_buffer:"
                f"\n  Send header: {send_header}"
                f"\n  Buffer available: {len(audio_buffer.get_buffer())} samples"
            )
            # Get current buffer contents
            buffer_data = audio_buffer.get_buffer()
            
            # Create WAV data in memory
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(4)  # 32-bit float
                wav_file.setframerate(48000)  # Sample rate
                wav_file.writeframes(buffer_data.tobytes())
            
            logger.debug("WAV buffer created, calling generate_gesture")

            try:
                # Generate gesture data
                gesture_generator = generate_gesture(
                    audio_file=wav_buffer,  # Pass the BytesIO object directly
                    styles=['Neutral'],
                    network_path=self.network_path,
                    data_path=self.data_path,
                    results_path=self.results_path,
                    style_encoding_type='label',
                    first_pose='data/clean/001_Neutral_0_mirror_x_1_0.bvh',
                    temperature=1.0,
                    send_header=send_header  # Pass to generate_gesture
                )

                logger.debug("Gesture generation completed, starting streaming")

                # Stream generated BVH data
                animation_data = await gesture_generator
                if animation_data and websocket.open:
                    logger.debug(f"Sending {len(animation_data.getvalue())} bytes of animation data")
                    await websocket.send(animation_data.getvalue())
                    logger.debug("Animation data sent successfully")

            except Exception as e:
                logger.error(f"Error in gesture generation: {str(e)}\n{traceback.format_exc()}")
                raise

        except asyncio.CancelledError:
            logger.info("Processing cancelled - new buffer ready")
        except Exception as e:
            logger.error(f"Error processing audio buffer: {str(e)}\n{traceback.format_exc()}")
            if websocket.open:
                await websocket.close(1011, f"Processing error: {str(e)}")


    @staticmethod
    def load_options(options_path: str) -> tuple:
        """Load and validate options file"""
        current_path = Path.cwd()
        
        # Find data directory
        data_dir = None
        if (current_path / 'data').exists():
            data_dir = current_path / 'data'
        elif (current_path.parent / 'data').exists():
            data_dir = current_path.parent / 'data'
        
        if data_dir is None:
            raise ValueError("'data' directory not found")
        
        # Modified path to match your structure
        options_file = data_dir / "outputs" / "v2" / "options.json"
        if not options_file.exists():
            raise ValueError(f"'options.json' not found in {options_file}")
        
        logger.info(f"Loading options from: {options_file}")
        
        with open(options_file, "r") as f:
            options = json.load(f)
        
        return options, data_dir

    async def start_server(self):
        """Start the WebSocket server"""
        logger.info(f"Starting WebSocket server on ws://{self.host}:{self.port}")
        
        try:
            async with websockets.serve(
                self.handle_client,
                self.host,
                self.port,
                max_size=None,
                ping_interval=None,
                ping_timeout=None
            ):
                logger.info(f"WebSocket server running at ws://{self.host}:{self.port}")
                await asyncio.Future()  # Run forever
        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            logger.error(traceback.format_exc())
            raise

def main():
    logger.info("Starting application...")
    try:
        server = GestureWebSocketServer(host="0.0.0.0", port=8000)
        asyncio.run(server.start_server())
    except Exception as e:
        logger.error(f"Main application error: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    logger.info("Initializing main application")
    main()