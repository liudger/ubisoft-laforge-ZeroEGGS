import io
import json
import logging
import os
import tempfile
import wave
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Tuple, Union

import numpy as np
from omegaconf import DictConfig
from pydub import AudioSegment


@dataclass
class AudioConfig:
    """Configuration for audio processing matching ZEGGS config"""
    sampling_rate: int = 16000
    pre_emphasis: bool = False
    pre_emph_coeff: float = 0.97
    
    @classmethod
    def from_config_file(cls, config_path: Union[str, Path]) -> 'AudioConfig':
        """Create AudioConfig from ZEGGS config file"""
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        return cls(
            sampling_rate=config_data['audio_conf']['sampling_rate'],
            pre_emphasis=config_data['audio_conf']['pre_emphasis'],
            pre_emph_coeff=config_data['audio_conf']['pre_emph_coeff']
        )
        
    @classmethod
    def from_dict_config(cls, config: DictConfig) -> 'AudioConfig':
        """Create AudioConfig from DictConfig"""
        return cls(
            sampling_rate=config.audio_conf.sampling_rate,
            pre_emphasis=config.audio_conf.pre_emphasis,
            pre_emph_coeff=config.audio_conf.pre_emph_coeff
        )

class AudioProcessor:
    """Handles basic audio file operations"""
    
    def __init__(self, config: Union[AudioConfig, str, Path, dict, DictConfig]):
        if isinstance(config, (str, Path)):
            self.config = AudioConfig.from_config_file(config)
        elif isinstance(config, dict):
            self.config = AudioConfig(**config)
        elif isinstance(config, DictConfig):
            self.config = AudioConfig.from_dict_config(config)
        else:
            self.config = config
            
        self.logger = logging.getLogger(__name__)
        self._temp_files = set()
        
    def __del__(self):
        """Cleanup temporary files on deletion"""
        self.cleanup_temp_files()
        
    def cleanup_temp_files(self):
        """Clean up any temporary files created during processing"""
        for temp_file in self._temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except Exception as e:
                self.logger.warning(f"Failed to delete temporary file {temp_file}: {e}")
        self._temp_files.clear()
        
    @contextmanager
    def _temp_file_handler(self, suffix: str = '.wav') -> Path:
        """Context manager for handling temporary files"""
        logging.debug(f"Creating temporary file with suffix {suffix}")
        temp_file = None
        try:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            temp_path = Path(temp_file.name)
            self._temp_files.add(temp_path)
            yield temp_path
        finally:
            if temp_file:
                temp_file.close()

    def _read_wav_file(self, file_obj: Union[str, Path, BinaryIO, io.BytesIO]) -> Tuple[np.ndarray, int]:
        """Internal method to read WAV file data"""
        try:
            if isinstance(file_obj, (str, Path)):
                with wave.open(str(file_obj), 'rb') as wav_file:
                    return self._extract_wav_data(wav_file)
            elif isinstance(file_obj, (io.BytesIO, BinaryIO)):
                # Reset position to start
                file_obj.seek(0)
                with wave.open(file_obj, 'rb') as wav_file:
                    return self._extract_wav_data(wav_file)
            else:
                raise ValueError(f"Unsupported file object type: {type(file_obj)}")
        except Exception as e:
            raise RuntimeError(f"Error reading WAV data: {e}")

    def _extract_wav_data(self, wav_file) -> Tuple[np.ndarray, int]:
        """Extract audio data from wave file object"""
        sample_rate = wav_file.getframerate()
        # Read the frames and convert to float32
        audio_data = np.frombuffer(
            wav_file.readframes(wav_file.getnframes()),
            dtype=np.int16
        ).astype(np.float32) / 32768.0  # Normalize to [-1, 1]
        return audio_data, sample_rate

    def convert_to_wav(self, input_source: Union[str, Path, BinaryIO, io.BytesIO]) -> io.BytesIO:
        """Convert audio data to WAV format with specified sample rate"""
        try:
            # Handle BytesIO input
            if isinstance(input_source, io.BytesIO):
                input_source.seek(0)
                audio = AudioSegment.from_wav(input_source)
            # Handle file-like object
            elif isinstance(input_source, BinaryIO):
                audio = AudioSegment.from_wav(input_source)
            # Handle file path
            else:
                audio = AudioSegment.from_file(str(input_source))
                    
            # Set sample rate
            if audio.frame_rate != self.config.sampling_rate:
                audio = audio.set_frame_rate(self.config.sampling_rate)
                    
            # Convert to mono
            if audio.channels > 1:
                audio = audio.set_channels(1)
                    
            # Export as WAV to BytesIO
            output = io.BytesIO()
            audio.export(output, format='wav')
            output.seek(0)
            return output
                
        except Exception as e:
            raise RuntimeError(f"Error converting audio to WAV: {e}")
            
    def load_audio(self, input_source: Union[str, Path, BinaryIO, io.BytesIO]) -> Tuple[np.ndarray, int]:
        """Load audio data with correct sample rate"""
        try:
            # For BytesIO or file-like objects containing raw PCM data
            if isinstance(input_source, (io.BytesIO, BinaryIO)):
                # Convert the raw PCM data to WAV format
                logging.debug("Converting raw PCM data to WAV format")
                wav_buffer = self.convert_to_wav(input_source)
                return self._read_wav_file(wav_buffer)
            
            # For file paths
            elif isinstance(input_source, (str, Path)):
                wav_path = input_source
                if not str(wav_path).lower().endswith('.wav'):
                    with self._temp_file_handler() as temp_path:
                        audio = AudioSegment.from_file(str(wav_path))
                        audio.export(str(temp_path), format='wav')
                        return self._read_wav_file(temp_path)
                else:
                    return self._read_wav_file(wav_path)
            else:
                raise ValueError(f"Unsupported input source type: {type(input_source)}")
            
        except Exception as e:
            raise RuntimeError(f"Error loading audio: {e}")
        finally:
            self.cleanup_temp_files()

# Example usage
if __name__ == "__main__":
    # Load config from file
    config_path = "data_pipeline_conf_v1.json"
    
    # Create processor
    processor = AudioProcessor(config_path)
    
    # Process a file
    try:
        audio_data, sample_rate = processor.load_audio("example.mp3")
        print(f"Loaded audio: {len(audio_data)} samples at {sample_rate}Hz")
    except Exception as e:
        print(f"Error processing audio: {e}")