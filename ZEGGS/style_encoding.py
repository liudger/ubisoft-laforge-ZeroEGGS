from dataclasses import dataclass
from typing import Union, List, Optional, Tuple
from pathlib import Path
import torch
import numpy as np
import logging
from enum import Enum

class StyleEncodingType(Enum):
    LABEL = "label"
    EXAMPLE = "example"
    
    @classmethod
    def from_string(cls, value: str) -> 'StyleEncodingType':
        try:
            return cls(value.lower())
        except ValueError:
            raise ValueError(f"Invalid style encoding type: {value}. Must be 'label' or 'example'")

@dataclass
class StyleConfig:
    """Configuration for style encoding"""
    encoding_type: StyleEncodingType
    temperature: float = 1.0
    blend_weights: Optional[List[float]] = None
    
    def __post_init__(self):
        if isinstance(self.encoding_type, str):
            self.encoding_type = StyleEncodingType.from_string(self.encoding_type)
            
        if self.temperature <= 0:
            raise ValueError(f"Temperature must be positive, got {self.temperature}")
            
        if self.blend_weights is not None:
            if not self.blend_weights:
                raise ValueError("blend_weights cannot be empty if provided")
            if abs(sum(self.blend_weights) - 1.0) > 1e-6:
                raise ValueError("blend_weights must sum to 1.0")

class StyleEncoder:
    """Enhanced style encoding system"""
    
    def __init__(self, label_names: List[str], device: torch.device):
        self.label_names = label_names
        self.device = device
        self.nlabels = len(label_names)
        self.logger = logging.getLogger(__name__)
        
    def encode_styles(self, 
                     styles: List[Union[str, Path, np.ndarray]],
                     config: StyleConfig,
                     network_style_encoder=None) -> torch.Tensor:
        """
        Encode multiple styles with optional blending
        
        Args:
            styles: List of styles (either labels, paths to BVH files, or numpy arrays)
            config: Style encoding configuration
            network_style_encoder: Neural network for example-based encoding
            
        Returns:
            Encoded styles tensor
        """
        try:
            # Validate inputs
            self._validate_inputs(styles, config, network_style_encoder)
            
            # Encode each style
            style_encodings = []
            for style in styles:
                if config.encoding_type == StyleEncodingType.LABEL:
                    encoding = self._encode_label(style)
                else:  # EXAMPLE
                    encoding = self._encode_example(style, network_style_encoder, config.temperature)
                style_encodings.append(encoding)
            
            # Blend styles if needed
            if len(style_encodings) > 1:
                return self._blend_styles(style_encodings, config.blend_weights)
            
            return style_encodings[0]
            
        except Exception as e:
            self.logger.error(f"Error encoding styles: {str(e)}")
            raise
            
    def _validate_inputs(self, 
                        styles: List[Union[str, Path, np.ndarray]], 
                        config: StyleConfig,
                        network_style_encoder) -> None:
        """Validate all inputs before processing"""
        if not styles:
            raise ValueError("No styles provided")
            
        if config.encoding_type == StyleEncodingType.EXAMPLE and network_style_encoder is None:
            raise ValueError("network_style_encoder required for example-based encoding")
            
        if config.encoding_type == StyleEncodingType.LABEL:
            invalid_styles = [s for s in styles if isinstance(s, str) and s not in self.label_names]
            if invalid_styles:
                raise ValueError(f"Invalid style labels: {invalid_styles}")
                
    def _encode_label(self, style: str) -> torch.Tensor:
        """Encode a single label-based style"""
        style_index = self.label_names.index(style)
        encoding = torch.zeros((1, self.nlabels), dtype=torch.float32, device=self.device)
        encoding[0, style_index] = 1.0
        return encoding
        
    def _encode_example(self, 
                       style: Union[Path, np.ndarray],
                       network_style_encoder,
                       temperature: float) -> torch.Tensor:
        """Encode a single example-based style"""
        if isinstance(style, (Path, str)):
            # Process BVH file
            example_features = self._process_bvh_file(style)
        else:
            # Direct numpy array input
            example_features = torch.as_tensor(style, dtype=torch.float32, device=self.device)
            
        # Encode using network
        with torch.no_grad():
            encoding, _, _ = network_style_encoder(example_features[None], temperature)
            
        return encoding
        
    def _process_bvh_file(self, bvh_path: Path) -> torch.Tensor:
        """Process BVH file to extract features"""
        from ZEGGS.anim import bvh  # Import here to avoid circular imports
        
        try:
            anim_data = bvh.load(str(bvh_path))
            # Extract features (implement specific feature extraction logic)
            # This is a placeholder - actual implementation would depend on your needs
            features = self._extract_features_from_anim(anim_data)
            return torch.as_tensor(features, dtype=torch.float32, device=self.device)
            
        except Exception as e:
            raise RuntimeError(f"Error processing BVH file {bvh_path}: {str(e)}")
            
    def _extract_features_from_anim(self, anim_data: dict) -> np.ndarray:
        """Extract features from animation data"""
        # Implement your feature extraction logic here
        # This is highly dependent on your specific needs
        raise NotImplementedError("Feature extraction not implemented")
        
    def _blend_styles(self, 
                     style_encodings: List[torch.Tensor],
                     blend_weights: Optional[List[float]] = None) -> torch.Tensor:
        """Blend multiple style encodings"""
        if blend_weights is None:
            # Equal weighting if not specified
            blend_weights = [1.0 / len(style_encodings)] * len(style_encodings)
            
        # Stack and blend
        stacked = torch.stack(style_encodings, dim=0)
        weights = torch.tensor(blend_weights, device=self.device).view(-1, 1, 1)
        return (stacked * weights).sum(dim=0)