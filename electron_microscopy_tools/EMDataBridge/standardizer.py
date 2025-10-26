#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standardizer module for electron microscopy data formats.

This module provides functionality to standardize various electron microscopy
data formats to a common schema using LLM-based approaches.
"""

import os
import json
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Union, List, Optional, Tuple
"""
Optional ML deps (transformers/hyperspy/h5py) are loaded lazily. This allows
the package to be imported in environments without heavy ML dependencies.
"""

# Lazy ML imports - avoid raising at import time
_TRANSFORMERS_AVAILABLE = False
try:
    # We don't import the specific classes at module import time to keep import
    # lightweight. The EMDataStandardizer will attempt to import when needed.
    import transformers  # type: ignore
    _TRANSFORMERS_AVAILABLE = True
except Exception:
    _TRANSFORMERS_AVAILABLE = False

# Import format-specific libraries
try:
    import mrcfile
except ImportError:
    print("Warning: mrcfile not installed. MRC format support will be limited.")

try:
    import hyperspy.api as hs
except ImportError:
    hs = None
    print("Warning: hyperspy not installed. DM3/DM4 format support will be limited.")

try:
    import h5py
except ImportError:
    print("Warning: h5py not installed. HDF5 format support will be limited.")


class EMDataStandardizer:
    """
    A class for standardizing electron microscopy data formats using LLM-based approaches.
    
    This standardizer can detect the format of input files and convert them to a
    standardized representation with consistent metadata schema.
    """
    
    # Standard schema for electron microscopy metadata
    STANDARD_SCHEMA = {
        "microscope": {
            "name": str,
            "voltage": float,  # kV
            "magnification": float,
            "camera_length": Optional[float],  # mm
            "spot_size": Optional[float],
        },
        "sample": {
            "name": str,
            "description": str,
            "preparation": Optional[str],
        },
        "acquisition": {
            "date": str,  # ISO format
            "operator": Optional[str],
            "exposure_time": Optional[float],  # seconds
            "tilt_angle": Optional[float],  # degrees
            "defocus": Optional[float],  # μm
        },
        "image": {
            "width": int,  # pixels
            "height": int,  # pixels
            "bit_depth": int,
            "pixel_size": Optional[float],  # nm
        },
        "processing": {
            "software": Optional[str],
            "operations": Optional[List[str]],
        }
    }
    
    def __init__(self, llm_model_name: str = "bert-base-uncased", llm_backend: str = "auto", llm_model_path: str | None = None):
        """
        Initialize the standardizer with an optional LLM model for metadata extraction.
        
        Args:
            llm_model_name: Name of the pre-trained model to use for metadata extraction
        """
        self.supported_formats = {
            ".mrc": self._process_mrc,
            ".em": self._process_em,
            ".dm3": self._process_dm,
            ".dm4": self._process_dm,
            ".tif": self._process_tiff,
            ".tiff": self._process_tiff,
            ".h5": self._process_hdf5,
            ".hdf5": self._process_hdf5,
            ".msa": self._process_msa,
            ".emd": self._process_emd,
            ".ser": self._process_ser,
            ".hds": self._process_hitachi,
            ".oip": self._process_oxford,
            ".inca": self._process_oxford,
            ".azw": self._process_oxford,
            ".azd": self._process_oxford,
        }
        
        # LLM configuration. The adapter is created lazily when needed.
        # llm_backend: 'auto' | 'llama' | 'mock' | 'none'
        self.llm_backend = llm_backend
        self.llm_model_path = llm_model_path or os.environ.get("LLAMA_MODEL_PATH")
        self.llm_adapter = None
        # Legacy transformers model name (kept for backward compatibility)
        self.llm_model_name = llm_model_name
    
    def standardize(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Standardize the given electron microscopy data file.
        
        Args:
            file_path: Path to the electron microscopy data file
            
        Returns:
            A dictionary with standardized data and metadata
        
        Raises:
            ValueError: If the file format is not supported
        """
        file_path = Path(file_path)
        suffix = file_path.suffix.lower()
        
        if suffix not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {suffix}")
        
        # Process the file using the appropriate handler
        data, metadata = self.supported_formats[suffix](file_path)
        
        # Optionally run LLM-based mapping if an adapter is available.
        # Lazy-initialize the adapter depending on configuration and environment.
        def _deep_update(d: dict, u: dict):
            """Recursively update dict d with u, merging nested dicts."""
            for k, v in u.items():
                if isinstance(v, dict) and isinstance(d.get(k), dict):
                    _deep_update(d[k], v)
                else:
                    d[k] = v

        if self.llm_backend and self.llm_backend.lower() != "none":
            # Lazily import adapter factory to avoid adding heavy deps to import time
            try:
                if self.llm_adapter is None:
                    from .llm_adapter import get_llm_adapter, LLMAdapterError

                    prefer = None
                    if self.llm_backend.lower() == "mock":
                        prefer = "mock"
                    elif self.llm_backend.lower() == "llama":
                        prefer = "llama"
                    elif self.llm_backend.lower() == "auto":
                        prefer = "llama" if self.llm_model_path else None

                    if prefer:
                        try:
                            # Pass model_path if available for llama adapter
                            if prefer == "llama":
                                self.llm_adapter = get_llm_adapter(prefer="llama", model_path=self.llm_model_path)
                            else:
                                self.llm_adapter = get_llm_adapter(prefer=prefer)
                        except Exception as e:
                            print(f"Warning: could not initialize LLM adapter ({prefer}): {e}")
                            self.llm_adapter = None

                if self.llm_adapter is not None:
                    try:
                        mapping = self.llm_adapter.map_metadata(metadata)
                        if isinstance(mapping, dict) and "canonical" in mapping:
                            # Merge LLM-proposed canonical fields into metadata
                            _deep_update(metadata, mapping.get("canonical", {}))
                        # Capture provenance info for later attachment (after validation)
                        prov = mapping.get("provenance") if isinstance(mapping, dict) else None
                    except Exception as e:
                        print(f"Warning: LLM mapping failed: {e}")
                        prov = None
            except Exception:
                # Any failure to import adapters shouldn't break standardization
                pass
        
        # Validate against standard schema
        standardized_metadata = self._validate_and_standardize_metadata(metadata)

        # Attach LLM provenance (if any) to the validated metadata for auditing
        if 'prov' in locals() and prov:
            standardized_metadata.setdefault("_llm_provenance", {}).update(prov)

        return {
            "data": data,
            "metadata": standardized_metadata,
            "original_format": suffix[1:],  # Remove leading dot
        }
    
    def _process_mrc(self, file_path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Process MRC format files.
        
        Args:
            file_path: Path to the MRC file
            
        Returns:
            Tuple of (data array, metadata dictionary)
        """
        with mrcfile.open(file_path) as mrc:
            data = mrc.data
            metadata = {
                "microscope": {
                    "name": "Unknown",
                    "voltage": 0.0,
                    "magnification": 0.0,
                },
                "sample": {
                    "name": file_path.stem,
                    "description": "",
                },
                "acquisition": {},
                "image": {
                    "width": data.shape[1] if len(data.shape) > 1 else data.shape[0],
                    "height": data.shape[0],
                    "bit_depth": data.dtype.itemsize * 8,
                },
                "processing": {},
            }
            
            # Extract additional metadata from MRC header
            if hasattr(mrc.header, 'exttyp'):
                metadata["processing"]["software"] = str(mrc.header.exttyp)
                
            if hasattr(mrc.header, 'mx') and hasattr(mrc.header, 'my'):
                metadata["image"]["width"] = mrc.header.mx
                metadata["image"]["height"] = mrc.header.my
                
            if hasattr(mrc.header, 'xlen') and hasattr(mrc.header, 'mx'):
                # Convert to nm (MRC stores in Å)
                pixel_size_x = (mrc.header.xlen / mrc.header.mx) / 10
                metadata["image"]["pixel_size"] = pixel_size_x
        
        return data, metadata
    
    def _process_em(self, file_path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Process EM format files.
        
        Args:
            file_path: Path to the EM file
            
        Returns:
            Tuple of (data array, metadata dictionary)
        """
        # This is a placeholder - would need pyem or similar library
        # For now, we'll just raise an error
        raise NotImplementedError("EM format processing is not yet implemented")
    
    def _process_dm(self, file_path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Process DM3/DM4 format files using HyperSpy.
        
        Args:
            file_path: Path to the DM3/DM4 file
            
        Returns:
            Tuple of (data array, metadata dictionary)
        """
        if 'hs' not in globals():
            raise ImportError("HyperSpy is required for DM3/DM4 processing")
        
        # Load the file with HyperSpy
        s = hs.load(file_path)
        data = s.data
        
        # Initialize metadata structure
        metadata = {
            "microscope": {
                "name": "Unknown",
                "voltage": 0.0,
                "magnification": 0.0,
            },
            "sample": {
                "name": file_path.stem,
                "description": "",
            },
            "acquisition": {},
            "image": {
                "width": data.shape[1] if len(data.shape) > 1 else data.shape[0],
                "height": data.shape[0],
                "bit_depth": data.dtype.itemsize * 8,
            },
            "processing": {},
        }
        
        # Extract metadata from HyperSpy signal
        if hasattr(s.metadata, 'General') and hasattr(s.metadata.General, 'title'):
            metadata["sample"]["name"] = s.metadata.General.title
        
        if hasattr(s.metadata, 'Acquisition_instrument'):
            acq_inst = s.metadata.Acquisition_instrument
            
            if hasattr(acq_inst, 'TEM'):
                tem = acq_inst.TEM
                
                if hasattr(tem, 'microscope'):
                    metadata["microscope"]["name"] = tem.microscope
                    
                if hasattr(tem, 'beam_energy'):
                    metadata["microscope"]["voltage"] = tem.beam_energy
                    
                if hasattr(tem, 'magnification'):
                    metadata["microscope"]["magnification"] = tem.magnification
        
        # Extract pixel size information
        if hasattr(s, 'axes_manager'):
            axes = s.axes_manager
            
            if len(axes) >= 2 and hasattr(axes[0], 'scale') and hasattr(axes[0], 'units'):
                # Convert to nm if necessary
                scale = axes[0].scale
                units = axes[0].units
                
                if units.lower() in ['µm', 'um']:
                    scale *= 1000  # Convert µm to nm
                elif units.lower() == 'Å' or units.lower() == 'a':
                    scale /= 10  # Convert Å to nm
                
                metadata["image"]["pixel_size"] = scale
        
        return data, metadata
    
    def _process_tiff(self, file_path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Process TIFF format files.
        
        Args:
            file_path: Path to the TIFF file
            
        Returns:
            Tuple of (data array, metadata dictionary)
        """
        from PIL import Image, TiffTags
        import numpy as np
        
        # Open the TIFF file
        with Image.open(file_path) as img:
            # Convert to numpy array
            data = np.array(img)
            
            # Initialize metadata structure
            metadata = {
                "microscope": {
                    "name": "Unknown",
                    "voltage": 0.0,
                    "magnification": 0.0,
                },
                "sample": {
                    "name": file_path.stem,
                    "description": "",
                },
                "acquisition": {},
                "image": {
                    "width": data.shape[1] if len(data.shape) > 1 else data.shape[0],
                    "height": data.shape[0],
                    "bit_depth": data.dtype.itemsize * 8,
                },
                "processing": {},
            }
            
            # Extract metadata from TIFF tags
            if hasattr(img, 'tag'):
                for tag_id, value in img.tag.items():
                    tag_name = TiffTags.TAGS.get(tag_id, {}).get('name', f"Tag_{tag_id}")
                    
                    # Map known TIFF tags to our metadata schema
                    if tag_name == 'ImageDescription':
                        metadata["sample"]["description"] = value[0]
                    elif tag_name == 'Software':
                        metadata["processing"]["software"] = value[0]
                    elif tag_name == 'DateTime':
                        metadata["acquisition"]["date"] = value[0]
                    elif tag_name == 'XResolution' and len(value) > 0:
                        # This is a simplification - proper resolution calculation would be more complex
                        metadata["image"]["pixel_size"] = 1000 / float(value[0])  # Rough conversion to nm
        
        return data, metadata
    
    def _process_hdf5(self, file_path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Process HDF5 format files.
        
        Args:
            file_path: Path to the HDF5 file
            
        Returns:
            Tuple of (data array, metadata dictionary)
        """
        if 'h5py' not in globals():
            raise ImportError("h5py is required for HDF5 processing")
        
        with h5py.File(file_path, 'r') as f:
            # Try to find the main dataset
            data = None
            metadata_dict = {}
            
            # Common dataset paths in EM HDF5 files
            dataset_paths = ['data', 'Data', 'image', 'Image', 'volume', 'Volume']
            metadata_paths = ['metadata', 'Metadata', 'info', 'Info']
            
            # Try to find the main dataset
            for path in dataset_paths:
                if path in f:
                    data = f[path][()]
                    break
            
            # If not found, take the first dataset that looks like image data
            if data is None:
                for key in f.keys():
                    if isinstance(f[key], h5py.Dataset) and len(f[key].shape) >= 2:
                        data = f[key][()]
                        break
            
            # If still not found, raise an error
            if data is None:
                raise ValueError("Could not find image data in HDF5 file")
            
            # Try to find metadata
            for path in metadata_paths:
                if path in f:
                    # Extract metadata as a dictionary
                    metadata_group = f[path]
                    self._extract_hdf5_metadata(metadata_group, metadata_dict)
            
            # Initialize standard metadata structure
            metadata = {
                "microscope": {
                    "name": metadata_dict.get('microscope_name', "Unknown"),
                    "voltage": metadata_dict.get('voltage', 0.0),
                    "magnification": metadata_dict.get('magnification', 0.0),
                },
                "sample": {
                    "name": metadata_dict.get('sample_name', file_path.stem),
                    "description": metadata_dict.get('sample_description', ""),
                },
                "acquisition": {
                    "date": metadata_dict.get('acquisition_date', ""),
                    "operator": metadata_dict.get('operator', None),
                },
                "image": {
                    "width": data.shape[1] if len(data.shape) > 1 else data.shape[0],
                    "height": data.shape[0],
                    "bit_depth": data.dtype.itemsize * 8,
                    "pixel_size": metadata_dict.get('pixel_size', None),
                },
                "processing": {
                    "software": metadata_dict.get('software', None),
                },
            }
        
        return data, metadata
    
    def _extract_hdf5_metadata(self, group, metadata_dict, prefix=''):
        """
        Recursively extract metadata from HDF5 groups and datasets.
        
        Args:
            group: HDF5 group or dataset
            metadata_dict: Dictionary to store extracted metadata
            prefix: Prefix for nested keys
        """
        if isinstance(group, h5py.Group):
            for key, item in group.items():
                full_key = f"{prefix}_{key}" if prefix else key
                if isinstance(item, h5py.Group):
                    self._extract_hdf5_metadata(item, metadata_dict, full_key)
                else:
                    try:
                        value = item[()]
                        if isinstance(value, np.ndarray) and value.size == 1:
                            value = value.item()
                        metadata_dict[full_key] = value
                    except Exception:
                        pass
        elif isinstance(group, h5py.Dataset):
            try:
                value = group[()]
                if isinstance(value, np.ndarray) and value.size == 1:
                    value = value.item()
                metadata_dict[prefix] = value
            except Exception:
                pass
    
    def _process_msa(self, file_path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Process MSA format files.
        
        Args:
            file_path: Path to the MSA file
            
        Returns:
            Tuple of (data array, metadata dictionary)
        """
        # This is a placeholder - would need specialized library
        # For now, we'll just raise an error
        raise NotImplementedError("MSA format processing is not yet implemented")
    
    def _process_emd(self, file_path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Process EMD format files (FEI/Thermo Fisher).
        
        Args:
            file_path: Path to the EMD file
            
        Returns:
            Tuple of (data array, metadata dictionary)
        """
        # EMD is based on HDF5, so we can use h5py
        if 'h5py' not in globals():
            raise ImportError("h5py is required for EMD processing")
        
        # This is a simplified implementation - a real one would need more specific parsing
        return self._process_hdf5(file_path)
    
    def _process_ser(self, file_path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Process SER format files (FEI/Thermo Fisher TIA).
        
        Args:
            file_path: Path to the SER file
            
        Returns:
            Tuple of (data array, metadata dictionary)
        """
        # This is a placeholder - would need specialized library
        # For now, we'll just raise an error
        raise NotImplementedError("SER format processing is not yet implemented")
    
    def _process_hitachi(self, file_path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Process Hitachi format files.
        
        Args:
            file_path: Path to the Hitachi format file
            
        Returns:
            Tuple of (data array, metadata dictionary)
        """
        # This is a placeholder - would need specialized library
        # For now, we'll just raise an error
        raise NotImplementedError("Hitachi format processing is not yet implemented")
    
    def _process_oxford(self, file_path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Process Oxford Instruments format files.
        
        Args:
            file_path: Path to the Oxford Instruments format file
            
        Returns:
            Tuple of (data array, metadata dictionary)
        """
        # This is a placeholder - would need specialized library
        # For now, we'll just raise an error
        raise NotImplementedError("Oxford format processing is not yet implemented")
    
    def _enhance_metadata_with_llm(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use LLM to enhance metadata extraction.
        
        Args:
            metadata: Extracted metadata dictionary
            
        Returns:
            Enhanced metadata dictionary
        """
        # This is a placeholder for LLM-based metadata enhancement
        # A real implementation would use the LLM to fill in missing fields
        # or improve existing ones based on context
        return metadata
    
    def _validate_and_standardize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and standardize metadata against the standard schema.
        
        Args:
            metadata: Extracted metadata dictionary
            
        Returns:
            Standardized metadata dictionary
        """
        standardized = {}
        
        # Ensure all top-level categories exist
        for category in self.STANDARD_SCHEMA.keys():
            if category not in metadata:
                metadata[category] = {}
            
            standardized[category] = {}
            
            # Process fields in each category
            for field, field_type in self.STANDARD_SCHEMA[category].items():
                if field in metadata[category] and metadata[category][field] is not None:
                    # Value exists, validate type
                    value = metadata[category][field]
                    
                    # Handle Optional types
                    if hasattr(field_type, '__origin__') and field_type.__origin__ is Union:
                        # This is an Optional type (Union[X, None])
                        inner_type = field_type.__args__[0]
                        if value is not None and not isinstance(value, inner_type):
                            # Try to convert
                            try:
                                value = inner_type(value)
                            except (ValueError, TypeError):
                                value = None
                    elif not isinstance(value, field_type) and field_type is not Optional[field_type]:
                        # Try to convert
                        try:
                            value = field_type(value)
                        except (ValueError, TypeError):
                            # Use default value
                            if field_type is str:
                                value = ""
                            elif field_type is int:
                                value = 0
                            elif field_type is float:
                                value = 0.0
                            elif field_type is bool:
                                value = False
                            else:
                                value = None
                                
                    standardized[category][field] = value
                else:
                    # Value doesn't exist, use default
                    if field_type is str:
                        standardized[category][field] = ""
                    elif field_type is int:
                        standardized[category][field] = 0
                    elif field_type is float:
                        standardized[category][field] = 0.0
                    elif field_type is bool:
                        standardized[category][field] = False
                    else:
                        standardized[category][field] = None
        
        return standardized
    
    def save_standardized(self, standardized: Dict[str, Any], output_file: Union[str, Path], 
                        format: str = 'json') -> None:
        """
        Save standardized metadata to a file.
        
        Args:
            standardized: Standardized data dictionary
            output_file: Path to save the output file
            format: Output format ('json' or 'yaml')
            
        Raises:
            ValueError: If the format is not supported
        """
        output_file = Path(output_file)
        metadata = standardized["metadata"]
        
        # Create parent directories if they don't exist
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == 'json':
            with open(output_file, 'w') as f:
                json.dump(metadata, f, indent=2)
        elif format.lower() == 'yaml':
            with open(output_file, 'w') as f:
                yaml.dump(metadata, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported output format: {format}")
