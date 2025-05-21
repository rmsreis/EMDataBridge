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
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Import format-specific libraries
try:
    import mrcfile
except ImportError:
    print("Warning: mrcfile not installed. MRC format support will be limited.")

try:
    import hyperspy.api as hs
except ImportError:
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
    
    def __init__(self, llm_model_name: str = "bert-base-uncased"):
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
        
        # Initialize LLM model for metadata extraction
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(llm_model_name)
            self.llm_available = True
        except Exception as e:
            print(f"Warning: Could not load LLM model: {e}")
            self.llm_available = False
    
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
        
        # Use LLM to enhance metadata extraction if available
        if self.llm_available:
            metadata = self._enhance_metadata_with_llm(metadata)
        
        # Validate against standard schema
        standardized_metadata = self._validate_and_standardize_metadata(metadata)
        
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
        # For now, we'll return a dummy implementation
        data = np.zeros((100, 100))  # Placeholder
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
                "width": 100,
                "height": 100,
                "bit_depth": 8,
            },
            "processing": {},
        }
        return data, metadata
    
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
        s = hs.load(str(file_path))
        data = s.data
        
        # Extract metadata from HyperSpy signal
        original_metadata = s.original_metadata.as_dictionary()
        
        # Create standardized metadata
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
            "processing": {
                "software": "Digital Micrograph",
            },
        }
        
        # Try to extract microscope info
        try:
            if 'Microscope' in original_metadata:
                if 'Name' in original_metadata['Microscope']:
                    metadata['microscope']['name'] = original_metadata['Microscope']['Name']
                if 'Voltage' in original_metadata['Microscope']:
                    metadata['microscope']['voltage'] = original_metadata['Microscope']['Voltage']
                if 'Magnification' in original_metadata['Microscope']:
                    metadata['microscope']['magnification'] = original_metadata['Microscope']['Magnification']
        except Exception:
            pass  # Ignore errors in metadata extraction
        
        # Try to extract pixel size
        try:
            if s.axes_manager[0].scale:
                # Convert to nm (HyperSpy typically uses µm)
                metadata['image']['pixel_size'] = s.axes_manager[0].scale * 1000
        except Exception:
            pass
        
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
            data = np.array(img)
            
            # Extract TIFF tags
            tags = {}
            for tag_id, value in img.tag.items():
                tag_name = TiffTags.TAGS.get(tag_id, {}).get('name', str(tag_id))
                tags[tag_name] = value[0] if len(value) == 1 else value
            
            # Create standardized metadata
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
                    "width": img.width,
                    "height": img.height,
                    "bit_depth": 8 * (data.dtype.itemsize),
                },
                "processing": {},
            }
            
            # Try to extract acquisition date
            if 'DateTime' in tags:
                metadata['acquisition']['date'] = tags['DateTime']
                
            # Try to extract software info
            if 'Software' in tags:
                metadata['processing']['software'] = tags['Software']
                
            # Try to extract resolution info (pixel size)
            if 'XResolution' in tags and 'YResolution' in tags:
                # This is a simplification - would need proper conversion
                x_res = tags['XResolution']
                if isinstance(x_res, tuple) and len(x_res) == 2:
                    # Convert to nm (assuming resolution is in pixels per inch)
                    pixel_size = 25400 / (x_res[0] / x_res[1])  # 25400 nm per inch
                    metadata['image']['pixel_size'] = pixel_size
        
        return data, metadata
    
    def _process_hdf5(self, file_path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Process HDF5 format files.
        
        Args:
            file_path: Path to the HDF5 file
            
        Returns:
            Tuple of (data array, metadata dictionary)
        """
        with h5py.File(file_path, 'r') as f:
            # Try to find the main dataset
            # This is a simplification - real implementation would need to handle
            # different HDF5 layouts used in EM
            data_keys = [k for k in f.keys() if isinstance(f[k], h5py.Dataset) and len(f[k].shape) >= 2]
            
            if not data_keys:
                raise ValueError("Could not find image data in HDF5 file")
                
            # Use the first dataset found
            data = f[data_keys[0]][()]
            
            # Create standardized metadata
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
            
            # Extract attributes from the dataset and groups
            for k, v in f[data_keys[0]].attrs.items():
                # This is a simplification - would need proper mapping
                if k.lower() in ['voltage', 'high_tension', 'ht']:
                    metadata['microscope']['voltage'] = float(v)
                elif k.lower() in ['magnification', 'mag']:
                    metadata['microscope']['magnification'] = float(v)
                elif k.lower() in ['pixel_size', 'pixelsize']:
                    metadata['image']['pixel_size'] = float(v)
                    
            # Look for metadata in other groups
            for group_name in ['metadata', 'info', 'attributes']:
                if group_name in f:
                    for k, v in f[group_name].attrs.items():
                        # Map attributes to our schema (simplified)
                        if k.lower() in ['microscope', 'scope', 'tem']:
                            metadata['microscope']['name'] = str(v)
                        elif k.lower() in ['date', 'acquisition_date']:
                            metadata['acquisition']['date'] = str(v)
                        elif k.lower() in ['operator', 'user']:
                            metadata['acquisition']['operator'] = str(v)
        
        return data, metadata
        
    def _process_msa(self, file_path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Process MSA (EMSA) format files.
        
        Args:
            file_path: Path to the MSA file
            
        Returns:
            Tuple of (data array, metadata dictionary)
        """
        # MSA is a text-based format for spectral data in electron microscopy
        # Format specification: http://www.amc.anl.gov/ANLSoftwareLibrary/02-MMSLib/XEDS/EMMFF/EMMFF.IBM/EMMFF.TXT
        
        # Read the file
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Parse header and data sections
        header_lines = []
        data_lines = []
        in_data_section = False
        
        for line in lines:
            line = line.strip()
            if line.startswith('#') and 'END' in line:
                in_data_section = True
                continue
            
            if not in_data_section:
                if line.startswith('#'):
                    header_lines.append(line[1:].strip())  # Remove # and whitespace
            else:
                if line and not line.startswith('#'):
                    data_lines.append(line)
        
        # Parse header into key-value pairs
        header = {}
        for line in header_lines:
            if ':' in line:
                key, value = line.split(':', 1)
                header[key.strip()] = value.strip()
        
        # Parse data
        data_values = []
        for line in data_lines:
            values = line.split(',')
            for val in values:
                try:
                    data_values.append(float(val.strip()))
                except ValueError:
                    pass  # Skip non-numeric values
        
        # Convert to numpy array
        data = np.array(data_values)
        
        # Create standardized metadata
        metadata = {
            "microscope": {
                "name": header.get('INSTRUMENT', 'Unknown'),
                "voltage": float(header.get('BEAMKV', 0.0)),
                "magnification": float(header.get('MAGNIFICATION', 0.0)),
            },
            "sample": {
                "name": header.get('TITLE', file_path.stem),
                "description": header.get('COMMENT', ''),
                "preparation": header.get('SPECIMEN', ''),
            },
            "acquisition": {
                "date": header.get('DATE', ''),
                "operator": header.get('OPERATOR', None),
                "tilt_angle": float(header.get('TILT', 0.0)) if 'TILT' in header else None,
            },
            "image": {
                "width": len(data),  # MSA is typically 1D spectral data
                "height": 1,
                "bit_depth": 32,  # Typically stored as 32-bit float
            },
            "processing": {
                "software": header.get('SIGNALTYPE', None),
            },
        }
        
        # Reshape data to 2D if it's 1D (for consistency with other formats)
        if len(data.shape) == 1:
            data = data.reshape(1, -1)
        
        return data, metadata
    
    def _process_emd(self, file_path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Process EMD format files (Thermo Fisher Scientific's electron microscopy data format, HDF5-based).
        
        Args:
            file_path: Path to the EMD file
            
        Returns:
            Tuple of (data array, metadata dictionary)
        """
        # EMD is an HDF5-based format used by Thermo Fisher Scientific microscopes (formerly FEI)
        # We'll use h5py to read it, but with EMD-specific structure knowledge
        
        with h5py.File(file_path, 'r') as f:
            # EMD files typically have a specific structure
            # Data is usually in /data/data, metadata in /data/metadata
            data_path = None
            metadata_path = None
            
            # Look for the standard EMD structure
            if 'data' in f:
                data_group = f['data']
                # Find the first dataset with data
                for i in range(len(data_group)):
                    group_name = f'data_{i}'
                    if group_name in data_group and 'data' in data_group[group_name]:
                        data_path = f'/data/{group_name}/data'
                        metadata_path = f'/data/{group_name}/metadata' if 'metadata' in data_group[group_name] else None
                        break
            
            # If standard structure not found, try to find any suitable dataset
            if data_path is None:
                # Fallback to generic HDF5 processing
                return self._process_hdf5(file_path)
            
            # Read the data
            data = f[data_path][()]
            
            # Create standardized metadata
            metadata = {
                "microscope": {
                    "name": "Thermo Fisher Scientific",  # Default for EMD files
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
                "processing": {
                    "software": "FEI",
                },
            }
            
            # Extract metadata if available
            if metadata_path and metadata_path in f:
                meta_group = f[metadata_path]
                
                # EMD metadata structure is complex and varies
                # Here we extract some common fields
                for k, v in meta_group.attrs.items():
                    if k.lower() in ['voltage', 'high_tension', 'ht']:
                        metadata['microscope']['voltage'] = float(v)
                    elif k.lower() in ['magnification', 'mag']:
                        metadata['microscope']['magnification'] = float(v)
                
                # Look for pixel size in metadata
                if 'pixelsize' in meta_group.attrs:
                    # Convert to nm if needed
                    pixel_size = float(meta_group.attrs['pixelsize'])
                    # EMD typically stores in meters, convert to nm
                    metadata['image']['pixel_size'] = pixel_size * 1e9
                
                # Look for acquisition date
                if 'acquisition_time' in meta_group.attrs:
                    metadata['acquisition']['date'] = str(meta_group.attrs['acquisition_time'])
        
        return data, metadata
    
    def _process_ser(self, file_path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Process SER format files (Thermo Fisher Scientific's TIA series data format).
        
        Args:
            file_path: Path to the SER file
            
        Returns:
            Tuple of (data array, metadata dictionary)
        """
        # SER is a binary format used by Thermo Fisher Scientific's TIA software (formerly FEI)
        # This is a simplified implementation - a full implementation would need
        # to parse the complex binary structure
        
        try:
            # Try to use hyperspy if available
            import hyperspy.api as hs
            
            # Load the file with HyperSpy
            s = hs.load(str(file_path))
            data = s.data
            
            # Extract metadata from HyperSpy signal
            original_metadata = s.original_metadata.as_dictionary()
            
            # Create standardized metadata
            metadata = {
                "microscope": {
                    "name": "Thermo Fisher Scientific",  # Default for SER files
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
                "processing": {
                    "software": "TIA",
                },
            }
            
            # Try to extract microscope info
            try:
                if 'Acquisition_instrument' in s.metadata:
                    acq = s.metadata.Acquisition_instrument
                    if 'TEM' in acq:
                        if 'beam_energy' in acq.TEM:
                            metadata['microscope']['voltage'] = acq.TEM.beam_energy
                        if 'magnification' in acq.TEM:
                            metadata['microscope']['magnification'] = acq.TEM.magnification
            except Exception:
                pass  # Ignore errors in metadata extraction
            
            # Try to extract pixel size
            try:
                if s.axes_manager[0].scale:
                    # Convert to nm (HyperSpy typically uses µm)
                    metadata['image']['pixel_size'] = s.axes_manager[0].scale * 1000
            except Exception:
                pass
                
            return data, metadata
            
        except (ImportError, Exception) as e:
            # Fallback to a simple binary read if hyperspy is not available
            # This is a very simplified approach and won't work for all SER files
            print(f"Warning: Could not process SER file with HyperSpy: {e}")
            print("Using simplified SER reader which may not extract all data correctly.")
            
            # Read the file as binary
            with open(file_path, 'rb') as f:
                # SER files have a complex header structure
                # For simplicity, we'll skip ahead to where the data typically starts
                # This is a gross simplification and will not work for all SER files
                f.seek(1024)  # Skip header (this offset is a guess)
                binary_data = f.read()
            
            # Try to interpret as 16-bit unsigned integers
            # This is a guess and may not work for all SER files
            try:
                data_array = np.frombuffer(binary_data, dtype=np.uint16)
                # Reshape to 2D (dimensions are a guess)
                width = int(np.sqrt(len(data_array)))
                data = data_array[:width*width].reshape(width, width)
            except Exception:
                # If reshaping fails, return as 1D array
                data = np.frombuffer(binary_data, dtype=np.uint16)
                if len(data) == 0:
                    # If still empty, try 8-bit
                    data = np.frombuffer(binary_data, dtype=np.uint8)
            
            # Create minimal metadata
            metadata = {
                "microscope": {
                    "name": "Thermo Fisher Scientific",  # Default for SER files
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
                    "height": data.shape[0] if len(data.shape) > 1 else 1,
                    "bit_depth": data.dtype.itemsize * 8,
                },
                "processing": {
                    "software": "TIA",
                },
            }
            
            return data, metadata
    
    def _process_hitachi(self, file_path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Process Hitachi format files (.hds).
        
        Args:
            file_path: Path to the Hitachi file
            
        Returns:
            Tuple of (data array, metadata dictionary)
        """
        # Hitachi HDS is a proprietary format
        # This is a simplified implementation that attempts to extract basic information
        
        try:
            # Try to use hyperspy if available
            import hyperspy.api as hs
            
            # Load the file with HyperSpy
            s = hs.load(str(file_path))
            data = s.data
            
            # Extract metadata from HyperSpy signal
            original_metadata = s.original_metadata.as_dictionary()
            
            # Create standardized metadata
            metadata = {
                "microscope": {
                    "name": "Hitachi",
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
                "processing": {
                    "software": "Hitachi",
                },
            }
            
            # Try to extract microscope info
            try:
                if 'Acquisition_instrument' in s.metadata:
                    acq = s.metadata.Acquisition_instrument
                    if 'SEM' in acq:
                        if 'beam_energy' in acq.SEM:
                            metadata['microscope']['voltage'] = acq.SEM.beam_energy
                        if 'magnification' in acq.SEM:
                            metadata['microscope']['magnification'] = acq.SEM.magnification
            except Exception:
                pass  # Ignore errors in metadata extraction
            
            # Try to extract pixel size
            try:
                if s.axes_manager[0].scale:
                    # Convert to nm (HyperSpy typically uses µm)
                    metadata['image']['pixel_size'] = s.axes_manager[0].scale * 1000
            except Exception:
                pass
                
            return data, metadata
            
        except (ImportError, Exception) as e:
            # Fallback to binary read if hyperspy is not available or fails
            print(f"Warning: Could not process Hitachi file with HyperSpy: {e}")
            
            # For Hitachi HDS files, we'll try to read the file as a TIFF if it fails with HyperSpy
            # Many Hitachi systems save data in TIFF format with metadata
            if file_path.suffix.lower() == '.hds':
                # Try to find an associated TIFF file
                tiff_path = file_path.with_suffix('.tif')
                if tiff_path.exists():
                    print(f"Found associated TIFF file: {tiff_path}, processing as TIFF")
                    return self._process_tiff(tiff_path)
            
            # If no TIFF file found or not an HDS file, try to read as binary
            try:
                with open(file_path, 'rb') as f:
                    # Read the first 1024 bytes to check for file type
                    header = f.read(1024)
                    
                    # Check if it's a TIFF file (starts with II or MM)
                    if header.startswith(b'II') or header.startswith(b'MM'):
                        from PIL import Image
                        with Image.open(file_path) as img:
                            data = np.array(img)
                    else:
                        # Read as raw binary data
                        f.seek(0)
                        binary_data = f.read()
                        
                        # Try different data types and shapes
                        # This is a guess and may not work for all files
                        try:
                            data = np.frombuffer(binary_data, dtype=np.uint16)
                            # Try to reshape to a square
                            width = int(np.sqrt(len(data)))
                            data = data[:width*width].reshape(width, width)
                        except:
                            # If reshaping fails, return as 1D array
                            data = np.frombuffer(binary_data, dtype=np.uint8)
            except Exception as e:
                print(f"Error reading Hitachi file as binary: {e}")
                # Return empty data as last resort
                data = np.zeros((100, 100), dtype=np.uint8)
            
            # Create minimal metadata
            metadata = {
                "microscope": {
                    "name": "Hitachi",
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
                    "height": data.shape[0] if len(data.shape) > 1 else 1,
                    "bit_depth": data.dtype.itemsize * 8,
                },
                "processing": {
                    "software": "Hitachi",
                },
            }
            
            return data, metadata
    
    def _process_oxford(self, file_path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Process Oxford Instruments format files (.oip, .inca, .azw, .azd).
        
        Args:
            file_path: Path to the Oxford Instruments file
            
        Returns:
            Tuple of (data array, metadata dictionary)
        """
        # Oxford Instruments formats are used for EDS/EBSD data
        # This is a simplified implementation that attempts to extract basic information
        
        # File extensions and their associated software
        oxford_software = {
            '.oip': 'Oxford INCA',
            '.inca': 'Oxford INCA',
            '.azw': 'Oxford AZtec',
            '.azd': 'Oxford AZtec',
        }
        
        software_name = oxford_software.get(file_path.suffix.lower(), 'Oxford Instruments')
        
        try:
            # Try to use hyperspy if available
            import hyperspy.api as hs
            
            # Load the file with HyperSpy
            s = hs.load(str(file_path))
            data = s.data
            
            # Extract metadata from HyperSpy signal
            original_metadata = s.original_metadata.as_dictionary()
            
            # Create standardized metadata
            metadata = {
                "microscope": {
                    "name": "Oxford Instruments",
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
                "processing": {
                    "software": software_name,
                },
            }
            
            # Try to extract microscope info
            try:
                if 'Acquisition_instrument' in s.metadata:
                    acq = s.metadata.Acquisition_instrument
                    if 'SEM' in acq:
                        if 'beam_energy' in acq.SEM:
                            metadata['microscope']['voltage'] = acq.SEM.beam_energy
                        if 'magnification' in acq.SEM:
                            metadata['microscope']['magnification'] = acq.SEM.magnification
            except Exception:
                pass  # Ignore errors in metadata extraction
            
            # Try to extract pixel size
            try:
                if s.axes_manager[0].scale:
                    # Convert to nm (HyperSpy typically uses µm)
                    metadata['image']['pixel_size'] = s.axes_manager[0].scale * 1000
            except Exception:
                pass
                
            return data, metadata
            
        except (ImportError, Exception) as e:
            print(f"Warning: Could not process Oxford Instruments file with HyperSpy: {e}")
            
            # For Oxford files, we'll try to read as XML if it's text-based
            try:
                # Check if it's a text-based format (XML)
                with open(file_path, 'r', errors='ignore') as f:
                    content = f.read(1024)  # Read first 1KB to check
                    
                if '<?xml' in content:
                    # It's an XML file, try to parse it
                    import xml.etree.ElementTree as ET
                    tree = ET.parse(file_path)
                    root = tree.getroot()
                    
                    # Extract some basic metadata if possible
                    # This is very simplified and would need to be adapted to the specific XML structure
                    metadata = {
                        "microscope": {
                            "name": "Oxford Instruments",
                            "voltage": 0.0,
                            "magnification": 0.0,
                        },
                        "sample": {
                            "name": file_path.stem,
                            "description": "",
                        },
                        "acquisition": {},
                        "image": {
                            "width": 100,  # Placeholder
                            "height": 100,  # Placeholder
                            "bit_depth": 8,  # Placeholder
                        },
                        "processing": {
                            "software": software_name,
                        },
                    }
                    
                    # Create a placeholder data array
                    # In a real implementation, you would extract the actual data from the XML
                    data = np.zeros((100, 100), dtype=np.uint8)
                    
                    return data, metadata
            except Exception as e:
                print(f"Error parsing Oxford Instruments file as XML: {e}")
            
            # If XML parsing fails or it's not XML, try to read as binary
            try:
                with open(file_path, 'rb') as f:
                    binary_data = f.read()
                    
                # Try different data types and shapes
                # This is a guess and may not work for all files
                try:
                    # First try as spectrum data (1D)
                    data = np.frombuffer(binary_data[1024:], dtype=np.float32)  # Skip header
                except:
                    # If that fails, try as image data
                    try:
                        data = np.frombuffer(binary_data[1024:], dtype=np.uint16)
                        # Try to reshape to a square
                        width = int(np.sqrt(len(data)))
                        data = data[:width*width].reshape(width, width)
                    except:
                        # Last resort: 8-bit data
                        data = np.frombuffer(binary_data, dtype=np.uint8)
                        if len(data) > 10000:  # If enough data to be an image
                            width = int(np.sqrt(len(data)))
                            data = data[:width*width].reshape(width, width)
            except Exception as e:
                print(f"Error reading Oxford Instruments file as binary: {e}")
                # Return empty data as last resort
                data = np.zeros((100, 100), dtype=np.uint8)
            
            # Create minimal metadata
            metadata = {
                "microscope": {
                    "name": "Oxford Instruments",
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
                    "height": data.shape[0] if len(data.shape) > 1 else 1,
                    "bit_depth": data.dtype.itemsize * 8,
                },
                "processing": {
                    "software": software_name,
                },
            }
            
            return data, metadata
    
    def _enhance_metadata_with_llm(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use LLM to enhance metadata extraction by inferring missing fields.
        
        Args:
            metadata: Original metadata dictionary
            
        Returns:
            Enhanced metadata dictionary
        """
        if not self.llm_available:
            return metadata
            
        # Convert metadata to text for LLM processing
        metadata_text = json.dumps(metadata, indent=2)
        
        # This is a placeholder for actual LLM processing
        # In a real implementation, you would:
        # 1. Identify missing fields in the metadata
        # 2. Generate prompts for the LLM to fill in these fields
        # 3. Parse the LLM responses and update the metadata
        
        # For now, we'll just return the original metadata
        return metadata
    
    def _validate_and_standardize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate metadata against the standard schema and fill in missing fields.
        
        Args:
            metadata: Metadata dictionary to validate
            
        Returns:
            Standardized metadata dictionary
        """
        standardized = {}
        
        # Process each top-level section
        for section, schema in self.STANDARD_SCHEMA.items():
            standardized[section] = {}
            
            # Get the section data from the input metadata, or use empty dict if missing
            section_data = metadata.get(section, {})
            
            # Process each field in the section
            for field, field_type in schema.items():
                if field in section_data:
                    # Field exists in input metadata
                    value = section_data[field]
                    
                    # Validate type
                    if field_type is not Optional and not isinstance(value, field_type):
                        # Try to convert
                        try:
                            if field_type is str:
                                value = str(value)
                            elif field_type is int:
                                value = int(value)
                            elif field_type is float:
                                value = float(value)
                            elif field_type is list:
                                value = list(value) if hasattr(value, '__iter__') else [value]
                            else:
                                # Can't convert, use default
                                value = None
                        except (ValueError, TypeError):
                            value = None
                    
                    standardized[section][field] = value
                else:
                    # Field doesn't exist in input metadata
                    if field_type is Optional:
                        standardized[section][field] = None
                    elif field_type is str:
                        standardized[section][field] = ""
                    elif field_type is int:
                        standardized[section][field] = 0
                    elif field_type is float:
                        standardized[section][field] = 0.0
                    elif field_type is list:
                        standardized[section][field] = []
                    else:
                        standardized[section][field] = None
        
        return standardized
    
    def save_standardized(self, standardized_data: Dict[str, Any], output_path: Union[str, Path], 
                         format: str = 'json') -> None:
        """
        Save standardized data to a file.
        
        Args:
            standardized_data: The standardized data dictionary
            output_path: Path to save the output file
            format: Output format ('json' or 'yaml')
            
        Raises:
            ValueError: If the format is not supported
        """
        output_path = Path(output_path)
        
        # Extract just the metadata for saving
        metadata = standardized_data['metadata']
        
        if format.lower() == 'json':
            with open(output_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        elif format.lower() == 'yaml':
            with open(output_path, 'w') as f:
                yaml.dump(metadata, f)
        else:
            raise ValueError(f"Unsupported output format: {format}")


def main():
    """Command line interface for the standardizer."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Standardize electron microscopy data formats')
    parser.add_argument('input_file', help='Input electron microscopy data file')
    parser.add_argument('--output', '-o', help='Output file for standardized metadata')
    parser.add_argument('--format', '-f', choices=['json', 'yaml'], default='json',
                      help='Output format for metadata')
    parser.add_argument('--model', '-m', default='bert-base-uncased',
                      help='LLM model to use for metadata extraction')
    
    args = parser.parse_args()
    
    # Create standardizer
    standardizer = EMDataStandardizer(llm_model_name=args.model)
    
    try:
        # Standardize the input file
        standardized = standardizer.standardize(args.input_file)
        
        # Print summary
        print(f"Standardized {args.input_file}")
        print(f"Original format: {standardized['original_format']}")
        print(f"Image dimensions: {standardized['data'].shape}")
        
        # Save metadata if output file specified
        if args.output:
            standardizer.save_standardized(standardized, args.output, format=args.format)
            print(f"Saved standardized metadata to {args.output}")
        else:
            # Print metadata to console
            if args.format == 'json':
                print(json.dumps(standardized['metadata'], indent=2))
            else:
                print(yaml.dump(standardized['metadata']))
                
    except Exception as e:
        print(f"Error: {e}")
        return 1
        
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
