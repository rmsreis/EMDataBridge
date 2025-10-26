#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Translator module for electron microscopy data formats.

This module provides functionality to translate between various electron microscopy
data formats using LLM-based approaches for metadata mapping.
"""

import os
import json
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Union, List, Optional, Tuple

# Lazy-transformers import to avoid requiring heavy ML deps at package import
_TRANSFORMERS_AVAILABLE = False
try:
    import transformers  # type: ignore
    _TRANSFORMERS_AVAILABLE = True
except Exception:
    _TRANSFORMERS_AVAILABLE = False

# Import the standardizer for common functionality
from .standardizer import EMDataStandardizer

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


class EMFormatTranslator:
    """
    A class for translating between electron microscopy data formats using LLM-based approaches.
    
    This translator can convert files between different formats while preserving
    metadata and applying appropriate transformations.
    """
    
    def __init__(self, llm_model_name: str = "bert-base-uncased", llm_backend: str = "auto", llm_model_path: str | None = None):
        """
        Initialize the translator with an optional LLM model for metadata mapping.
        
        Args:
            llm_model_name: Name of the pre-trained model to use for metadata mapping
        """
        # Initialize the standardizer for common functionality, forwarding LLM config
        self.standardizer = EMDataStandardizer(llm_model_name=llm_model_name, llm_backend=llm_backend, llm_model_path=llm_model_path)
        
        # Define supported output formats and their handlers
        self.output_formats = {
            "mrc": self._write_mrc,
            "em": self._write_em,
            "dm4": self._write_dm4,
            "tiff": self._write_tiff,
            "tif": self._write_tiff,
            "hdf5": self._write_hdf5,
            "h5": self._write_hdf5,
            "msa": self._write_msa,
            "emd": self._write_emd,
            "ser": self._write_ser,
            "hds": self._write_hitachi,
            "oip": self._write_oxford,
            "inca": self._write_oxford,
            "azw": self._write_oxford,
            "azd": self._write_oxford,
        }
        
        # Initialize LLM model lazily. Only attempt to load if transformers is
        # available and needed; this avoids requiring heavy ML deps on import.
        self.llm_available = False
        self.tokenizer = None
        self.model = None
        self.llm_model_name = llm_model_name

    def _ensure_llm_loaded(self):
        """Load transformers AutoTokenizer/AutoModel on demand."""
        if self.llm_available:
            return
        if not _TRANSFORMERS_AVAILABLE:
            return
        try:
            from transformers import AutoTokenizer, AutoModel

            self.tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name)
            self.model = AutoModel.from_pretrained(self.llm_model_name)
            self.llm_available = True
        except Exception as e:
            print(f"Warning: Could not load LLM model: {e}")
    
    def translate(self, input_file: Union[str, Path], output_file: Union[str, Path], 
                 output_format: Optional[str] = None) -> None:
        """
        Translate an electron microscopy data file to another format.
        
        Args:
            input_file: Path to the input electron microscopy data file
            output_file: Path to save the output file
            output_format: Output format to use (if None, inferred from output_file extension)
            
        Raises:
            ValueError: If the input or output format is not supported
        """
        input_file = Path(input_file)
        output_file = Path(output_file)
        
        # Determine output format
        if output_format is None:
            output_format = output_file.suffix.lower()[1:]  # Remove leading dot
        else:
            output_format = output_format.lower()
        
        # Check if output format is supported
        if output_format not in self.output_formats:
            raise ValueError(f"Unsupported output format: {output_format}")
        
        # Standardize the input file
        standardized = self.standardizer.standardize(input_file)
        
        # Write to the output format
        self.output_formats[output_format](standardized, output_file)
        
        print(f"Translated {input_file} to {output_file} (format: {output_format})")
    
    def _write_mrc(self, standardized: Dict[str, Any], output_file: Path) -> None:
        """
        Write data to MRC format.
        
        Args:
            standardized: Standardized data dictionary
            output_file: Path to save the output file
        """
        data = standardized["data"]
        metadata = standardized["metadata"]
        
        # Ensure data is in a format compatible with MRC
        if data.dtype not in [np.int8, np.int16, np.float32]:
            # Convert to float32 for compatibility
            data = data.astype(np.float32)
        
        # Write the MRC file
        with mrcfile.new(output_file, overwrite=True) as mrc:
            mrc.set_data(data)
            
            # Set header values based on metadata
            if "pixel_size" in metadata["image"] and metadata["image"]["pixel_size"] is not None:
                # Convert from nm to Å (MRC uses Å)
                pixel_size_angstrom = metadata["image"]["pixel_size"] * 10
                mrc.voxel_size = (pixel_size_angstrom, pixel_size_angstrom, pixel_size_angstrom)
            
            # Add extended header information if available
            # This is a simplification - real implementation would need more mapping
            if "software" in metadata["processing"] and metadata["processing"]["software"] is not None:
                mrc.header.exttyp = metadata["processing"]["software"][:4].encode()  # Limited to 4 chars
    
    def _write_em(self, standardized: Dict[str, Any], output_file: Path) -> None:
        """
        Write data to EM format.
        
        Args:
            standardized: Standardized data dictionary
            output_file: Path to save the output file
        """
        # This is a placeholder - would need pyem or similar library
        # For now, we'll just raise an error
        raise NotImplementedError("EM format writing is not yet implemented")
    
    def _write_dm4(self, standardized: Dict[str, Any], output_file: Path) -> None:
        """
        Write data to DM4 format using HyperSpy.
        
        Args:
            standardized: Standardized data dictionary
            output_file: Path to save the output file
        """
        if 'hs' not in globals():
            raise ImportError("HyperSpy is required for DM4 processing")
        
        data = standardized["data"]
        metadata = standardized["metadata"]
        
        # Create a HyperSpy signal
        s = hs.signals.Signal2D(data)
        
        # Set metadata
        if "pixel_size" in metadata["image"] and metadata["image"]["pixel_size"] is not None:
            # Convert from nm to µm (HyperSpy typically uses µm)
            pixel_size_um = metadata["image"]["pixel_size"] / 1000
            s.axes_manager[0].scale = pixel_size_um
            s.axes_manager[1].scale = pixel_size_um
            s.axes_manager[0].units = "µm"
            s.axes_manager[1].units = "µm"
        
        # Set additional metadata
        s.metadata.General.title = metadata["sample"]["name"]
        
        if "name" in metadata["microscope"] and metadata["microscope"]["name"] is not None:
            s.metadata.Acquisition_instrument.TEM.microscope = metadata["microscope"]["name"]
            
        if "voltage" in metadata["microscope"] and metadata["microscope"]["voltage"] is not None:
            s.metadata.Acquisition_instrument.TEM.beam_energy = metadata["microscope"]["voltage"]
            
        if "magnification" in metadata["microscope"] and metadata["microscope"]["magnification"] is not None:
            s.metadata.Acquisition_instrument.TEM.magnification = metadata["microscope"]["magnification"]
        
        # Save the file
        s.save(output_file)
    
    def _write_tiff(self, standardized: Dict[str, Any], output_file: Path) -> None:
        """
        Write data to TIFF format.
        
        Args:
            standardized: Standardized data dictionary
            output_file: Path to save the output file
        """
        from PIL import Image
        import numpy as np
        
        data = standardized["data"]
        metadata = standardized["metadata"]
        
        # Ensure data is in a format compatible with PIL
        if data.dtype == np.float32 or data.dtype == np.float64:
            # Normalize floating point data to 0-255 range
            data_min = np.min(data)
            data_max = np.max(data)
            if data_max > data_min:
                data = ((data - data_min) / (data_max - data_min) * 255).astype(np.uint8)
            else:
                data = np.zeros_like(data, dtype=np.uint8)
        elif data.dtype == np.uint16:
            # 16-bit data is supported directly
            pass
        elif data.dtype not in [np.uint8, np.int8]:
            # Convert other types to 8-bit
            data = data.astype(np.uint8)
        
        # Create PIL image
        img = Image.fromarray(data)
        
        # Add metadata as TIFF tags
        # This is a simplification - real implementation would need more mapping
        tags = {}
        
        if metadata["sample"]["description"]:
            tags[270] = metadata["sample"]["description"]  # ImageDescription tag
            
        if metadata["processing"]["software"]:
            tags[305] = metadata["processing"]["software"]  # Software tag
            
        if metadata["acquisition"]["date"]:
            tags[306] = metadata["acquisition"]["date"]  # DateTime tag
        
        # Save the image with metadata
        img.save(output_file, tiffinfo=tags)
    
    def _write_hdf5(self, standardized: Dict[str, Any], output_file: Path) -> None:
        """
        Write data to HDF5 format.
        
        Args:
            standardized: Standardized data dictionary
            output_file: Path to save the output file
        """
        if 'h5py' not in globals():
            raise ImportError("h5py is required for HDF5 processing")
        
        data = standardized["data"]
        metadata = standardized["metadata"]
        
        with h5py.File(output_file, 'w') as f:
            # Create main dataset
            f.create_dataset('data', data=data)
            
            # Create metadata group
            meta_group = f.create_group('metadata')
            
            # Add metadata as attributes
            for category, category_data in metadata.items():
                category_group = meta_group.create_group(category)
                
                for key, value in category_data.items():
                    if value is not None:
                        # Convert non-scalar values to JSON strings
                        if isinstance(value, (list, dict)):
                            value = json.dumps(value)
                        
                        category_group.attrs[key] = value
    
    def _write_msa(self, standardized: Dict[str, Any], output_file: Path) -> None:
        """
        Write data to MSA format.
        
        Args:
            standardized: Standardized data dictionary
            output_file: Path to save the output file
        """
        # This is a placeholder - would need specialized library
        # For now, we'll just raise an error
        raise NotImplementedError("MSA format writing is not yet implemented")
    
    def _write_emd(self, standardized: Dict[str, Any], output_file: Path) -> None:
        """
        Write data to EMD format (FEI/Thermo Fisher).
        
        Args:
            standardized: Standardized data dictionary
            output_file: Path to save the output file
        """
        # EMD is based on HDF5, but with a specific structure
        # This is a simplified implementation - a real one would need more specific formatting
        if 'h5py' not in globals():
            raise ImportError("h5py is required for EMD processing")
        
        data = standardized["data"]
        metadata = standardized["metadata"]
        
        with h5py.File(output_file, 'w') as f:
            # Create EMD group structure
            emd_group = f.create_group('EMData')
            data_group = emd_group.create_group('Data')
            image_group = data_group.create_group('Image')
            
            # Add data
            image_group.create_dataset('Data', data=data)
            
            # Add metadata
            meta_group = image_group.create_group('Metadata')
            
            # Add microscope metadata
            microscope_group = meta_group.create_group('Microscope')
            for key, value in metadata["microscope"].items():
                if value is not None:
                    microscope_group.attrs[key] = value
            
            # Add sample metadata
            sample_group = meta_group.create_group('Sample')
            for key, value in metadata["sample"].items():
                if value is not None:
                    sample_group.attrs[key] = value
            
            # Add acquisition metadata
            acquisition_group = meta_group.create_group('Acquisition')
            for key, value in metadata["acquisition"].items():
                if value is not None:
                    acquisition_group.attrs[key] = value
    
    def _write_ser(self, standardized: Dict[str, Any], output_file: Path) -> None:
        """
        Write data to SER format (FEI/Thermo Fisher TIA).
        
        Args:
            standardized: Standardized data dictionary
            output_file: Path to save the output file
        """
        # This is a placeholder - would need specialized library
        # For now, we'll just raise an error
        raise NotImplementedError("SER format writing is not yet implemented")
    
    def _write_hitachi(self, standardized: Dict[str, Any], output_file: Path) -> None:
        """
        Write data to Hitachi format.
        
        Args:
            standardized: Standardized data dictionary
            output_file: Path to save the output file
        """
        # This is a placeholder - would need specialized library
        # For now, we'll just raise an error
        raise NotImplementedError("Hitachi format writing is not yet implemented")
    
    def _write_oxford(self, standardized: Dict[str, Any], output_file: Path) -> None:
        """
        Write data to Oxford Instruments format.
        
        Args:
            standardized: Standardized data dictionary
            output_file: Path to save the output file
        """
        # This is a placeholder - would need specialized library
        # For now, we'll just raise an error
        raise NotImplementedError("Oxford format writing is not yet implemented")
    
    def batch_translate(self, input_dir: Union[str, Path], output_dir: Union[str, Path],
                       output_format: str, file_pattern: str = "*") -> None:
        """
        Batch translate multiple files from a directory.
        
        Args:
            input_dir: Directory containing input files
            output_dir: Directory to save output files
            output_format: Output format to use
            file_pattern: Glob pattern to match files
            
        Raises:
            ValueError: If the output format is not supported
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        
        # Check if output format is supported
        if output_format not in self.output_formats:
            raise ValueError(f"Unsupported output format: {output_format}")
        
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all files matching the pattern
        files = list(input_dir.glob(file_pattern))
        
        # Filter for supported file extensions
        supported_extensions = set(ext.lower() for ext in self.standardizer.supported_formats.keys())
        supported_files = [f for f in files if f.suffix.lower() in supported_extensions]
        
        if not supported_files:
            print(f"No supported files found in {input_dir} matching pattern {file_pattern}")
            return
        
        print(f"Found {len(supported_files)} files to translate")
        
        # Process each file
        for i, file_path in enumerate(supported_files):
            try:
                # Determine output file path
                output_file = output_dir / f"{file_path.stem}.{output_format}"
                
                # Translate the file
                self.translate(file_path, output_file, output_format)
                
                print(f"Translated {i+1}/{len(supported_files)}: {file_path.name} -> {output_file.name}")
            except Exception as e:
                print(f"Error translating {file_path.name}: {e}")
    
    def list_supported_formats(self) -> List[Dict[str, Any]]:
        """
        List all supported formats with details.
        
        Returns:
            List of dictionaries with format details
        """
        formats = [
            {
                "name": "MRC",
                "extensions": [".mrc", ".map", ".rec"],
                "description": "Medical Research Council format for electron density maps",
                "libraries": ["mrcfile"],
                "metadata_support": "Limited",
                "typical_use": "Cryo-EM, electron tomography"
            },
            {
                "name": "EM",
                "extensions": [".em"],
                "description": "IMAGIC format for electron microscopy",
                "libraries": ["pyem"],
                "metadata_support": "Limited",
                "typical_use": "Single particle analysis"
            },
            {
                "name": "Digital Micrograph",
                "extensions": [".dm3", ".dm4"],
                "description": "Gatan Digital Micrograph format",
                "libraries": ["hyperspy"],
                "metadata_support": "Extensive",
                "typical_use": "TEM, STEM imaging"
            },
            {
                "name": "TIFF",
                "extensions": [".tif", ".tiff"],
                "description": "Tagged Image File Format",
                "libraries": ["pillow"],
                "metadata_support": "Moderate (via tags)",
                "typical_use": "General purpose, publication"
            },
            {
                "name": "HDF5",
                "extensions": [".h5", ".hdf5"],
                "description": "Hierarchical Data Format version 5",
                "libraries": ["h5py"],
                "metadata_support": "Extensive (hierarchical)",
                "typical_use": "Large datasets, multidimensional data"
            },
            {
                "name": "EMD",
                "extensions": [".emd"],
                "description": "FEI/Thermo Fisher Scientific EMD format (based on HDF5)",
                "libraries": ["h5py"],
                "metadata_support": "Extensive",
                "typical_use": "FEI/Thermo Fisher microscopes"
            },
            {
                "name": "MSA",
                "extensions": [".msa"],
                "description": "EMSA/MAS Spectral Data File Format",
                "libraries": ["hyperspy"],
                "metadata_support": "Moderate",
                "typical_use": "Spectroscopy data"
            },
            {
                "name": "SER",
                "extensions": [".ser"],
                "description": "TIA Series Data File Format (FEI/Thermo Fisher)",
                "libraries": ["hyperspy"],
                "metadata_support": "Moderate",
                "typical_use": "TEM image series"
            },
            {
                "name": "Hitachi",
                "extensions": [".hds"],
                "description": "Hitachi Data File Format",
                "libraries": ["custom"],
                "metadata_support": "Limited",
                "typical_use": "Hitachi microscopes"
            },
            {
                "name": "Oxford Instruments",
                "extensions": [".oip", ".inca", ".azw", ".azd"],
                "description": "Oxford Instruments File Formats (AZtec, INCA)",
                "libraries": ["custom"],
                "metadata_support": "Moderate",
                "typical_use": "EDS, EBSD data"
            }
        ]
        
        return formats
