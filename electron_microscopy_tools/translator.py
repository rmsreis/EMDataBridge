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
"""
Translator module: guard heavy ML and IO libs (transformers, hyperspy) so the
package can be imported without those optional dependencies. Load them lazily
when translation or format-specific functions are invoked.
"""

# Lazy checks for optional deps
_TRANSFORMERS_AVAILABLE = False
try:
    import transformers  # type: ignore
    _TRANSFORMERS_AVAILABLE = True
except Exception:
    _TRANSFORMERS_AVAILABLE = False

try:
    import hyperspy.api as hs
except ImportError:
    hs = None
    print("Warning: hyperspy not installed. DM3/DM4 format support will be limited.")

# Import the standardizer for common functionality
from standardizer import EMDataStandardizer

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
    
    def __init__(self, llm_model_name: str = "bert-base-uncased"):
        """
        Initialize the translator with an optional LLM model for metadata mapping.
        
        Args:
            llm_model_name: Name of the pre-trained model to use for metadata mapping
        """
        # Initialize the standardizer for common functionality
        self.standardizer = EMDataStandardizer(llm_model_name=llm_model_name)
        
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
        
        # Initialize LLM model for metadata mapping
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
            self.model = AutoModel.from_pretrained(llm_model_name)
            self.llm_available = True
        except Exception as e:
            print(f"Warning: Could not load LLM model: {e}")
            self.llm_available = False
    
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
        from PIL import Image, TiffImagePlugin
        
        data = standardized["data"]
        metadata = standardized["metadata"]
        
        # Normalize data for TIFF format if needed
        if data.dtype in [np.float32, np.float64]:
            # Normalize to 0-65535 for 16-bit TIFF
            data_min = data.min()
            data_max = data.max()
            if data_max > data_min:  # Avoid division by zero
                data = ((data - data_min) * 65535 / (data_max - data_min)).astype(np.uint16)
            else:
                data = np.zeros_like(data, dtype=np.uint16)
        elif data.dtype == np.uint8:
            # Already in 8-bit format, no change needed
            pass
        else:
            # Convert to 16-bit for other types
            data = data.astype(np.uint16)
        
        # Create PIL Image
        img = Image.fromarray(data)
        
        # Prepare TIFF tags with metadata
        tags = {}
        
        # Add acquisition date if available
        if "date" in metadata["acquisition"] and metadata["acquisition"]["date"] is not None:
            tags[TiffImagePlugin.TAGS["DateTime"]] = metadata["acquisition"]["date"]
        
        # Add software info if available
        if "software" in metadata["processing"] and metadata["processing"]["software"] is not None:
            tags[TiffImagePlugin.TAGS["Software"]] = metadata["processing"]["software"]
        
        # Add resolution info if pixel size is available
        if "pixel_size" in metadata["image"] and metadata["image"]["pixel_size"] is not None:
            # Convert from nm to dpi (dots per inch)
            # 1 inch = 25.4 mm = 25,400,000 nm
            dpi = int(25400000 / metadata["image"]["pixel_size"])
            tags[TiffImagePlugin.TAGS["XResolution"]] = (dpi, 1)
            tags[TiffImagePlugin.TAGS["YResolution"]] = (dpi, 1)
            tags[TiffImagePlugin.TAGS["ResolutionUnit"]] = 2  # inches
        
        # Add description with JSON metadata
        tags[TiffImagePlugin.TAGS["ImageDescription"]] = json.dumps(metadata)
        
        # Save the image with metadata
        img.save(output_file, tiffinfo=tags)
    
    def _write_hdf5(self, standardized: Dict[str, Any], output_file: Path) -> None:
        """
        Write data to HDF5 format.
        
        Args:
            standardized: Standardized data dictionary
            output_file: Path to save the output file
        """
        data = standardized["data"]
        metadata = standardized["metadata"]
        
        with h5py.File(output_file, 'w') as f:
            # Create main dataset
            dset = f.create_dataset("data", data=data)
            
            # Add metadata as attributes to the dataset
            if "pixel_size" in metadata["image"] and metadata["image"]["pixel_size"] is not None:
                dset.attrs["pixel_size"] = metadata["image"]["pixel_size"]
                
            if "voltage" in metadata["microscope"] and metadata["microscope"]["voltage"] is not None:
                dset.attrs["voltage"] = metadata["microscope"]["voltage"]
                
            if "magnification" in metadata["microscope"] and metadata["microscope"]["magnification"] is not None:
                dset.attrs["magnification"] = metadata["microscope"]["magnification"]
            
            # Create metadata group with all metadata
            meta_group = f.create_group("metadata")
            
            # Add metadata to groups by section
            for section, section_data in metadata.items():
                section_group = meta_group.create_group(section)
                
                for key, value in section_data.items():
                    if value is not None:
                        # Convert complex types to JSON strings
                        if isinstance(value, (dict, list)):
                            section_group.attrs[key] = json.dumps(value)
                        else:
                            section_group.attrs[key] = value
                            
    def _write_msa(self, standardized: Dict[str, Any], output_file: Path) -> None:
        """
        Write data to MSA (EMSA) format.
        
        Args:
            standardized: Standardized data dictionary
            output_file: Path to save the output file
        """
        data = standardized["data"]
        metadata = standardized["metadata"]
        
        # MSA is a text-based format for spectral data
        # Format specification: http://www.amc.anl.gov/ANLSoftwareLibrary/02-MMSLib/XEDS/EMMFF/EMMFF.IBM/EMMFF.TXT
        
        # Ensure data is 1D or 2D
        if len(data.shape) > 2:
            # If data is 3D or higher, take the first 2D slice
            data = data[0]
        
        # If data is 2D, take the first row (MSA is primarily for 1D spectra)
        if len(data.shape) == 2:
            # Check if it's a single row with multiple columns
            if data.shape[0] == 1:
                data = data[0]
            # Otherwise take the first row
            else:
                data = data[0]
        
        # Create MSA header
        header_lines = [
            "#FORMAT      : EMSA/MAS Spectral Data File",
            "#VERSION     : 1.0",
            f"#TITLE       : {metadata['sample']['name']}",
            f"#DATE        : {metadata['acquisition'].get('date', '')}",
            f"#TIME        : 00:00",  # Placeholder, not typically available
            f"#OWNER       : {metadata['acquisition'].get('operator', '')}",
            f"#NPOINTS     : {len(data)}",
            f"#NCOLUMNS    : 1",
            f"#XUNITS      : eV",  # Default for EDS/EELS, could be different
            f"#YUNITS      : counts",  # Default for spectral data
            f"#DATATYPE    : Y",
            f"#XPERCHAN    : 1.0",  # Default, would need actual energy calibration
            f"#OFFSET      : 0.0",
        ]
        
        # Add microscope information
        if metadata["microscope"]["name"] != "Unknown":
            header_lines.append(f"#INSTRUMENT  : {metadata['microscope']['name']}")
        if metadata["microscope"]["voltage"] > 0:
            header_lines.append(f"#BEAMKV      : {metadata['microscope']['voltage']}")
        if metadata["microscope"]["magnification"] > 0:
            header_lines.append(f"#MAGNIFICATION: {metadata['microscope']['magnification']}")
        
        # Add sample information
        if metadata["sample"]["description"]:
            header_lines.append(f"#COMMENT     : {metadata['sample']['description']}")
        if metadata["sample"]["preparation"]:
            header_lines.append(f"#SPECIMEN    : {metadata['sample']['preparation']}")
        
        # Add acquisition information
        if "tilt_angle" in metadata["acquisition"] and metadata["acquisition"]["tilt_angle"] is not None:
            header_lines.append(f"#TILT        : {metadata['acquisition']['tilt_angle']}")
        
        # Add end of header marker
        header_lines.append("#SPECTRUM    : Spectral Data Follows")
        
        # Format data values
        data_lines = []
        for i in range(0, len(data), 4):  # 4 values per line
            chunk = data[i:i+4]
            line = ", ".join(f"{val:.6g}" for val in chunk)
            data_lines.append(line)
        
        # Write to file
        with open(output_file, 'w') as f:
            f.write("\n".join(header_lines) + "\n")
            f.write("\n".join(data_lines) + "\n")
            f.write("#ENDOFDATA   : End of Data")
    
    def _write_emd(self, standardized: Dict[str, Any], output_file: Path) -> None:
        """
        Write data to EMD format (Thermo Fisher Scientific's electron microscopy data format, HDF5-based).
        
        Args:
            standardized: Standardized data dictionary
            output_file: Path to save the output file
        """
        data = standardized["data"]
        metadata = standardized["metadata"]
        
        # EMD is an HDF5-based format with a specific structure
        with h5py.File(output_file, 'w') as f:
            # Create the standard EMD structure
            data_group = f.create_group("data")
            data_0_group = data_group.create_group("data_0")
            
            # Add the data
            data_0_group.create_dataset("data", data=data)
            
            # Create metadata group
            meta_group = data_0_group.create_group("metadata")
            
            # Add pixel size if available (convert from nm to m for EMD format)
            if "pixel_size" in metadata["image"] and metadata["image"]["pixel_size"] is not None:
                pixel_size_m = metadata["image"]["pixel_size"] * 1e-9  # nm to m
                meta_group.attrs["pixelsize"] = pixel_size_m
            
            # Add microscope information
            if metadata["microscope"]["voltage"] > 0:
                meta_group.attrs["high_tension"] = metadata["microscope"]["voltage"]
            if metadata["microscope"]["magnification"] > 0:
                meta_group.attrs["magnification"] = metadata["microscope"]["magnification"]
            
            # Add acquisition date if available
            if "date" in metadata["acquisition"] and metadata["acquisition"]["date"]:
                meta_group.attrs["acquisition_time"] = metadata["acquisition"]["date"]
            
            # Add other metadata as attributes
            for section, section_data in metadata.items():
                for key, value in section_data.items():
                    if value is not None:
                        # Use a flattened naming scheme for attributes
                        attr_name = f"{section}_{key}"
                        # Convert complex types to JSON strings
                        if isinstance(value, (dict, list)):
                            meta_group.attrs[attr_name] = json.dumps(value)
                        else:
                            meta_group.attrs[attr_name] = value
    
    def _write_ser(self, standardized: Dict[str, Any], output_file: Path) -> None:
        """
        Write data to SER format (Thermo Fisher Scientific's TIA series data format).
        
        Args:
            standardized: Standardized data dictionary
            output_file: Path to save the output file
        """
        data = standardized["data"]
        metadata = standardized["metadata"]
        
        try:
            # Try to use hyperspy if available
            import hyperspy.api as hs
            
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
            
            # Save the file in SER format
            s.save(output_file, extension="ser")
            
        except (ImportError, Exception) as e:
            # If HyperSpy is not available, we can't write SER format directly
            # SER is a complex binary format that requires specialized libraries
            print(f"Error: Cannot write to SER format without HyperSpy: {e}")
            print("Falling back to HDF5 format with .ser extension")
            
            # Fall back to HDF5 format
            self._write_hdf5(standardized, output_file)
    
    def _write_hitachi(self, standardized: Dict[str, Any], output_file: Path) -> None:
        """
        Write data to Hitachi format (.hds).
        
        Args:
            standardized: Standardized data dictionary
            output_file: Path to save the output file
        """
        data = standardized["data"]
        metadata = standardized["metadata"]
        
        # Hitachi HDS is a proprietary format
        # For writing, we'll use TIFF format with metadata, which is commonly used by Hitachi systems
        try:
            # First try to use HyperSpy if available
            import hyperspy.api as hs
            
            # Create a HyperSpy signal
            if len(data.shape) == 1:
                # 1D data (spectrum)
                s = hs.signals.Signal1D(data)
            else:
                # 2D data (image)
                s = hs.signals.Signal2D(data)
            
            # Set metadata
            if "pixel_size" in metadata["image"] and metadata["image"]["pixel_size"] is not None:
                # Convert from nm to µm (HyperSpy typically uses µm)
                pixel_size_um = metadata["image"]["pixel_size"] / 1000
                s.axes_manager[0].scale = pixel_size_um
                if len(data.shape) > 1:
                    s.axes_manager[1].scale = pixel_size_um
                    s.axes_manager[0].units = "µm"
                    s.axes_manager[1].units = "µm"
                else:
                    s.axes_manager[0].units = "eV"  # For spectrum data
            
            # Set additional metadata
            s.metadata.General.title = metadata["sample"]["name"]
            
            if "name" in metadata["microscope"] and metadata["microscope"]["name"] is not None:
                s.metadata.Acquisition_instrument.SEM.microscope = metadata["microscope"]["name"]
                
            if "voltage" in metadata["microscope"] and metadata["microscope"]["voltage"] is not None:
                s.metadata.Acquisition_instrument.SEM.beam_energy = metadata["microscope"]["voltage"]
                
            if "magnification" in metadata["microscope"] and metadata["microscope"]["magnification"] is not None:
                s.metadata.Acquisition_instrument.SEM.magnification = metadata["microscope"]["magnification"]
            
            # Save as TIFF with metadata
            # If the output is .hds, change to .tif
            if output_file.suffix.lower() == '.hds':
                output_file = output_file.with_suffix('.tif')
                
            s.save(output_file)
            print(f"Saved as TIFF format with Hitachi metadata: {output_file}")
            
        except (ImportError, Exception) as e:
            print(f"Error: Cannot write to Hitachi format with HyperSpy: {e}")
            print("Falling back to standard TIFF format")
            
            # Fall back to TIFF format
            if output_file.suffix.lower() == '.hds':
                output_file = output_file.with_suffix('.tif')
                
            self._write_tiff(standardized, output_file)
    
    def _write_oxford(self, standardized: Dict[str, Any], output_file: Path) -> None:
        """
        Write data to Oxford Instruments format (.oip, .inca, .azw, .azd).
        
        Args:
            standardized: Standardized data dictionary
            output_file: Path to save the output file
        """
        data = standardized["data"]
        metadata = standardized["metadata"]
        
        # Oxford Instruments formats are used for EDS/EBSD data
        # For writing, we'll use HDF5 format with Oxford-specific metadata structure
        try:
            # First try to use HyperSpy if available
            import hyperspy.api as hs
            
            # Create a HyperSpy signal based on data dimensionality
            if len(data.shape) == 1 or (len(data.shape) == 2 and data.shape[0] == 1):
                # 1D data (spectrum)
                if len(data.shape) == 2:
                    data = data[0]  # Extract 1D array from 2D with single row
                s = hs.signals.EDSTEMSpectrum(data)  # Use EDS spectrum signal type
            else:
                # 2D data (image or map)
                s = hs.signals.Signal2D(data)
            
            # Set metadata
            if "pixel_size" in metadata["image"] and metadata["image"]["pixel_size"] is not None:
                # Convert from nm to µm (HyperSpy typically uses µm)
                pixel_size_um = metadata["image"]["pixel_size"] / 1000
                s.axes_manager[0].scale = pixel_size_um
                if len(s.data.shape) > 1:
                    s.axes_manager[1].scale = pixel_size_um
                    s.axes_manager[0].units = "µm"
                    s.axes_manager[1].units = "µm"
                else:
                    s.axes_manager[0].units = "eV"  # For spectrum data
            
            # Set additional metadata
            s.metadata.General.title = metadata["sample"]["name"]
            
            # Set Oxford-specific metadata
            if "name" in metadata["microscope"] and metadata["microscope"]["name"] is not None:
                s.metadata.Acquisition_instrument.SEM.microscope = metadata["microscope"]["name"]
                
            if "voltage" in metadata["microscope"] and metadata["microscope"]["voltage"] is not None:
                s.metadata.Acquisition_instrument.SEM.beam_energy = metadata["microscope"]["voltage"]
                
            if "magnification" in metadata["microscope"] and metadata["microscope"]["magnification"] is not None:
                s.metadata.Acquisition_instrument.SEM.magnification = metadata["microscope"]["magnification"]
            
            # For EDS data, set detector-specific metadata
            if isinstance(s, hs.signals._signal_1d.EDSSpectrum):
                # Set EDS detector metadata
                s.metadata.Acquisition_instrument.SEM.Detector.EDS.energy_resolution_MnKa = 130.0  # Default value in eV
                s.metadata.Acquisition_instrument.SEM.Detector.EDS.azimuth_angle = 0.0
                s.metadata.Acquisition_instrument.SEM.Detector.EDS.elevation_angle = 35.0
                s.metadata.Acquisition_instrument.SEM.Detector.EDS.live_time = 100.0  # Default in seconds
            
            # Save the file in appropriate format based on extension
            extension = output_file.suffix.lower()
            if extension in ['.azw', '.azd']:
                # For AZtec files, save as HDF5
                output_file = output_file.with_suffix('.h5')
                s.save(output_file)
                print(f"Saved as HDF5 format with Oxford AZtec metadata: {output_file}")
            elif extension in ['.oip', '.inca']:
                # For INCA files, save as HDF5
                output_file = output_file.with_suffix('.h5')
                s.save(output_file)
                print(f"Saved as HDF5 format with Oxford INCA metadata: {output_file}")
            else:
                # Save with original extension
                s.save(output_file)
            
        except (ImportError, Exception) as e:
            print(f"Error: Cannot write to Oxford format with HyperSpy: {e}")
            print("Falling back to HDF5 format")
            
            # Fall back to HDF5 format
            output_file = output_file.with_suffix('.h5')
            self._write_hdf5(standardized, output_file)
    
    def batch_translate(self, input_dir: Union[str, Path], output_dir: Union[str, Path],
                       output_format: str, file_pattern: str = "*") -> None:
        """
        Batch translate multiple files from a directory.
        
        Args:
            input_dir: Directory containing input files
            output_dir: Directory to save output files
            output_format: Format to convert files to
            file_pattern: Glob pattern to match input files
        """
        from glob import glob
        import os
        
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get list of input files matching the pattern
        input_files = list(input_dir.glob(file_pattern))
        
        if not input_files:
            print(f"No files matching pattern '{file_pattern}' found in {input_dir}")
            return
        
        print(f"Found {len(input_files)} files to translate")
        
        # Process each file
        for input_file in input_files:
            # Create output filename with new extension
            output_file = output_dir / f"{input_file.stem}.{output_format}"
            
            try:
                self.translate(input_file, output_file, output_format)
                print(f"Translated {input_file.name} to {output_file.name}")
            except Exception as e:
                print(f"Error translating {input_file.name}: {e}")
    
    def get_format_info(self, format_name: str) -> Dict[str, Any]:
        """
        Get information about a specific format.
        
        Args:
            format_name: Name of the format to get information about
            
        Returns:
            Dictionary with format information
        """
        # Define format information
        format_info = {
            "mrc": {
                "name": "MRC (Medical Research Council)",
                "description": "Common format for electron microscopy and tomography",
                "extensions": [".mrc", ".map", ".rec"],
                "libraries": ["mrcfile"],
                "metadata_support": "Limited",
                "dimensions": "2D/3D",
                "typical_use": "Cryo-EM, tomography reconstructions"
            },
            "em": {
                "name": "EM Format",
                "description": "Simple binary format for electron microscopy",
                "extensions": [".em"],
                "libraries": ["pyem"],
                "metadata_support": "Minimal",
                "dimensions": "2D/3D",
                "typical_use": "TEM images"
            },
            "dm3": {
                "name": "Digital Micrograph 3",
                "description": "Gatan Digital Micrograph format",
                "extensions": [".dm3"],
                "libraries": ["hyperspy"],
                "metadata_support": "Extensive",
                "dimensions": "2D/3D",
                "typical_use": "TEM/STEM images with metadata"
            },
            "dm4": {
                "name": "Digital Micrograph 4",
                "description": "Newer version of Gatan Digital Micrograph format",
                "extensions": [".dm4"],
                "libraries": ["hyperspy"],
                "metadata_support": "Extensive",
                "dimensions": "2D/3D/4D",
                "typical_use": "TEM/STEM images and spectrum images"
            },
            "tiff": {
                "name": "Tagged Image File Format",
                "description": "Flexible image format with metadata tags",
                "extensions": [".tif", ".tiff"],
                "libraries": ["pillow", "tifffile"],
                "metadata_support": "Good",
                "dimensions": "2D/3D",
                "typical_use": "General purpose, SEM images"
            },
            "hdf5": {
                "name": "Hierarchical Data Format 5",
                "description": "Flexible container format for complex data",
                "extensions": [".h5", ".hdf5"],
                "libraries": ["h5py"],
                "metadata_support": "Excellent",
                "dimensions": "Any",
                "typical_use": "4D-STEM, complex datasets with metadata"
            },
            "msa": {
                "name": "EMSA Format",
                "description": "Text-based format for spectral data in electron microscopy",
                "extensions": [".msa"],
                "libraries": ["numpy"],
                "metadata_support": "Good",
                "dimensions": "1D/2D",
                "typical_use": "EDS/EELS spectra"
            },
            "emd": {
                "name": "EMD Format",
                "description": "FEI's electron microscopy data format (HDF5-based)",
                "extensions": [".emd"],
                "libraries": ["hyperspy", "h5py"],
                "metadata_support": "Excellent",
                "dimensions": "Any",
                "typical_use": "Data from FEI/Thermo Fisher microscopes"
            },
            "ser": {
                "name": "FEI SER Format",
                "description": "FEI's TIA series data format",
                "extensions": [".ser"],
                "libraries": ["hyperspy"],
                "metadata_support": "Good",
                "dimensions": "2D/3D",
                "typical_use": "TEM/STEM images from FEI microscopes"
            },
            "emd": {
                "name": "EMD Format",
                "description": "FEI's electron microscopy data format (HDF5-based)",
                "extensions": [".emd"],
                "libraries": ["hyperspy", "h5py"],
                "metadata_support": "Excellent",
                "dimensions": "Any",
                "typical_use": "Data from FEI/Thermo Fisher microscopes"
            }
        }
        
        # Return information for the requested format
        format_name = format_name.lower()
        if format_name in format_info:
            return format_info[format_name]
        else:
            raise ValueError(f"Unknown format: {format_name}")
    
    def list_supported_formats(self) -> List[Dict[str, Any]]:
        """
        List all supported formats with their information.
        
        Returns:
            List of dictionaries with format information
        """
        formats = []
        
        for format_name in set(list(self.standardizer.supported_formats.keys()) + 
                             list(self.output_formats.keys())):
            format_name = format_name.lstrip('.')
            try:
                format_info = self.get_format_info(format_name)
                formats.append(format_info)
            except ValueError:
                # Skip unknown formats
                pass
        
        return formats


def main():
    """Command line interface for the translator."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Translate between electron microscopy data formats')
    parser.add_argument('input_file', help='Input electron microscopy data file')
    parser.add_argument('output_file', help='Output file path')
    parser.add_argument('--format', '-f', help='Output format (if not specified, inferred from output file extension)')
    parser.add_argument('--model', '-m', default='bert-base-uncased',
                      help='LLM model to use for metadata mapping')
    parser.add_argument('--list-formats', '-l', action='store_true',
                      help='List supported formats and exit')
    parser.add_argument('--batch', '-b', action='store_true',
                      help='Batch mode: input_file is a directory, output_file is a directory')
    parser.add_argument('--pattern', '-p', default='*',
                      help='File pattern for batch mode (default: *)')
    
    args = parser.parse_args()
    
    # Create translator
    translator = EMFormatTranslator(llm_model_name=args.model)
    
    # List formats if requested
    if args.list_formats:
        formats = translator.list_supported_formats()
        print("Supported formats:")
        for fmt in formats:
            print(f"\n{fmt['name']} ({', '.join(fmt['extensions'])})")
            print(f"  Description: {fmt['description']}")
            print(f"  Libraries: {', '.join(fmt['libraries'])}")
            print(f"  Metadata support: {fmt['metadata_support']}")
            print(f"  Typical use: {fmt['typical_use']}")
        return 0
    
    try:
        if args.batch:
            # Batch mode
            if not args.format:
                print("Error: Output format must be specified in batch mode")
                return 1
                
            translator.batch_translate(args.input_file, args.output_file, 
                                     args.format, file_pattern=args.pattern)
        else:
            # Single file mode
            translator.translate(args.input_file, args.output_file, args.format)
                
    except Exception as e:
        print(f"Error: {e}")
        return 1
        
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
