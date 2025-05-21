#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmarking module for electron microscopy tools.

This module provides functionality to benchmark the performance of format
conversions and metadata extraction in the electron microscopy tools.
"""

import time
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Union
from datetime import datetime

from standardizer import EMDataStandardizer
from translator import EMFormatTranslator


class EMBenchmark:
    """A class for benchmarking electron microscopy data processing tools."""
    
    def __init__(self, output_dir: Union[str, Path] = "benchmark_results"):
        """
        Initialize the benchmarking tool.
        
        Args:
            output_dir: Directory to save benchmark results
        """
        self.output_dir = Path(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize the tools to benchmark
        self.standardizer = EMDataStandardizer()
        self.translator = EMFormatTranslator()
        
        # Store results
        self.results = {}
    
    def benchmark_standardization(self, input_files: List[Union[str, Path]]) -> Dict[str, Any]:
        """
        Benchmark the standardization process for multiple files.
        
        Args:
            input_files: List of files to benchmark
            
        Returns:
            Dictionary with benchmark results
        """
        results = {}
        
        for file_path in input_files:
            file_path = Path(file_path)
            format_name = file_path.suffix.lower()[1:]  # Remove leading dot
            file_size = file_path.stat().st_size / (1024 * 1024)  # Size in MB
            
            print(f"Benchmarking standardization of {file_path}...")
            
            # Measure time for standardization
            start_time = time.time()
            try:
                standardized = self.standardizer.standardize(file_path)
                success = True
            except Exception as e:
                print(f"Error standardizing {file_path}: {e}")
                success = False
                standardized = None
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Record results
            result = {
                "file": str(file_path),
                "format": format_name,
                "size_mb": file_size,
                "processing_time_sec": processing_time,
                "success": success,
            }
            
            # Add metadata extraction stats if successful
            if success and standardized:
                result["data_shape"] = str(standardized["data"].shape)
                result["data_type"] = str(standardized["data"].dtype)
                result["metadata_keys"] = list(standardized["metadata"].keys())
                result["metadata_completeness"] = self._calculate_metadata_completeness(standardized["metadata"])
            
            # Store in results dictionary
            if format_name not in results:
                results[format_name] = []
            
            results[format_name].append(result)
        
        # Store in overall results
        self.results["standardization"] = results
        return results
    
    def benchmark_translation(self, input_files: List[Union[str, Path]], 
                             output_formats: List[str]) -> Dict[str, Any]:
        """
        Benchmark the translation process for multiple files and formats.
        
        Args:
            input_files: List of files to benchmark
            output_formats: List of output formats to test
            
        Returns:
            Dictionary with benchmark results
        """
        results = {}
        
        for file_path in input_files:
            file_path = Path(file_path)
            input_format = file_path.suffix.lower()[1:]  # Remove leading dot
            file_size = file_path.stat().st_size / (1024 * 1024)  # Size in MB
            
            for output_format in output_formats:
                print(f"Benchmarking translation of {file_path} to {output_format}...")
                
                # Create temporary output file
                output_file = self.output_dir / f"temp_output.{output_format}"
                
                # Measure time for translation
                start_time = time.time()
                try:
                    self.translator.translate(file_path, output_file, output_format)
                    success = True
                except Exception as e:
                    print(f"Error translating {file_path} to {output_format}: {e}")
                    success = False
                
                end_time = time.time()
                processing_time = end_time - start_time
                
                # Calculate output file size if successful
                output_size = output_file.stat().st_size / (1024 * 1024) if success and output_file.exists() else 0
                
                # Record results
                result = {
                    "input_file": str(file_path),
                    "input_format": input_format,
                    "output_format": output_format,
                    "input_size_mb": file_size,
                    "output_size_mb": output_size,
                    "processing_time_sec": processing_time,
                    "success": success,
                }
                
                # Store in results dictionary
                key = f"{input_format}_to_{output_format}"
                if key not in results:
                    results[key] = []
                
                results[key].append(result)
                
                # Clean up temporary file
                if output_file.exists():
                    try:
                        os.remove(output_file)
                    except Exception:
                        pass
        
        # Store in overall results
        self.results["translation"] = results
        return results
    
    def benchmark_batch_processing(self, input_dir: Union[str, Path], 
                                  output_formats: List[str],
                                  file_pattern: str = "*") -> Dict[str, Any]:
        """
        Benchmark batch processing of files.
        
        Args:
            input_dir: Directory containing input files
            output_formats: List of output formats to test
            file_pattern: Glob pattern to match input files
            
        Returns:
            Dictionary with benchmark results
        """
        input_dir = Path(input_dir)
        results = {}
        
        for output_format in output_formats:
            print(f"Benchmarking batch processing to {output_format}...")
            
            # Create temporary output directory
            temp_output_dir = self.output_dir / f"temp_batch_{output_format}"
            os.makedirs(temp_output_dir, exist_ok=True)
            
            # Measure time for batch translation
            start_time = time.time()
            try:
                self.translator.batch_translate(input_dir, temp_output_dir, 
                                              output_format, file_pattern)
                success = True
            except Exception as e:
                print(f"Error in batch processing to {output_format}: {e}")
                success = False
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Count processed files
            processed_files = list(temp_output_dir.glob(f"*.{output_format}"))
            file_count = len(processed_files)
            
            # Record results
            result = {
                "input_dir": str(input_dir),
                "output_format": output_format,
                "file_pattern": file_pattern,
                "file_count": file_count,
                "processing_time_sec": processing_time,
                "success": success,
                "avg_time_per_file": processing_time / file_count if file_count > 0 else 0,
            }
            
            results[output_format] = result
            
            # Clean up temporary directory
            for file in processed_files:
                try:
                    os.remove(file)
                except Exception:
                    pass
            try:
                os.rmdir(temp_output_dir)
            except Exception:
                pass
        
        # Store in overall results
        self.results["batch_processing"] = results
        return results
    
    def _calculate_metadata_completeness(self, metadata: Dict[str, Any]) -> float:
        """
        Calculate a score for metadata completeness.
        
        Args:
            metadata: Metadata dictionary
            
        Returns:
            Completeness score between 0 and 1
        """
        # Define expected metadata fields for a complete record
        expected_fields = {
            "microscope": ["name", "voltage", "magnification"],
            "sample": ["name", "description"],
            "acquisition": ["date", "operator", "mode"],
            "image": ["width", "height", "pixel_size", "bit_depth"],
        }
        
        # Count present fields
        total_fields = 0
        present_fields = 0
        
        for section, fields in expected_fields.items():
            if section in metadata:
                for field in fields:
                    total_fields += 1
                    if field in metadata[section] and metadata[section][field] is not None:
                        present_fields += 1
        
        # Calculate completeness score
        return present_fields / total_fields if total_fields > 0 else 0
    
    def generate_report(self, report_file: Optional[Union[str, Path]] = None) -> None:
        """
        Generate a benchmark report.
        
        Args:
            report_file: Path to save the report (if None, uses timestamp)
        """
        if not self.results:
            print("No benchmark results to report.")
            return
        
        # Create report filename if not provided
        if report_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = self.output_dir / f"benchmark_report_{timestamp}.json"
        else:
            report_file = Path(report_file)
        
        # Add timestamp to results
        self.results["timestamp"] = datetime.now().isoformat()
        
        # Save results as JSON
        with open(report_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"Benchmark report saved to {report_file}")
        
        # Generate visualizations
        self._generate_visualizations()
    
    def _generate_visualizations(self) -> None:
        """
        Generate visualizations of benchmark results.
        """
        # Create visualizations directory
        vis_dir = self.output_dir / "visualizations"
        os.makedirs(vis_dir, exist_ok=True)
        
        # Generate standardization performance chart
        if "standardization" in self.results:
            self._plot_standardization_performance(vis_dir)
        
        # Generate translation performance chart
        if "translation" in self.results:
            self._plot_translation_performance(vis_dir)
        
        # Generate batch processing performance chart
        if "batch_processing" in self.results:
            self._plot_batch_performance(vis_dir)
    
    def _plot_standardization_performance(self, vis_dir: Path) -> None:
        """
        Plot standardization performance by format.
        
        Args:
            vis_dir: Directory to save visualizations
        """
        std_results = self.results["standardization"]
        formats = list(std_results.keys())
        
        if not formats:
            return
        
        # Calculate average processing time per format
        avg_times = []
        for fmt in formats:
            times = [r["processing_time_sec"] for r in std_results[fmt] if r["success"]]
            avg_times.append(np.mean(times) if times else 0)
        
        # Create bar chart
        plt.figure(figsize=(10, 6))
        plt.bar(formats, avg_times)
        plt.xlabel('Format')
        plt.ylabel('Average Processing Time (seconds)')
        plt.title('Standardization Performance by Format')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save figure
        plt.savefig(vis_dir / "standardization_performance.png")
        plt.close()
    
    def _plot_translation_performance(self, vis_dir: Path) -> None:
        """
        Plot translation performance by format pair.
        
        Args:
            vis_dir: Directory to save visualizations
        """
        trans_results = self.results["translation"]
        format_pairs = list(trans_results.keys())
        
        if not format_pairs:
            return
        
        # Calculate average processing time per format pair
        avg_times = []
        for pair in format_pairs:
            times = [r["processing_time_sec"] for r in trans_results[pair] if r["success"]]
            avg_times.append(np.mean(times) if times else 0)
        
        # Create bar chart
        plt.figure(figsize=(12, 6))
        plt.bar(format_pairs, avg_times)
        plt.xlabel('Format Conversion')
        plt.ylabel('Average Processing Time (seconds)')
        plt.title('Translation Performance by Format Pair')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save figure
        plt.savefig(vis_dir / "translation_performance.png")
        plt.close()
    
    def _plot_batch_performance(self, vis_dir: Path) -> None:
        """
        Plot batch processing performance by output format.
        
        Args:
            vis_dir: Directory to save visualizations
        """
        batch_results = self.results["batch_processing"]
        formats = list(batch_results.keys())
        
        if not formats:
            return
        
        # Get average time per file for each format
        avg_times = [batch_results[fmt]["avg_time_per_file"] for fmt in formats]
        
        # Create bar chart
        plt.figure(figsize=(10, 6))
        plt.bar(formats, avg_times)
        plt.xlabel('Output Format')
        plt.ylabel('Average Time per File (seconds)')
        plt.title('Batch Processing Performance by Format')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save figure
        plt.savefig(vis_dir / "batch_processing_performance.png")
        plt.close()


def main():
    """Command line interface for the benchmarking tool."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark electron microscopy tools")
    subparsers = parser.add_subparsers(dest="command", help="Benchmark command")
    
    # Standardization benchmark parser
    std_parser = subparsers.add_parser("standardize", help="Benchmark standardization")
    std_parser.add_argument("--files", nargs="+", required=True, help="Files to benchmark")
    std_parser.add_argument("--output-dir", default="benchmark_results", help="Output directory")
    
    # Translation benchmark parser
    trans_parser = subparsers.add_parser("translate", help="Benchmark translation")
    trans_parser.add_argument("--files", nargs="+", required=True, help="Files to benchmark")
    trans_parser.add_argument("--formats", nargs="+", required=True, help="Output formats to test")
    trans_parser.add_argument("--output-dir", default="benchmark_results", help="Output directory")
    
    # Batch processing benchmark parser
    batch_parser = subparsers.add_parser("batch", help="Benchmark batch processing")
    batch_parser.add_argument("--input-dir", required=True, help="Input directory")
    batch_parser.add_argument("--formats", nargs="+", required=True, help="Output formats to test")
    batch_parser.add_argument("--pattern", default="*", help="File pattern")
    batch_parser.add_argument("--output-dir", default="benchmark_results", help="Output directory")
    
    # Full benchmark parser
    full_parser = subparsers.add_parser("full", help="Run all benchmarks")
    full_parser.add_argument("--files", nargs="+", required=True, help="Files to benchmark")
    full_parser.add_argument("--formats", nargs="+", required=True, help="Output formats to test")
    full_parser.add_argument("--input-dir", required=True, help="Input directory for batch tests")
    full_parser.add_argument("--pattern", default="*", help="File pattern for batch tests")
    full_parser.add_argument("--output-dir", default="benchmark_results", help="Output directory")
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 1
    
    # Initialize benchmark tool
    benchmark = EMBenchmark(output_dir=args.output_dir)
    
    # Run requested benchmark
    if args.command == "standardize":
        benchmark.benchmark_standardization(args.files)
    elif args.command == "translate":
        benchmark.benchmark_translation(args.files, args.formats)
    elif args.command == "batch":
        benchmark.benchmark_batch_processing(args.input_dir, args.formats, args.pattern)
    elif args.command == "full":
        benchmark.benchmark_standardization(args.files)
        benchmark.benchmark_translation(args.files, args.formats)
        benchmark.benchmark_batch_processing(args.input_dir, args.formats, args.pattern)
    
    # Generate report
    benchmark.generate_report()
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
