#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmarking module for EMDataBridge.

This module provides functionality to benchmark the performance of
EMDataBridge operations on electron microscopy data formats.
"""

import os
import time
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Union, List, Optional, Tuple
import matplotlib.pyplot as plt
from datetime import datetime

# Import EMDataBridge components
from .standardizer import EMDataStandardizer
from .translator import EMFormatTranslator
from .discovery import EMDataDiscovery


class EMDataBenchmark:
    """
    A class for benchmarking EMDataBridge operations.
    
    This benchmark tool can measure performance of format conversions,
    metadata extraction, and discovery operations.
    """
    
    def __init__(self, output_dir: Union[str, Path] = None):
        """
        Initialize the benchmark tool.
        
        Args:
            output_dir: Directory to save benchmark results
        """
        self.standardizer = EMDataStandardizer()
        self.translator = EMFormatTranslator()
        self.discovery = EMDataDiscovery()
        
        # Set output directory
        if output_dir is None:
            self.output_dir = Path("benchmark_results")
        else:
            self.output_dir = Path(output_dir)
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize results storage
        self.results = {
            "standardize": [],
            "translate": [],
            "discover": [],
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "version": "0.0.1"
            }
        }
    
    def benchmark_standardize(self, files: List[Union[str, Path]]):
        """
        Benchmark standardization of electron microscopy files.
        
        Args:
            files: List of files to benchmark
        """
        print(f"Benchmarking standardization of {len(files)} files...")
        
        for file_path in files:
            file_path = Path(file_path)
            
            try:
                # Measure time to standardize
                start_time = time.time()
                standardized = self.standardizer.standardize(file_path)
                end_time = time.time()
                
                duration = end_time - start_time
                
                # Record result
                result = {
                    "file": str(file_path),
                    "format": standardized["original_format"],
                    "size_mb": file_path.stat().st_size / (1024 * 1024),
                    "duration_seconds": duration,
                    "success": True
                }
                
                print(f"Standardized {file_path.name} ({result['format']}) in {duration:.2f} seconds")
            
            except Exception as e:
                # Record failure
                result = {
                    "file": str(file_path),
                    "format": file_path.suffix[1:],
                    "size_mb": file_path.stat().st_size / (1024 * 1024),
                    "duration_seconds": 0,
                    "success": False,
                    "error": str(e)
                }
                
                print(f"Failed to standardize {file_path.name}: {e}")
            
            self.results["standardize"].append(result)
    
    def benchmark_translate(self, files: List[Union[str, Path]], formats: List[str]):
        """
        Benchmark translation of electron microscopy files to different formats.
        
        Args:
            files: List of files to benchmark
            formats: List of output formats to test
        """
        print(f"Benchmarking translation of {len(files)} files to {len(formats)} formats...")
        
        # Create temporary directory for outputs
        temp_dir = self.output_dir / "temp_translations"
        temp_dir.mkdir(exist_ok=True)
        
        for file_path in files:
            file_path = Path(file_path)
            
            for output_format in formats:
                output_file = temp_dir / f"{file_path.stem}.{output_format}"
                
                try:
                    # Measure time to translate
                    start_time = time.time()
                    self.translator.translate(file_path, output_file, output_format)
                    end_time = time.time()
                    
                    duration = end_time - start_time
                    
                    # Record result
                    result = {
                        "input_file": str(file_path),
                        "input_format": file_path.suffix[1:],
                        "output_format": output_format,
                        "input_size_mb": file_path.stat().st_size / (1024 * 1024),
                        "output_size_mb": output_file.stat().st_size / (1024 * 1024),
                        "duration_seconds": duration,
                        "success": True
                    }
                    
                    print(f"Translated {file_path.name} to {output_format} in {duration:.2f} seconds")
                
                except Exception as e:
                    # Record failure
                    result = {
                        "input_file": str(file_path),
                        "input_format": file_path.suffix[1:],
                        "output_format": output_format,
                        "input_size_mb": file_path.stat().st_size / (1024 * 1024),
                        "output_size_mb": 0,
                        "duration_seconds": 0,
                        "success": False,
                        "error": str(e)
                    }
                    
                    print(f"Failed to translate {file_path.name} to {output_format}: {e}")
                
                self.results["translate"].append(result)
    
    def benchmark_discover(self, directories: List[Union[str, Path]]):
        """
        Benchmark discovery of electron microscopy datasets.
        
        Args:
            directories: List of directories to benchmark
        """
        print(f"Benchmarking discovery of {len(directories)} directories...")
        
        for directory in directories:
            directory = Path(directory)
            
            try:
                # Measure time to discover
                start_time = time.time()
                dataset_structure = self.discovery.discover_dataset_structure(directory)
                end_time = time.time()
                
                duration = end_time - start_time
                
                # Record result
                result = {
                    "directory": str(directory),
                    "file_count": len(dataset_structure["files"]),
                    "relationship_count": len(dataset_structure["relationships"]),
                    "duration_seconds": duration,
                    "success": True
                }
                
                print(f"Discovered {result['file_count']} files in {directory.name} in {duration:.2f} seconds")
            
            except Exception as e:
                # Record failure
                result = {
                    "directory": str(directory),
                    "file_count": 0,
                    "relationship_count": 0,
                    "duration_seconds": 0,
                    "success": False,
                    "error": str(e)
                }
                
                print(f"Failed to discover {directory.name}: {e}")
            
            self.results["discover"].append(result)
    
    def save_results(self, filename: str = "benchmark_results.json"):
        """
        Save benchmark results to a file.
        
        Args:
            filename: Name of the output file
        """
        output_file = self.output_dir / filename
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"Benchmark results saved to {output_file}")
    
    def generate_report(self, filename: str = "benchmark_report.html"):
        """
        Generate a report of benchmark results.
        
        Args:
            filename: Name of the output file
        """
        output_file = self.output_dir / filename
        
        # Create visualizations
        self._create_standardize_chart()
        self._create_translate_chart()
        self._create_discover_chart()
        
        # Generate HTML report
        html = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            "    <title>EMDataBridge Benchmark Report</title>",
            "    <style>",
            "        body { font-family: Arial, sans-serif; margin: 20px; }",
            "        h1 { color: #2c3e50; }",
            "        h2 { color: #3498db; }",
            "        .chart { margin: 20px 0; max-width: 800px; }",
            "        table { border-collapse: collapse; width: 100%; }",
            "        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
            "        th { background-color: #f2f2f2; }",
            "        tr:nth-child(even) { background-color: #f9f9f9; }",
            "    </style>",
            "</head>",
            "<body>",
            "    <h1>EMDataBridge Benchmark Report</h1>",
            f"    <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>",
            f"    <p>EMDataBridge version: {self.results['metadata']['version']}</p>",
            "",
            "    <h2>Standardization Performance</h2>",
            "    <div class='chart'><img src='standardize_chart.png' alt='Standardization Performance'></div>",
            self._generate_table("standardize"),
            "",
            "    <h2>Translation Performance</h2>",
            "    <div class='chart'><img src='translate_chart.png' alt='Translation Performance'></div>",
            self._generate_table("translate"),
            "",
            "    <h2>Discovery Performance</h2>",
            "    <div class='chart'><img src='discover_chart.png' alt='Discovery Performance'></div>",
            self._generate_table("discover"),
            "",
            "</body>",
            "</html>"
        ]
        
        with open(output_file, 'w') as f:
            f.write('\n'.join(html))
        
        print(f"Benchmark report saved to {output_file}")
    
    def _create_standardize_chart(self):
        """
        Create a chart of standardization performance.
        """
        if not self.results["standardize"]:
            return
        
        # Filter successful results
        successful = [r for r in self.results["standardize"] if r["success"]]
        
        if not successful:
            return
        
        # Group by format
        formats = {}
        for result in successful:
            fmt = result["format"]
            if fmt not in formats:
                formats[fmt] = []
            formats[fmt].append(result["duration_seconds"])
        
        # Create chart
        plt.figure(figsize=(10, 6))
        
        # Plot average duration by format
        avg_durations = [(fmt, np.mean(durations)) for fmt, durations in formats.items()]
        avg_durations.sort(key=lambda x: x[1])
        
        x = [fmt for fmt, _ in avg_durations]
        y = [duration for _, duration in avg_durations]
        
        plt.bar(x, y)
        plt.xlabel('Format')
        plt.ylabel('Average Duration (seconds)')
        plt.title('Standardization Performance by Format')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plt.savefig(self.output_dir / "standardize_chart.png")
        plt.close()
    
    def _create_translate_chart(self):
        """
        Create a chart of translation performance.
        """
        if not self.results["translate"]:
            return
        
        # Filter successful results
        successful = [r for r in self.results["translate"] if r["success"]]
        
        if not successful:
            return
        
        # Group by output format
        formats = {}
        for result in successful:
            fmt = result["output_format"]
            if fmt not in formats:
                formats[fmt] = []
            formats[fmt].append(result["duration_seconds"])
        
        # Create chart
        plt.figure(figsize=(10, 6))
        
        # Plot average duration by format
        avg_durations = [(fmt, np.mean(durations)) for fmt, durations in formats.items()]
        avg_durations.sort(key=lambda x: x[1])
        
        x = [fmt for fmt, _ in avg_durations]
        y = [duration for _, duration in avg_durations]
        
        plt.bar(x, y)
        plt.xlabel('Output Format')
        plt.ylabel('Average Duration (seconds)')
        plt.title('Translation Performance by Output Format')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plt.savefig(self.output_dir / "translate_chart.png")
        plt.close()
    
    def _create_discover_chart(self):
        """
        Create a chart of discovery performance.
        """
        if not self.results["discover"]:
            return
        
        # Filter successful results
        successful = [r for r in self.results["discover"] if r["success"]]
        
        if not successful:
            return
        
        # Create scatter plot of file count vs. duration
        plt.figure(figsize=(10, 6))
        
        x = [r["file_count"] for r in successful]
        y = [r["duration_seconds"] for r in successful]
        
        plt.scatter(x, y)
        plt.xlabel('Number of Files')
        plt.ylabel('Duration (seconds)')
        plt.title('Discovery Performance by File Count')
        
        # Add trend line
        if len(x) > 1:
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            plt.plot(x, p(x), "r--")
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "discover_chart.png")
        plt.close()
    
    def _generate_table(self, result_type):
        """
        Generate an HTML table for the specified result type.
        
        Args:
            result_type: Type of result to generate table for
            
        Returns:
            HTML table as a string
        """
        results = self.results[result_type]
        
        if not results:
            return "<p>No results available.</p>"
        
        # Determine columns based on result type
        if result_type == "standardize":
            columns = ["file", "format", "size_mb", "duration_seconds", "success"]
        elif result_type == "translate":
            columns = ["input_file", "input_format", "output_format", "input_size_mb", "output_size_mb", "duration_seconds", "success"]
        elif result_type == "discover":
            columns = ["directory", "file_count", "relationship_count", "duration_seconds", "success"]
        else:
            return "<p>Unknown result type.</p>"
        
        # Generate table
        html = ["<table>", "<tr>"]
        
        # Add headers
        for column in columns:
            html.append(f"<th>{column.replace('_', ' ').title()}</th>")
        html.append("</tr>")
        
        # Add rows
        for result in results:
            html.append("<tr>")
            for column in columns:
                value = result.get(column, "")
                
                # Format numeric values
                if isinstance(value, float):
                    if column.endswith("_mb"):
                        value = f"{value:.2f} MB"
                    elif column.endswith("_seconds"):
                        value = f"{value:.2f} s"
                
                html.append(f"<td>{value}</td>")
            html.append("</tr>")
        
        html.append("</table>")
        return "\n".join(html)


def main():
    """Main entry point for the benchmark CLI."""
    parser = argparse.ArgumentParser(
        description='EMDataBridge Benchmark Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Standardize benchmark
    standardize_parser = subparsers.add_parser('standardize', help='Benchmark standardization')
    standardize_parser.add_argument('--files', nargs='+', required=True, help='Files to benchmark')
    standardize_parser.add_argument('--output-dir', help='Output directory for results')
    
    # Translate benchmark
    translate_parser = subparsers.add_parser('translate', help='Benchmark translation')
    translate_parser.add_argument('--files', nargs='+', required=True, help='Files to benchmark')
    translate_parser.add_argument('--formats', nargs='+', required=True, help='Output formats to test')
    translate_parser.add_argument('--output-dir', help='Output directory for results')
    
    # Batch benchmark
    batch_parser = subparsers.add_parser('batch', help='Benchmark batch translation')
    batch_parser.add_argument('--input-dir', required=True, help='Input directory')
    batch_parser.add_argument('--formats', nargs='+', required=True, help='Output formats to test')
    batch_parser.add_argument('--output-dir', help='Output directory for results')
    
    # Discovery benchmark
    discover_parser = subparsers.add_parser('discover', help='Benchmark discovery')
    discover_parser.add_argument('--directories', nargs='+', required=True, help='Directories to benchmark')
    discover_parser.add_argument('--output-dir', help='Output directory for results')
    
    # Full benchmark
    full_parser = subparsers.add_parser('full', help='Run all benchmarks')
    full_parser.add_argument('--files', nargs='+', help='Files to benchmark')
    full_parser.add_argument('--formats', nargs='+', help='Output formats to test')
    full_parser.add_argument('--input-dir', help='Input directory for batch benchmark')
    full_parser.add_argument('--directories', nargs='+', help='Directories to benchmark for discovery')
    full_parser.add_argument('--output-dir', help='Output directory for results')
    
    args = parser.parse_args()
    
    # Create benchmark instance
    benchmark = EMDataBenchmark(output_dir=args.output_dir if hasattr(args, 'output_dir') else None)
    
    if args.command == 'standardize':
        benchmark.benchmark_standardize(args.files)
    elif args.command == 'translate':
        benchmark.benchmark_translate(args.files, args.formats)
    elif args.command == 'batch':
        # Find all files in input directory
        files = list(Path(args.input_dir).glob('*.*'))
        benchmark.benchmark_translate(files, args.formats)
    elif args.command == 'discover':
        benchmark.benchmark_discover(args.directories)
    elif args.command == 'full':
        # Run all benchmarks if parameters are provided
        if hasattr(args, 'files') and args.files:
            benchmark.benchmark_standardize(args.files)
            if hasattr(args, 'formats') and args.formats:
                benchmark.benchmark_translate(args.files, args.formats)
        
        if hasattr(args, 'input_dir') and args.input_dir and hasattr(args, 'formats') and args.formats:
            files = list(Path(args.input_dir).glob('*.*'))
            benchmark.benchmark_translate(files, args.formats)
        
        if hasattr(args, 'directories') and args.directories:
            benchmark.benchmark_discover(args.directories)
    else:
        parser.print_help()
        return 1
    
    # Save results and generate report
    benchmark.save_results()
    benchmark.generate_report()
    
    return 0


if __name__ == "__main__":
    main()
