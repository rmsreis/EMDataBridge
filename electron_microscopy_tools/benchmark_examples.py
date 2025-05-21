#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Examples of using the benchmarking module for specific format tests.

This script provides concrete examples of benchmarking the newly added
formats (Thermo Fisher Scientific, Oxford Instruments, and Hitachi).
"""

import os
from pathlib import Path
from benchmark import EMBenchmark


def benchmark_thermo_fisher_formats():
    """Benchmark Thermo Fisher Scientific formats (formerly FEI)."""
    print("\nBenchmarking Thermo Fisher Scientific formats...")
    benchmark = EMBenchmark(output_dir="benchmark_results/thermo_fisher")
    
    # Sample data paths - update these to your actual file paths
    emd_files = list(Path("sample_data").glob("*.emd"))
    ser_files = list(Path("sample_data").glob("*.ser"))
    
    # Combine all Thermo Fisher files
    thermo_files = emd_files + ser_files
    
    if not thermo_files:
        print("No Thermo Fisher Scientific files found for benchmarking.")
        return
    
    print(f"Found {len(thermo_files)} Thermo Fisher Scientific files.")
    
    # Test standardization
    benchmark.benchmark_standardization(thermo_files)
    
    # Test translation to other formats
    benchmark.benchmark_translation(thermo_files, 
                                  ["tiff", "hdf5", "mrc", "emd", "ser"])
    
    # Generate report
    benchmark.generate_report("thermo_fisher_benchmark_report.json")
    print("Thermo Fisher Scientific benchmark complete!")


def benchmark_oxford_formats():
    """Benchmark Oxford Instruments formats."""
    print("\nBenchmarking Oxford Instruments formats...")
    benchmark = EMBenchmark(output_dir="benchmark_results/oxford")
    
    # Sample data paths - update these to your actual file paths
    oxford_files = []
    for ext in [".oip", ".inca", ".azw", ".azd"]:
        oxford_files.extend(list(Path("sample_data").glob(f"*{ext}")))
    
    if not oxford_files:
        print("No Oxford Instruments files found for benchmarking.")
        return
    
    print(f"Found {len(oxford_files)} Oxford Instruments files.")
    
    # Test standardization
    benchmark.benchmark_standardization(oxford_files)
    
    # Test translation to other formats
    benchmark.benchmark_translation(oxford_files, 
                                  ["tiff", "hdf5", "mrc", "oip", "azw"])
    
    # Generate report
    benchmark.generate_report("oxford_benchmark_report.json")
    print("Oxford Instruments benchmark complete!")


def benchmark_hitachi_formats():
    """Benchmark Hitachi formats."""
    print("\nBenchmarking Hitachi formats...")
    benchmark = EMBenchmark(output_dir="benchmark_results/hitachi")
    
    # Sample data paths - update these to your actual file paths
    hitachi_files = list(Path("sample_data").glob("*.hds"))
    
    if not hitachi_files:
        print("No Hitachi files found for benchmarking.")
        return
    
    print(f"Found {len(hitachi_files)} Hitachi files.")
    
    # Test standardization
    benchmark.benchmark_standardization(hitachi_files)
    
    # Test translation to other formats
    benchmark.benchmark_translation(hitachi_files, 
                                  ["tiff", "hdf5", "mrc", "hds"])
    
    # Generate report
    benchmark.generate_report("hitachi_benchmark_report.json")
    print("Hitachi benchmark complete!")


def benchmark_format_comparison():
    """Compare performance across all formats."""
    print("\nRunning format comparison benchmark...")
    benchmark = EMBenchmark(output_dir="benchmark_results/comparison")
    
    # Get all sample files
    sample_dir = Path("sample_data")
    all_files = []
    
    # Thermo Fisher Scientific formats
    all_files.extend(list(sample_dir.glob("*.emd")))
    all_files.extend(list(sample_dir.glob("*.ser")))
    
    # Oxford Instruments formats
    all_files.extend(list(sample_dir.glob("*.oip")))
    all_files.extend(list(sample_dir.glob("*.inca")))
    all_files.extend(list(sample_dir.glob("*.azw")))
    all_files.extend(list(sample_dir.glob("*.azd")))
    
    # Hitachi formats
    all_files.extend(list(sample_dir.glob("*.hds")))
    
    # Common formats for comparison
    all_files.extend(list(sample_dir.glob("*.dm4")))
    all_files.extend(list(sample_dir.glob("*.tif")))
    all_files.extend(list(sample_dir.glob("*.mrc")))
    
    if not all_files:
        print("No files found for benchmarking.")
        return
    
    print(f"Found {len(all_files)} files for format comparison.")
    
    # Common output formats to test
    output_formats = ["tiff", "hdf5", "mrc", "emd", "ser", "hds", "oip"]
    
    # Run standardization benchmark
    benchmark.benchmark_standardization(all_files)
    
    # Run translation benchmark
    benchmark.benchmark_translation(all_files, output_formats)
    
    # Generate report
    benchmark.generate_report("format_comparison_report.json")
    print("Format comparison benchmark complete!")


def main():
    """Main function to run all benchmarks."""
    print("Electron Microscopy Tools - Format-Specific Benchmarks")
    print("-" * 60)
    
    # Create benchmark results directory
    os.makedirs("benchmark_results", exist_ok=True)
    
    # Run specific format benchmarks
    benchmark_thermo_fisher_formats()
    benchmark_oxford_formats()
    benchmark_hitachi_formats()
    
    # Run comparison benchmark
    benchmark_format_comparison()
    
    print("\nAll benchmarks complete!")
    print("Results saved in the benchmark_results directory.")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
