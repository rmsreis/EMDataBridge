#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Discovery module for electron microscopy data organization.

This module provides functionality to automatically discover and extract
self-describing data organization from electron microscopy files using LLM-based approaches.
"""

import os
import json
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Union, List, Optional, Tuple, Set
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Import the standardizer for common functionality
from standardizer import EMDataStandardizer


class EMDataDiscovery:
    """
    A class for discovering and extracting self-describing data organization
    from electron microscopy files using LLM-based approaches.
    
    This discovery tool can analyze collections of electron microscopy files,
    extract metadata patterns, and infer relationships between files.
    """
    
    def __init__(self, llm_model_name: str = "t5-small"):
        """
        Initialize the discovery tool with an optional LLM model for metadata analysis.
        
        Args:
            llm_model_name: Name of the pre-trained model to use for metadata analysis
        """
        # Initialize the standardizer for common functionality
        self.standardizer = EMDataStandardizer()
        
        # Initialize LLM model for metadata analysis
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(llm_model_name)
            self.llm_available = True
        except Exception as e:
            print(f"Warning: Could not load LLM model: {e}")
            self.llm_available = False
        
        # Initialize TF-IDF vectorizer for text similarity
        self.vectorizer = TfidfVectorizer()
    
    def discover_dataset_structure(self, directory: Union[str, Path], 
                                 recursive: bool = True,
                                 file_pattern: str = "*") -> Dict[str, Any]:
        """
        Discover the structure of a dataset in a directory.
        
        Args:
            directory: Directory containing electron microscopy files
            recursive: Whether to search recursively in subdirectories
            file_pattern: Glob pattern to match files
            
        Returns:
            Dictionary with dataset structure information
        """
        directory = Path(directory)
        
        # Find all files matching the pattern
        if recursive:
            files = list(directory.glob(f"**/{file_pattern}"))
        else:
            files = list(directory.glob(file_pattern))
        
        # Filter for known file extensions
        supported_extensions = set(ext.lower() for ext in [
            ".mrc", ".map", ".rec", ".em", ".dm3", ".dm4", 
            ".tif", ".tiff", ".h5", ".hdf5", ".ser", ".emd"
        ])
        
        em_files = [f for f in files if f.suffix.lower() in supported_extensions]
        
        if not em_files:
            print(f"No supported electron microscopy files found in {directory}")
            return {"files": [], "patterns": {}, "relationships": []}
        
        print(f"Found {len(em_files)} electron microscopy files")
        
        # Extract metadata from files
        metadata_collection = []
        for file_path in em_files:
            try:
                # Standardize the file to extract metadata
                standardized = self.standardizer.standardize(file_path)
                metadata = standardized["metadata"]
                
                # Add file information
                file_info = {
                    "path": str(file_path),
                    "name": file_path.name,
                    "format": standardized["original_format"],
                    "relative_path": str(file_path.relative_to(directory)),
                    "metadata": metadata
                }
                
                metadata_collection.append(file_info)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        
        # Analyze metadata patterns
        patterns = self._analyze_metadata_patterns(metadata_collection)
        
        # Discover relationships between files
        relationships = self._discover_relationships(metadata_collection)
        
        # Discover dataset hierarchy
        hierarchy = self._discover_hierarchy(metadata_collection)
        
        return {
            "files": metadata_collection,
            "patterns": patterns,
            "relationships": relationships,
            "hierarchy": hierarchy
        }
    
    def _analyze_metadata_patterns(self, metadata_collection: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze patterns in metadata across files.
        
        Args:
            metadata_collection: List of file metadata dictionaries
            
        Returns:
            Dictionary with metadata pattern information
        """
        # Count occurrences of metadata fields
        field_counts = defaultdict(int)
        field_values = defaultdict(set)
        
        for file_info in metadata_collection:
            metadata = file_info["metadata"]
            
            # Recursively process nested metadata
            def process_metadata(data, prefix=""):
                if isinstance(data, dict):
                    for key, value in data.items():
                        full_key = f"{prefix}.{key}" if prefix else key
                        
                        if value is not None:
                            field_counts[full_key] += 1
                            
                            # Store unique values (convert non-hashable types to strings)
                            if isinstance(value, (list, dict)):
                                field_values[full_key].add(str(value))
                            else:
                                field_values[full_key].add(value)
                            
                            # Recurse into nested structures
                            process_metadata(value, full_key)
            
            process_metadata(metadata)
        
        # Calculate statistics
        total_files = len(metadata_collection)
        field_stats = {}
        
        for field, count in field_counts.items():
            coverage = count / total_files
            unique_values = field_values[field]
            cardinality = len(unique_values)
            
            # Determine if field is an identifier
            is_identifier = cardinality == total_files and coverage == 1.0
            
            # Determine if field is categorical
            is_categorical = cardinality < total_files * 0.5 and cardinality > 1
            
            # Determine field type
            if all(isinstance(v, (int, float)) for v in unique_values if not isinstance(v, bool)):
                field_type = "numeric"
            elif all(isinstance(v, str) for v in unique_values):
                field_type = "text"
            elif all(isinstance(v, bool) for v in unique_values):
                field_type = "boolean"
            else:
                field_type = "mixed"
            
            field_stats[field] = {
                "coverage": coverage,
                "cardinality": cardinality,
                "is_identifier": is_identifier,
                "is_categorical": is_categorical,
                "type": field_type,
                "example_values": list(unique_values)[:5]  # Show up to 5 examples
            }
        
        # Use LLM to enhance pattern analysis if available
        if self.llm_available:
            field_stats = self._enhance_patterns_with_llm(field_stats, metadata_collection)
        
        return field_stats
    
    def _enhance_patterns_with_llm(self, field_stats: Dict[str, Any], 
                                 metadata_collection: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Use LLM to enhance pattern analysis.
        
        Args:
            field_stats: Dictionary with field statistics
            metadata_collection: List of file metadata dictionaries
            
        Returns:
            Enhanced field statistics dictionary
        """
        # This is a placeholder for actual LLM processing
        # In a real implementation, you would:
        # 1. Generate prompts for the LLM to analyze patterns
        # 2. Parse the LLM responses and update the field statistics
        
        # For now, we'll just return the original field statistics
        return field_stats
    
    def _discover_relationships(self, metadata_collection: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Discover relationships between files based on metadata.
        
        Args:
            metadata_collection: List of file metadata dictionaries
            
        Returns:
            List of relationship dictionaries
        """
        relationships = []
        
        # Extract file paths for easy reference
        file_paths = [file_info["path"] for file_info in metadata_collection]
        
        # Create a graph to represent relationships
        graph = nx.Graph()
        
        # Add nodes for each file
        for i, file_info in enumerate(metadata_collection):
            graph.add_node(i, path=file_info["path"], name=file_info["name"])
        
        # Find relationships based on common metadata values
        for field in ["microscope.name", "sample.name", "acquisition.date"]:
            field_values = {}
            
            # Group files by field value
            for i, file_info in enumerate(metadata_collection):
                metadata = file_info["metadata"]
                
                # Navigate to nested field
                value = metadata
                for part in field.split("."):
                    if part in value:
                        value = value[part]
                    else:
                        value = None
                        break
                
                if value is not None:
                    if value not in field_values:
                        field_values[value] = []
                    field_values[value].append(i)
            
            # Create relationships for files with common values
            for value, file_indices in field_values.items():
                if len(file_indices) > 1:  # Only create relationship if multiple files share the value
                    for i in range(len(file_indices)):
                        for j in range(i+1, len(file_indices)):
                            idx1 = file_indices[i]
                            idx2 = file_indices[j]
                            
                            # Add edge to graph
                            if graph.has_edge(idx1, idx2):
                                # Increment weight if edge already exists
                                graph[idx1][idx2]["weight"] += 1
                                graph[idx1][idx2]["fields"].append(field)
                            else:
                                # Create new edge
                                graph.add_edge(idx1, idx2, weight=1, fields=[field])
        
        # Extract relationships from graph
        for u, v, data in graph.edges(data=True):
            relationships.append({
                "file1": file_paths[u],
                "file2": file_paths[v],
                "strength": data["weight"],
                "fields": data["fields"]
            })
        
        return relationships
    
    def _discover_hierarchy(self, metadata_collection: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Discover hierarchical organization of the dataset.
        
        Args:
            metadata_collection: List of file metadata dictionaries
            
        Returns:
            Dictionary with hierarchy information
        """
        # Extract relative paths
        paths = [Path(file_info["relative_path"]) for file_info in metadata_collection]
        
        # Build directory tree
        tree = {}
        
        for path in paths:
            current = tree
            for part in path.parts[:-1]:  # Process directories only
                if part not in current:
                    current[part] = {}
                current = current[part]
            
            # Add file to current directory
            filename = path.parts[-1]
            if "__files__" not in current:
                current["__files__"] = []
            current["__files__"].append(str(path))
        
        # Analyze directory structure for patterns
        patterns = self._analyze_directory_patterns(tree)
        
        return {
            "tree": tree,
            "patterns": patterns
        }
    
    def _analyze_directory_patterns(self, tree: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Analyze patterns in directory structure.
        
        Args:
            tree: Directory tree dictionary
            
        Returns:
            List of pattern dictionaries
        """
        patterns = []
        
        # Look for common naming patterns in directories
        all_dirs = self._extract_all_directories(tree)
        
        if len(all_dirs) > 1:
            # Use TF-IDF to find similar directory names
            try:
                vectors = self.vectorizer.fit_transform(all_dirs)
                similarity_matrix = cosine_similarity(vectors)
                
                # Cluster similar directory names
                clustering = DBSCAN(eps=0.5, min_samples=2, metric="precomputed")
                distance_matrix = 1 - similarity_matrix  # Convert similarity to distance
                clusters = clustering.fit_predict(distance_matrix)
                
                # Extract patterns from clusters
                for cluster_id in set(clusters):
                    if cluster_id >= 0:  # Ignore noise points (-1)
                        cluster_dirs = [all_dirs[i] for i in range(len(all_dirs)) if clusters[i] == cluster_id]
                        
                        patterns.append({
                            "type": "directory_naming",
                            "directories": cluster_dirs,
                            "pattern": self._extract_naming_pattern(cluster_dirs)
                        })
            except Exception as e:
                print(f"Error clustering directory names: {e}")
        
        return patterns
    
    def _extract_all_directories(self, tree: Dict[str, Any], prefix: str = "") -> List[str]:
        """
        Extract all directory names from the tree.
        
        Args:
            tree: Directory tree dictionary
            prefix: Current path prefix
            
        Returns:
            List of directory names
        """
        directories = []
        
        for key, value in tree.items():
            if key != "__files__" and isinstance(value, dict):
                path = f"{prefix}/{key}" if prefix else key
                directories.append(path)
                directories.extend(self._extract_all_directories(value, path))
        
        return directories
    
    def _extract_naming_pattern(self, names: List[str]) -> str:
        """
        Extract a naming pattern from a list of similar names.
        
        Args:
            names: List of similar names
            
        Returns:
            String describing the naming pattern
        """
        if not names:
            return ""
            
        # Simple pattern extraction - find common prefix and suffix
        name_parts = [Path(name).parts[-1] for name in names]  # Get last part of path
        
        # Find common prefix
        prefix = os.path.commonprefix(name_parts)
        
        # Find common suffix
        reversed_parts = [part[::-1] for part in name_parts]
        suffix = os.path.commonprefix(reversed_parts)[::-1]
        
        if prefix and suffix and prefix != suffix:
            return f"{prefix}*{suffix}"
        elif prefix:
            return f"{prefix}*"
        elif suffix:
            return f"*{suffix}"
        else:
            return "*"  # No common pattern
    
    def visualize_relationships(self, relationships: List[Dict[str, Any]], 
                              output_file: Optional[Union[str, Path]] = None) -> None:
        """
        Visualize relationships between files as a graph.
        
        Args:
            relationships: List of relationship dictionaries
            output_file: Path to save the visualization (if None, display interactively)
        """
        # Create a graph from relationships
        graph = nx.Graph()
        
        # Add edges from relationships
        for rel in relationships:
            file1 = Path(rel["file1"]).name  # Use filename only for clarity
            file2 = Path(rel["file2"]).name
            weight = rel["strength"]
            
            graph.add_edge(file1, file2, weight=weight)
        
        # Create visualization
        plt.figure(figsize=(12, 10))
        
        # Use spring layout with weights
        pos = nx.spring_layout(graph, weight="weight", seed=42)
        
        # Draw nodes and edges
        nx.draw_networkx_nodes(graph, pos, node_size=500, alpha=0.8)
        nx.draw_networkx_edges(graph, pos, width=[graph[u][v]["weight"] for u, v in graph.edges()], alpha=0.5)
        nx.draw_networkx_labels(graph, pos, font_size=10)
        
        plt.title("File Relationships")
        plt.axis("off")
        
        # Save or display
        if output_file:
            plt.savefig(output_file, bbox_inches="tight")
            print(f"Saved visualization to {output_file}")
        else:
            plt.show()
    
    def generate_schema(self, dataset_structure: Dict[str, Any], 
                      output_file: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """
        Generate a schema from the discovered dataset structure.
        
        Args:
            dataset_structure: Dataset structure dictionary from discover_dataset_structure
            output_file: Path to save the schema (if None, return without saving)
            
        Returns:
            Dictionary with the generated schema
        """
        # Extract patterns from dataset structure
        patterns = dataset_structure["patterns"]
        
        # Create schema based on patterns
        schema = {
            "title": "Electron Microscopy Dataset Schema",
            "description": "Automatically generated schema for electron microscopy dataset",
            "type": "object",
            "properties": {}
        }
        
        # Add properties based on field patterns
        for field, stats in patterns.items():
            field_schema = {
                "description": f"{field} field",
                "coverage": stats["coverage"],
                "examples": stats["example_values"][:3]  # Show up to 3 examples
            }
            
            # Set type based on field statistics
            if stats["type"] == "numeric":
                field_schema["type"] = "number"
            elif stats["type"] == "boolean":
                field_schema["type"] = "boolean"
            else:
                field_schema["type"] = "string"
            
            # Add enum for categorical fields
            if stats["is_categorical"] and len(stats["example_values"]) <= 10:
                field_schema["enum"] = stats["example_values"]
            
            # Add field to schema
            schema["properties"][field] = field_schema
        
        # Add required fields (those with high coverage)
        required_fields = [field for field, stats in patterns.items() 
                         if stats["coverage"] > 0.9]
        
        if required_fields:
            schema["required"] = required_fields
        
        # Save schema if output file specified
        if output_file:
            output_path = Path(output_file)
            
            with open(output_path, "w") as f:
                if output_path.suffix.lower() == ".yaml" or output_path.suffix.lower() == ".yml":
                    yaml.dump(schema, f)
                else:
                    json.dump(schema, f, indent=2)
                    
            print(f"Saved schema to {output_file}")
        
        return schema
    
    def generate_report(self, dataset_structure: Dict[str, Any], 
                       output_file: Union[str, Path]) -> None:
        """
        Generate a comprehensive report about the dataset structure.
        
        Args:
            dataset_structure: Dataset structure dictionary from discover_dataset_structure
            output_file: Path to save the report
        """
        output_path = Path(output_file)
        
        # Extract information from dataset structure
        files = dataset_structure["files"]
        patterns = dataset_structure["patterns"]
        relationships = dataset_structure["relationships"]
        hierarchy = dataset_structure["hierarchy"]
        
        # Generate report content
        report = []
        
        # Add title
        report.append("# Electron Microscopy Dataset Report\n")
        
        # Add summary
        report.append("## Summary\n")
        report.append(f"- Total files: {len(files)}")
        report.append(f"- File formats: {', '.join(set(f['format'] for f in files))}")
        report.append(f"- Relationships discovered: {len(relationships)}\n")
        
        # Add file information
        report.append("## Files\n")
        report.append("| File | Format | Size |")
        report.append("| --- | --- | --- |")
        
        for file_info in files[:20]:  # Show only first 20 files
            file_path = Path(file_info["path"])
            file_size = file_path.stat().st_size / (1024 * 1024)  # Convert to MB
            report.append(f"| {file_info['name']} | {file_info['format']} | {file_size:.2f} MB |")
            
        if len(files) > 20:
            report.append(f"*...and {len(files) - 20} more files*\n")
        else:
            report.append("\n")
        
        # Add metadata patterns
        report.append("## Metadata Patterns\n")
        report.append("| Field | Coverage | Type | Cardinality | Examples |")
        report.append("| --- | --- | --- | --- | --- |")
        
        for field, stats in patterns.items():
            examples = ", ".join(str(v) for v in stats["example_values"][:3])
            report.append(f"| {field} | {stats['coverage']:.2f} | {stats['type']} | {stats['cardinality']} | {examples} |")
            
        report.append("\n")
        
        # Add relationships
        report.append("## Relationships\n")
        
        if relationships:
            report.append("| File 1 | File 2 | Strength | Common Fields |")
            report.append("| --- | --- | --- | --- |")
            
            for rel in relationships[:20]:  # Show only first 20 relationships
                file1 = Path(rel["file1"]).name
                file2 = Path(rel["file2"]).name
                strength = rel["strength"]
                fields = ", ".join(rel["fields"])
                
                report.append(f"| {file1} | {file2} | {strength} | {fields} |")
                
            if len(relationships) > 20:
                report.append(f"*...and {len(relationships) - 20} more relationships*\n")
            else:
                report.append("\n")
        else:
            report.append("No significant relationships discovered.\n")
        
        # Add directory patterns
        report.append("## Directory Structure Patterns\n")
        
        dir_patterns = hierarchy["patterns"]
        if dir_patterns:
            for pattern in dir_patterns:
                report.append(f"### {pattern['type']}\n")
                report.append(f"Pattern: `{pattern['pattern']}`\n")
                report.append("Examples:")
                
                for dir_name in pattern["directories"][:5]:  # Show only first 5 directories
                    report.append(f"- {dir_name}")
                    
                if len(pattern["directories"]) > 5:
                    report.append(f"*...and {len(pattern['directories']) - 5} more*\n")
                else:
                    report.append("\n")
        else:
            report.append("No significant directory patterns discovered.\n")
        
        # Add recommendations
        report.append("## Recommendations\n")
        
        # Generate recommendations based on discovered patterns
        recommendations = []
        
        # Check for missing metadata
        low_coverage_fields = [field for field, stats in patterns.items() 
                             if 0.1 < stats["coverage"] < 0.9]
        
        if low_coverage_fields:
            recommendations.append("- **Improve metadata consistency**: The following fields have inconsistent coverage across files:")
            for field in low_coverage_fields[:5]:  # Show only first 5 fields
                recommendations.append(f"  - `{field}` (coverage: {patterns[field]['coverage']:.2f})")
        
        # Check for potential identifiers
        identifiers = [field for field, stats in patterns.items() 
                     if stats["is_identifier"]]
        
        if identifiers:
            recommendations.append("- **Use consistent identifiers**: The following fields appear to be unique identifiers:")
            for field in identifiers:
                recommendations.append(f"  - `{field}`")
        
        # Add directory structure recommendations
        if not dir_patterns:
            recommendations.append("- **Improve directory organization**: Consider organizing files into a more structured hierarchy based on metadata attributes.")
        
        if recommendations:
            report.extend(recommendations)
        else:
            report.append("No specific recommendations at this time.")
        
        # Write report to file
        with open(output_path, "w") as f:
            f.write("\n".join(report))
            
        print(f"Saved report to {output_file}")


def main():
    """Command line interface for the discovery tool."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Discover self-describing data organization in electron microscopy files')
    parser.add_argument('directory', help='Directory containing electron microscopy files')
    parser.add_argument('--output', '-o', help='Output file for report')
    parser.add_argument('--schema', '-s', help='Output file for generated schema')
    parser.add_argument('--visualize', '-v', help='Output file for relationship visualization')
    parser.add_argument('--recursive', '-r', action='store_true', default=True,
                      help='Search recursively in subdirectories')
    parser.add_argument('--pattern', '-p', default='*',
                      help='File pattern to match (default: *)')
    parser.add_argument('--model', '-m', default='t5-small',
                      help='LLM model to use for metadata analysis')
    
    args = parser.parse_args()
    
    # Create discovery tool
    discovery = EMDataDiscovery(llm_model_name=args.model)
    
    try:
        # Discover dataset structure
        print(f"Analyzing directory: {args.directory}")
        dataset_structure = discovery.discover_dataset_structure(
            args.directory, recursive=args.recursive, file_pattern=args.pattern)
        
        print(f"Found {len(dataset_structure['files'])} files")
        
        # Generate report if output file specified
        if args.output:
            discovery.generate_report(dataset_structure, args.output)
        
        # Generate schema if schema file specified
        if args.schema:
            discovery.generate_schema(dataset_structure, args.schema)
        
        # Visualize relationships if visualization file specified
        if args.visualize:
            discovery.visualize_relationships(dataset_structure["relationships"], args.visualize)
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
        
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
