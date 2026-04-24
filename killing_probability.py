#!/usr/bin/env python3
"""
Killing Probability Calculator for DL Mutants Realism Analysis

This script calculates killing probabilities for each test input and each mutant/faulty model based on the execution matrix generated in the previous step. It also saves the confidence intervals.
Developed to process individual bug files like the execution matrix script.
"""

import argparse
import numpy as np
import pandas as pd
import os
import pickle
from tqdm import tqdm
from collections import defaultdict

# =============================================================================
# CONFIGURATION SECTION - ADJUST THESE PARAMETERS
# =============================================================================

# Import bug configuration from execution matrix script
# You can also manually set BUG_LIST here if preferred
try:
    from execution_matrix import BUG_CONFIG
    BUG_LIST = list(BUG_CONFIG.keys())
except ImportError:
    # Fallback: manual configuration
    BUG_LIST = ['cleanml001', 'cleanml002']

# Root directories
RESULTS_ROOT = 'results/'

# Mutant types to process - should match execution matrix runs
MUTANT_TYPES = [
    'pre-training',
    #'post-training/scenario1_mutants', 
    #'post-training/scenario2_mutants'
]

# =============================================================================

def load_execution_matrix(file_path):
    """Load the execution matrix and metadata from pickle file."""
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    return data['execution_matrix'], data['column_metadata'], data.get('bug_id'), data.get('mutant_type')

def calculate_killing_probabilities(execution_matrix, column_metadata):
    """
    Calculate killing probability for each test input and mutant type.
    
    Returns:
    - killing_prob_matrix: Matrix of killing probabilities
    - mutant_types: List of mutant type identifiers
    - confidence_intervals: 90% Wilson confidence intervals for each probability
    """
    num_test_inputs = execution_matrix.shape[0]
    
    # Group columns by mutant type (keeping original name format but removing instance number)
    mutant_type_columns = defaultdict(list)
    mutant_type_info = {}  # Additional metadata about each mutant type
    
    for col_idx, meta in column_metadata.items():
        # Extract filename but remove instance indicator (e.g., "_0.h5", "_1.h5")
        filename = meta['filename']
        # Keep the mutant type prefix (MM_ for mutant models, FM_ for faulty models)
        prefix = "MM_" if meta['type'] == 'mutant_model' else "FM_"
        
        # Create a key for this mutant type (preserving original naming format)
        mutant_type = f"{prefix}{filename}"
        # Remove instance number by splitting at last underscore and joining everything before it
        # This assumes your filenames follow a pattern like "iris_mutant_0.h5", "iris_mutant_1.h5"
        base_name = '_'.join(mutant_type.split('_')[:-1])
        
        mutant_type_columns[base_name].append(col_idx)
        
        # Store additional info about this mutant type
        if base_name not in mutant_type_info:
            mutant_type_info[base_name] = {
                'type': meta['type'],
                'group': meta['group'],
                'base_filename': '_'.join(filename.split('_')[:-1])  # Original filename without instance
            }
    
    # Sort the mutant types for consistent ordering
    mutant_types = sorted(mutant_type_columns.keys())
    num_mutant_types = len(mutant_types)
    
    # Initialize killing probability matrix and confidence intervals
    killing_prob_matrix = np.zeros((num_test_inputs, num_mutant_types))
    confidence_intervals = np.zeros((num_test_inputs, num_mutant_types, 2))  # [lower, upper]
    
    # For each test input and mutant type, calculate killing probability
    for i in tqdm(range(num_test_inputs), desc="Calculating killing probabilities"):
        for j, mutant_type in enumerate(mutant_types):
            # Get columns for this mutant type
            columns = mutant_type_columns[mutant_type]
            n_instances = len(columns)
            
            # Count how many instances this test input kills
            kills = sum(execution_matrix[i, col] for col in columns)
            
            # Calculate killing probability
            prob = kills / n_instances if n_instances > 0 else 0
            killing_prob_matrix[i, j] = prob
            
            # Calculate Wilson confidence interval
            if n_instances > 0:
                # Wilson score interval with z=1.645 for 90% confidence
                z = 1.645
                numerator = prob + z*z/(2*n_instances)
                denominator = 1 + z*z/n_instances
                
                p_tilde = numerator / denominator
                
                c = z * np.sqrt((prob*(1-prob) + z*z/(4*n_instances)) / n_instances) / denominator
                
                lower_bound = max(0, p_tilde - c)
                upper_bound = min(1, p_tilde + c)
                
                confidence_intervals[i, j, 0] = lower_bound
                confidence_intervals[i, j, 1] = upper_bound
            else:
                confidence_intervals[i, j, 0] = 0
                confidence_intervals[i, j, 1] = 0
    
    return killing_prob_matrix, mutant_types, confidence_intervals, mutant_type_info

def process_single_bug_mutant_type(bug_id, mutant_type, results_root):
    """Process killing probabilities for a single bug and mutant type."""
    print(f"\n=== Processing Bug: {bug_id}, Mutant Type: {mutant_type} ===")
    
    # Generate mutant type suffix for filename
    mutant_suffix = mutant_type.replace('/', '_').replace('-', '_')
    
    # Load execution matrix for this bug and mutant type
    execution_matrix_file = os.path.join(results_root, "execution_matrix", 
                                       f"execution_matrix_{bug_id}_{mutant_suffix}.pkl")
    
    if not os.path.exists(execution_matrix_file):
        print(f"Warning: Execution matrix file not found: {execution_matrix_file}")
        return None
        
    execution_matrix, column_metadata, loaded_bug_id, loaded_mutant_type = load_execution_matrix(execution_matrix_file)
    print(f"Loaded execution matrix with shape {execution_matrix.shape}")
    
    # Verify we loaded the correct file
    if loaded_bug_id != bug_id:
        print(f"Warning: Bug ID mismatch. Expected {bug_id}, got {loaded_bug_id}")
    if loaded_mutant_type != mutant_type:
        print(f"Warning: Mutant type mismatch. Expected {mutant_type}, got {loaded_mutant_type}")
    
    # Calculate killing probabilities
    killing_prob_matrix, mutant_types, confidence_intervals, mutant_type_info = calculate_killing_probabilities(
        execution_matrix, column_metadata
    )
    
    return killing_prob_matrix, mutant_types, confidence_intervals, mutant_type_info

def save_bug_results(killing_prob_matrix, mutant_types, confidence_intervals, mutant_type_info, 
                    bug_id, mutant_type, results_root):
    """Save the killing probability matrix and metadata for a single bug and mutant type."""
    # Create killing_probability subdirectory
    output_dir = os.path.join(results_root, "killing_probability")
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate mutant type suffix for filename
    mutant_suffix = mutant_type.replace('/', '_').replace('-', '_')
    
    # Generate output filename with bug ID and mutant type
    output_file = os.path.join(output_dir, f"killing_probability_{bug_id}_{mutant_suffix}.pkl")
    
    # Save as pickle for preserving the full structure
    with open(output_file, 'wb') as f:
        pickle.dump({
            'bug_id': bug_id,
            'mutant_type': mutant_type,
            'killing_prob_matrix': killing_prob_matrix,
            'mutant_types': mutant_types,
            'confidence_intervals': confidence_intervals,
            'mutant_type_info': mutant_type_info
        }, f)
    
    # Create a DataFrame where each cell contains "prob [lower, upper]"
    prob_with_ci = []
    for i in range(killing_prob_matrix.shape[0]):
        row = {}
        for j, mt in enumerate(mutant_types):
            prob = killing_prob_matrix[i, j]
            lower = confidence_intervals[i, j, 0]
            upper = confidence_intervals[i, j, 1]
            row[mt] = f"{prob:.4f} [{lower:.4f}, {upper:.4f}]"
        prob_with_ci.append(row)
    
    # Convert to DataFrame
    df = pd.DataFrame(prob_with_ci)
    
    # Save as CSV
    csv_file = os.path.join(output_dir, f"killing_probability_{bug_id}_{mutant_suffix}.csv")
    df.to_csv(csv_file, index=False)
    
    # Also save a clean version with just the probabilities for easier processing
    prob_df = pd.DataFrame(killing_prob_matrix, columns=mutant_types)
    prob_csv = os.path.join(output_dir, f"killing_probability_{bug_id}_{mutant_suffix}_clean.csv")
    prob_df.to_csv(prob_csv, index=False)
    
    print(f"Bug {bug_id} ({mutant_type}) killing probability matrix saved to {output_file}")
    print(f"Bug {bug_id} ({mutant_type}) CSV with CIs saved to {csv_file}")
    print(f"Bug {bug_id} ({mutant_type}) clean CSV saved to {prob_csv}")
    
    return output_file

def main():
    print(f"Configuration:")
    print(f"  Bug list: {BUG_LIST}")
    print(f"  Results root: {RESULTS_ROOT}")
    print(f"  Mutant types: {MUTANT_TYPES}")
    
    print(f"\nProcessing killing probabilities for all bugs and mutant types...")
    
    # Process each combination of bug and mutant type
    processed_count = 0
    for bug_id in BUG_LIST:
        for mutant_type in MUTANT_TYPES:
            result = process_single_bug_mutant_type(bug_id, mutant_type, RESULTS_ROOT)
            
            if result is not None:
                killing_prob_matrix, mutant_types, confidence_intervals, mutant_type_info = result
                
                # Save individual bug results
                save_bug_results(killing_prob_matrix, mutant_types, confidence_intervals, mutant_type_info,
                               bug_id, mutant_type, RESULTS_ROOT)
                
                # Print summary for this combination
                print(f"Bug {bug_id} ({mutant_type}) - Killing Probability Matrix Shape: {killing_prob_matrix.shape}")
                print(f"Bug {bug_id} ({mutant_type}) - Mutant types: {len(mutant_types)}")
                print(f"Bug {bug_id} ({mutant_type}) - Mean killing probability: {np.mean(killing_prob_matrix):.4f}")
                
                processed_count += 1
            else:
                print(f"Skipping bug {bug_id}, mutant type {mutant_type} due to errors")
    
    print(f"\n=== PROCESSING COMPLETE ===")
    print(f"Successfully processed {processed_count} bug-mutant type combinations")
    print(f"All results saved in: {os.path.join(RESULTS_ROOT, 'killing_probability')}")

if __name__ == "__main__":
    main()