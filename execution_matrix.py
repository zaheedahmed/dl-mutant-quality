#!/usr/bin/env python3
"""
Execution Matrix Generator - Classification Tasks Only
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import os
import gc
from tqdm import tqdm
import pickle
tf.get_logger().setLevel('ERROR')

# =============================================================================
# CONFIGURATION SECTION
# =============================================================================
'''
# Bug-level configuration with mixed task types
BUG_CONFIG = {
    'cleanml002': 'auto',              # Auto-detect task type
    'deepfd010': 'classification',     # Known single-label classification
    'deeplocalize015': 'multi-label',  # Known multi-label classification
    'defect4ml044': 'auto',            # Auto-detect task type
}
'''
#============ CleanML Dataset ============
'''
BUG_CONFIG = {
    'cleanml001': 'classification',
    'cleanml002': 'classification',
    'cleanml003': 'classification',
    'cleanml004': 'classification', 
    'cleanml005': 'classification', 
    'cleanml006': 'classification', 
    'cleanml007': 'classification',
    'cleanml008': 'classification', 
    'cleanml009': 'classification',
    'cleanml010': 'classification', 
    'cleanml011': 'classification',
    'cleanml012': 'classification',
    'cleanml013': 'classification', 
    'cleanml014': 'classification', 
    'cleanml015': 'classification',
    'cleanml016': 'classification',
    'cleanml017': 'classification', 
    'cleanml018': 'classification',
    'cleanml019': 'classification'
}
'''
#============ DeepFD Dataset ============
'''
BUG_CONFIG = {
    'deepfd002': 'classification',
    'deepfd006': 'multi-label',
    'deepfd008': 'classification',
    'deepfd010': 'classification',
    'deepfd011': 'classification',
    'deepfd014': 'classification',
    'deepfd015': 'classification',
    'deepfd016': 'classification',
    'deepfd021': 'classification',
    'deepfd022': 'classification',
    'deepfd023': 'classification',
    'deepfd024': 'classification',
    'deepfd026': 'classification',
    'deepfd027': 'classification',
    'deepfd028': 'classification',
    'deepfd029': 'classification',
    'deepfd030': 'classification', 
    'deepfd031': 'multi-label',
    'deepfd034': 'classification',
    'deepfd035': 'classification',
    'deepfd036': 'classification',
    'deepfd037': 'classification',
    'deepfd047': 'classification',
    'deepfd048': 'classification',
    'deepfd050': 'classification'
}
'''
#============ DeepLocalize Dataset ============
'''
BUG_CONFIG = {
    'deeplocalize002': 'classification',
    'deeplocalize006': 'multi-label',
    'deeplocalize008': 'classification', 
    'deeplocalize010': 'classification', 
    'deeplocalize011': 'classification',
    'deeplocalize015': 'classification',
    'deeplocalize021': 'classification',
    'deeplocalize024': 'classification',
    'deeplocalize026': 'classification',
    'deeplocalize027': 'classification',
    'deeplocalize028': 'classification',
    'deeplocalize030': 'classification',
    'deeplocalize031': 'classification', 
    'deeplocalize034': 'classification',
    'deeplocalize035': 'classification',
    'deeplocalize038': 'classification',
    'deeplocalize039': 'classification',
    'deeplocalize040': 'classification'
}
'''
#============ defect4ML Dataset ============

BUG_CONFIG = {
    'defect4ml010': 'classification',
    'defect4ml011': 'classification',
    'defect4ml026': 'classification',
    'defect4ml038': 'classification',
    'defect4ml041': 'classification',
    'defect4ml044': 'classification',
    'defect4ml047': 'classification', 
    'defect4ml048': 'classification',
    'defect4ml050': 'classification',
    'defect4ml051': 'classification', 
    'defect4ml052': 'classification',
    'defect4ml053': 'classification',
    'defect4ml054': 'classification',
    'defect4ml061': 'classification',
    'defect4ml067': 'multi-label',
    'defect4ml068': 'classification',
    'defect4ml069': 'classification',
    'defect4ml074': 'classification',
    'defect4ml075': 'classification',
    'defect4ml088': 'classification',
    'defect4ml091': 'classification',
    'defect4ml095': 'classification',
    'defect4ml098': 'classification',
    'defect4ml100': 'classification' 
}

MODELS_ROOT = 'models/'
RESULTS_ROOT = 'results/'
MULTILABEL_THRESHOLD = 0.5

BATCH_SIZE = 1024

MUTANT_TYPE = 'pre-training'
#MUTANT_TYPE = 'post-training/scenario1_mutants'
#MUTANT_TYPE = 'post-training/scenario2_mutants'

# =============================================================================

# Global cache for detected task types
TASK_TYPE_CACHE = {}

def detect_task_type_from_labels(y_data, bug_id):
    """
    Detect task type by analyzing the full label dataset structure.
    Returns 'classification' for single-label or 'multi-label'
    Uses caching to avoid re-detection.
    """
    if bug_id in TASK_TYPE_CACHE:
        return TASK_TYPE_CACHE[bug_id]
    
    detected_type = 'classification'  # Default
    
    if isinstance(y_data, np.ndarray):
        if y_data.ndim == 1:
            # 1D array - single-label classification
            detected_type = 'classification'
        elif y_data.ndim == 2:
            # 2D array - check if one-hot or multi-label
            if y_data.shape[1] == 1:
                # Single column - single-label
                detected_type = 'classification'
            else:
                # Multiple columns - check row sums
                row_sums = np.sum(y_data, axis=1)
                if np.all(row_sums == 1):
                    detected_type = 'classification'  # One-hot encoded
                else:
                    detected_type = 'multi-label'   # Multiple labels per sample
    
    # Cache the result
    TASK_TYPE_CACHE[bug_id] = detected_type
    print(f"Auto-detected task type: {detected_type}")
    
    return detected_type

def get_task_type(bug_id, y_data):
    """
    Get task type from config or auto-detect if 'auto' is specified.
    """
    config_value = BUG_CONFIG[bug_id]
    
    if config_value == 'auto':
        return detect_task_type_from_labels(y_data, bug_id)
    else:
        # Use explicit configuration
        TASK_TYPE_CACHE[bug_id] = config_value
        print(f"Using configured task type: {config_value}")
        return config_value

def validate_model_output(output, model_type, model_name):
    """
    Validate model output for NaN/Inf and raise exception if found.
    """
    if np.any(np.isnan(output)):
        raise ValueError(f"NaN detected in {model_type} model output: {model_name}")
    if np.any(np.isinf(output)):
        raise ValueError(f"Inf detected in {model_type} model output: {model_name}")

def is_kill_classification(original_output, mutant_output, expected_output, task_type, multilabel_threshold=0.5):
    """
    Classification kill detection for both single-label and multi-label.
    Returns 1 if original correct and mutant wrong, 0 otherwise.
    """
    if task_type == 'multi-label':
        # Multi-label classification
        # Convert probabilities to binary predictions using threshold
        if isinstance(original_output, np.ndarray) and len(original_output.shape) > 0:
            original_pred = (original_output >= multilabel_threshold).astype(int)
        else:
            original_pred = int(original_output >= multilabel_threshold)

        if isinstance(mutant_output, np.ndarray) and len(mutant_output.shape) > 0:
            mutant_pred = (mutant_output >= multilabel_threshold).astype(int)
        else:
            mutant_pred = int(mutant_output >= multilabel_threshold)

        # Ensure expected_output is in the right format
        if isinstance(expected_output, np.ndarray):
            expected_binary = expected_output.astype(int)
        else:
            expected_binary = int(expected_output)

        # Check if predictions match expected output exactly
        if isinstance(expected_binary, np.ndarray):
            original_correct = np.array_equal(original_pred, expected_binary)
            mutant_correct = np.array_equal(mutant_pred, expected_binary)
        else:
            original_correct = (original_pred == expected_binary)
            mutant_correct = (mutant_pred == expected_binary)

        # Input contributes to killing if original is correct but mutant is wrong
        return 1 if (original_correct and not mutant_correct) else 0

    else:
        # Single-label classification
        if isinstance(expected_output, np.ndarray) and expected_output.size > 1:
            expected_class = np.argmax(expected_output)
        else:
            expected_class = int(expected_output)

        # Get predicted class from model outputs
        if isinstance(original_output, np.ndarray) and original_output.size > 1:
            original_class = np.argmax(original_output)
        else:
            original_val = float(np.ravel(original_output)[0])
            original_class = int(original_val >= 0.5)

        if isinstance(mutant_output, np.ndarray) and mutant_output.size > 1:
            mutant_class = np.argmax(mutant_output)
        else:
            mutant_val = float(np.ravel(mutant_output)[0])
            mutant_class = int(mutant_val >= 0.5)

        # Input contributes to killing if original model is correct but mutant is wrong
        return 1 if (original_class == expected_class and mutant_class != expected_class) else 0

def load_model_safely(model_path):
    """Load model with proper error handling."""
    try:
        return keras.models.load_model(model_path)
    except Exception as e:
        raise ValueError(f"Failed to load model from {model_path}: {str(e)}")

def process_bug_sequential(bug_id):
    """
    Process bug using sequential model loading to avoid OOM.
    """
    print(f"\n=== Processing Bug: {bug_id} ===")
    
    bug_path = os.path.join(MODELS_ROOT, bug_id)
    
    # Load test data
    test_data_path = os.path.join(bug_path, f"test_data_{bug_id}.npz")
    data = np.load(test_data_path)
    X, y = data['X'], data['y']
    print(f"Loaded {len(X)} test inputs")
    
    # Get task type from config or auto-detect
    task_type = get_task_type(bug_id, y)
    
    # Get model paths
    original_path = os.path.join(bug_path, 'original')
    faulty_path = os.path.join(bug_path, 'faulty') 
    mutants_path = os.path.join(bug_path, 'mutants', MUTANT_TYPE)
    
    # Count models for matrix dimensions
    original_files = sorted([f for f in os.listdir(original_path) if f.endswith('.h5')])
    faulty_files = sorted([f for f in os.listdir(faulty_path) if f.endswith('.h5')])
    mutant_dirs = sorted([d for d in os.listdir(mutants_path) 
                         if os.path.isdir(os.path.join(mutants_path, d))])
    
    n_instances = len(original_files)
    total_cols = len(mutant_dirs) * n_instances + n_instances  # mutants + faulty
    
    print(f"Matrix dimensions: {len(X)} x {total_cols}")
    print(f"  {len(mutant_dirs)} mutant models x {n_instances} instances = {len(mutant_dirs) * n_instances}")
    print(f"  1 faulty model x {n_instances} instances = {1 * n_instances}")
    
    execution_matrix = np.zeros((len(X), total_cols), dtype=int)
    column_metadata = {}
    
    # Step 1: Precompute original predictions
    print("\nComputing original predictions...")
    original_preds = []
    
    for i, model_file in enumerate(original_files):
        print(f"  Original model {i+1}/{len(original_files)}: {model_file}")
        tf.keras.backend.clear_session()
        gc.collect()
        
        model = load_model_safely(os.path.join(original_path, model_file))
        pred = model.predict(X, batch_size=BATCH_SIZE, verbose=0)
        
        # Validate output - throw exception if NaN/Inf found
        validate_model_output(pred, "original", model_file)
        
        original_preds.append(pred)
        del model
    
    # Step 2: Process mutants sequentially
    print(f"\nProcessing {len(mutant_dirs)} mutants sequentially...")
    col_idx = 0
    
    for j, mutant_dir in enumerate(tqdm(mutant_dirs, desc="Mutants")):
        mutant_path = os.path.join(mutants_path, mutant_dir)
        mutant_files = sorted([f for f in os.listdir(mutant_path) if f.endswith('.h5')])
        
        for k, model_file in enumerate(mutant_files):
            tf.keras.backend.clear_session()
            gc.collect()
            
            # Load single mutant model
            model = load_model_safely(os.path.join(mutant_path, model_file))
            mutant_pred = model.predict(X, batch_size=BATCH_SIZE, verbose=0)
            
            # Validate output - throw exception if NaN/Inf found
            validate_model_output(mutant_pred, "mutant", f"{mutant_dir}/{model_file}")
            
            # Store metadata
            column_metadata[col_idx] = {
                'type': 'mutant_model',
                'group': j,
                'instance': k,
                'filename': model_file,
                'mutant_name': mutant_dir,
                'bug_id': bug_id
            }
            
            # Compute kills for this mutant
            for i in range(len(X)):
                execution_matrix[i, col_idx] = is_kill_classification(
                    original_preds[k][i], mutant_pred[i], y[i], task_type, MULTILABEL_THRESHOLD
                )
            
            col_idx += 1
            del model
    
    # Step 3: Process faulty model's instances
    print("\nProcessing faulty model's instances...")
    
    for k, model_file in enumerate(faulty_files):
        print(f"  Faulty model {k+1}/{len(faulty_files)}: {model_file}")
        tf.keras.backend.clear_session()
        gc.collect()
        
        model = load_model_safely(os.path.join(faulty_path, model_file))
        faulty_pred = model.predict(X, batch_size=BATCH_SIZE, verbose=0)
        
        # Validate output - throw exception if NaN/Inf found
        validate_model_output(faulty_pred, "faulty", model_file)
        
        # Store metadata
        column_metadata[col_idx] = {
            'type': 'faulty_model',
            'group': 0,
            'instance': k,
            'filename': model_file,
            'bug_id': bug_id
        }
        
        # Compute kills for this faulty model
        for i in range(len(X)):
            execution_matrix[i, col_idx] = is_kill_classification(
                original_preds[k][i], faulty_pred[i], y[i], task_type, MULTILABEL_THRESHOLD
            )
        
        col_idx += 1
        del model
    
    return execution_matrix, column_metadata

def save_results(execution_matrix, column_metadata, bug_id):
    """Save execution matrix and metadata."""
    output_dir = os.path.join(RESULTS_ROOT, "execution_matrix")
    os.makedirs(output_dir, exist_ok=True)

    # Generate mutant type suffix for filename (replace both '/' and '-')
    mutant_suffix = MUTANT_TYPE.replace('/', '_').replace('-', '_')
    
    # Generate output filename with bug ID and mutant type and save as pickle
    pickle_file = os.path.join(output_dir, f"execution_matrix_{bug_id}_{mutant_suffix}.pkl")
    with open(pickle_file, 'wb') as f:
        pickle.dump({
            'bug_id': bug_id,
            'mutant_type': MUTANT_TYPE,
            'execution_matrix': execution_matrix,
            'column_metadata': column_metadata
        }, f)
    
    # Create a DataFrame with informative column names for the CSV
    column_names = []
    for col_idx in range(execution_matrix.shape[1]):
        meta = column_metadata[col_idx]
        if meta['type'] == 'mutant_model':
            col_name = f"MM_{meta['mutant_name']}_{meta['instance']}"
        else: # faulty_model
            col_name = f"FM_{meta['filename']}"
        column_names.append(col_name)

    # Convert to DataFrame with proper column names
    csv_df = pd.DataFrame(execution_matrix, columns=column_names)
    
    # Save as CSV for inspection
    csv_file = os.path.join(output_dir, f"execution_matrix_{bug_id}_{mutant_suffix}.csv")
    csv_df.to_csv(csv_file, index=False)
    
    print(f"Results saved:")
    print(f"  Pickle: {pickle_file}")
    print(f"  CSV: {csv_file}")

def main():
    """Main execution function."""
    print("Execution Matrix Configuration:")
    print(f"  Bugs: {list(BUG_CONFIG.keys())}")
    print(f"  Models root: {MODELS_ROOT}")
    print(f"  Results root: {RESULTS_ROOT}")
    print(f"  Mutant type: {MUTANT_TYPE}")
    print(f"  Multi-label threshold: {MULTILABEL_THRESHOLD}")
    
    print(f"\nIMPORTANT: Ensure you have run model_validator.py first!")
    
    for bug_id in BUG_CONFIG.keys():
        try:
            execution_matrix, column_metadata = process_bug_sequential(bug_id)
            
            # Save results
            save_results(execution_matrix, column_metadata, bug_id)
            
            # Print summary
            print(f"\nSummary for {bug_id}:")
            print(f"  Task type: {TASK_TYPE_CACHE.get(bug_id, 'unknown')}")
            print(f"  Matrix shape: {execution_matrix.shape}")
            print(f"  Non-zero elements: {np.count_nonzero(execution_matrix)}")
            print(f"  Kill percentage: {np.mean(execution_matrix) * 100:.2f}%")
            
        except Exception as e:
            print(f"ERROR processing {bug_id}: {e}")
            print(f"This indicates a problem with model filtering or data quality.")
            raise  # Re-raise to stop execution
    
    print("\n=== PROCESSING COMPLETE ===")
    print(f"Mutant type processed: {MUTANT_TYPE}")

if __name__ == "__main__":
    main()