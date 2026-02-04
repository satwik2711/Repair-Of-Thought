import os
import json
import asyncio
import argparse
from datetime import datetime
import re
from typing import List, Dict, Any, Optional

# Import the evaluation function from the existing code
from eval.auto_eval import evaluate_single_patch

def read_patch_file(file_path: str) -> str:
    """Read the content of a patch file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read().strip()

def extract_bug_name(file_name: str) -> str:
    """Extract the bug name from a patch file name.
    Example: 'Chart-1_1.txt' -> 'Chart-1'
    """
    match = re.match(r"(\w+-\d+)_\d+\.txt", file_name)
    if match:
        return match.group(1)
    return None

async def evaluate_patch(bug_name: str, patch_file: str, api_key: str) -> Dict[str, Any]:
    """
    Evaluate a single patch using the given API key.
    
    Returns:
        Dict with the evaluation result and any error information
    """
    patch_number = os.path.basename(patch_file).split("_")[1].split(".")[0]
    print(f"  Evaluating patch {patch_number} using API key ending with ...{api_key[-4:]}")
    
    # Read the patch content
    patch_content = read_patch_file(patch_file)
    
    try:
        # Call the evaluation function for the patch
        eval_result = await evaluate_single_patch(bug_name, patch_content, api_key)
        
        return {
            "patch_file": os.path.basename(patch_file),
            "patch_content": patch_content,
            "evaluation": eval_result,
            "error": None
        }
        
    except Exception as e:
        print(f"    Error evaluating patch {patch_number} with API key ending with ...{api_key[-4:]}: {str(e)}")
        return {
            "patch_file": os.path.basename(patch_file),
            "patch_content": patch_content,
            "evaluation": {
                "status": "ERROR",
                "error": str(e)
            },
            "error": str(e)
        }

async def evaluate_bug_with_retry(bug_name: str, patch_files: List[str], api_keys: List[str], output_dir: str) -> Dict[str, Any]:
    """
    Evaluate all patches for a single bug, retrying with different API keys on error.
    
    Args:
        bug_name: Name of the bug (e.g., "Chart-1")
        patch_files: List of patch file paths for this bug
        api_keys: List of API keys to try
        output_dir: Directory to store evaluation results
        
    Returns:
        Dictionary with evaluation results
    """
    print(f"\nEvaluating {len(patch_files)} patches for bug: {bug_name}")
    
    # Get dataset for ground truth comparison
    dataset_path = "datasets/defects4j-sf.json"
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    if bug_name not in dataset:
        print(f"  Warning: Bug {bug_name} not found in dataset, skipping")
        return {
            "bug_name": bug_name,
            "evaluation_status": "ERROR",
            "error": f"Bug {bug_name} not found in dataset"
        }
    
    # Get the ground truth patch from the dataset for comparison
    ground_truth = dataset[bug_name].get("fix", "")
    buggy_code = dataset[bug_name].get("buggy", "")
    
    patch_results = []
    patch_statuses = []
    
    # Evaluate each patch
    for patch_file in sorted(patch_files, key=lambda x: int(os.path.basename(x).split("_")[1].split(".")[0])):
        patch_number = os.path.basename(patch_file).split("_")[1].split(".")[0]
        print(f"  Processing patch {patch_number} for {bug_name}")
        
        # Try each API key until success or all keys fail
        result = None
        for api_key in api_keys:
            result = await evaluate_patch(bug_name, patch_file, api_key)
            
            # If there's no error, break the loop
            if result["error"] is None:
                print(f"    Successfully evaluated patch {patch_number} with API key ending with ...{api_key[-4:]}")
                break
            else:
                print(f"    Trying next API key for patch {patch_number}...")
        
        # If we've tried all API keys and still have an error, use the last result
        patch_results.append(result)
        
        # Extract the validation status
        if result["error"] is None:
            patch_status = result["evaluation"].get("patch_validation_status", "UNKNOWN")
        else:
            patch_status = "ERROR"
            
        patch_statuses.append(patch_status)
        print(f"    Patch {patch_number} final status: {patch_status}")
    
    # Determine overall status according to the criteria
    overall_status = "INCORRECT"
    if "CORRECT" in patch_statuses:
        overall_status = "CORRECT"
    elif "PLAUSIBLE" in patch_statuses:
        overall_status = "PLAUSIBLE"
    
    print(f"  Overall status for {bug_name}: {overall_status}")
    
    # Create the result dictionary
    bug_result = {
        "bug_name": bug_name,
        "buggy_code": buggy_code,
        "ground_truth": ground_truth,
        "overall_status": overall_status,
        "patch_results": patch_results,
        "processing_time": datetime.now().isoformat()
    }
    
    # Save the results to a file
    output_file = os.path.join(output_dir, f"{bug_name}_eval.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(bug_result, f, indent=2)
    
    return bug_result

def get_all_bug_names(patch_dir: str) -> List[str]:
    """Get a list of all unique bug names from the patch directory."""
    bug_names = set()
    for file_name in os.listdir(patch_dir):
        bug_name = extract_bug_name(file_name)
        if bug_name:
            bug_names.add(bug_name)
    return sorted(list(bug_names))

def get_patch_files_for_bug(bug_name: str, patch_dir: str) -> List[str]:
    """Get all patch files for a specific bug."""
    patch_files = []
    for file_name in os.listdir(patch_dir):
        if file_name.startswith(f"{bug_name}_") and file_name.endswith(".txt"):
            patch_files.append(os.path.join(patch_dir, file_name))
    return patch_files

async def main(start_idx: int, end_idx: int, api_key1: str, api_key2: str, api_key3: str, 
             patch_dir: str = "correct_patch/defects4j-sf", 
             output_dir: str = "outputs/correct_patch_eval"):
    """
    Main function to evaluate patches within a specified range.
    
    Args:
        start_idx: Starting index for the bugs to evaluate
        end_idx: Ending index for the bugs to evaluate
        api_key1, api_key2, api_key3: GEMINI API keys to use for evaluation
        patch_dir: Directory containing the patch files
        output_dir: Directory to save evaluation results
    """
    # Set up the API keys
    api_keys = [api_key1, api_key2, api_key3]
    print(f"Using {len(api_keys)} API keys for evaluation with retry")
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all bug names from the patch directory
    all_bug_names = get_all_bug_names(patch_dir)
    print(f"Found {len(all_bug_names)} total bugs in the patch directory")
    
    # Apply the index bounds
    if end_idx is None or end_idx > len(all_bug_names):
        end_idx = len(all_bug_names)
    
    bugs_to_evaluate = all_bug_names[start_idx:end_idx]
    print(f"Evaluating bugs {start_idx} to {end_idx-1}: {bugs_to_evaluate}")
    
    # Process the bugs one at a time
    results = []
    for bug_name in bugs_to_evaluate:
        patch_files = get_patch_files_for_bug(bug_name, patch_dir)
        
        if not patch_files:
            print(f"No patch files found for bug {bug_name}, skipping")
            results.append({
                "bug_name": bug_name,
                "evaluation_status": "ERROR",
                "error": "No patch files found"
            })
            continue
        
        # Evaluate this bug with retry logic
        bug_result = await evaluate_bug_with_retry(bug_name, patch_files, api_keys, output_dir)
        results.append(bug_result)
    
    # Generate a summary
    summary = {
        "total_bugs": len(bugs_to_evaluate),
        "correct_count": sum(1 for r in results if r.get("overall_status") == "CORRECT"),
        "plausible_count": sum(1 for r in results if r.get("overall_status") == "PLAUSIBLE"),
        "incorrect_count": sum(1 for r in results if r.get("overall_status") == "INCORRECT"),
        "error_count": sum(1 for r in results if r.get("evaluation_status") == "ERROR"),
        "timestamp": datetime.now().isoformat(),
        "bugs_evaluated": [r.get("bug_name") for r in results]
    }
    
    # Save the summary
    summary_file = os.path.join(output_dir, f"summary_{start_idx}_to_{end_idx-1}.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    print("\nEvaluation Summary:")
    print(f"Total bugs evaluated: {summary['total_bugs']}")
    print(f"CORRECT: {summary['correct_count']}")
    print(f"PLAUSIBLE: {summary['plausible_count']}")
    print(f"INCORRECT: {summary['incorrect_count']}")
    print(f"Errors: {summary['error_count']}")
    print(f"Summary saved to: {summary_file}")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate correct patches from the defects4j-sf dataset")
    parser.add_argument('--start', type=int, default=0, help="Starting index for bugs to evaluate")
    parser.add_argument('--end', type=int, default=None, help="Ending index for bugs to evaluate (exclusive)")
    parser.add_argument('--api-key1', type=str, required=True, help="First GEMINI API key")
    parser.add_argument('--api-key2', type=str, required=True, help="Second GEMINI API key")
    parser.add_argument('--api-key3', type=str, required=True, help="Third GEMINI API key")
    parser.add_argument('--patch-dir', type=str, default="correct_patch/defects4j-sf", 
                       help="Directory containing the patch files")
    parser.add_argument('--output-dir', type=str, default="outputs/correct_patch_eval",
                       help="Directory to store evaluation results")
    
    args = parser.parse_args()
    
    # Run the evaluation
    asyncio.run(main(
        start_idx=args.start,
        end_idx=args.end,
        api_key1=args.api_key1,
        api_key2=args.api_key2,
        api_key3=args.api_key3,
        patch_dir=args.patch_dir,
        output_dir=args.output_dir
    ))