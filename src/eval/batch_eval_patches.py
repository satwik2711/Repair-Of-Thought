import os
import json
import asyncio
import argparse
from tqdm import tqdm
from datetime import datetime

# Import your evaluation function
from eval.auto_eval import evaluate_single_patch

async def evaluate_ground_truth_patches(dataset_path="datasets/defects4j-sf.json", 
                                      results_dir="outputs/ground_truth_eval",
                                      start_idx=0, end_idx=None,
                                      api_key=None):
    """
    Evaluate ground truth patches from the dataset and extract component scores.
    
    Args:
        dataset_path: Path to the dataset JSON
        results_dir: Directory to store evaluation results
        start_idx: Starting index for batch processing
        end_idx: Ending index for batch processing (None for all bugs)
        api_key: Gemini API key
    """
    # Create output directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    # Load dataset
    print(f"Loading dataset from {dataset_path}")
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    # Get list of bug names and apply bounds
    bug_names = sorted(list(dataset.keys()))
    total_bugs = len(bug_names)
    
    if end_idx is None:
        end_idx = total_bugs
    
    print(f"Processing bugs {start_idx} to {end_idx-1} out of {total_bugs} total bugs")
    bug_names = bug_names[start_idx:end_idx]
    
    # Process each bug
    all_results = {}
    
    for bug_idx, bug_name in enumerate(tqdm(bug_names, desc="Processing bugs")):
        print(f"\n[{bug_idx + start_idx + 1}/{end_idx}] Processing {bug_name}")
        
        # Get the ground truth patch (fix)
        ground_truth_patch = dataset[bug_name].get("fix", "")
        if not ground_truth_patch:
            print(f"  No fix found for {bug_name}, skipping")
            continue
            
        # Get the buggy code for reference
        buggy_code = dataset[bug_name].get("buggy", "")
        
        bug_results = {
            "bug_name": bug_name,
            "buggy_code": buggy_code,
            "ground_truth": ground_truth_patch,
            "processing_time": datetime.now().isoformat()
        }
        
        # Evaluate the ground truth patch
        print(f"  Evaluating ground truth patch for {bug_name}")
        try:
            # Call your evaluation function for the ground truth patch
            eval_result = await evaluate_single_patch(bug_name, ground_truth_patch, api_key)
            
            # Extract and store relevant information
            patch_status = eval_result.get("patch_validation_status", "UNKNOWN")
            semantic_equivalence = eval_result.get("semantic_equivalence", {})
            
            # Extract component scores
            reasoning = semantic_equivalence.get("reasoning", "")
            ast_score, symbolic_score, llm_score = extract_validation_scores(reasoning)
            
            bug_results["evaluation"] = {
                "status": patch_status,
                "confidence": semantic_equivalence.get("confidence", 0.0),
                "is_equivalent": semantic_equivalence.get("is_equivalent", False),
                "reasoning": reasoning,
                "component_scores": {
                    "ast": ast_score,
                    "symbolic": symbolic_score,
                    "llm": llm_score
                }
            }
            
            print(f"    Patch status: {patch_status}")
            print(f"    Component scores: AST={ast_score:.2f}, Symbolic={symbolic_score:.2f}, LLM={llm_score:.2f}")
            
        except Exception as e:
            print(f"    Error evaluating patch: {str(e)}")
            bug_results["evaluation"] = {
                "status": "ERROR",
                "error": str(e)
            }
        
        # Save results for this bug
        result_file = os.path.join(results_dir, f"{bug_name}_gt_eval.json")
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(bug_results, f, indent=2)
            
        all_results[bug_name] = bug_results
        
        # Save batch results so far
        batch_file = os.path.join(results_dir, f"batch_{start_idx}_to_{end_idx}.json")
        with open(batch_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2)
            
    print(f"\nCompleted evaluation of {len(bug_names)} bugs")
    
    # Calculate and save summary statistics
    calculate_summary_statistics(all_results, results_dir, f"summary_{start_idx}_to_{end_idx}.json")
    
    return all_results

def extract_validation_scores(reasoning_text):
    """
    Extract the individual validation scores from the reasoning text.
    Returns tuple of (ast_score, symbolic_score, llm_score)
    """
    import re
    
    # Initialize default values
    ast_score = 0.5
    symbolic_score = 0.5
    llm_score = 0.5
    
    # Extract AST score - look for either "structure similarity score" or "token similarity"
    ast_match = re.search(r'structure similarity score: (\d+\.\d+)', reasoning_text)
    if not ast_match:
        ast_match = re.search(r'token similarity: (\d+\.\d+)', reasoning_text)
    if not ast_match:
        ast_match = re.search(r'text similarity: (\d+\.\d+)', reasoning_text)
    if not ast_match:
        ast_match = re.search(r'Overall structure similarity score: (\d+\.\d+)', reasoning_text)
    if ast_match:
        try:
            ast_score = float(ast_match.group(1))
        except ValueError:
            pass
    
    # Extract Symbolic score - look for "Control flow similarity"
    sym_match = re.search(r'Control flow similarity: (\d+\.\d+)', reasoning_text)
    if sym_match:
        try:
            symbolic_score = float(sym_match.group(1))
        except ValueError:
            pass
    
    # Extract LLM score - look for "LLM assessed equivalence"
    llm_match = re.search(r'LLM assessed equivalence: (\d+\.\d+)', reasoning_text)
    if llm_match:
        try:
            llm_score = float(llm_match.group(1))
        except ValueError:
            pass
    
    return ast_score, symbolic_score, llm_score

def calculate_summary_statistics(results, results_dir, summary_filename):
    """
    Calculate summary statistics from the evaluation results.
    """
    summary = {
        "total_bugs": len(results),
        "component_scores": {
            "ast": {
                "min": 1.0, "max": 0.0, "avg": 0.0, "med": 0.0,
                "above_threshold_count": 0
            },
            "symbolic": {
                "min": 1.0, "max": 0.0, "avg": 0.0, "med": 0.0,
                "above_threshold_count": 0
            },
            "llm": {
                "min": 1.0, "max": 0.0, "avg": 0.0, "med": 0.0,
                "above_threshold_count": 0
            }
        },
        "combined_scores": {
            "equal_weights": {
                "min": 1.0, "max": 0.0, "avg": 0.0, "med": 0.0,
                "above_threshold_count": 0
            },
            "ast_only": {
                "above_threshold_count": 0
            },
            "symbolic_only": {
                "above_threshold_count": 0
            },
            "llm_only": {
                "above_threshold_count": 0
            }
        },
        "status_counts": {}
    }
    
    # Collect all scores
    ast_scores = []
    symbolic_scores = []
    llm_scores = []
    equal_weights_scores = []
    status_counts = {}
    
    for bug_name, bug_data in results.items():
        eval_data = bug_data.get("evaluation", {})
        component_scores = eval_data.get("component_scores", {})
        
        status = eval_data.get("status", "UNKNOWN")
        status_counts[status] = status_counts.get(status, 0) + 1
        
        # Get component scores
        ast = component_scores.get("ast", 0.5)
        symbolic = component_scores.get("symbolic", 0.5)
        llm = component_scores.get("llm", 0.5)
        
        # Calculate combined score with equal weights
        equal_weights = (ast + symbolic + llm) / 3
        
        # Add to lists for statistics
        ast_scores.append(ast)
        symbolic_scores.append(symbolic)
        llm_scores.append(llm)
        equal_weights_scores.append(equal_weights)
        
        # Count scores above threshold
        if ast >= 0.5:
            summary["component_scores"]["ast"]["above_threshold_count"] += 1
            summary["combined_scores"]["ast_only"]["above_threshold_count"] += 1
            
        if symbolic >= 0.5:
            summary["component_scores"]["symbolic"]["above_threshold_count"] += 1
            summary["combined_scores"]["symbolic_only"]["above_threshold_count"] += 1
            
        if llm >= 0.5:
            summary["component_scores"]["llm"]["above_threshold_count"] += 1
            summary["combined_scores"]["llm_only"]["above_threshold_count"] += 1
            
        if equal_weights >= 0.5:
            summary["combined_scores"]["equal_weights"]["above_threshold_count"] += 1
    
    # Calculate statistics if we have data
    if ast_scores:
        summary["component_scores"]["ast"]["min"] = min(ast_scores)
        summary["component_scores"]["ast"]["max"] = max(ast_scores)
        summary["component_scores"]["ast"]["avg"] = sum(ast_scores) / len(ast_scores)
        summary["component_scores"]["ast"]["med"] = sorted(ast_scores)[len(ast_scores) // 2]
        
    if symbolic_scores:
        summary["component_scores"]["symbolic"]["min"] = min(symbolic_scores)
        summary["component_scores"]["symbolic"]["max"] = max(symbolic_scores)
        summary["component_scores"]["symbolic"]["avg"] = sum(symbolic_scores) / len(symbolic_scores)
        summary["component_scores"]["symbolic"]["med"] = sorted(symbolic_scores)[len(symbolic_scores) // 2]
        
    if llm_scores:
        summary["component_scores"]["llm"]["min"] = min(llm_scores)
        summary["component_scores"]["llm"]["max"] = max(llm_scores)
        summary["component_scores"]["llm"]["avg"] = sum(llm_scores) / len(llm_scores)
        summary["component_scores"]["llm"]["med"] = sorted(llm_scores)[len(llm_scores) // 2]
        
    if equal_weights_scores:
        summary["combined_scores"]["equal_weights"]["min"] = min(equal_weights_scores)
        summary["combined_scores"]["equal_weights"]["max"] = max(equal_weights_scores)
        summary["combined_scores"]["equal_weights"]["avg"] = sum(equal_weights_scores) / len(equal_weights_scores)
        summary["combined_scores"]["equal_weights"]["med"] = sorted(equal_weights_scores)[len(equal_weights_scores) // 2]
    
    summary["status_counts"] = status_counts
    
    # Save summary
    summary_path = os.path.join(results_dir, summary_filename)
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    print("\nSummary Statistics:")
    print(f"Total bugs evaluated: {summary['total_bugs']}")
    
    for component, data in summary["component_scores"].items():
        print(f"{component.upper()} scores: Min={data['min']:.2f}, Max={data['max']:.2f}, Avg={data['avg']:.2f}, Above 0.5={data['above_threshold_count']}")
    
    print("\nCombined scores with equal weights:")
    data = summary["combined_scores"]["equal_weights"]
    print(f"Min={data['min']:.2f}, Max={data['max']:.2f}, Avg={data['avg']:.2f}, Above 0.5={data['above_threshold_count']}")
    
    print("\nIndividual component threshold counts:")
    print(f"AST only: {summary['combined_scores']['ast_only']['above_threshold_count']}")
    print(f"Symbolic only: {summary['combined_scores']['symbolic_only']['above_threshold_count']}")
    print(f"LLM only: {summary['combined_scores']['llm_only']['above_threshold_count']}")
    
    print("\nStatus counts:")
    for status, count in summary["status_counts"].items():
        print(f"  {status}: {count}")
    
    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ground truth patches from dataset")
    parser.add_argument('--dataset', type=str, default="datasets/defects4j-sf.json", 
                       help="Path to dataset JSON")
    parser.add_argument('--results-dir', type=str, default="outputs/ground_truth_eval",
                       help="Directory to store evaluation results")
    parser.add_argument('--start', type=int, default=0,
                       help="Starting index for batch processing")
    parser.add_argument('--end', type=int, default=None,
                       help="Ending index for batch processing (None for all bugs)")
    parser.add_argument('--api-key', type=str, required=True,
                       help="Gemini API key")
    
    args = parser.parse_args()
    
    # Run the evaluation
    asyncio.run(evaluate_ground_truth_patches(
        dataset_path=args.dataset,
        results_dir=args.results_dir,
        start_idx=args.start,
        end_idx=args.end,
        api_key=args.api_key
    ))