import os
import json
import subprocess
import concurrent.futures
import asyncio
import time
import random
from datetime import datetime
from sf_auto_eval import evaluate_patches
import pandas as pd

def process_bug(bug_name, patch_num, sample_size=1, retry_count=3, base_delay=5):
    """
    Generate solutions and patches for a single bug, then run auto_eval on the patch file.
    Includes retry logic with exponential backoff to handle rate limiting.
    
    Args:
        bug_name (str): Name of the bug (e.g. "Math-2")
        patch_num (int): Number of patches to generate
        sample_size (int): Number of solution samples to generate
        retry_count (int): Maximum number of retries for each step
        base_delay (int): Base delay in seconds before retrying (will grow exponentially)
    
    Returns:
        dict: Contains processing info, file paths, and evaluation results
    """
    sol_dir = "outputs/sol"
    patches_dir = "outputs/patches"
    os.makedirs(sol_dir, exist_ok=True)
    os.makedirs(patches_dir, exist_ok=True)
    
    start_time = datetime.now()
    solution_file = os.path.join(sol_dir, f"{bug_name}.json")
    extracted_solution_file = os.path.join(sol_dir, f"{bug_name}_extracted.json")
    patch_file = os.path.join(patches_dir, f"{bug_name}_patch.json")
    
    bug_result = {
        "bug_name": bug_name,
        "processing_start_time": start_time.isoformat(),
        "solution_file": solution_file,
        "extracted_solution_file": extracted_solution_file,
        "patch_file": patch_file,
        "solution_generation": {
            "status": None,
            "attempt_count": 0,
            "error": None,
            "start_time": None,
            "end_time": None
        },
        "patch_generation": {
            "status": None,
            "attempt_count": 0,
            "error": None,
            "start_time": None,
            "end_time": None
        },
        "evaluation": {
            "status": None,
            "attempt_count": 0,
            "error": None,
            "start_time": None,
            "end_time": None,
            "results": None
        }
    }

    # 1. Generate Solutions
    bug_result["solution_generation"]["start_time"] = datetime.now().isoformat()
    print(f"[INFO] Generating solutions for bug: {bug_name}")
    
    for attempt in range(1, retry_count + 1):
        bug_result["solution_generation"]["attempt_count"] = attempt
        try:
            cmd_solution = (
                f"python src/sf_gen_solution_reasoned.py "
                f"-d datasets/defects4j-sf.json "
                f"-o {solution_file} "
                f"-s {sample_size} "
                f"-bug {bug_name} "
                f"-patch_num {patch_num}"
            )
            subprocess.run(cmd_solution, shell=True, check=True)
            bug_result["solution_generation"]["status"] = "Success"
            print(f"[SUCCESS] Generated solutions for {bug_name} on attempt {attempt}")
            break
        except subprocess.CalledProcessError as e:
            bug_result["solution_generation"]["error"] = f"Error: {str(e)}"
            print(f"[ERROR] Solution generation failed for {bug_name} on attempt {attempt}/{retry_count}")
            
            if attempt < retry_count:
                # Exponential backoff with jitter
                delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 2)
                print(f"[INFO] Retrying in {delay:.2f} seconds...")
                time.sleep(delay)
    
    bug_result["solution_generation"]["end_time"] = datetime.now().isoformat()
    
    # Only proceed if solution generation was successful
    if bug_result["solution_generation"]["status"] == "Success":
        # 2. Generate Patches
        time.sleep(base_delay)  # Add delay between steps to avoid rate limiting
        bug_result["patch_generation"]["start_time"] = datetime.now().isoformat()
        print(f"[INFO] Generating patches for bug: {bug_name}")
        
        for attempt in range(1, retry_count + 1):
            bug_result["patch_generation"]["attempt_count"] = attempt
            try:
                cmd_patch = (
                    f"python src/sf_gen_patch.py "
                    f"-d datasets/defects4j-sf.json "
                    f"-s {extracted_solution_file} "
                    f"-o {patch_file} "
                    f"-bug {bug_name}"
                )
                subprocess.run(cmd_patch, shell=True, check=True)
                bug_result["patch_generation"]["status"] = "Success"
                print(f"[SUCCESS] Generated patches for {bug_name} on attempt {attempt}")
                break
            except subprocess.CalledProcessError as e:
                bug_result["patch_generation"]["error"] = f"Error: {str(e)}"
                print(f"[ERROR] Patch generation failed for {bug_name} on attempt {attempt}/{retry_count}")
                
                if attempt < retry_count:
                    delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 2)
                    print(f"[INFO] Retrying in {delay:.2f} seconds...")
                    time.sleep(delay)
        
        bug_result["patch_generation"]["end_time"] = datetime.now().isoformat()
        
        # 3. Evaluate Patches
        if bug_result["patch_generation"]["status"] == "Success" and os.path.exists(patch_file):
            time.sleep(base_delay)  # Add delay before evaluation
            bug_result["evaluation"]["start_time"] = datetime.now().isoformat()
            print(f"[INFO] Evaluating patches for bug: {bug_name}")
            
            try:
                for attempt in range(1, retry_count + 1):
                    bug_result["evaluation"]["attempt_count"] = attempt
                    try:
                        api_key = os.getenv("GEMINI_API_KEY")
                        auto_eval_dict = asyncio.run(evaluate_patches(bug_name, patch_file, api_key))
                        bug_result["evaluation"]["results"] = auto_eval_dict.get(bug_name, [])
                        bug_result["evaluation"]["status"] = "Success"
                        print(f"[SUCCESS] Evaluated patches for {bug_name} on attempt {attempt}")
                        break
                    except Exception as e:
                        bug_result["evaluation"]["error"] = f"Error: {str(e)}"
                        print(f"[ERROR] Evaluation failed for {bug_name} on attempt {attempt}/{retry_count}: {e}")
                        
                        if attempt < retry_count:
                            delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 2)
                            print(f"[INFO] Retrying evaluation in {delay:.2f} seconds...")
                            time.sleep(delay)
            except Exception as e:
                bug_result["evaluation"]["error"] = f"Error during evaluation: {str(e)}"
                print(f"[ERROR] Evaluation completely failed for {bug_name}: {e}")
            
            bug_result["evaluation"]["end_time"] = datetime.now().isoformat()
        else:
            bug_result["evaluation"]["status"] = "Skipped"
            bug_result["evaluation"]["error"] = "No patch file found or patch generation failed"
    
    bug_result["processing_end_time"] = datetime.now().isoformat()
    
    # Calculate processing duration
    start = datetime.fromisoformat(bug_result["processing_start_time"])
    end = datetime.fromisoformat(bug_result["processing_end_time"])
    bug_result["processing_duration_seconds"] = (end - start).total_seconds()
    
    return bug_result


def generate_dataset_results(patch_num, bug_names, chunk_size=1, delay_between_bugs=10, results_dir="results"):
    """
    Process a list of bug names sequentially or in small batches, with configurable delays
    between processing to avoid rate limiting.
    
    Args:
        patch_num (int): Number of patches to generate
        bug_names (list[str]): List of bug names (e.g. ["Math-2", "Csv-3", "Chart-5"])
        chunk_size (int): Number of bugs to process concurrently in one batch (default=1)
        delay_between_bugs (int): Delay in seconds between processing each bug (default=10)
        results_dir (str): Directory to store results
    
    Returns:
        dict: The final results dictionary with summary metrics
    """
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    # Time-based identifiers for this run
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results_summary = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "configuration": {
            "patch_num": patch_num,
            "chunk_size": chunk_size,
            "delay_between_bugs": delay_between_bugs,
            "total_bugs": len(bug_names)
        },
        "bugs": [],
        "summary_metrics": {
            "total_bugs": len(bug_names),
            "completed_bugs": 0,
            "solution_success_count": 0,
            "patch_success_count": 0,
            "evaluation_success_count": 0,
            "patch_validation_counts": {
                "CORRECT": 0,
                "PLAUSIBLE": 0,
                "INCORRECT": 0
            }
        }
    }

    # Directory structure for this specific run
    run_dir = os.path.join(results_dir, f"run_{run_id}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Individual bugs will also have their own result files
    bugs_dir = os.path.join(run_dir, "bugs")
    os.makedirs(bugs_dir, exist_ok=True)
    
    # Progress tracking
    total_bugs = len(bug_names)
    bug_chunks = [bug_names[i:i+chunk_size] for i in range(0, total_bugs, chunk_size)]
    
    print(f"\n[INFO] Starting processing run {run_id} with {total_bugs} bugs in {len(bug_chunks)} chunks")
    print(f"[CONFIG] chunk_size={chunk_size}, delay_between_bugs={delay_between_bugs}, patch_num={patch_num}")
    
    # Process each chunk
    for chunk_idx, chunk in enumerate(bug_chunks, 1):
        print(f"\n[INFO] Processing chunk {chunk_idx}/{len(bug_chunks)}: {chunk}")
        
        # Process bugs in this chunk concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=chunk_size) as executor:
            future_to_bug = {
                executor.submit(process_bug, bug_name, patch_num): bug_name
                for bug_name in chunk
            }
            
            for future in concurrent.futures.as_completed(future_to_bug):
                bug_name = future_to_bug[future]
                try:
                    bug_result = future.result()
                    
                    # Save individual bug result
                    bug_file = os.path.join(bugs_dir, f"{bug_name}.json")
                    with open(bug_file, "w", encoding="utf-8") as f:
                        json.dump(bug_result, f, indent=2)
                    
                    # Add to results list
                    results_summary["bugs"].append(bug_result)
                    
                    # Update metrics
                    results_summary["summary_metrics"]["completed_bugs"] += 1
                    
                    if bug_result["solution_generation"]["status"] == "Success":
                        results_summary["summary_metrics"]["solution_success_count"] += 1
                        
                    if bug_result["patch_generation"]["status"] == "Success":
                        results_summary["summary_metrics"]["patch_success_count"] += 1
                        
                    if bug_result["evaluation"]["status"] == "Success":
                        results_summary["summary_metrics"]["evaluation_success_count"] += 1
                        
                        # Count validation statuses
                        for eval_result in bug_result["evaluation"]["results"]:
                            status = eval_result.get("patch_validation_status")
                            if status in ["CORRECT", "PLAUSIBLE", "INCORRECT"]:
                                results_summary["summary_metrics"]["patch_validation_counts"][status] += 1
                    
                    print(f"[SUCCESS] Completed processing for {bug_name}")
                    
                except Exception as e:
                    print(f"[ERROR] Failed to process bug {bug_name}: {e}")
                    error_result = {
                        "bug_name": bug_name,
                        "error": f"Failed to process: {str(e)}",
                        "processing_start_time": datetime.now().isoformat(),
                        "processing_end_time": datetime.now().isoformat(),
                    }
                    results_summary["bugs"].append(error_result)
        
        # Save intermediate results after each chunk, but only as a JSON file
        intermediate_file = os.path.join(run_dir, f"intermediate_results_chunk_{chunk_idx}.json")
        with open(intermediate_file, "w", encoding="utf-8") as f:
            json.dump(results_summary, f, indent=2)
        
        # Delay before next chunk (unless it's the last chunk)
        if chunk_idx < len(bug_chunks):
            delay = delay_between_bugs * chunk_size
            print(f"[INFO] Completed chunk {chunk_idx}/{len(bug_chunks)}. Waiting {delay} seconds before next chunk...")
            time.sleep(delay)
    
    # Final results
    final_file = os.path.join(run_dir, "final_results.json")
    with open(final_file, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)
    
    # Generate only one summary report at the end of the run
    final_report = os.path.join(run_dir, "summary_report.html")
    generate_summary_report(results_summary, final_report)
    
    print(f"\n[INFO] Run {run_id} completed!")
    print(f"[SUMMARY] Total bugs: {total_bugs}")
    print(f"[SUMMARY] Successful solutions: {results_summary['summary_metrics']['solution_success_count']}")
    print(f"[SUMMARY] Successful patches: {results_summary['summary_metrics']['patch_success_count']}")
    print(f"[SUMMARY] Successful evaluations: {results_summary['summary_metrics']['evaluation_success_count']}")
    print(f"[SUMMARY] CORRECT patches: {results_summary['summary_metrics']['patch_validation_counts']['CORRECT']}")
    print(f"[SUMMARY] PLAUSIBLE patches: {results_summary['summary_metrics']['patch_validation_counts']['PLAUSIBLE']}")
    print(f"[SUMMARY] INCORRECT patches: {results_summary['summary_metrics']['patch_validation_counts']['INCORRECT']}")
    print(f"[INFO] Final results saved to: {final_file}")
    print(f"[INFO] Summary report: {final_report}")
    
    return results_summary


def generate_summary_report(results, output_file):
    """
    Generate an HTML summary report from the results dictionary.
    
    Args:
        results (dict): The results dictionary
        output_file (str): Path to save the HTML report
    """
    try:
        # Create DataFrame from bug results
        rows = []
        
        for bug in results["bugs"]:
            row = {
                "Bug": bug.get("bug_name", "Unknown"),
                "Solution Status": bug.get("solution_generation", {}).get("status", "N/A"),
                "Patch Status": bug.get("patch_generation", {}).get("status", "N/A"), 
                "Evaluation Status": bug.get("evaluation", {}).get("status", "N/A"),
                "Duration (s)": bug.get("processing_duration_seconds", 0),
            }
            
            # Count patch validation statuses
            if bug.get("evaluation", {}).get("results"):
                statuses = [r.get("patch_validation_status", "UNKNOWN") for r in bug["evaluation"]["results"]]
                row["CORRECT"] = statuses.count("CORRECT")
                row["PLAUSIBLE"] = statuses.count("PLAUSIBLE")
                row["INCORRECT"] = statuses.count("INCORRECT")
                row["UNKNOWN"] = statuses.count("UNKNOWN")
            else:
                row["CORRECT"] = 0
                row["PLAUSIBLE"] = 0
                row["INCORRECT"] = 0
                row["UNKNOWN"] = 0
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Create summary HTML
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Repair-of-Thought Results Summary</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2 {{ color: #333; }}
                .summary {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                .metrics {{ display: flex; flex-wrap: wrap; gap: 20px; }}
                .metric {{ background-color: white; border: 1px solid #ddd; padding: 10px; border-radius: 5px; flex: 1; min-width: 150px; }}
                .success {{ color: green; }}
                .warning {{ color: orange; }}
                .error {{ color: red; }}
                table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
            </style>
        </head>
        <body>
            <h1>Repair-of-Thought Results Summary</h1>
            
            <div class="summary">
                <h2>Run Information</h2>
                <p><strong>Run ID:</strong> {results.get("run_id", "Unknown")}</p>
                <p><strong>Timestamp:</strong> {results.get("timestamp", "Unknown")}</p>
                <p><strong>Configuration:</strong></p>
                <ul>
                    <li>Patch Number: {results.get("configuration", {}).get("patch_num", "Unknown")}</li>
                    <li>Chunk Size: {results.get("configuration", {}).get("chunk_size", "Unknown")}</li>
                    <li>Delay Between Bugs: {results.get("configuration", {}).get("delay_between_bugs", "Unknown")} seconds</li>
                    <li>Total Bugs: {results.get("configuration", {}).get("total_bugs", "Unknown")}</li>
                </ul>
            </div>
            
            <h2>Summary Metrics</h2>
            <div class="metrics">
                <div class="metric">
                    <h3>Progress</h3>
                    <p>Completed: {results["summary_metrics"]["completed_bugs"]} / {results["summary_metrics"]["total_bugs"]}</p>
                    <p>Progress: {results["summary_metrics"]["completed_bugs"] / results["summary_metrics"]["total_bugs"] * 100:.1f}%</p>
                </div>
                
                <div class="metric">
                    <h3>Success Rates</h3>
                    <p>Solutions: {results["summary_metrics"]["solution_success_count"]} / {results["summary_metrics"]["completed_bugs"]} 
                       ({results["summary_metrics"]["solution_success_count"] / max(1, results["summary_metrics"]["completed_bugs"]) * 100:.1f}%)</p>
                    <p>Patches: {results["summary_metrics"]["patch_success_count"]} / {results["summary_metrics"]["completed_bugs"]}
                       ({results["summary_metrics"]["patch_success_count"] / max(1, results["summary_metrics"]["completed_bugs"]) * 100:.1f}%)</p>
                    <p>Evaluations: {results["summary_metrics"]["evaluation_success_count"]} / {results["summary_metrics"]["completed_bugs"]}
                       ({results["summary_metrics"]["evaluation_success_count"] / max(1, results["summary_metrics"]["completed_bugs"]) * 100:.1f}%)</p>
                </div>
                
                <div class="metric">
                    <h3>Patch Quality</h3>
                    <p class="success">CORRECT: {results["summary_metrics"]["patch_validation_counts"]["CORRECT"]}</p>
                    <p class="warning">PLAUSIBLE: {results["summary_metrics"]["patch_validation_counts"]["PLAUSIBLE"]}</p>
                    <p class="error">INCORRECT: {results["summary_metrics"]["patch_validation_counts"]["INCORRECT"]}</p>
                </div>
            </div>
            
            <h2>Bug Details</h2>
            {df.to_html(index=False)}
        </body>
        </html>
        """
        
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html)
            
    except Exception as e:
        print(f"[ERROR] Failed to generate summary report: {e}")
        # Write a simple error report instead
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"<html><body><h1>Error generating report</h1><p>{str(e)}</p></body></html>")


if __name__ == '__main__':
    """
    Example usage: Just run this script directly, or call the function 
    generate_dataset_results in your own code.
    """
    file_path = r"./datasets/defects4j-sf.json"
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    subset_data = list(data.keys())[400:440]  # Reduced to 20 bugs for testing --- 40
    
    final_results = generate_dataset_results(
        patch_num=3,
        bug_names=subset_data,
        chunk_size=1,       # Process 1 bug at a time to avoid rate limiting
        delay_between_bugs=5  # 5 second delay between bugs
    )
    
    print(f"[INFO] All done! Results saved to results directory.")