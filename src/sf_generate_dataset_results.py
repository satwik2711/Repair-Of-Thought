import os
import json
import subprocess
import concurrent.futures
import asyncio
from datetime import datetime
from sf_auto_eval import evaluate_patches
import groq
from dotenv import load_dotenv

load_dotenv()

def create_groq_client(key_index):
    """Create a Groq client with the specified API key index"""
    key = os.getenv(f"GROQ_API_KEY_{key_index}")
    if not key:
        raise ValueError(f"GROQ_API_KEY_{key_index} not found in environment variables")
    return groq.Groq(api_key=key)

def get_available_client_count():
    """Count how many Groq API keys are available"""
    count = 0
    while os.getenv(f"GROQ_API_KEY_{count + 1}"):
        count += 1
    return count

def process_bug(bug_name, patch_num, client_index, sample_size=1):
    """
    Generate solutions and patches for a single bug, then run auto_eval on the patch file.
    
    Returns a dictionary with:
      - bug_name
      - solution_file
      - extracted_solution_file
      - patch_file
      - solution_generation_status
      - patch_generation_status
      - auto_eval_results (list of evaluation results or error message)
    """
    sol_dir = "outputs/sol"
    patches_dir = "outputs/patches"
    os.makedirs(sol_dir, exist_ok=True)
    os.makedirs(patches_dir, exist_ok=True)
    
    solution_file = os.path.join(sol_dir, f"{bug_name}.json")
    extracted_solution_file = os.path.join(sol_dir, f"{bug_name}_extracted.json")
    patch_file = os.path.join(patches_dir, f"{bug_name}_patch.json")
    
    bug_result = {
        "bug_name": bug_name,
        "solution_file": solution_file,
        "extracted_solution_file": extracted_solution_file,
        "patch_file": patch_file,
        "solution_generation_status": None,
        "patch_generation_status": None,
        "auto_eval_results": None,  # We'll store the evaluation results here
    }

    # 1. Generate Solutions
    print(f"[INFO] Generating solutions for bug: {bug_name} using client {client_index}")
    try:
        client = create_groq_client(client_index)
        
        from sf_gen_solution_reasoned import get_solutions_with_reasoning, SolInfo, extract_solutions
        
        sol_info = SolInfo(
            dataset_path="datasets/defects4j-sf.json",
            solution_path=solution_file,
            extracted_solution_path=extracted_solution_file,
            sample_size=sample_size,
            target_bug=bug_name,
            patch_num=patch_num
        )
        solutions = get_solutions_with_reasoning(sol_info, client)
        with open(solution_file, 'w') as f:
            json.dump(solutions, f, indent=2)
            
        extracted_solutions = extract_solutions(solutions)
        with open(extracted_solution_file, 'w') as f:
            json.dump(extracted_solutions, f, indent=2)
            
        bug_result["solution_generation_status"] = "Success"
    except subprocess.CalledProcessError as e:
        bug_result["solution_generation_status"] = f"Error: {e}"
        print(f"[ERROR] Solution generation failed for {bug_name}: {e}")

    # 2. Generate Patches
    print(f"[INFO] Generating patches for bug: {bug_name}")
    try:
        cmd_patch = (
            f"python src/sf_gen_patch.py "
            f"-d datasets/defects4j-sf.json "
            f"-s {extracted_solution_file} "
            f"-o {patch_file} "
            f"-bug {bug_name}"
        )
        subprocess.run(cmd_patch, shell=True, check=True)
        bug_result["patch_generation_status"] = "Success"
    except subprocess.CalledProcessError as e:
        bug_result["patch_generation_status"] = f"Error: {e}"
        print(f"[ERROR] Patch generation failed for {bug_name}")

    if os.path.exists(patch_file):
        try:
            auto_eval_dict = asyncio.run(evaluate_patches(bug_name, patch_file))
            bug_result["auto_eval_results"] = auto_eval_dict.get(bug_name, [])
            
            validation_statuses = [result.get("patch_validation_status","").upper() for result in bug_result["auto_eval_results"]]
            
            if "CORRECT" in validation_statuses:
                bug_result["final_result"] = "CORRECT"
            elif "PLAUSIBLE" in validation_statuses:
                bug_result["final_result"] = "PLAUSIBLE"
            else:
                bug_result["final_result"] = "INCORRECT"
                
        except Exception as e:
            bug_result["auto_eval_results"] = f"Error during evaluation: {e}"
            bug_result["final_result"] = "INCORRECT"
            print(f"[ERROR] Evaluation failed for {bug_name}: {e}")
    else:
        bug_result["auto_eval_results"] = "No patch file found, skipping evaluation."
        bug_result["final_result"] = "INCORRECT"
    
    
    
    return bug_result


def generate_dataset_results(patch_num, bug_names, chunk_size=3):
    """
    Process a list of bug names in batches of `chunk_size`, generating solutions and patches,
    and then evaluating them via auto_eval. Partial and final JSON summaries are stored.
    
    Args:
        patch_num (int): Number of patches to generate
        bug_names (list[str]): List of bug names (e.g. ["Math-2", "Csv-3", "Chart-5"])
        chunk_size (int): Number of bugs to process concurrently in one batch (default=10)
    
    Returns:
        str: Path to the final JSON results file
    """
    load_dotenv()
    num_clients = get_available_client_count()
    if num_clients == 0:
        raise ValueError("No Groq API keys found in environment variables")
    
    results_summary = {
        "timestamp": datetime.now().isoformat(),
        "patch_num": patch_num,
        "bugs": []
    }

    os.makedirs("results", exist_ok=True)

    bug_chunks = [
        bug_names[i : i + chunk_size]
        for i in range(0, len(bug_names), chunk_size)
    ]

    # We'll use a time-based suffix for partial and final outputs
    time_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Process each chunk in parallel
    for chunk_index, chunk in enumerate(bug_chunks, start=1):
        print(f"\n[INFO] Processing chunk {chunk_index} (up to {chunk_size} bugs): {chunk}\n")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=chunk_size) as executor:
            future_to_bug = {
                executor.submit(
                    process_bug, 
                    b, 
                    patch_num, 
                    (chunk.index(b) % num_clients) + 1, 
                    1
                ): b
                for b in chunk
            }
            for future in concurrent.futures.as_completed(future_to_bug):
                bug = future_to_bug[future]
                try:
                    bug_result = future.result()
                    results_summary["bugs"].append(bug_result)
                except Exception as e:
                    print(f"[ERROR] Unexpected error with bug {bug}: {e}")
                    error_result = {
                        "bug_name": bug,
                        "solution_generation_status": f"Error: {e}",
                        "patch_generation_status": None,
                        "auto_eval_results": None,
                    }
                    results_summary["bugs"].append(error_result)

        # After finishing each chunk, save partial results
        partial_output_path = os.path.join(
            "results",
            f"patch_results_llama33_qwen32_3pn_partial_chunk{chunk_index}_{time_suffix}.json"
        )
        with open(partial_output_path, "w", encoding="utf-8") as f:
            json.dump(results_summary, f, indent=2)
        print(f"[INFO] Wrote partial results after chunk {chunk_index} to: {partial_output_path}")

    # Finally, write the complete results
    final_output_path = os.path.join(
        "results", f"patch_results_final_llama33_qwen32_3pn_{time_suffix}.json"
    )
    with open(final_output_path, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)
    print(f"[INFO] Final results written to {final_output_path}")

    return final_output_path


if __name__ == '__main__':
    """
    Example usage: Just run this script directly, or call the function 
    generate_dataset_results in your own code.
    """
    file_path = r"D:\Repair-Of-Thought\datasets\defects4j-sf.json"
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    subset_data=list(data.keys())[100:150]
    final_path = generate_dataset_results(
        patch_num=3,
        bug_names=subset_data
    )
    print(f"[INFO] All done! Results in {final_path}")
