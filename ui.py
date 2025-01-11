import streamlit as st
import json
import os
import subprocess
from datetime import datetime

def main():
    st.title("Repair of Thought")
    st.write(
        """
        An APR Framework that addresses function-level bug repair using a dual model approach, offering a streamlined approach that reduces the need for costly statement-level fault localization.
        This app performs the complete Automated Program Repair (APR) workflow with detailed reasoning steps.
        Input a bug name and sample size to generate solutions and patches, or view existing results.
        """
    )
    
    # User inputs
    sample_size = st.number_input("Number of solution samples to generate", min_value=1, max_value=5, value=1)
    bug_name = st.text_input("Specify the bug ID to generate a solution for (e.g., Math-2)")

    # Define output directories
    sol_dir = "outputs/sol"
    patches_dir = "outputs/patches"
    val_dir = "outputs/val"
    os.makedirs(sol_dir, exist_ok=True)
    os.makedirs(patches_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    solution_file = os.path.join(sol_dir, f"{bug_name}.json")
    extracted_solution_file = os.path.join(sol_dir, f"{bug_name}_extracted.json")
    patch_file = os.path.join(patches_dir, f"{bug_name}_patch.json")
    validation_file = os.path.join(val_dir, f"{bug_name}_patch_val.json")

    # Button to generate solutions and patches
    if st.button("Generate Solution and Patch"):
        if bug_name:
            # Step 1: Generate Solutions with Reasoning
            if not os.path.exists(solution_file) or not os.path.exists(extracted_solution_file):
                st.write(f"Generating solutions for bug: {bug_name}...")
                try:
                    command = f"python src/sf_gen_solution_reasoned.py -d datasets/defects4j-sf.json -o {solution_file} -s {sample_size} -bug {bug_name}"
                    subprocess.run(command, shell=True, check=True)
                    st.success("Solutions generated with reasoning steps")
                except subprocess.CalledProcessError as e:
                    st.error(f"Error during solution generation: {e}")
                    return
            else:
                st.write(f"Solution files for bug '{bug_name}' already exist. Using existing files.")

            # Step 2: Generate Patches
            if not os.path.exists(patch_file):
                st.write(f"Generating patches for bug: {bug_name}...")
                try:
                    command = f"python src/sf_gen_patch.py -d datasets/defects4j-sf.json -s {extracted_solution_file} -o {patch_file} -bug {bug_name}"
                    subprocess.run(command, shell=True, check=True)
                    st.success("Patches generated successfully")
                except subprocess.CalledProcessError as e:
                    st.error(f"Error during patch generation: {e}")
                    return
            else:
                st.write(f"Patch file for bug '{bug_name}' already exists. Using existing file.")

    # Button to validate patches after they are generated
    if os.path.exists(patch_file):
        if st.button("Validate Patches"):
            if not os.path.exists(validation_file):
                st.write(f"Validating patches for bug: {bug_name}...")
                try:
                    from src.sf_auto_eval import evaluate_patches
                    import asyncio
                    
                    with open(patch_file, 'r') as f:
                        patch_data = json.load(f)
                    
                    results = asyncio.run(evaluate_patches(bug_name, patch_file))
                    st.success("Patches validated successfully")
                except Exception as e:
                    st.error(f"Error during patch validation: {e}")
                    st.exception(e)  # This will print the full traceback
                    return
            else:
                st.write(f"Validation file for bug '{bug_name}' already exists. Using existing file.")

    # Display Results
    if bug_name:
        if os.path.exists(solution_file):
            st.write("### Generated Solutions with Reasoning")
            with open(solution_file, "r") as f:
                solutions = json.load(f)
                
                # Display original bug information
                if 'prompt' in solutions.get(bug_name, {}):
                    with st.expander("Original Bug Information", expanded=False):
                        st.code(solutions[bug_name]['prompt'], language='text')
                
                # Display solutions with reasoning
                for idx, solution in enumerate(solutions.get(bug_name, {}).get('solutions', []), 1):
                    st.write(f"#### Solution {idx}")
                    
                    # Display reasoning steps
                    reasoning_steps = solution.get('reasoning_steps', "").split('\n\n')
                    for step in reasoning_steps:
                        if step.strip():
                            parts = step.split('\n', 1)
                            title = parts[0]
                            content = parts[1] if len(parts) > 1 else "No additional content provided."
                            with st.expander(title, expanded=True):
                                st.markdown(content)
                    
                    # Display final solution
                    with st.expander("Final Solution", expanded=True):
                        st.markdown(solution.get('final_solution', "No final solution provided."))
                    
                    # Display timing information
                    st.info(f"Total thinking time: {solution.get('total_time', 0):.2f} seconds")
                    st.divider()

        if os.path.exists(validation_file):
            st.write("### Patch Validation Results")
            with open(validation_file, "r") as f:
                validation_results = json.load(f)

                if validation_results.get(bug_name):
                    for i, result in enumerate(validation_results[bug_name], 1):
                        with st.expander(f"Validation Result for Patch {i}", expanded=False):
                            st.write(f"**Validation Status**: {result['patch_validation_status']}")
                            st.write(f"**Bug Name**: {result['bug_name']}")
                            st.write(f"**Timestamp**: {result['timestamp']}")
                            st.code(result['generated_patch'], language='java')
                            
                            # Instead of nested expander, use columns and a toggle
                            st.write("---")  # Add a separator
                            show_details = st.toggle('Show Prompt and Analysis', key=f"toggle_{i}")
                            if show_details:
                                st.write("#### Prompt")
                                st.code(result['prompt'], language='text')
                                st.write("#### Analysis")
                                st.markdown(result['analysis'])

if __name__ == "__main__":
    main()
