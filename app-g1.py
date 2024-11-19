import streamlit as st
import json
import os
import subprocess
from datetime import datetime

def main():
    # st.set_page_config(layout="wide")
    st.title("Repair of Thought")
    st.write(
        """
        An APR Framework that addresses function level bug repair using a dual model approach, offering a streamlined approach that reduces the need for costly statement level fault localization
        This app performs the complete Automated Program Repair (APR) workflow with detailed reasoning steps.
        Input a bug name and sample size to generate solutions and patches, or view existing results.
        """
    )
    

    # User inputs
    sample_size = st.number_input("Number of solution samples to generate", min_value=1, max_value=5, value=1)
    bug_name = st.text_input("Specify the bug ID to generate a solution for (e.g., Math-2)")

    # Define output directories
    sol_dir = "outputs-g1/sol"
    patches_dir = "outputs-g1/patches"
    os.makedirs(sol_dir, exist_ok=True)
    os.makedirs(patches_dir, exist_ok=True)

    if st.button("Generate Solution and Patch"):
        if bug_name:
            solution_file = os.path.join(sol_dir, f"{bug_name}.json")
            extracted_solution_file = os.path.join(sol_dir, f"{bug_name}_extracted.json")
            patch_file = os.path.join(patches_dir, f"{bug_name}_patch.json")

            # Step 1: Generate Solutions with Reasoning
            if not os.path.exists(solution_file) or not os.path.exists(extracted_solution_file):
                st.write(f"Generating solutions for bug: {bug_name}...")
                try:
                    command = f"python src/sf_gen_solution_reasoned.py -d datasets/defects4j-sf.json -o {solution_file} -s {sample_size} -bug {bug_name}"
                    subprocess.run(command, shell=True, check=True)
                    st.success(f"Solutions generated with reasoning steps")
                except subprocess.CalledProcessError as e:
                    st.error(f"Error during solution generation: {e}")
                    return
            else:
                st.write(f"Solution files for bug '{bug_name}' already exist. Using existing files.")

            # Step 2: Generate Patches (unchanged)
            if not os.path.exists(patch_file):
                st.write(f"Generating patches for bug: {bug_name}...")
                try:
                    command = f"python src/sf_gen_patch.py -d datasets/defects4j-sf.json -s {extracted_solution_file} -o {patch_file} -bug {bug_name}"
                    subprocess.run(command, shell=True, check=True)
                    st.success(f"Patches generated successfully")
                except subprocess.CalledProcessError as e:
                    st.error(f"Error during patch generation: {e}")
                    return
            else:
                st.write(f"Patch file for bug '{bug_name}' already exists. Using existing file.")

    # Display Results
    if bug_name:
        solution_file = os.path.join(sol_dir, f"{bug_name}.json")
        patch_file = os.path.join(patches_dir, f"{bug_name}_patch.json")

        if os.path.exists(solution_file):
            st.write("### Generated Solutions with Reasoning")
            with open(solution_file, "r") as f:
                solutions = json.load(f)
                
                # Display original bug information
                if 'prompt' in solutions[bug_name]:
                    with st.expander("Original Bug Information", expanded=False):
                        st.code(solutions[bug_name]['prompt'], language='text')
                
                # Display solutions with reasoning
                for idx, solution in enumerate(solutions[bug_name]['solutions'], 1):
                    st.write(f"#### Solution {idx}")
                    
                    # Display reasoning steps
                    reasoning_steps = solution['reasoning_steps'].split('\n\n')
                    for step in reasoning_steps:
                        if step.strip():
                            title, content = step.split('\n', 1)
                            with st.expander(title, expanded=True):
                                st.markdown(content)
                    
                    # Display final solution
                    with st.expander("Final Solution", expanded=True):
                        st.markdown(solution['final_solution'])
                    
                    # Display timing information
                    st.info(f"Total thinking time: {solution['total_time']:.2f} seconds")
                    st.divider()

        if os.path.exists(patch_file):
            st.write("### Generated Patches")
            with open(patch_file, "r") as f:
                patch_result = json.load(f)
                
                patches = patch_result.get(bug_name, {}).get("patches", [])
                if patches:
                    for i, patch in enumerate(patches, 1):
                        with st.expander(f"Patch {i}", expanded=False):
                            st.code(patch, language='java')
                else:
                    st.write("No patches generated.")

if __name__ == "__main__":
    main()