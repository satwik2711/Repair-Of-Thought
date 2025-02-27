import streamlit as st
import json
import os
import subprocess
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

def main():
    st.title("Repair of Thought")
    st.write(
        """
        An APR Framework that addresses function-level bug repair using a dual model approach, offering a streamlined approach that reduces the need for costly statement-level fault localization.
        This app performs the complete Automated Program Repair (APR) workflow with detailed reasoning steps.
        Input a bug name and sample size to generate solutions and patches, or view existing results.
        """
    )
    
    # API Keys section
    st.subheader("API Keys Configuration")
    
    # Create columns for API key inputs
    col1, col2 = st.columns(2)
    
    with col1:
        # Gemini API Key
        default_gemini_key = os.getenv('GEMINI_API_KEY', '')
        gemini_key = st.text_input(
            "Gemini API Key", 
            value=default_gemini_key,
            type="password",
            help="Enter your Gemini API key for patch validation.",
            key="gemini_key"
        )
        
        # Together API Key
        default_together_key = os.getenv('TOGETHER_API_KEY', '')
        together_key = st.text_input(
            "Together API Key",
            value=default_together_key,
            type="password",
            help="Enter your Together API key for patch generation.",
            key="together_key"
        )
    
    with col2:
        # Groq API Key
        default_self_url = os.getenv('SELF_API_URL', '')
        self_api_url = st.text_input("Self-hosted LLaMA API URL", value=default_self_url, help="Enter your self-hosted LLaMA API URL.", key="self_api_url")

        default_self_id = os.getenv('SELF_API_ID', '')
        self_api_id = st.text_input("Self-hosted LLaMA API ID", value=default_self_id, help="Enter your self-hosted LLaMA API ID.", key="self_api_id")

        default_self_token = os.getenv('SELF_API_TOKEN', '')
        self_api_token = st.text_input("Self-hosted LLaMA API Token", value=default_self_token, type="password", help="Enter your self-hosted LLaMA API Token.", key="self_api_token")

    # Then update the environment variables accordingly:
    os.environ['SELF_API_URL'] = self_api_url
    os.environ['SELF_API_ID'] = self_api_id
    os.environ['SELF_API_TOKEN'] = self_api_token

    os.environ['GEMINI_API_KEY'] = gemini_key
    os.environ['TOGETHER_API_KEY'] = together_key
    
    # User inputs
    sample_size = 1
    num_patches = st.number_input("Number of patches to generate", min_value=1, max_value=10, value=3)
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


    if st.button("Generate Solution and Patch"):
        if not together_key:
            st.error("Please provide a Together API key for patch generation.")
            return
            
        if bug_name:
            # Step 1: Generate Solutions with Reasoning
            if not os.path.exists(solution_file) or not os.path.exists(extracted_solution_file):
                st.write(f"Generating solutions for bug: {bug_name}...")
                try:
                    command = f"python src/sf_gen_solution_reasoned.py -d datasets/defects4j-sf.json -o {solution_file} -s {sample_size} -bug {bug_name} -patch_num {num_patches}"
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
            if not gemini_key:
                st.error("Please provide a Gemini API key for patch validation.")
                return
                
            if not os.path.exists(validation_file):
                st.write(f"Validating patches for bug: {bug_name}...")
                try:
                    from src.sf_auto_eval2 import evaluate_patches
                    import asyncio
                    
                    with open(patch_file, 'r') as f:
                        patch_data = json.load(f)
                    
                    results = asyncio.run(evaluate_patches(bug_name, patch_file, gemini_key))
                    st.success("Patches validated successfully")
                except Exception as e:
                    st.error(f"Error during patch validation: {e}")
                    st.exception(e)  # This will print the full traceback
                    return
            else:
                st.write(f"Validation file for bug '{bug_name}' already exists. Using existing file.")

    # Display Results section remains unchanged
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

        if os.path.exists(validation_file):
            st.write("### Patch Validation Results")
            with open(validation_file, "r") as f:
                validation_results = json.load(f)

                if validation_results.get(bug_name):
                    for i, result in enumerate(validation_results[bug_name], 1):
                        st.write(f"## Patch {i}")
                        st.write(f"**Validation Status**: {result['patch_validation_status']}")
                        st.divider()

if __name__ == "__main__":
    main()