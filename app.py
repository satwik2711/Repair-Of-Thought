import streamlit as st
import json
import os
import subprocess
from datetime import datetime

# Set page configuration
st.set_page_config(page_title="Repair of Thought", layout="wide")

def main():
    st.title("🔧 Repair of Thought")
    st.markdown(
        """
        An APR Framework that addresses function-level bug repair using a dual model approach, offering a streamlined approach that reduces the need for costly statement-level fault localization.
        """
    )
    st.markdown("---")

    # Sidebar for user inputs
    with st.sidebar:
        st.header("Configuration")
        sample_size = st.number_input("Number of solution samples to generate", min_value=1, max_value=5, value=1)
        bug_name = st.text_input("Specify the bug ID to generate a solution for (e.g., Math-2)")
        st.markdown("---")
        st.write("© 2023 Repair of Thought")

    # Define output directories
    sol_dir = "outputs-g1/sol"
    patches_dir = "outputs-g1/patches"
    val_dir = "outputs-g1/val"
    os.makedirs(sol_dir, exist_ok=True)
    os.makedirs(patches_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    solution_file = os.path.join(sol_dir, f"{bug_name}.json")
    extracted_solution_file = os.path.join(sol_dir, f"{bug_name}_extracted.json")
    patch_file = os.path.join(patches_dir, f"{bug_name}_patch.json")
    validation_file = os.path.join(val_dir, f"{bug_name}_patch_val.json")

    st.markdown("### Actions")

    # Buttons for generating solutions and patches
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Generate Solution and Patch"):
            if bug_name:
                # Step 1: Generate Solutions with Reasoning
                if not os.path.exists(solution_file) or not os.path.exists(extracted_solution_file):
                    st.write(f"Generating solutions for bug: {bug_name}...")
                    try:
                        with st.spinner("Generating solutions..."):
                            command = f"python src/sf_gen_solution_reasoned.py -d datasets/defects4j-sf.json -o {solution_file} -s {sample_size} -bug {bug_name}"
                            subprocess.run(command, shell=True, check=True)
                        st.success("Solutions generated with reasoning steps")
                    except subprocess.CalledProcessError as e:
                        st.error(f"Error during solution generation: {e}")
                        return
                else:
                    st.info(f"Solution files for bug '{bug_name}' already exist. Using existing files.")

                # Step 2: Generate Patches
                if not os.path.exists(patch_file):
                    st.write(f"Generating patches for bug: {bug_name}...")
                    try:
                        with st.spinner("Generating patches..."):
                            command = f"python src/sf_gen_patch.py -d datasets/defects4j-sf.json -s {extracted_solution_file} -o {patch_file} -bug {bug_name}"
                            subprocess.run(command, shell=True, check=True)
                        st.success("Patches generated successfully")
                    except subprocess.CalledProcessError as e:
                        st.error(f"Error during patch generation: {e}")
                        return
                else:
                    st.info(f"Patch file for bug '{bug_name}' already exists. Using existing file.")

    with col2:
        if os.path.exists(patch_file):
            if st.button("Validate Patches"):
                if not os.path.exists(validation_file):
                    st.write(f"Validating patches for bug: {bug_name}...")
                    try:
                        with st.spinner("Validating patches..."):
                            command = f"python3 src/sf_val2.py -i {patch_file} -o {validation_file} -d datasets/defects4j-sf.json -p correct_patch/defects4j-sf"
                            subprocess.run(command, shell=True, check=True)
                        st.success("Patches validated successfully")
                    except subprocess.CalledProcessError as e:
                        st.error(f"Error during patch validation: {e}")
                        return
                else:
                    st.info(f"Validation file for bug '{bug_name}' already exists. Using existing file.")

    st.markdown("---")

    # Display Results
    if bug_name:
        tab1, tab2, tab3 = st.tabs(["Solutions", "Patches", "Validation Results"])

        # Solutions Tab
        with tab1:
            st.write("### Generated Solutions with Reasoning")
            if os.path.exists(solution_file):
                with open(solution_file, "r") as f:
                    solutions = json.load(f)

                    # Display original bug information
                    bug_info = solutions.get(bug_name, {})
                    if 'prompt' in bug_info:
                        with st.expander("Original Bug Information", expanded=False):
                            st.code(bug_info['prompt'], language='java')

                    # Display solutions
                    for idx, solution in enumerate(bug_info.get('solutions', []), 1):
                        st.markdown(f"#### Solution {idx}")

                        # Use expanders for reasoning steps
                        reasoning_steps = solution.get('reasoning_steps', "").split('\n\n')
                        for step in reasoning_steps:
                            if step.strip():
                                parts = step.split('\n', 1)
                                title = parts[0].strip()
                                content = parts[1].strip() if len(parts) > 1 else "No additional content provided."
                                with st.expander(title, expanded=False):
                                    st.markdown(content)

                        # Display final solution
                        st.markdown("**Final Solution:**")
                        st.code(solution.get('final_solution', "No final solution provided."), language='java')

                        # Display timing information
                        st.info(f"**Total thinking time:** {solution.get('total_time', 0):.2f} seconds")
                        st.markdown("---")
            else:
                st.warning("No solution file found. Please generate solutions first.")

        # Patches Tab
        with tab2:
            st.write("### Generated Patches")
            if os.path.exists(patch_file):
                with open(patch_file, "r") as f:
                    patch_result = json.load(f)

                    patches = patch_result.get(bug_name, {}).get("patches", [])
                    if patches:
                        for i, patch in enumerate(patches, 1):
                            with st.expander(f"Patch {i}", expanded=False):
                                st.code(patch, language='java')
                    else:
                        st.warning("No patches generated.")
            else:
                st.warning("No patch file found. Please generate patches first.")

        # Validation Results Tab
        with tab3:
            st.write("### Patch Validation Results")
            if os.path.exists(validation_file):
                with open(validation_file, "r") as f:
                    validation_results = json.load(f)

                    results = validation_results.get(bug_name, [])
                    if results:
                        for i, result in enumerate(results, 1):
                            with st.expander(f"Validation Result for Patch {i}", expanded=False):
                                status = result['patch_validation_status']
                                if status.lower() == 'success':
                                    st.success(f"**Validation Status**: ✅ {status}")
                                else:
                                    st.error(f"**Validation Status**: {status}")
                                st.write(f"**Bug Name**: {result['bug_name']}")
                                st.write(f"**Timestamp**: {result['timestamp']}")
                                st.code(result['patch_code'], language='java')
                                st.write(f"**Validation Tool**: TogetherAI")
                    else:
                        st.warning("No validation results available.")
            else:
                st.warning("No validation file found. Please validate patches first.")

if __name__ == "__main__":
    main()
