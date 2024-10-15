import streamlit as st
import json
import os
import subprocess


def main():
    st.title("Automated Program Repair (APR) End-to-End Workflow")
    st.write(
        """
        This app performs the complete Automated Program Repair (APR) workflow.
        Input a bug name and sample size to generate solutions and patches, or view existing results if they are available.
        """
    )

    # User inputs for parameters
    sample_size = st.number_input("Number of solution samples to generate", min_value=1, max_value=5, value=1)
    bug_name = st.text_input("Specify the bug ID to generate a solution for (e.g., Math-2)")

    # Define output directories
    sol_dir = "outputs/sol"
    patches_dir = "outputs/patches"
    os.makedirs(sol_dir, exist_ok=True)
    os.makedirs(patches_dir, exist_ok=True)

    # Generate Solution and Patch Workflow
    if st.button("Generate Solution and Patch"):
        if bug_name:
            # Define file paths
            solution_file = os.path.join(sol_dir, f"{bug_name}.json")
            extracted_solution_file = os.path.join(sol_dir, f"{bug_name}_extracted.json")
            patch_file = os.path.join(patches_dir, f"{bug_name}_patch.json")

            # Step 1: Generate Solutions
            if not os.path.exists(solution_file) or not os.path.exists(extracted_solution_file):
                st.write(f"Generating solutions for bug: {bug_name}...")
                try:
                    command = f"python src/sf_gen_solution.py -d datasets/defects4j-sf.json -o {solution_file} -s {sample_size} -bug {bug_name}"
                    subprocess.run(command, shell=True, check=True)
                    st.success(f"Solutions generated and saved in {solution_file} and {extracted_solution_file}.")
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
                    st.success(f"Patches generated and saved in {patch_file}.")
                except subprocess.CalledProcessError as e:
                    st.error(f"Error during patch generation: {e}")
                    return
            else:
                st.write(f"Patch file for bug '{bug_name}' already exist. Using existing file.")

    # Display Generated Solutions and Patches
    if bug_name:
        solution_file = os.path.join(sol_dir, f"{bug_name}.json")
        extracted_solution_file = os.path.join(sol_dir, f"{bug_name}_extracted.json")
        patch_file = os.path.join(patches_dir, f"{bug_name}_patch.json")

        if os.path.exists(solution_file) and os.path.exists(extracted_solution_file):
            st.write("### Generated Solutions")
            with open(solution_file, "r") as f:
                solutions = json.load(f)
                st.json(solutions)

        if os.path.exists(patch_file):
            st.write("### Generated Patches")
            with open(patch_file, "r") as f:
                patch_result = json.load(f)
                st.write("#### Bug Name: ", bug_name)
                st.write("##### Prompt")
                st.code(patch_result.get(bug_name, {}).get("prompt", ""), language='markdown')

                st.write("##### Patches")
                patches = patch_result.get(bug_name, {}).get("patches", [])
                if patches:
                    for i, patch in enumerate(patches, 1):
                        st.write(f"**Patch {i}:**")
                        st.code(patch, language='java')
                else:
                    st.write("No patches generated.")

if __name__ == "__main__":
    main()