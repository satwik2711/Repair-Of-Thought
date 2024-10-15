import streamlit as st
import json
import os
import subprocess

import regex as re



# Streamlit Application
def main():
    st.title("Automated Program Repair (APR) End-to-End Workflow")
    st.write("""
    This app allows you to perform the complete workflow of Automated Program Repair (APR) using the provided dataset.
    Upload a dataset, generate solutions, create patches, and view the results—all within this interface.
    """)

    # File uploader for dataset
    dataset_file = st.file_uploader("Upload Dataset File (JSON)", type="json")

    # User inputs for parameters
    sample_size = st.number_input("Number of solution samples to generate", min_value=1, max_value=5, value=1)
    target_bug = st.text_input("Specify the bug ID to generate a solution for (Optional)")

    # Step 1: Generate Solution
    if st.button("Generate Solution"):
        if dataset_file is not None:
            with open("uploaded_dataset.json", "wb") as f:
                f.write(dataset_file.read())

            # Run the solution generation script
            try:
                command = f"python src/sf_gen_solution.py -d uploaded_dataset.json -o output.json -s {sample_size}"
                if target_bug:
                    command += f" -bug {target_bug}"
                
                subprocess.run(command, shell=True, check=True)
                st.success("Solution generation completed. Solutions saved in output.json.")
            except subprocess.CalledProcessError as e:
                st.error(f"Error during solution generation: {e}")

    # Step 2: Extract Solutions
    if st.button("Extract Solutions"):
        try:
            with open("output2.json", "r") as f:
                raw_solutions = json.load(f)
            
            extracted_solutions = extract_solutions(raw_solutions)
            with open("output2_extracted.json", "w") as f:
                json.dump(extracted_solutions, f, indent=2)

            st.success("Solutions extracted successfully. Extracted solutions saved in output_extracted.json.")
        except Exception as e:
            st.error(f"Error during solution extraction: {e}")

    # Step 3: Generate Patch
    if st.button("Generate Patch"):
        if os.path.exists("output_extracted.json"):
            try:
                command = f"python src/sf_gen_patch.py -d uploaded_dataset.json -s output_extracted.json -o patch_result.json"
                if target_bug:
                    command += f" -bug {target_bug}"

                subprocess.run(command, shell=True, check=True)
                st.success("Patch generation completed. Patches saved in patch_result.json.")
            except subprocess.CalledProcessError as e:
                st.error(f"Error during patch generation: {e}")
        else:
            st.warning("Please extract solutions before generating patches.")

    # Step 4: Display Patch Results
    if os.path.exists("patch_result.json"):
        st.write("### Generated Patches Review")

        with open("timepatch2.json", "r") as f:
            patch_result = json.load(f)

        bug_names = list(patch_result.keys())
        selected_bug = st.selectbox("Select a Bug to Review", bug_names)

        if selected_bug:
            st.subheader(f"Bug: {selected_bug}")
            st.write("### Prompt")
            st.code(patch_result[selected_bug].get("prompt", ""), language='markdown')

            st.write("### Generated Patches")
            patches = patch_result[selected_bug].get("patches", [])

            if patches:
                for i, patch in enumerate(patches, 1):
                    st.write(f"**Patch {i}:**")
                    st.code(patch, language='java')
            else:
                st.write("No patches generated.")

# Extract solutions helper function
def extract_solutions(raw_solution):
    extracted_solutions = {}
    for bug_name, solution_data in raw_solution.items():
        extracted_solutions[bug_name] = {}
        solutions = solution_data['solutions']

        for solution in solutions:
            split_solution_list = split_solutions(solution)
            root_cause = extract_root_cause(solution)

            if root_cause not in extracted_solutions[bug_name]:
                extracted_solutions[bug_name][root_cause] = split_solution_list
            else:
                extracted_solutions[bug_name][root_cause].extend(split_solution_list)

    return extracted_solutions

# Function to split solutions based on patterns
def split_solutions(text):
    pattern = r'Suggestion \d+:.*?(?=Suggestion \d+:|$)'
    solutions = re.findall(pattern, text, re.DOTALL)
    return [solution.split(':', 1)[1].strip() if ':' in solution else solution.strip() for solution in solutions]

# Function to extract the root cause from the solution text
def extract_root_cause(text):
    match = re.search(r'Root Cause:.*?(?=Suggestion \d+:|$)', text, re.DOTALL)
    return match.group().split(':', 1)[1].strip() if match else None

if __name__ == "__main__":
    main()
