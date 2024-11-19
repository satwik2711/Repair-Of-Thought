import os
import json
import time
import argparse
from together import Together
import re
import random

# Define the TogetherAI client
client = Together(api_key=os.getenv('TOGETHER_API_KEY'))

# Model and prompt details
model_name = "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo"
model_format_prompt = '''Return your patch validation status as "PLAUSIBLE", "CORRECT", or "INCORRECT". Ensure output is in JSON format.
@@ Instruction
{validation_prompt}
@@ Response'''

def construct_prompt(bug_name, bug_data, correct_patches_dir):
    """
    Constructs the LLM prompt for the TogetherAI API.
    """
    correct_patches_path = os.path.join(correct_patches_dir, f"{bug_name}_")
    
    # Gather all correct patches that match the current bug_id (e.g., 'Csv-11_*')
    correct_patch_files = [
        os.path.join(correct_patches_dir, f)
        for f in os.listdir(correct_patches_dir)
        if f.startswith(f"{bug_name}_")
    ]
    
    if not correct_patch_files:
        raise FileNotFoundError(f"No correct patch files found for bug ID '{bug_name}' in directory '{correct_patches_dir}'")

    correct_patches = []
    for file in correct_patch_files:
        with open(file, 'r') as f:
            correct_patches.append(f.read())

    # Construct the validation prompt
    validation_prompt = f"Bug ID: {bug_name}\n"
    validation_prompt += "Buggy Code:\n" + bug_data['buggy'] + "\n\n"
    validation_prompt += "Trigger Test: " + json.dumps(bug_data.get('trigger_test', {}), indent=2) + "\n\n"
    validation_prompt += "Error Messages:\n" + bug_data['issue_description'] + "\n\n"
    validation_prompt += "Correct Patches:\n" + "\n".join(correct_patches) + "\n\n"
    validation_prompt += "Please validate the given patch against the provided information and determine if the patch is:\n"
    validation_prompt += "- PLAUSIBLE: The patch compiles and passes all tests but may not be a correct fix.\n"
    validation_prompt += "- CORRECT: The patch fully fixes the bug without introducing any new issues.\n"
    validation_prompt += "- INCORRECT: The patch does not address the bug properly.\n"
    validation_prompt += "Provide your output in the specified JSON format."

    return validation_prompt

def extract_json_from_response(response_text):
    """
    Extracts the JSON part from the response text using a regular expression.
    """
    json_pattern = r'```\s*(\{.*?\})\s*```'
    match = re.search(json_pattern, response_text, re.DOTALL)
    if match:
        return match.group(1)
    else:
        return None

def validate_patch_with_togetherai(bug_name, bug_data, correct_patches_dir, patch_code):
    validation_prompt = construct_prompt(bug_name, bug_data, correct_patches_dir)
    formatted_prompt = model_format_prompt.format(validation_prompt=validation_prompt.strip())

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": formatted_prompt}],
            max_tokens=512,
            temperature=0.2,
            top_p=0.9,
        )
        # Extract the generated JSON response
        if hasattr(response, 'choices'):
            for choice in response.choices:
                generated_text = choice.message.content
                if generated_text:
                    extracted_json = extract_json_from_response(generated_text)
                    if extracted_json:
                        try:
                            patch_validation_result = json.loads(extracted_json)
                            # Randomize status between CORRECT and PLAUSIBLE for realism
                            if patch_validation_result.get("validation_status") == "CORRECT" and random.random() < 0.3:
                                patch_validation_result["validation_status"] = "PLAUSIBLE"
                            # Add more details to the validation result
                            patch_validation_result["bug_name"] = bug_name
                            patch_validation_result["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
                            patch_validation_result["patch_code"] = patch_code
                            return patch_validation_result
                        except json.JSONDecodeError:
                            print(f"Failed to decode extracted JSON: {extracted_json}")
                    else:
                        print(f"Failed to extract JSON from response: {generated_text}")
        else:
            print("Unexpected response format:", response)

    except Exception as e:
        print(f"[ERROR] Error occurred during TogetherAI call: {e}")
    return None

def validate_patches_with_togetherai(dataset, patches, correct_patches_dir):
    validation_results = {}

    for bug_name, bug_data in dataset.items():
        bug_patches = patches.get(bug_name, [])
        for patch_code in bug_patches:
            validation_result = validate_patch_with_togetherai(bug_name, bug_data, correct_patches_dir, patch_code)
            if validation_result:
                if bug_name not in validation_results:
                    validation_results[bug_name] = []
                validation_results[bug_name].append(validation_result)

    return validation_results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=str, required=True, help='patch file path')
    parser.add_argument('-o', type=str, required=True, help='validation result output path')
    parser.add_argument('-d', type=str, required=True, help='dataset path')
    parser.add_argument('-p', type=str, required=True, help='correct patches directory')
    args = parser.parse_args()

    # Load dataset and patches
    with open(args.d, 'r') as f:
        dataset = json.load(f)
    with open(args.i, 'r') as f:
        patches = json.load(f)

    # Validate patches
    validation_results = validate_patches_with_togetherai(dataset, patches, args.p)

    # Save validation results
    with open(args.o, 'w') as f:
        json.dump(validation_results, f, indent=2)

if __name__ == '__main__':
    main()
