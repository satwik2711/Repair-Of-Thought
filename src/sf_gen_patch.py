import re
import json
import argparse
import os
from together import Together
from gen_patch_prompt import sf_build_apr_prompt_auto

# Use a model suitable for code generation
model_name = "Qwen/Qwen2.5-Coder-32B-Instruct"
model_format_prompt = '''You are a Senior Level Programmer who is an expert in fixing bugs. Review the following function and provide a corrected implementation.

@@ Context
{apr_prompt}

@@ Instructions
1. Analyze the provided root cause and suggestion carefully
2. Ensure your fix addresses the identified issue
3. Maintain the original method signature
4. Make MINIMAL changes to fix the identified issue
5. Preserve the original code structure and style
6. Only modify the specific lines needed to address the root cause
7. Keep all working parts of the original implementation
8. Follow the provided fix suggestion without over-engineering

@@ Response Format
Return ONLY the complete fixed function wrapped in ```java
<your implementation>
``` tags.

@@ Important Notes
- DO NOT rewrite or restructure the entire function
- DO NOT add unnecessary optimizations
- DO NOT change working code that isn't related to the bug
- DO NOT modify the method signature
- DO maintain the original variable names and coding style

Your response:'''

def extract_all_patch_codes(orig_patch, dataset, bug_name):
    patch_code_lst = []
    code_patch_pattern = r'```(?:java)?\n(.*?)\n```'
    extracted_lst = re.findall(code_patch_pattern, orig_patch, re.DOTALL)

    if extracted_lst:
        patch_code_lst.extend(extracted_lst)
        return patch_code_lst

    # Fallback logic
    return [orig_patch.strip()]

class AprInfo:
    def __init__(self, dataset_path, suggestions_path, out_path, target_bug):
        with open(dataset_path, 'r') as f:
            self.dataset = json.load(f)
        with open(suggestions_path, 'r') as f:
            self.suggestions = json.load(f)
        if target_bug is not None:
            self.dataset = {target_bug: self.dataset[target_bug]}
            self.suggestions = {target_bug: self.suggestions[target_bug]}
        self.out_path = out_path
        self.target_bug = target_bug

def togetherai_model_apr(apr_info):
    dataset = apr_info.dataset
    suggestions = apr_info.suggestions

    client = Together(api_key=os.getenv('TOGETHER_API_KEY'))

    patches = {}
    for bug_name in dataset:
        curr_patch = {'prompt': [], 'patches': []}

        for root_cause in suggestions[bug_name]:
            for suggestion in suggestions[bug_name][root_cause]:
                apr_prompt = sf_build_apr_prompt_auto(
                    dataset[bug_name]['buggy'], root_cause, suggestion)
                prompt = model_format_prompt.format(
                    apr_prompt=apr_prompt.strip())
                curr_patch['prompt'].append(apr_prompt)
                suggest_patch = []

                try:
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=512,
                        temperature=0.8,
                        top_p=0.9,
                    )

                    # Access the generated text
                    if hasattr(response, 'choices'):
                        for choice in response.choices:
                            generated_text = choice.message.content
                            if generated_text:
                                print(f"\n {generated_text} \n")
                                suggest_patch.append(generated_text)
                    else:
                        print("Unexpected response format:", response)

                    if not suggest_patch:
                        print(f"No suggestions returned for bug: {bug_name}")

                    curr_patch_cnt = len(curr_patch['patches']) + len(suggest_patch)
                    print(f'### [APR]: bug_name: {bug_name:25}  |  curr_patch_cnt: {curr_patch_cnt:>3}  |  patches_cnt: {len(patches):3} ###')

                except Exception as e:
                    print(f"Error occurred during TogetherAI call: {e}")

                curr_patch['patches'].extend(suggest_patch)

        patches[bug_name] = curr_patch

    return patches

def main():
    apr_result = {}
    apr_info = AprInfo(
        dataset_path=args.d,
        suggestions_path=args.s,
        out_path=args.o,
        target_bug=args.bug
    )
    apr_result = togetherai_model_apr(apr_info)
    apr_result = extract_patch(apr_info.dataset, apr_result)
    with open(apr_info.out_path, 'w') as f:
        json.dump(apr_result, f, indent=2)

def extract_patch(dataset, raw_patch_result):
    extracted_patch_result = {}
    for bug_name in raw_patch_result.keys():
        extracted_patch_result[bug_name] = {'prompt': raw_patch_result[bug_name]['prompt'], 'patches': []}
        for raw_patch in raw_patch_result[bug_name]['patches']:
            extracted_patches = extract_all_patch_codes(raw_patch, dataset, bug_name)
            for extracted_patch in extracted_patches:
                extracted_patch_result[bug_name]['patches'].append(extracted_patch)
    return extracted_patch_result

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, required=True, help='dataset path')
    parser.add_argument('-s', type=str, required=True, help='suggestions path')
    parser.add_argument('-o', type=str, required=True, help='patch_result path')
    parser.add_argument('-bug', type=str, required=False, help='bug')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    main()
