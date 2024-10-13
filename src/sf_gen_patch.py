import re
import json
import argparse
import os
from together import Together
from gen_patch_prompt import sf_build_apr_prompt_auto


model_name = "mistralai/Mistral-7B-v0.1"
model_format_prompt = '''Return your fixed function surrounded with ```java\\n```.\n@@ Instruction\n{apr_prompt}\n@@ Response'''


# Function to extract the test method from test cases
def extract_test_method(testcase_lst):
    method_info_lst = []

    bracket_cnt = 0
    bracket_flag = False
    for line in testcase_lst:
        method_info_lst.append(line)
        bracket_cnt += line.count('{')
        if bracket_cnt:
            bracket_flag = True
        bracket_cnt -= line.count('}')
        if bracket_cnt == 0 and bracket_flag:
            return method_info_lst
    return None


# Function to extract all patch codes
def extract_all_patch_codes(orig_patch, dataset, bug_name):
    patch_code_lst = []
    code_patch_pattern = r'```(?:java\n)?(.*?)\n```'
    extracted_lst = re.findall(code_patch_pattern, orig_patch, re.DOTALL)

    function_name = ' ' + dataset[bug_name]['method_signature']['method_name'] + '('
    function_return_type = dataset[bug_name]['method_signature']['return_type']

    if extracted_lst:
        for patch_code in extracted_lst:
            if function_name in patch_code and function_return_type in patch_code:
                patch_code_lst.append(patch_code)
        if len(patch_code_lst) > 0:
            return patch_code_lst

    orig_patch_lines = orig_patch.split('\n')
    len_orig_patch_lines = len(orig_patch_lines)
    for idx in range(len_orig_patch_lines - 1, -1, -1):
        curr_rline = orig_patch_lines[idx]
        if function_name not in curr_rline or function_return_type not in curr_rline:
            continue
        patch_code = extract_test_method(orig_patch_lines[idx:])
        if patch_code:
            patch_code_lst.append('\n'.join(patch_code))
            break
    return patch_code_lst


# Function to extract patch
def extract_patch(dataset, raw_patch_result):
    extracted_patch_result = {}
    for bug_name in raw_patch_result.keys():
        extracted_patch_result[bug_name] = {'prompt': raw_patch_result[bug_name]['prompt'], 'patches': []}
        for raw_patch in raw_patch_result[bug_name]['patches']:
            extracted_patches = extract_all_patch_codes(raw_patch, dataset, bug_name)
            for extracted_patch in extracted_patches:
                if extracted_patch.startswith(':'):
                    extracted_patch = extracted_patch[1:]
                elif extracted_patch.startswith('@@ Response:'):
                    extracted_patch = extracted_patch[12:]
                extracted_patch_result[bug_name]['patches'].append(extracted_patch)
    return extracted_patch_result


# APR Info Class
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


# TogetherAI Model APR function
def togetherai_model_apr(apr_info):
    dataset = apr_info.dataset
    suggestions = apr_info.suggestions

    # Initialize TogetherAI client
    client = Together(api_key=os.getenv('TOGETHER_API_KEY'))
    
    patches = {}
    for bug_name in dataset:
        curr_patch = {'prompt': [], 'patches': []}

        for root_cause in suggestions[bug_name].keys():
            for suggestion in suggestions[bug_name][root_cause]:
                apr_prompt = sf_build_apr_prompt_auto(dataset[bug_name]['buggy'], root_cause, suggestion)
                prompt = model_format_prompt.format(apr_prompt=apr_prompt.strip())
                prompt_token_len = len(prompt.split())  # Rough approximation for token count

                if prompt_token_len > 4000:
                    continue

                curr_patch['prompt'].append(apr_prompt)
                suggest_patch = []

                try:
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=512,
                        temperature=0.8,
                        top_p=0.9,
                        top_k=50,
                        repetition_penalty=1.1,
                        stop=["<|eot_id|>", "<|eom_id|>"],
                    )

                    for choice in response['choices']:
                        generated_text = choice['message']['content']
                        prompt_end_idx = generated_text.find(apr_prompt.strip()) + len(apr_prompt.strip())
                        fixed_result = generated_text[prompt_end_idx:].strip() if prompt_end_idx != -1 else generated_text
                        fixed_result = fixed_result.replace('[/INST]  ', '')
                        suggest_patch.append(fixed_result)

                    curr_patch_cnt = len(curr_patch['patches']) + len(suggest_patch)
                    print(f'### [APR]: bug_name: {bug_name:25}  |  curr_patch_cnt: {curr_patch_cnt:>3}  |  patches_cnt: {len(patches):3} ###')

                except Exception as e:
                    print(f"Error occurred during TogetherAI call: {e}")

                curr_patch['patches'].extend(suggest_patch)

        patches[bug_name] = curr_patch

    return patches


# Main function
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


# Argument parser for command-line usage
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
