import time
import re
import json
import openai
import argparse
from tqdm import tqdm
from gen_solution_prompt import sf_construct_prompt
import os

# Function to query the OpenAI model with reasoning steps
def query_model_with_reasoning(prompt, sample_size):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API Key not set. Please set the 'OPENAI_API_KEY' environment variable.")
    
    delay = 10
    while(True):
        try:
            response = api_gpt_response(prompt, sample_size)
            break
        except openai.APIError as e:
            if "Please reduce " in str(e):
                raise ValueError("Over Length")
            print(f"OpenAI API returned an API Error: {e}")
            time.sleep(delay)
        except Exception as e:
            print(f"Exception in api_query_model {e}")
            if "Please reduce " in str(e):
                raise ValueError("Over Length")
            time.sleep(delay)

    response_list = []
    for choice in response.choices:
        if choice.message:
            response_list.append(choice.message.content)
    return response_list

from openai import OpenAI
# Function to interact with OpenAI's chat completion API (latest version)
def api_gpt_response(prompt, n):
    client = OpenAI()
    return client.chat.completions.create(
        model="gpt-4o",  # You can switch to "gpt-3.5-turbo" if needed
        messages=[{"role": "user", "content": prompt}],
        n=n,
        temperature=0.8
    )

# Solution Information class to handle the dataset, paths, and extraction
class SolInfo:
    def __init__(self, dataset_path, solution_path, extracted_solution_path=None, sample_size=1, target_bug=None):
        with open(dataset_path, 'r') as f:
            self.dataset = json.load(f)

        if target_bug:
            self.dataset = {target_bug: self.dataset[target_bug]}

        self.solution_path = solution_path
        self.extracted_solution_path = extracted_solution_path or self._get_extracted_solution_path(solution_path)
        self.sample_size = sample_size
        self.target_bug = target_bug

    def _get_extracted_solution_path(self, solution_path):
        base, ext = solution_path.rsplit('.', 1)
        return f"{base}_extracted.{ext}"

# Function to split solutions based on patterns
def split_solutions(text):
    pattern = r'Suggestion \d+:.*?(?=Suggestion \d+:|$)'
    solutions = re.findall(pattern, text, re.DOTALL)
    return [solution.split(':', 1)[1].strip() if ':' in solution else solution.strip() for solution in solutions]

# Function to extract the root cause from the solution text
def extract_root_cause(text):
    match = re.search(r'Root Cause:.*?(?=Suggestion \d+:|$)', text, re.DOTALL)
    return match.group().split(':', 1)[1].strip() if match else None

# Function to get solutions with reasoning steps by querying the model
def get_solutions_with_reasoning(sol_info):
    solutions = {}
    for bug_name in tqdm(sol_info.dataset.keys()):
        solutions[bug_name] = {}
        prompt = sf_construct_prompt(sol_info.dataset, bug_name)
        solutions[bug_name]['prompt'] = prompt
        raw_responses = query_model_with_reasoning(prompt, sol_info.sample_size)
        solutions[bug_name]['solutions'] = []
        for response in raw_responses:
            reasoning_steps = extract_reasoning_steps(response)
            final_solution_with_reasoning = f"Reasoning Steps:\n" + "\n".join(reasoning_steps) + "\n\nFinal Solution:\n" + response
            solutions[bug_name]['solutions'].append(final_solution_with_reasoning)
    return solutions

# Function to extract reasoning steps from the solution text
def extract_reasoning_steps(text):
    pattern = r'Reasoning Step \d+:.*?(?=Reasoning Step \d+:|Final Answer|$)'
    reasoning_steps = re.findall(pattern, text, re.DOTALL)
    return [step.strip() for step in reasoning_steps]

# Function to extract solutions from the raw output
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

# Main execution flow
def main():
    args = parse_arguments()

    sol_info = SolInfo(
        dataset_path=args.d,
        solution_path=args.o,
        extracted_solution_path=args.eo,
        sample_size=args.s,
        target_bug=args.bug
    )

    solutions = get_solutions_with_reasoning(sol_info)
    with open(sol_info.solution_path, 'w') as f:
        json.dump(solutions, f, indent=2)

    extracted_solutions = extract_solutions(solutions)
    with open(sol_info.extracted_solution_path, 'w') as f:
        json.dump(extracted_solutions, f, indent=2)

# Argument parser for command-line usage
def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate and extract solutions for single-function bugs.")
    parser.add_argument('-d', type=str, required=True, help='Path to the dataset file.')
    parser.add_argument('-o', type=str, required=True, help='Path to save the raw solution output.')
    parser.add_argument('-eo', type=str, required=False, help='Path to save the extracted solutions output.')
    parser.add_argument('-s', type=int, required=False, default=1, help='Number of solution samples to generate.')
    parser.add_argument('-bug', type=str, required=False, help='Specific bug to generate a solution for.')
    return parser.parse_args()

if __name__ == '__main__':
    main()