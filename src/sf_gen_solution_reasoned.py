import time
import re
import json
import groq
import argparse
from tqdm import tqdm
import os
from gen_solution_prompt import sf_construct_prompt
from dotenv import load_dotenv
import requests
from typing import Optional,Dict,Any



def send_llama_request(
    prompt: str,
    api_url: str = os.getenv('SELF_API_URL'),
    app_id: str = os.getenv('SELF_API_ID'),
    auth_token: str = f"Access {os.getenv('SELF_API_TOKEN')}"
) -> Optional[Dict[str, Any]]:
    """Send a request to the LLaMA model API."""
    
    try:
        headers = {
            'Content-Type': 'application/json',
            'x-app-id': app_id,
            'authorization': f"Access {auth_token}"
        }
        
        data = {
            "question": prompt,
            "temperature": 0
        }
        
        
        response = requests.post(api_url, headers=headers, data=json.dumps(data))
        response.raise_for_status()
        
        json_response = response.json()
        print(f"[DEBUG] Response received and parsed successfully")
        
        return json_response["messages"][0].get("answer")
        
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] API Request failed: {str(e)}")
        if hasattr(e.response, 'text'):
            print(f"[ERROR] Response text: {e.response.text}")
        return None
    except json.JSONDecodeError as e:
        print(f"[ERROR] Failed to parse API response: {str(e)}")
        return None
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred: {str(e)}")
        print(f"[ERROR] Error type: {type(e)}")
        return None

def get_api_details():
    """Get API details from environment variables"""
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dotenv_path = os.path.join(root_dir, '.env')
    load_dotenv(dotenv_path)
    
    api_url = os.getenv('SELF_API_URL')
    app_id = os.getenv('SELF_API_ID')
    auth_token = os.getenv('SELF_API_TOKEN')
    
    missing_vars = []
    if not api_url:
        missing_vars.append('SELF_API_URL')
    if not app_id:
        missing_vars.append('SELF_API_ID')
    if not auth_token:
        missing_vars.append('SELF_API_TOKEN')
    
    if missing_vars:
        error_msg = f"Missing required API environment variables: {', '.join(missing_vars)}"
        print(f"[ERROR] {error_msg}")
        raise ValueError(error_msg)
    
    return api_url, app_id, auth_token

def make_api_call(messages, max_tokens, is_final_answer=False, custom_client=None):
    api_url, app_id, auth_token = get_api_details()
    # Convert messages to prompt format
    prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
    
    last_error = None  # Add this to track the last error
    
    for attempt in range(3):
        try:
            response = send_llama_request(
                prompt=prompt,
                api_url=api_url,
                app_id=app_id,
                auth_token=auth_token
            )
            
            if response is None:
                raise ValueError("Empty response from API")
                
            if is_final_answer:
                return response
            else:
                return json.loads(response)
                
        except Exception as e:
            last_error = e  # Store the error
            if attempt < 2:
                time.sleep(1)
            continue
            
    # Now we can safely reference last_error
    if is_final_answer:
        return f"Error: Failed to generate response after 3 attempts. Last error: {str(last_error)}"
    else:
        return {
            "title": "Error",
            "content": f"Failed to generate response after 3 attempts. Last error: {str(last_error)}",
            "next_action": "final_answer"
        }

def generate_reasoned_solution(prompt, client=None, num_patches=3):

    messages = [
        {"role": "system", "content": """You are an expert software developer and debugger that explains your reasoning step by step. For each step, provide a title that describes what you're doing in that step, along with the content. Decide if you need another step or if you're ready to give the final solution. Respond in JSON format with 'title', 'content', and 'next_action' (either 'continue' or 'final_answer') keys. 

USE AT LEAST 3 REASONING STEPS. Consider:
1. Understanding the bug and test case
2. Analyzing root causes
3. Exploring potential fixes
4. Validating proposed solutions
5. Considering edge cases and potential issues

BE THOROUGH IN YOUR ANALYSIS:
- Consider multiple approaches
- Evaluate trade-offs
- Identify potential pitfalls
- Validate assumptions
- Consider corner cases

Example response format:
{
    "title": "Analyzing Bug Context",
    "content": "First, let's understand the buggy function and its intended behavior...",
    "next_action": "continue"
}"""},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": "I will analyze this bug step by step, considering all aspects carefully."}
    ]
    
    steps = []
    step_count = 1
    total_thinking_time = 0
    
    while True:
        start_time = time.time()
        step_data = make_api_call(messages, 500, custom_client=client)
        end_time = time.time()
        thinking_time = end_time - start_time
        total_thinking_time += thinking_time
        
        steps.append((f"Step {step_count}: {step_data['title']}", step_data['content'], thinking_time))
        
        messages.append({"role": "assistant", "content": json.dumps(step_data)})
        
        if step_data['next_action'] == 'final_answer' or step_count > 10:
            break
        
        step_count += 1
        yield steps, None

    messages.append({
        "role": "user", 
        "content": f"Based on your reasoning above, provide a final detailed solution with:\n1. Root Cause Analysis\n2. ONLY {num_patches} distinct repair suggestions, NOTHING MORE, NOTHING LESS\nFormat as 'Root Cause: {{description}}' followed by 'Suggestion 1: {{title}}\n{{details}}' etc."
    })
    
    start_time = time.time()
    final_solution = make_api_call(messages, 1500, is_final_answer=True, custom_client=client)
    end_time = time.time()
    thinking_time = end_time - start_time
    total_thinking_time += thinking_time
    
    steps.append(("Final Solution", final_solution, thinking_time))
    yield steps, total_thinking_time

class SolInfo:
    def __init__(self, dataset_path, solution_path, extracted_solution_path=None, sample_size=1, target_bug=None, patch_num=3):
        with open(dataset_path, 'r') as f:
            self.dataset = json.load(f)
        
        if target_bug:
            self.dataset = {target_bug: self.dataset[target_bug]}
            
        self.solution_path = solution_path
        self.extracted_solution_path = extracted_solution_path or self._get_extracted_solution_path(solution_path)
        self.sample_size = sample_size
        self.target_bug = target_bug
        self.patch_num = patch_num

    def _get_extracted_solution_path(self, solution_path):
        base, ext = solution_path.rsplit('.', 1)
        return f"{base}_extracted.{ext}"

def get_solutions_with_reasoning(sol_info, client=None):
    solutions = {}
    for bug_name in tqdm(sol_info.dataset.keys()):
        solutions[bug_name] = {}
        prompt = sf_construct_prompt(sol_info.dataset, bug_name)
        solutions[bug_name]['prompt'] = prompt
        solutions[bug_name]['solutions'] = []
        
        for _ in range(sol_info.sample_size):
            steps = []
            final_time = None
            for step_result, total_time in generate_reasoned_solution(prompt, client, num_patches=sol_info.patch_num):
                steps = step_result
                final_time = total_time
            
            reasoning_content = "\n\n".join([f"{title}\n{content}" for title, content, _ in steps[:-1]])
            final_solution = steps[-1][1]
            
            solution_with_reasoning = {
                "reasoning_steps": reasoning_content,
                "final_solution": final_solution,
                "total_time": final_time
            }
            solutions[bug_name]['solutions'].append(solution_with_reasoning)
    
    return solutions

def extract_solutions(raw_solution):
    extracted_solutions = {}
    for bug_name, solution_data in raw_solution.items():
        extracted_solutions[bug_name] = {}
        
        for solution in solution_data['solutions']:
            final_solution = solution['final_solution']
            root_cause = extract_root_cause(final_solution)
            solution_suggestions = split_solutions(final_solution)
            
            if root_cause not in extracted_solutions[bug_name]:
                extracted_solutions[bug_name][root_cause] = solution_suggestions
            else:
                extracted_solutions[bug_name][root_cause].extend(solution_suggestions)
    
    return extracted_solutions

def split_solutions(text):
    pattern = r'Suggestion \d+:.*?(?=Suggestion \d+:|$)'
    solutions = re.findall(pattern, text, re.DOTALL)
    return [solution.split(':', 1)[1].strip() if ':' in solution else solution.strip() 
            for solution in solutions]

def extract_root_cause(text):
    match = re.search(r'Root Cause:.*?(?=Suggestion \d+:|$)', text, re.DOTALL)
    return match.group().split(':', 1)[1].strip() if match else None

def main():
    parser = argparse.ArgumentParser(description="Generate and extract solutions with reasoning for single-function bugs.")
    parser.add_argument('-d', type=str, required=True, help='Path to the dataset file.')
    parser.add_argument('-o', type=str, required=True, help='Path to save the raw solution output.')
    parser.add_argument('-eo', type=str, required=False, help='Path to save the extracted solutions output.')
    parser.add_argument('-s', type=int, required=False, default=1, help='Number of solution samples to generate.')
    parser.add_argument('-bug', type=str, required=False, help='Specific bug to generate a solution for.')
    parser.add_argument('-patch_num', type=int, required=False, default=3, help="Number of patches to generate.")
    
    args = parser.parse_args()
    
    sol_info = SolInfo(
        dataset_path=args.d,
        solution_path=args.o,
        extracted_solution_path=args.eo,
        sample_size=args.s,
        target_bug=args.bug,
        patch_num=args.patch_num
    )
    
    solutions = get_solutions_with_reasoning(sol_info)
    with open(sol_info.solution_path, 'w') as f:
        json.dump(solutions, f, indent=2)
        
    extracted_solutions = extract_solutions(solutions)
    with open(sol_info.extracted_solution_path, 'w') as f:
        json.dump(extracted_solutions, f, indent=2)

if __name__ == '__main__':
    main()