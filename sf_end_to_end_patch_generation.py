import os
import json
import time
import re
from typing import List, Dict, Any
import groq
from together import Together
from dotenv import load_dotenv
import random
import tiktoken

load_dotenv()

SF_START_ASSIST_PROMPT = '''
You need to first analyse the buggy code, trigger test and error message. Then analyse the root cause and finally try to provide a repair suggestions to fix the buggy.
Note that the bug can be fixed by modifying only the given buggy code; do not attempt to modify the class, add new functions, or conduct further testing.'''

SF_END_ASSIST_PROMPT = '''
First, analyze the trigger test and error message, and then analyse the root cause of the buggy function in the format 'Root Cause: {content}'. Provide multiple distinct and detailed patch suggestions for resolving this bug.
You suggestions should be in the format 'Suggestion 1: {suggestion title}\\n{detailed description}', etc.'''

APR_LABEL = '// Provide a fix for the buggy function.'
BUGGY_LABEL = '// Buggy Function'
FIXED_LABEL = '// Fixed Function'

class AutoPatchGenerator:
    def __init__(self, dataset_path: str = "datasets/defects4j-sf.json"):
        """Initialize the patch generator with necessary clients and dataset."""
        self.groq_client = groq.Groq(api_key=os.getenv('GROQ_API_KEY'))
        self.together_client = Together(api_key=os.getenv('TOGETHER_API_KEY'))
        
        # Load dataset
        with open(dataset_path, 'r') as f:
            self.dataset = json.load(f)
            
    def _make_groq_call(self, messages: List[Dict], max_tokens: int, is_final_answer: bool = False) -> Any:
        """Make API call to Groq with retry logic."""
        for attempt in range(3):
            try:
                if is_final_answer:
                    response = self.groq_client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=0.9,
                    )
                    return response.choices[0].message.content
                else:
                    response = self.groq_client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=0.9,
                        response_format={"type": "json_object"}
                    )
                    return json.loads(response.choices[0].message.content)
            except Exception as e:
                if attempt == 2:
                    if is_final_answer:
                        return {"error": f"Failed after 3 attempts: {str(e)}"}
                    else:
                        return {"title": "Error", "content": str(e), "next_action": "final_answer"}
                time.sleep(1)

    def _construct_prompt(self, bug_name: str) -> str:
        """Construct the prompt for solution generation."""
        bug_data = self.dataset[bug_name]
        random_trigger_test = random.choice(list(bug_data['trigger_test'].keys()))
        err_msg = bug_data['trigger_test'][random_trigger_test]['clean_error_msg']
        err_msg = self._slim_content_token(err_msg)
        trigger_src = bug_data['trigger_test'][random_trigger_test]['src']

        comment = bug_data['buggy_code_comment']
        buggy_function = bug_data['buggy']
        
        prompt_parts = [
            SF_START_ASSIST_PROMPT,
            f'\n1. Buggy Function: \n{comment}\n{buggy_function}',
            f'\n2. Trigger Test: \n{trigger_src}',
            f'\n3. Error Message: \n{err_msg}\n',
            SF_END_ASSIST_PROMPT
        ]
        return '\n'.join(prompt_parts)

    def _slim_content_token(self, content: str, token_limit: int = 200) -> str:
        """Limit the token count of content."""
        encoding = tiktoken.get_encoding("cl100k_base")
        lines = content.split('\n')
        slim_lines = []
        remaining_tokens = token_limit
        
        for line in lines:
            tokens = len(encoding.encode(line))
            if remaining_tokens - tokens <= 0:
                break
            slim_lines.append(line)
            remaining_tokens -= tokens
            
        return '\n'.join(slim_lines)

    def _generate_solutions(self, bug_name: str) -> Dict[str, Any]:
        """Generate solutions using Groq API."""
        prompt = self._construct_prompt(bug_name)
        
        messages = [
            {"role": "system", "content": '''You are an expert software developer and debugger that explains your reasoning step by step. For each step, provide a title that describes what you're doing in that step, along with the content. Decide if you need another step or if you're ready to give the final solution.'''},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "I will analyze this bug step by step, considering all aspects carefully."}
        ]
        
        steps = []
        step_count = 1
        total_thinking_time = 0
        
        while True:
            start_time = time.time()
            step_data = self._make_groq_call(messages, 500)
            thinking_time = time.time() - start_time
            total_thinking_time += thinking_time
            
            if isinstance(step_data, dict) and 'error' in step_data:
                return {'error': step_data['error']}
                
            steps.append({
                "title": step_data['title'],
                "content": step_data['content'],
                "thinking_time": thinking_time
            })
            
            messages.append({"role": "assistant", "content": json.dumps(step_data)})
            
            if step_data['next_action'] == 'final_answer' or step_count >= 5:
                break
                
            step_count += 1
        
        # Get final solution
        messages.append({
            "role": "user",
            "content": "Based on your reasoning above, provide a final detailed solution with:\n1. Root Cause Analysis\n2. Five distinct repair suggestions\nFormat as 'Root Cause: {description}' followed by 'Suggestion 1: {title}\n{details}' etc."
        })
        
        final_solution = self._make_groq_call(messages, 1500, is_final_answer=True)
        
        return {
            "steps": steps,
            "final_solution": final_solution,
            "total_time": total_thinking_time
        }

    def _extract_solutions(self, solution_text: str) -> Dict[str, List[str]]:
        """Extract root cause and suggestions from solution text."""
        root_cause_match = re.search(r'Root Cause:.*?(?=Suggestion \d+:|$)', solution_text, re.DOTALL)
        root_cause = root_cause_match.group().split(':', 1)[1].strip() if root_cause_match else ""
        
        suggestion_pattern = r'Suggestion \d+:.*?(?=Suggestion \d+:|$)'
        suggestions = re.findall(suggestion_pattern, solution_text, re.DOTALL)
        suggestions = [s.split(':', 1)[1].strip() if ':' in s else s.strip() for s in suggestions]
        
        return {"root_cause": root_cause, "suggestions": suggestions}

    def _generate_patches(self, bug_name: str, solutions: Dict) -> List[str]:
        """Generate patches using Together AI."""
        extracted = self._extract_solutions(solutions['final_solution'])
        patches = []
        
        for suggestion in extracted['suggestions']:
            prompt = '\n'.join([
                APR_LABEL,
                'Root cause: ' + extracted['root_cause'],
                'Suggestion: ' + suggestion,
                BUGGY_LABEL,
                self.dataset[bug_name]['buggy'],
                FIXED_LABEL
            ])
            print(prompt)
            try:
                response = self.together_client.chat.completions.create(
                    model="Qwen/Qwen2.5-Coder-32B-Instruct",
                    #model="codellama/CodeLlama-34b-Instruct-hf",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=4000,
                    temperature=0.8,
                    top_p=0.9,
                )
                
                if hasattr(response, 'choices'):
                    for choice in response.choices:
                        generated_text = choice.message.content
                        if generated_text:
                            # Extract code between ```java and ``` if present
                            code_match = re.search(r'```(?:java)?\n(.*?)\n```', generated_text, re.DOTALL)
                            if code_match:
                                patches.append(code_match.group(1))
                            else:
                                patches.append(generated_text.strip())
                
            except Exception as e:
                print(f"Error generating patch: {str(e)}")
                continue
                
        return patches

    def generate_patches(self, bug_name: str) -> Dict[str, Any]:
        """
        Generate patches for a specific bug.
        
        Args:
            bug_name: Name of the bug (e.g., "Math-2")
            
        Returns:
            Dictionary containing the generated patches and related information
        """
        if bug_name not in self.dataset:
            raise ValueError(f"Bug {bug_name} not found in dataset")
            
        # Step 1: Generate solutions with reasoning
        solutions = self._generate_solutions(bug_name)
        if 'error' in solutions:
            return {"error": solutions['error']}
            
        # Step 2: Generate patches from solutions
        patches = self._generate_patches(bug_name, solutions)
        
        return {
            "bug_name": bug_name,
            "solutions": solutions,
            "patches": patches
        }

def get_patches(bug_name: str, dataset_path: str = "datasets/defects4j-sf.json") -> Dict[str, Any]:
    """
    Convenience function to get patches for a specific bug.
    
    Args:
        bug_name: Name of the bug (e.g., "Math-2")
        dataset_path: Path to the dataset JSON file
        
    Returns:
        Dictionary containing:
        - bug_name: The name of the bug
        - solutions: The reasoning steps and final solution
        - patches: List of generated patches
        
    Example:
        >>> result = get_patches("Math-2")
        >>> print(f"Generated {len(result['patches'])} patches")
        >>> for patch in result['patches']:
        ...     print(patch)
    """
    generator = AutoPatchGenerator(dataset_path)
    return generator.generate_patches(bug_name)