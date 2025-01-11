import json

import os
from google import genai
from google.genai import types
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

class PatchInfo(BaseModel):
  patch_type: str

api_key = os.getenv('GEMINI_API_KEY')
client = genai.Client(api_key=api_key)

def evaluate_patch(bug_name: str, generated_patch: str) -> dict:
    """
    Evaluate a generated patch against a bug's ground truth
    
    Args:
        bug_name: Name of the bug (e.g. "Math-2")
        generated_patch: The patch code to evaluate
        
    Returns:
        dict: Contains response text and classification JSON
    """
    file_path = r"D:\Repair-Of-Thought\datasets\defects4j-sf.json"
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    test_sections = []
    for test_key in data[bug_name]['trigger_test']:
        test_case = data[bug_name]['trigger_test'][test_key]
        test_section = f"""
[Trigger Test {len(test_sections) + 1}]
Test Name: {test_case['function_name']}
Test Source: {test_case['src']}
Error Message: {test_case['clean_error_msg']}
"""
        test_sections.append(test_section)

    prompt = f"""You are an expert at evaluating program patches for correctness.

You will be given:
1. Original buggy code
2. The Github Issue name,description of the bug
3. Trigger test name, code and error message which is causing the issue to be caught
2. Ground truth patch from benchmark 
3. Generated patch to evaluate

 You will systematically analyze patches in a step by step manner using the following structure:
[Analysis]
1. Core Functionality and Behavior
- How does each patch handle the main issue?
- What are the behavioral differences between patches?
- How do edge cases differ?

2. Return Value Patterns & Edge Cases
- Compare return values for key scenarios:
  * Normal case (valid input)
  * Missing/null values
  * Out of bounds values
  * Invalid inputs
- Note any differences in behavior

3. Error Handling Patterns
- Compare exception types and messages
- Compare validation approaches
- Identify differences in error handling strategies

4. Test Implications 
- Will patch pass the trigger test?
- Are there potential issues with other tests?
- Could behavior changes affect dependent code?

[Classification]
Based on the analysis, classify the patch as:
- CORRECT: Semantically equivalent to ground truth
- PLAUSIBLE: Fixes core issue but has behavioral differences
- INCORRECT: Wrong behavior or fails tests

Provide clear reasoning for your classification based on the behavioral differences identified.

Explain your reasoning step by step, then conclude with your classification.

[Buggy Code]: {data[bug_name]['buggy_fl']}
[Buggy Code's Comments]: {data[bug_name]['buggy_code_comment']}
[Issue Title]: {data[bug_name]['issue_title']}
[Issue Description]: {data[bug_name]['issue_description']}
{''.join(test_sections)}
[Ground Truth Patch]: {data[bug_name]['fix']} 
[Generated Patch]: {generated_patch}
"""

    generation_config = types.GenerateContentConfig(
        temperature=0.9,
        top_p=0.95,
        top_k=20,
        candidate_count=1,
        stop_sequences=["STOP!"]
    )

    response = client.models.generate_content(
        model='gemini-2.0-flash-thinking-exp-1219',
        #model='gemini-2.0-flash-exp',
        contents=types.Part.from_text(prompt),
        config=generation_config
    )

    output_prompt = f"""This is a big reasoning response from gemini-2.0-flash-thinking, output the classification type in JSON.- CORRECT: Semantically equivalent to ground truth
- PLAUSIBLE: Fixes core issue but has behavioral differences
- INCORRECT: Wrong behavior or fails tests
Output: {response.text}"""

    response2 = client.models.generate_content(
        model='gemini-2.0-flash-exp',
        contents=types.Part.from_text(output_prompt),
        config=types.GenerateContentConfig(
            response_mime_type='application/json',
            response_schema=PatchInfo,
        )
    )

    return {
        "prompt": prompt,
        "analysis": response.text,
        "classification": response2.text
    }
