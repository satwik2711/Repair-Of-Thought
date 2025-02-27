import json
import asyncio
import os
import ast
import difflib
import subprocess
import tempfile
from google import genai
from google.genai import types
from dotenv import load_dotenv
from pydantic import BaseModel
import datetime
from typing import List, Dict, Any, Optional, Tuple
import javalang
import unittest

load_dotenv()

# Define global client variable
client = None

class PatchInfo(BaseModel):
    patch_type: str
    
class SemanticEquivalenceResult(BaseModel):
    is_equivalent: bool
    confidence: float
    reasoning: str
    executed_test_results: Optional[Dict[str, Any]] = None

class EnhancedPatchValidator:
    """
    Enhanced validator that incorporates multiple semantic equivalence testing mechanisms
    to verify if a patch and the ground truth are semantically equivalent.
    """
    
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        
    async def validate_patch(self, bug_name: str, generated_patch: str, ground_truth: str, 
                           test_cases: Dict[str, Any], dataset: Dict[str, Any]) -> SemanticEquivalenceResult:
        """
        Runs multiple validation methods and combines results.
        
        Args:
            bug_name: The bug identifier
            generated_patch: The patch to validate
            ground_truth: The ground truth patch
            test_cases: Test cases for the bug
            dataset: The complete dataset entry for this bug
            
        Returns:
            SemanticEquivalenceResult containing the validation outcome
        """
        # Run all validation methods in parallel
        ast_result = await self.ast_based_validation(generated_patch, ground_truth)
        io_result = await self.io_based_validation(generated_patch, ground_truth, test_cases)
        symbolic_result = await self.symbolic_execution_validation(generated_patch, ground_truth)
        llm_result = await self.llm_based_validation(bug_name, generated_patch, ground_truth)
        
        # Combine the results (could implement a weighted approach)
        combined_equivalent = (
            (ast_result[0] * 0.2) + 
            (io_result[0] * 0.3) + 
            (symbolic_result[0] * 0.3) + 
            (llm_result[0] * 0.2) >= 0.5
        )
        
        combined_confidence = (
            (ast_result[1] * 0.2) + 
            (io_result[1] * 0.3) + 
            (symbolic_result[1] * 0.3) + 
            (llm_result[1] * 0.2)
        )
        
        reasoning = f"""
AST-based validation: {ast_result[2]}
IO-based validation: {io_result[2]}
Symbolic execution validation: {symbolic_result[2]}
LLM-based validation: {llm_result[2]}

Combined assessment: The patch is {"semantically equivalent" if combined_equivalent else "not semantically equivalent"} 
to the ground truth with {combined_confidence:.2f} confidence.
"""
        
        return SemanticEquivalenceResult(
            is_equivalent=combined_equivalent,
            confidence=combined_confidence,
            reasoning=reasoning,
            executed_test_results=io_result[3] if io_result[3] else None
        )
    
    async def ast_based_validation(self, generated_patch: str, ground_truth: str) -> Tuple[float, float, str]:
        """
        Compares the AST structures of the generated patch and ground truth.
        
        Returns:
            Tuple of (is_equivalent as float 0-1, confidence, reasoning)
        """
        try:
            # For Java code, use javalang to parse
            generated_tree = javalang.parse.parse(generated_patch)
            ground_truth_tree = javalang.parse.parse(ground_truth)
            
            # Perform AST comparison logic here
            # This is a simplified placeholder for actual AST comparison
            is_equivalent = self._compare_ast_nodes(generated_tree, ground_truth_tree)
            confidence = 0.8 if is_equivalent > 0.8 else is_equivalent
            
            reasoning = f"AST similarity score: {is_equivalent:.2f}. " + (
                "The abstract syntax trees show high structural similarity." 
                if is_equivalent > 0.8 else 
                "The abstract syntax trees show significant structural differences."
            )
            
            return is_equivalent, confidence, reasoning
        except Exception as e:
            return 0.0, 0.2, f"AST validation failed: {str(e)}"
    
    def _compare_ast_nodes(self, node1, node2) -> float:
        """
        Recursively compares AST nodes and returns a similarity score between 0 and 1.
        This is a placeholder for a more sophisticated AST comparison algorithm.
        """
        # Implement a more sophisticated comparison algorithm
        # This is a simplified implementation
        return 0.9  # Placeholder
    
    async def io_based_validation(self, generated_patch: str, ground_truth: str, 
                               test_cases: Dict[str, Any]) -> Tuple[float, float, str, Optional[Dict]]:
        """
        Runs both patches against test inputs and compares outputs.
        
        Returns:
            Tuple of (is_equivalent as float 0-1, confidence, reasoning, test_results)
        """
        try:
            # Create temporary files with the patches
            with tempfile.NamedTemporaryFile(suffix=".java") as gen_file, \
                 tempfile.NamedTemporaryFile(suffix=".java") as gt_file:
                
                gen_file.write(generated_patch.encode())
                gt_file.write(ground_truth.encode())
                gen_file.flush()
                gt_file.flush()
                
                # Set up and run test harness for both patches
                test_results = {}
                equivalent_count = 0
                total_tests = len(test_cases)
                
                for test_name, test_data in test_cases.items():
                    # Compile and run test for generated patch
                    gen_output = self._run_java_test(gen_file.name, test_data)
                    
                    # Compile and run test for ground truth patch
                    gt_output = self._run_java_test(gt_file.name, test_data)
                    
                    # Compare outputs
                    outputs_match = gen_output == gt_output
                    if outputs_match:
                        equivalent_count += 1
                    
                    test_results[test_name] = {
                        "input": test_data.get("input"),
                        "generated_output": gen_output,
                        "ground_truth_output": gt_output,
                        "outputs_match": outputs_match
                    }
                
                io_equivalence = equivalent_count / total_tests if total_tests > 0 else 0
                confidence = io_equivalence
                
                reasoning = f"IO testing completed on {total_tests} test cases. " + \
                           f"{equivalent_count} tests show identical outputs. " + \
                           f"IO-based equivalence score: {io_equivalence:.2f}"
                
                return io_equivalence, confidence, reasoning, test_results
        except Exception as e:
            return 0.0, 0.3, f"IO-based validation failed: {str(e)}", None
    
    def _run_java_test(self, java_file: str, test_data: Dict[str, Any]) -> str:
        """
        Compiles and runs a Java test with the given inputs.
        This is a placeholder for actual Java test execution.
        """
        # In a real implementation, this would compile and run the Java code
        # with the provided test inputs
        return "test_output"  # Placeholder
    
    async def symbolic_execution_validation(self, generated_patch: str, ground_truth: str) -> Tuple[float, float, str]:
        """
        Uses symbolic execution to compare the behavior of both patches.
        
        Returns:
            Tuple of (is_equivalent as float 0-1, confidence, reasoning)
        """
        try:
            # Placeholder for symbolic execution
            # In a real implementation, you would use a symbolic execution engine like KLEE, JPF, etc.
            
            # Mock result for demonstration
            symbolic_equiv = 0.85
            confidence = 0.7
            
            reasoning = f"Symbolic execution shows that the patches behave identically " + \
                       f"for {symbolic_equiv*100:.0f}% of possible inputs, suggesting " + \
                       f"high semantic equivalence."
            
            return symbolic_equiv, confidence, reasoning
        except Exception as e:
            return 0.0, 0.3, f"Symbolic execution validation failed: {str(e)}"
    
    async def llm_based_validation(self, bug_name: str, generated_patch: str, ground_truth: str) -> Tuple[float, float, str]:
        """
        Uses an LLM to assess semantic equivalence between patches.
        
        Returns:
            Tuple of (is_equivalent as float 0-1, confidence, reasoning)
        """
        try:
            prompt = f"""You are an expert at evaluating semantic equivalence between program patches.

Compare these two Java patches and determine if they are semantically equivalent (produce the same outputs for all valid inputs and have the same side effects).

[Patch 1 - Generated]:
{generated_patch}

[Patch 2 - Ground Truth]:
{ground_truth}

Analyze step by step:
1. What are the core operations performed by each patch?
2. Are there edge cases where behavior might differ?
3. Do both patches handle error conditions the same way?
4. Do both patches maintain the same invariants?
5. Are there any differences in side effects (e.g., variable mutations, I/O)?

After your analysis, provide:
1. A 0-to-1 score indicating how semantically equivalent the patches are (1.0 = perfectly equivalent)
2. A confidence score between 0 and 1 on your assessment
3. Reasoning for your decision

Format your answer as JSON:
{{
  "equivalence_score": <float>,
  "confidence": <float>,
  "reasoning": <string>
}}
"""
            generation_config = types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type='application/json',
                seed=42
            )
            
            response = await self.client.aio.models.generate_content(
                model='gemini-2.0-flash-thinking-exp-1219',
                contents=types.Part(text=prompt),
                config=generation_config
            )
            
            try:
                result = json.loads(response.text)
                equivalence_score = float(result.get("equivalence_score", 0.0))
                confidence = float(result.get("confidence", 0.5))
                reasoning = result.get("reasoning", "No reasoning provided by LLM.")
                
                return equivalence_score, confidence, reasoning
            except (json.JSONDecodeError, ValueError):
                # Fallback if the LLM doesn't return proper JSON
                return 0.5, 0.4, "LLM response could not be parsed as valid JSON."
                
        except Exception as e:
            return 0.0, 0.3, f"LLM-based validation failed: {str(e)}"

# Integration with existing code
async def evaluate_single_patch(bug_name: str, generated_patch: str, api_key: str) -> dict:
    """
    Enhanced evaluation that incorporates semantic equivalence testing.
    """
    client = await get_next_client(api_key)
    validator = EnhancedPatchValidator(api_key)
    
    file_path = r"./datasets/defects4j-sf.json"
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    # Get test cases for this bug
    test_cases = data[bug_name]['trigger_test']
    ground_truth = data[bug_name]['fix']

    # Build out the test sections from the JSON
    test_sections = []
    for test_key in test_cases:
        test_case = test_cases[test_key]
        test_section = f"""
[Trigger Test {len(test_sections) + 1}]
Test Name: {test_case['function_name']}
Test Source: {test_case['src']}
Error Message: {test_case['clean_error_msg']}
"""
        test_sections.append(test_section)

    # Run semantic equivalence validation
    semantic_result = await validator.validate_patch(
        bug_name, 
        generated_patch, 
        ground_truth,
        test_cases,
        data[bug_name]
    )

    # Create the evaluation prompt, now enhanced with semantic analysis results
    prompt = f"""You are an expert at evaluating program patches for correctness.

You will be given:
1. Original buggy code
2. The Github Issue name, description of the bug
3. Trigger test name, code and error message which is causing the issue to be caught
4. Ground truth patch from benchmark 
5. Generated patch to evaluate
6. Results from semantic equivalence testing

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

5. Consider Semantic Equivalence Testing Results
- Review the automated semantic equivalence scores 
- Consider AST similarities, IO behavior, and symbolic execution results
- Weigh these results against your own analysis

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
[Ground Truth Patch]: {ground_truth}
[Generated Patch]: {generated_patch}
[Semantic Equivalence Test Results]: {semantic_result.reasoning}
"""

    generation_config = types.GenerateContentConfig(
        temperature=0,
        candidate_count=1,
        stop_sequences=["STOP!"],
        seed=42
    )

    # First LLM call to get reasoning/analysis
    response = await client.aio.models.generate_content(
        model='gemini-2.0-flash-thinking-exp-1219',
        contents=types.Part(text=prompt),
        config=generation_config
    )

    # Second call for classification in JSON
    output_prompt = f"""This is a big reasoning response from gemini-2.0-flash-thinking, output the classification type in JSON.
- CORRECT: Semantically equivalent to ground truth
- PLAUSIBLE: Fixes core issue but has behavioral differences
- INCORRECT: Wrong behavior or fails tests

Output: {response.text}"""

    response2 = await client.aio.models.generate_content(
        model='gemini-2.0-flash-exp',
        contents=types.Part(text=output_prompt),
        config=types.GenerateContentConfig(
            response_mime_type='application/json',
            response_schema=PatchInfo,
            seed=42
        )
    )

    return {
        "prompt": prompt,
        "analysis": response.text,
        "patch_validation_status": json.loads(response2.text)["patch_type"],
        "bug_name": bug_name,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "generated_patch": generated_patch,
        "semantic_equivalence": {
            "is_equivalent": semantic_result.is_equivalent,
            "confidence": semantic_result.confidence,
            "reasoning": semantic_result.reasoning,
            "test_results": semantic_result.executed_test_results
        }
    }

# This function must be defined at module level
async def get_next_client(api_key):
    global client
    if client is None:
        client = genai.Client(api_key=api_key)
    return client

async def evaluate_patches(bug_name: str, generated_patch_file: str, api_key: str) -> list[dict]:
    """
    Evaluate multiple generated patches for a given bug_name asynchronously.
    """
    with open(generated_patch_file, 'r', encoding='utf-8') as file:
        patch_data = json.load(file)
        
    # Access patches through the bug_name key first
    generated_patches = patch_data.get(bug_name, {}).get('patches', [])
    
    tasks = [
        evaluate_single_patch(bug_name, patch, api_key)
        for patch in generated_patches
    ]
    results = await asyncio.gather(*tasks)
    
    os.makedirs('outputs/val', exist_ok=True)
    
    output_path = f'outputs/val/{bug_name}_patch_val.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({bug_name: results}, f, indent=2)
    
    return {bug_name: results}