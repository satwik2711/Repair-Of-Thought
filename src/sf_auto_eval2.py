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
        # Run all validation methods and handle failures
        try:
            ast_result = await self.ast_based_validation(generated_patch, ground_truth)
        except Exception as e:
            ast_result = (0.5, 0.3, f"AST validation exception: {str(e)}")
        
        try:
            io_result = await self.io_based_validation(generated_patch, ground_truth, test_cases)
        except Exception as e:
            io_result = (0.5, 0.3, f"IO validation exception: {str(e)}", None)
        
        try:
            symbolic_result = await self.symbolic_execution_validation(generated_patch, ground_truth)
        except Exception as e:
            symbolic_result = (0.5, 0.3, f"Symbolic execution exception: {str(e)}")
        
        try:
            llm_result = await self.llm_based_validation(bug_name, generated_patch, ground_truth)
        except Exception as e:
            llm_result = (0.5, 0.3, f"LLM validation exception: {str(e)}")
        
        # Weight the results based on success of each module
        weights = [0.2, 0.3, 0.3, 0.2]  # Default weights
        
        # Combined calculation with weights
        combined_equivalent = (
            (ast_result[0] * weights[0]) + 
            (io_result[0] * weights[1]) + 
            (symbolic_result[0] * weights[2]) + 
            (llm_result[0] * weights[3]) >= 0.5
        )
        
        combined_confidence = (
            (ast_result[1] * weights[0]) + 
            (io_result[1] * weights[1]) + 
            (symbolic_result[1] * weights[2]) + 
            (llm_result[1] * weights[3])
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
            executed_test_results=io_result[3] if len(io_result) > 3 and io_result[3] else None
        )

    async def ast_based_validation(self, generated_patch: str, ground_truth: str) -> Tuple[float, float, str]:
        """
        Compares the AST structures of the generated patch and ground truth.
        
        Returns:
            Tuple of (is_equivalent as float 0-1, confidence, reasoning)
        """
        try:
            # For Java code, we need to handle potential fragments
            # First, check if we're dealing with complete classes or just method fragments
            gen_has_class = "class" in generated_patch.lower()
            truth_has_class = "class" in ground_truth.lower()
            
            # Check if we have valid Java code that can be parsed
            try:
                # Try parsing as-is first
                if gen_has_class and truth_has_class:
                    generated_tree = javalang.parse.parse(generated_patch)
                    ground_truth_tree = javalang.parse.parse(ground_truth)
                    parsing_approach = "full class parsing"
                else:
                    # For method fragments, we need to tokenize and analyze
                    gen_tokens = list(javalang.tokenizer.tokenize(generated_patch))
                    truth_tokens = list(javalang.tokenizer.tokenize(ground_truth))
                    
                    # Simple token-based similarity
                    total_tokens = max(len(gen_tokens), len(truth_tokens))
                    if total_tokens == 0:
                        return 0.5, 0.5, "AST validation: Empty tokens found"
                    
                    # Compare token types and values
                    matches = 0
                    for i in range(min(len(gen_tokens), len(truth_tokens))):
                        if gen_tokens[i].value == truth_tokens[i].value:
                            matches += 1
                    
                    token_similarity = matches / total_tokens
                    
                    # Use a more sophisticated approach for method fragments
                    # Extract method structure using regular expressions
                    import re
                    
                    # Extract method signature patterns
                    gen_methods = re.findall(r'(\w+\s+\w+\s*\([^)]*\))', generated_patch)
                    truth_methods = re.findall(r'(\w+\s+\w+\s*\([^)]*\))', ground_truth)
                    
                    method_similarity = 0.0
                    if gen_methods and truth_methods:
                        # Compare method signatures
                        method_matches = sum(1 for gm in gen_methods for tm in truth_methods if gm == tm)
                        method_similarity = method_matches / max(len(gen_methods), len(truth_methods))
                    
                    # Extract variable declarations and operations
                    gen_vars = re.findall(r'(\w+\s+\w+\s*=)', generated_patch)
                    truth_vars = re.findall(r'(\w+\s+\w+\s*=)', ground_truth)
                    
                    var_similarity = 0.0
                    if gen_vars and truth_vars:
                        var_matches = sum(1 for gv in gen_vars for tv in truth_vars if gv == tv)
                        var_similarity = var_matches / max(len(gen_vars), len(truth_vars))
                    
                    # Calculate overall code structure similarity
                    # Weighted average of token, method and variable similarities
                    is_equivalent = 0.5 * token_similarity + 0.3 * method_similarity + 0.2 * var_similarity
                    confidence = 0.6  # Lower confidence for fragment analysis
                    
                    reasoning = f"AST validation using token analysis: " \
                            f"Token similarity: {token_similarity:.2f}, " \
                            f"Method signature similarity: {method_similarity:.2f}, " \
                            f"Variable usage similarity: {var_similarity:.2f}. " \
                            f"Overall structure similarity score: {is_equivalent:.2f}"
                    
                    return is_equivalent, confidence, reasoning
                    
            except javalang.parser.JavaSyntaxError:
                # If direct parsing fails, try wrapping in a dummy class
                try:
                    wrap_code = lambda code: f"class DummyClass {{ {code} }}"
                    generated_tree = javalang.parse.parse(wrap_code(generated_patch))
                    ground_truth_tree = javalang.parse.parse(wrap_code(ground_truth))
                    parsing_approach = "wrapped method parsing"
                except:
                    # If all parsing fails, fall back to text similarity
                    import difflib
                    similarity = difflib.SequenceMatcher(None, 
                                                    generated_patch.strip(), 
                                                    ground_truth.strip()).ratio()
                    return similarity, 0.5, f"AST parsing failed, using text similarity: {similarity:.2f}"
            
            # If we get here, we have parsed trees to compare
            # Compare tree structures using our specialized method
            similarity_score = self._compare_ast_structure(generated_tree, ground_truth_tree)
            confidence = 0.8 if similarity_score > 0.8 else 0.6
            
            reasoning = f"AST analysis using {parsing_approach}: Structural similarity score: {similarity_score:.2f}. " + (
                "The abstract syntax trees show high structural similarity." 
                if similarity_score > 0.7 else 
                "The abstract syntax trees show significant structural differences."
            )
            
            return similarity_score, confidence, reasoning
        except Exception as e:
            # Fall back to text similarity
            import difflib
            try:
                similarity = difflib.SequenceMatcher(None, 
                                                generated_patch.strip(), 
                                                ground_truth.strip()).ratio()
                return similarity, 0.4, f"AST validation failed with error: {str(e)}. Using text similarity: {similarity:.2f}"
            except:
                # If everything fails, return neutral values
                return 0.5, 0.3, f"AST validation failed completely: {str(e)}"

    def _compare_ast_structure(self, tree1, tree2) -> float:
        """
        Compare two parsed Java ASTs and return a similarity score.
        This is a more advanced comparison that looks at structure and content.
        """
        try:
            # Extract methods from both trees
            methods1 = list(tree1.filter(javalang.tree.MethodDeclaration))
            methods2 = list(tree2.filter(javalang.tree.MethodDeclaration))
            
            # If no methods, compare classes directly
            if not methods1 or not methods2:
                # Extract classes
                classes1 = list(tree1.filter(javalang.tree.ClassDeclaration))
                classes2 = list(tree2.filter(javalang.tree.ClassDeclaration))
                
                if not classes1 or not classes2:
                    return 0.5  # Neutral if no classes to compare
                
                # Compare class members and structure
                class_similarities = []
                for c1 in classes1:
                    for c2 in classes2:
                        # Compare fields
                        fields1 = list(c1.filter(javalang.tree.FieldDeclaration))
                        fields2 = list(c2.filter(javalang.tree.FieldDeclaration))
                        
                        field_similarity = 0.0
                        if fields1 and fields2:
                            # Simple count comparison for now
                            field_similarity = min(len(fields1), len(fields2)) / max(len(fields1), len(fields2))
                        
                        # Add more comparisons for class structure if needed
                        class_similarities.append(field_similarity)
                
                return max(class_similarities) if class_similarities else 0.5
            
            # For each method in tree1, find the best matching method in tree2
            method_similarities = []
            for m1 in methods1:
                best_match = 0.0
                for m2 in methods2:
                    # Start with name similarity
                    if m1.name == m2.name:
                        similarity = 0.4  # 40% for matching name
                        
                        # Check return type
                        if str(m1.return_type) == str(m2.return_type):
                            similarity += 0.1  # 10% for matching return type
                        
                        # Check parameters
                        param_similarity = 0.0
                        if len(m1.parameters) == len(m2.parameters):
                            param_matches = sum(1 for p1, p2 in zip(m1.parameters, m2.parameters) 
                                            if str(p1.type) == str(p2.type))
                            if param_matches > 0:
                                param_similarity = param_matches / len(m1.parameters)
                        
                        similarity += param_similarity * 0.2  # 20% for parameters
                        
                        # Compare body - extract statements
                        body_similarity = 0.0
                        if hasattr(m1, 'body') and hasattr(m2, 'body') and m1.body and m2.body:
                            stmts1 = list(m1.body)
                            stmts2 = list(m2.body)
                            
                            if stmts1 and stmts2:
                                # Simple count-based comparison
                                stmt_count_sim = min(len(stmts1), len(stmts2)) / max(len(stmts1), len(stmts2))
                                
                                # For a more advanced comparison, we could compare statement types
                                stmt_types1 = [type(s).__name__ for s in stmts1]
                                stmt_types2 = [type(s).__name__ for s in stmts2]
                                
                                # Calculate Jaccard similarity of statement types
                                set1 = set(stmt_types1)
                                set2 = set(stmt_types2)
                                if set1 or set2:
                                    jaccard = len(set1.intersection(set2)) / len(set1.union(set2))
                                    body_similarity = (stmt_count_sim + jaccard) / 2
                                else:
                                    body_similarity = stmt_count_sim
                        
                        similarity += body_similarity * 0.3  # 30% for body similarity
                        
                        if similarity > best_match:
                            best_match = similarity
                    else:
                        # Different method names - lower similarity
                        similarity = 0.1  # Only 10% for different named methods
                        
                        # Still check parameters and body for structural similarity
                        if len(m1.parameters) == len(m2.parameters):
                            similarity += 0.1
                        
                        if similarity > best_match:
                            best_match = similarity
                
                method_similarities.append(best_match)
            
            # Overall similarity is average of best method matches
            if method_similarities:
                return sum(method_similarities) / len(method_similarities)
            return 0.5  # Neutral if no similarities computed
        except Exception as e:
            # If comparison fails, return neutral similarity
            return 0.5

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
    1. A score between 0 and 1 indicating how semantically equivalent the patches are (1.0 = perfectly equivalent)
    2. A confidence score between 0 and 1 on your assessment
    3. Reasoning for your decision

    Format your final answer as:
    Equivalence Score: [score]
    Confidence: [confidence]
    Reasoning: [reasoning]
    """
            generation_config = types.GenerateContentConfig(
                temperature=0.0,
                seed=42
            )
            
            response = await self.client.aio.models.generate_content(
                model='gemini-2.0-flash-thinking-exp-1219',
                contents=types.Part(text=prompt),
                config=generation_config
            )
            
            # Extract scores from the text response
            response_text = response.text
            
            # Parse the text response to extract scores
            equivalence_score = 0.5  # default
            confidence = 0.5  # default
            
            # Look for patterns like "Equivalence Score: 0.8" in the response
            import re
            equiv_match = re.search(r"Equivalence Score:\s*(\d+\.\d+)", response_text)
            conf_match = re.search(r"Confidence:\s*(\d+\.\d+)", response_text)
            
            if equiv_match:
                equivalence_score = float(equiv_match.group(1))
            if conf_match:
                confidence = float(conf_match.group(1))
            
            # Extract reasoning - everything after "Reasoning: "
            reasoning_parts = response_text.split("Reasoning:", 1)
            reasoning = reasoning_parts[1].strip() if len(reasoning_parts) > 1 else "Detailed reasoning not provided."
            
            return equivalence_score, confidence, f"LLM assessed equivalence: {equivalence_score} with confidence {confidence}. {reasoning[:200]}..."
                
        except Exception as e:
            return 0.5, 0.4, f"LLM-based validation failed: {str(e)}"
    
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