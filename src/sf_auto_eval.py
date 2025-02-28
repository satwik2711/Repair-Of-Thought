import json
import asyncio
import os
import re
from google import genai
from google.genai import types
from dotenv import load_dotenv
from pydantic import BaseModel
import datetime
from typing import Dict, Any, Optional, Tuple
import javalang

load_dotenv()

# Global client variable
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
    Enhanced validator that now uses AST-based, symbolic execution, 
    and LLM-based semantic equivalence testing. IO-based validation
    has been removed.
    """
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        
    async def validate_patch(self, bug_name: str, generated_patch: str, ground_truth: str, 
                             test_cases: Dict[str, Any], dataset: Dict[str, Any]) -> SemanticEquivalenceResult:
        # Run each validation method and handle failures gracefully
        try:
            ast_result = await self.ast_based_validation(generated_patch, ground_truth)
        except Exception as e:
            ast_result = (0.5, 0.3, f"AST validation exception: {str(e)}")
        
        try:
            symbolic_result = await self.symbolic_execution_validation(generated_patch, ground_truth)
        except Exception as e:
            symbolic_result = (0.5, 0.3, f"Symbolic execution exception: {str(e)}")
        
        try:
            llm_result = await self.llm_based_validation(bug_name, generated_patch, ground_truth)
        except Exception as e:
            llm_result = (0.5, 0.3, f"LLM validation exception: {str(e)}")
        
        w_ast = 0.333
        w_sym = 0.333
        w_llm = 0.333
        
        combined_equivalent = ((ast_result[0] * w_ast) + 
                               (symbolic_result[0] * w_sym) + 
                               (llm_result[0] * w_llm)) >= 0.5
        
        combined_confidence = (ast_result[1] * w_ast) + (symbolic_result[1] * w_sym) + (llm_result[1] * w_llm)
        
        reasoning = f"""
AST-based validation: {ast_result[2]}
Symbolic execution validation: {symbolic_result[2]}
LLM-based validation: {llm_result[2]}

Combined assessment: The patch is {"semantically equivalent" if combined_equivalent else "not semantically equivalent"} 
to the ground truth with {combined_confidence:.2f} confidence.
"""
        
        return SemanticEquivalenceResult(
            is_equivalent=combined_equivalent,
            confidence=combined_confidence,
            reasoning=reasoning,
            executed_test_results=None
        )

    async def ast_based_validation(self, generated_patch: str, ground_truth: str) -> Tuple[float, float, str]:
        """
        Compares the AST structures of the generated patch and ground truth.
        Returns a tuple: (similarity score, confidence, reasoning).
        """
        try:
            gen_has_class = "class" in generated_patch.lower()
            truth_has_class = "class" in ground_truth.lower()
            
            try:
                if gen_has_class and truth_has_class:
                    generated_tree = javalang.parse.parse(generated_patch)
                    ground_truth_tree = javalang.parse.parse(ground_truth)
                    parsing_approach = "full class parsing"
                else:
                    gen_tokens = list(javalang.tokenizer.tokenize(generated_patch))
                    truth_tokens = list(javalang.tokenizer.tokenize(ground_truth))
                    
                    total_tokens = max(len(gen_tokens), len(truth_tokens))
                    if total_tokens == 0:
                        return 0.5, 0.5, "AST validation: Empty tokens found"
                    
                    matches = 0
                    for i in range(min(len(gen_tokens), len(truth_tokens))):
                        if gen_tokens[i].value == truth_tokens[i].value:
                            matches += 1
                    
                    token_similarity = matches / total_tokens
                    
                    gen_methods = re.findall(r'(\w+\s+\w+\s*\([^)]*\))', generated_patch)
                    truth_methods = re.findall(r'(\w+\s+\w+\s*\([^)]*\))', ground_truth)
                    
                    method_similarity = 0.0
                    if gen_methods and truth_methods:
                        method_matches = sum(1 for gm in gen_methods for tm in truth_methods if gm == tm)
                        method_similarity = method_matches / max(len(gen_methods), len(truth_methods))
                    
                    gen_vars = re.findall(r'(\w+\s+\w+\s*=)', generated_patch)
                    truth_vars = re.findall(r'(\w+\s+\w+\s*=)', ground_truth)
                    
                    var_similarity = 0.0
                    if gen_vars and truth_vars:
                        var_matches = sum(1 for gv in gen_vars for tv in truth_vars if gv == tv)
                        var_similarity = var_matches / max(len(gen_vars), len(truth_vars))
                    
                    is_equivalent = 0.5 * token_similarity + 0.3 * method_similarity + 0.2 * var_similarity
                    confidence = 0.6
                    reasoning = (f"AST validation using token analysis: Token similarity: {token_similarity:.2f}, "
                                f"Method signature similarity: {method_similarity:.2f}, "
                                f"Variable usage similarity: {var_similarity:.2f}. "
                                f"Overall structure similarity score: {is_equivalent:.2f}")
                    
                    return is_equivalent, confidence, reasoning
                    
            except javalang.parser.JavaSyntaxError:
                try:
                    wrap_code = lambda code: f"class DummyClass {{ {code} }}"
                    generated_tree = javalang.parse.parse(wrap_code(generated_patch))
                    ground_truth_tree = javalang.parse.parse(wrap_code(ground_truth))
                    parsing_approach = "wrapped method parsing"
                except:
                    import difflib
                    similarity = difflib.SequenceMatcher(None, generated_patch.strip(), ground_truth.strip()).ratio()
                    return similarity, 0.5, f"AST parsing failed, using text similarity: {similarity:.2f}"
            
            similarity_score = self._compare_ast_structure(generated_tree, ground_truth_tree)
            confidence = 0.8 if similarity_score > 0.8 else 0.6
            reasoning = (f"AST analysis using {parsing_approach}: Structural similarity score: {similarity_score:.2f}. " +
                         ("The abstract syntax trees show high structural similarity." 
                          if similarity_score > 0.7 else "The abstract syntax trees show significant structural differences."))
            
            return similarity_score, confidence, reasoning
        except Exception as e:
            import difflib
            try:
                similarity = difflib.SequenceMatcher(None, generated_patch.strip(), ground_truth.strip()).ratio()
                return similarity, 0.4, f"AST validation failed with error: {str(e)}. Using text similarity: {similarity:.2f}"
            except:
                return 0.5, 0.3, f"AST validation failed completely: {str(e)}"
    
    def _compare_ast_structure(self, tree1, tree2) -> float:
        """
        Compare two parsed Java ASTs and return a similarity score.
        """
        try:
            methods1 = list(tree1.filter(javalang.tree.MethodDeclaration))
            methods2 = list(tree2.filter(javalang.tree.MethodDeclaration))
            
            if not methods1 or not methods2:
                classes1 = list(tree1.filter(javalang.tree.ClassDeclaration))
                classes2 = list(tree2.filter(javalang.tree.ClassDeclaration))
                if not classes1 or not classes2:
                    return 0.5
                class_similarities = []
                for c1 in classes1:
                    for c2 in classes2:
                        fields1 = list(c1.filter(javalang.tree.FieldDeclaration))
                        fields2 = list(c2.filter(javalang.tree.FieldDeclaration))
                        field_similarity = 0.0
                        if fields1 and fields2:
                            field_similarity = min(len(fields1), len(fields2)) / max(len(fields1), len(fields2))
                        class_similarities.append(field_similarity)
                return max(class_similarities) if class_similarities else 0.5
            
            method_similarities = []
            for m1 in methods1:
                best_match = 0.0
                for m2 in methods2:
                    if m1.name == m2.name:
                        similarity = 0.4
                        if str(m1.return_type) == str(m2.return_type):
                            similarity += 0.1
                        param_similarity = 0.0
                        if len(m1.parameters) == len(m2.parameters):
                            param_matches = sum(1 for p1, p2 in zip(m1.parameters, m2.parameters) 
                                                if str(p1.type) == str(p2.type))
                            if param_matches > 0:
                                param_similarity = param_matches / len(m1.parameters)
                        similarity += param_similarity * 0.2
                        body_similarity = 0.0
                        if hasattr(m1, 'body') and hasattr(m2, 'body') and m1.body and m2.body:
                            stmts1 = list(m1.body)
                            stmts2 = list(m2.body)
                            if stmts1 and stmts2:
                                stmt_count_sim = min(len(stmts1), len(stmts2)) / max(len(stmts1), len(stmts2))
                                stmt_types1 = [type(s).__name__ for s in stmts1]
                                stmt_types2 = [type(s).__name__ for s in stmts2]
                                set1 = set(stmt_types1)
                                set2 = set(stmt_types2)
                                if set1 or set2:
                                    jaccard = len(set1.intersection(set2)) / len(set1.union(set2))
                                    body_similarity = (stmt_count_sim + jaccard) / 2
                                else:
                                    body_similarity = stmt_count_sim
                        similarity += body_similarity * 0.3
                        if similarity > best_match:
                            best_match = similarity
                    else:
                        similarity = 0.1
                        if len(m1.parameters) == len(m2.parameters):
                            similarity += 0.1
                        if similarity > best_match:
                            best_match = similarity
                method_similarities.append(best_match)
            if method_similarities:
                return sum(method_similarities) / len(method_similarities)
            return 0.5
        except Exception as e:
            return 0.5

    async def symbolic_execution_validation(self, generated_patch: str, ground_truth: str) -> Tuple[float, float, str]:
        """
        Uses a fallback symbolic analysis based on control flow constructs
        to compare the behavior of both patches.
        Returns a tuple: (equivalence score, confidence, reasoning).
        """
        return self._fallback_symbolic_analysis(generated_patch, ground_truth)


    def _create_wrapper_class(self, patch_code: str, class_name: str) -> str:
        """
        Wrap the patch code in a class if it isn't already a class.
        """
        if "class" not in patch_code.lower():
            return f"""
                public class {class_name} {{
                    // Wrapped patch code
                    {patch_code}
                }}
        """
        else:
            # If already a class, rename it
            return patch_code.replace("class", f"class {class_name}", 1)

    def _create_comparator_class(self) -> str:
        """
        Create a Java comparator class for symbolic execution.
        """
        return """
import gov.nasa.jpf.vm.Verify;

public class PatchComparator {
    public static void main(String[] args) {
        int symbolicInt = Verify.getInt(0, 100);
        boolean symbolicBool = Verify.getBoolean();
        
        GeneratedPatch genPatch = new GeneratedPatch();
        GroundTruth gtPatch = new GroundTruth();
        
        Object genResult = genPatch.method(symbolicInt, symbolicBool);
        Object gtResult = gtPatch.method(symbolicInt, symbolicBool);
        
        boolean equivalent = (genResult == null && gtResult == null) ||
                              (genResult != null && genResult.equals(gtResult));
        
        assert equivalent : "Patches produce different results for same input";
    }
}
"""

    def _create_jpf_config(self, target_class: str) -> str:
        """
        Create configuration file for Java PathFinder.
        """
        return f"""
target = {target_class}
classpath = .

# Enable symbolic execution for main method
symbolic.method = {target_class}.main(symbolic)
listener = gov.nasa.jpf.listener.AssertionProperty
"""

    def _extract_violation_details(self, jpf_output: str) -> str:
        """
        Extract details of property violation from JPF output.
        """
        lines = jpf_output.split('\n')
        for i, line in enumerate(lines):
            if "AssertionError" in line and i+1 < len(lines):
                return lines[i+1].strip()
        return "Behavioral difference detected but details not available"

    def _analyze_path_coverage(self, jpf_output: str) -> Dict[str, Any]:
        """
        Analyze JPF output to extract path coverage and compute similarity.
        """
        path_count = 0
        for line in jpf_output.split('\n'):
            if "paths explored" in line.lower():
                try:
                    path_count = int(line.split(':')[1].strip())
                except:
                    pass
        return {
            "explored_paths": path_count,
            "path_similarity": 0.95 if path_count > 0 else 0.5,
            "equivalence_score": 0.9 if path_count > 0 else 0.5,
            "confidence": 0.8 if path_count > 10 else 0.6
        }

    def _fallback_symbolic_analysis(self, generated_patch: str, ground_truth: str) -> Tuple[float, float, str]:
        """
        Fallback symbolic analysis using control flow constructs.
        """
        def count_constructs(code):
            return {
                "if": len(re.findall(r'\bif\s*\(', code)),
                "else": len(re.findall(r'\belse\b', code)),
                "for": len(re.findall(r'\bfor\s*\(', code)),
                "while": len(re.findall(r'\bwhile\s*\(', code)),
                "return": len(re.findall(r'\breturn\b', code)),
                "try": len(re.findall(r'\btry\b', code)),
                "catch": len(re.findall(r'\bcatch\b', code))
            }
        
        gen_counts = count_constructs(generated_patch)
        gt_counts = count_constructs(ground_truth)
        
        total_constructs = sum(max(gen_counts[k], gt_counts[k]) for k in gen_counts.keys())
        if total_constructs == 0:
            similarity = 0.5
        else:
            differences = sum(abs(gen_counts[k] - gt_counts[k]) for k in gen_counts.keys())
            similarity = 1.0 - (differences / (2 * total_constructs))
        
        confidence = 0.5
        reasoning = f"Fallback symbolic analysis: Control flow similarity: {similarity:.2f}"
        return similarity, confidence, reasoning

    async def llm_based_validation(self, bug_name: str, generated_patch: str, ground_truth: str) -> Tuple[float, float, str]:
        """
        Uses an LLM to assess semantic equivalence between patches.
        """
        try:
            prompt = f"""You are an expert at evaluating semantic equivalence between program patches.

Compare these two Java patches and determine if they are semantically equivalent (i.e., produce the same outputs for all valid inputs and have the same side effects).

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
1. A score between 0 and 1 indicating semantic equivalence (1.0 = perfectly equivalent)
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
            
            response_text = response.text
            equivalence_score = 0.5
            confidence = 0.5
            equiv_match = re.search(r"Equivalence Score:\s*(\d+\.\d+)", response_text)
            conf_match = re.search(r"Confidence:\s*(\d+\.\d+)", response_text)
            
            if equiv_match:
                equivalence_score = float(equiv_match.group(1))
            if conf_match:
                confidence = float(conf_match.group(1))
            
            reasoning_parts = response_text.split("Reasoning:", 1)
            reasoning = reasoning_parts[1].strip() if len(reasoning_parts) > 1 else "Detailed reasoning not provided."
            
            return equivalence_score, confidence, f"LLM assessed equivalence: {equivalence_score} with confidence {confidence}. {reasoning[:200]}..."
                
        except Exception as e:
            return 0.5, 0.4, f"LLM-based validation failed: {str(e)}"

# Integration with existing code
async def evaluate_single_patch(bug_name: str, generated_patch: str, api_key: str) -> dict:
    client = await get_next_client(api_key)
    validator = EnhancedPatchValidator(api_key)
    
    file_path = r"./datasets/defects4j-sf.json"
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    test_cases = data[bug_name]['trigger_test']
    ground_truth = data[bug_name]['fix']

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

    semantic_result = await validator.validate_patch(
        bug_name, 
        generated_patch, 
        ground_truth,
        test_cases,
        data[bug_name]
    )

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
- Consider AST, symbolic execution, and LLM-based validation results
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

    response = await client.aio.models.generate_content(
        model='gemini-2.0-flash-thinking-exp-1219',
        contents=types.Part(text=prompt),
        config=generation_config
    )

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

async def get_next_client(api_key):
    global client
    if client is None:
        client = genai.Client(api_key=api_key)
    return client

async def evaluate_patches(bug_name: str, generated_patch_file: str, api_key: str) -> list[dict]:
    with open(generated_patch_file, 'r', encoding='utf-8') as file:
        patch_data = json.load(file)
        
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
