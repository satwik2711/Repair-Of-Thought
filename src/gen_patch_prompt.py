apr_label = '// Provide a fix for the buggy function.'
buggy_label = '// Buggy Function'
fixed_label = '// Fixed Function'

def sf_build_apr_prompt_auto(buggy_function, root_cause, suggestion):
    parts = []
    parts.append(apr_label)
    parts.append('Root cause: ' + root_cause)
    parts.append('Suggestion: ' + suggestion)
    parts.append(buggy_label)
    parts.append(buggy_function)
    parts.append(fixed_label)
    return '\n'.join(parts)

def build_function_level_repair_prompt(buggy_function, root_cause, suggestion):
    """
    Builds a prompt specifically designed for function-level repair, focusing on
    complete function replacement rather than statement-level patching.
    
    Args:
        buggy_function: The complete buggy function code
        root_cause: Identified root cause of the bug
        suggestion: Suggestion for fixing the bug
        
    Returns:
        A formatted prompt string for function-level repair
    """
    prompt_template = '''You are a Senior Level Programmer with expertise in debugging and code repair. Your task is to review the provided function and produce a corrected, function-level implementation that fully addresses the identified bug.

@@ Context
{apr_prompt}

@@ Instructions
1. Carefully analyze the provided root cause and fix suggestion.
2. Ensure your implementation completely resolves the identified issue.
3. Maintain the original method signature.
4. Preserve the original functionality and coding style as much as possible.
5. Incorporate all working parts of the original implementation.
6. Follow the provided fix suggestion without over-engineering or introducing unnecessary changes.

@@ Response Format
Return ONLY the complete fixed function wrapped in ```java
<your implementation>
``` tags.

@@ Important Notes
- Do not add unnecessary optimizations.
- Do not modify the method signature.
- Maintain the original variable names and overall coding style wherever possible.'''
    
    # Build the basic APR prompt first
    apr_prompt = sf_build_apr_prompt_auto(buggy_function, root_cause, suggestion)
    
    # Format the full function-level repair prompt
    return prompt_template.format(apr_prompt=apr_prompt.strip())