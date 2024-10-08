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

