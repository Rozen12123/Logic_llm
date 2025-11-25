import re
from typing import List, Optional


def clean_logic_program(raw_program: str, dataset_name: str) -> str:
    """
    Remove narration/markdown noise that language models sometimes append
    to the generated logic programs. When nothing can be salvaged, the
    original text is returned so upstream code can continue to handle the
    failure case.
    """
    if not isinstance(raw_program, str):
        return raw_program

    if dataset_name == 'FOLIO':
        cleaned = _clean_folio_program(raw_program)
        if cleaned:
            raw_program = cleaned

    if dataset_name == 'AR-LSAT':
        raw_program = _normalize_intsort_ranges(raw_program)

    return raw_program


def _clean_folio_program(raw_program: str) -> Optional[str]:
    text = raw_program.strip()
    if not text:
        return None

    lower_text = text.lower()
    if 'premises:' not in lower_text:
        return None
    if 'conclusion:' not in lower_text and 'question:' not in lower_text:
        return None

    premises_split = re.split(r'(?i)premises\s*:\s*', text, maxsplit=1)
    if len(premises_split) != 2:
        return None

    premises_and_tail = premises_split[1]

    # Prefer explicit Conclusion blocks; otherwise fall back to the last Question block.
    conclusion_match = re.search(r'(?i)conclusion\s*:\s*', premises_and_tail)
    if conclusion_match:
        premises_block = premises_and_tail[:conclusion_match.start()]
        conclusion_block = premises_and_tail[conclusion_match.end():]
    else:
        question_matches = list(re.finditer(r'(?i)question\s*:\s*', premises_and_tail))
        if not question_matches:
            return None
        last_question = question_matches[-1]
        premises_block = premises_and_tail[:last_question.start()]
        conclusion_block = premises_and_tail[last_question.end():]

    premises = _extract_clause_lines(premises_block)
    conclusion_lines = _extract_clause_lines(conclusion_block)
    if not conclusion_lines:
        inferred_conclusion = _infer_conclusion_clause(conclusion_block)
        conclusion_lines = [inferred_conclusion] if inferred_conclusion else []

    if not premises or not conclusion_lines:
        return None

    sanitized = 'Premises:\n'
    sanitized += '\n'.join(premises)
    sanitized += '\nConclusion:\n'
    sanitized += conclusion_lines[0]
    return sanitized


def _normalize_intsort_ranges(raw_program: str) -> str:
    """
    Some AR-LSAT generations emit bare `IntSort()` declarations.
    Without explicit domains the downstream Z3 translator cannot
    unroll quantifiers such as Count/ForAll that iterate over the
    corresponding scope. We attempt to infer a contiguous integer
    range for each bare IntSort by scanning constraints for bounds
    on variables typed with that sort (e.g. `ForAll([l:lockers], l>=1, l<=5)`).
    When both a minimum and maximum can be recovered we rewrite the
    declaration as `IntSort([min, ..., max])`.
    """
    intsort_pattern = re.compile(
        r'^(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*IntSort\(\s*\)\s*$',
        re.MULTILINE
    )

    def infer_domain(sort_name: str) -> Optional[List[int]]:
        scope_var_pattern = re.compile(rf'\[([A-Za-z0-9_]+):{sort_name}\]')
        scoped_vars = {m.group(1) for m in scope_var_pattern.finditer(raw_program)}
        if not scoped_vars:
            return None

        numbers = set()
        for var in scoped_vars:
            forward = re.compile(rf'\b{var}\b\s*(==|>=|<=|>|<)\s*(-?\d+)')
            reverse = re.compile(rf'(-?\d+)\s*(==|>=|<=|>|<)\s*\b{var}\b')
            for match in forward.finditer(raw_program):
                numbers.add(int(match.group(2)))
            for match in reverse.finditer(raw_program):
                numbers.add(int(match.group(1)))

        if not numbers:
            return None

        lo, hi = min(numbers), max(numbers)
        if hi < lo or hi - lo > 20:
            return None

        return list(range(lo, hi + 1))

    def replace_decl(match: re.Match) -> str:
        sort_name = match.group('name')
        domain = infer_domain(sort_name)
        if not domain:
            return match.group(0)
        domain_str = ", ".join(str(x) for x in domain)
        # Use EnumSort for numeric values instead of IntSort to avoid code generation issues
        return f"{sort_name} = EnumSort([{domain_str}])"

    result = re.sub(intsort_pattern, replace_decl, raw_program)
    
    # Also convert IntSort([...]) to EnumSort([...]) if it exists
    intsort_with_list_pattern = re.compile(
        r'^(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*IntSort\(\s*\[(?P<values>[^\]]+)\]\s*\)\s*$',
        re.MULTILINE
    )
    
    def replace_intsort_with_list(match: re.Match) -> str:
        sort_name = match.group('name')
        values = match.group('values')
        # Use EnumSort for numeric values instead of IntSort
        return f"{sort_name} = EnumSort([{values}])"
    
    result = re.sub(intsort_with_list_pattern, replace_intsort_with_list, result)
    return result


def _extract_clause_lines(block: str) -> List[str]:
    clauses: List[str] = []
    for raw_line in block.splitlines():
        if ':::' not in raw_line:
            continue

        cleaned = _clean_clause_line(raw_line)
        if cleaned:
            clauses.append(cleaned)
    return clauses


def _clean_clause_line(line: str) -> Optional[str]:
    parts = line.split(':::', 1)
    if len(parts) != 2:
        return None

    formula, description = parts
    formula = _strip_leading_markup(formula)
    description = description.strip()

    if not formula or not description:
        return None

    normalized_formula = _normalize_formula(formula)
    return f'{normalized_formula} ::: {description}'


def _strip_leading_markup(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r'^\*{1,3}\s*', '', cleaned)  # markdown bullets
    cleaned = re.sub(r'^#+\s*', '', cleaned)  # markdown headings
    cleaned = re.sub(r'^\d+[\).:]\s*', '', cleaned)  # numbered lists
    cleaned = re.sub(r'^[ivxlcdm]+\.\s*', '', cleaned, flags=re.IGNORECASE)  # roman numerals
    cleaned = re.sub(r'^[\-\u2022•]+\s*', '', cleaned)  # bullet chars
    cleaned = re.sub(r'^\(?[a-z]\)\s*', '', cleaned, flags=re.IGNORECASE)  # lettered lists
    cleaned = re.sub(r'^Step\s*\d+:\s*', '', cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def _infer_conclusion_clause(block: str) -> Optional[str]:
    formula = _extract_formula_candidate(block)
    if not formula:
        return None

    description = _infer_conclusion_description(block, formula)
    if not description:
        description = 'Conclusion inferred from text.'

    return f'{formula} ::: {description}'


def _extract_formula_candidate(block: str) -> Optional[str]:
    # Prefer inline code fragments that usually highlight formulas
    inline_matches = re.findall(r'`([^`]+)`', block)
    for match in inline_matches:
        candidate = _normalize_formula(match)
        if _looks_like_formula(candidate):
            return candidate

    lines = [_strip_leading_markup(line).strip() for line in block.splitlines()]
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        if not line:
            idx += 1
            continue

        if _line_looks_like_expression(line):
            candidate, next_idx = _collect_expression_from_lines(lines, idx)
            if candidate and _looks_like_formula(candidate):
                return candidate
            idx = next_idx
            continue

        candidate = _extract_simple_formula_from_line(line)
        if candidate:
            return candidate

        idx += 1

    return None


def _collect_expression_from_lines(lines: List[str], start_idx: int) -> (Optional[str], int):
    parts: List[str] = []
    idx = start_idx
    while idx < len(lines):
        segment = lines[idx]
        if idx != start_idx and (not segment or not _line_can_continue_expression(segment)):
            break
        parts.append(segment)
        idx += 1
    candidate = _normalize_formula(' '.join(parts)) if parts else None
    return candidate, idx


def _extract_simple_formula_from_line(line: str) -> Optional[str]:
    if not line:
        return None

    predicate_match = re.search(r'[A-Za-z][A-Za-z0-9_]*\s*\([^()]*\)', line)
    if predicate_match:
        candidate = _normalize_formula(predicate_match.group(0))
        if _looks_like_formula(candidate):
            return candidate

    quantifier_match = re.search(r'[∃∀][^.:]+', line)
    if quantifier_match:
        candidate = _normalize_formula(quantifier_match.group(0))
        if _looks_like_formula(candidate):
            return candidate

    return None


def _infer_conclusion_description(block: str, formula: str) -> str:
    for line in block.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if formula in stripped:
            stripped = stripped.replace(formula, '').strip()
        normalized = re.sub(r'[*`_#>\-]+', '', stripped).strip()
        if normalized:
            return normalized.rstrip('.')
    return ''


def _normalize_formula(text: str) -> str:
    cleaned = text.strip().strip('.')
    if '(' in cleaned and ')' in cleaned:
        cleaned = _flatten_nested_terms(cleaned)
    return cleaned


def _flatten_nested_terms(formula: str) -> str:
    """
    Replace nested functional arguments such as Foo(Bar(x)) with constants so
    downstream parsers that only support predicate-style atoms can still handle
    the clauses.
    """
    replacements = []
    stack: List[tuple[int, int, bool]] = []
    for idx, ch in enumerate(formula):
        if ch == '(':
            symbol_start, symbol = _find_symbol(formula, idx)
            stack.append((idx, symbol_start, bool(symbol)))
        elif ch == ')':
            if not stack:
                continue
            open_idx, symbol_start, has_symbol = stack.pop()
            if has_symbol and any(parent_has_symbol for _, _, parent_has_symbol in stack):
                segment_start = symbol_start
                segment_end = idx + 1
                segment_text = formula[segment_start:segment_end]
                inner_start = segment_text.find('(') + 1
                inner_content = segment_text[inner_start:-1] if inner_start > 0 else ''
                if '(' not in inner_content:
                    continue
                placeholder = _make_placeholder(segment_text, len(replacements))
                replacements.append((segment_start, segment_end, placeholder))
    if not replacements:
        return formula
    # Apply replacements from right to left so recorded indices remain valid.
    normalized = formula
    for start, end, placeholder in sorted(replacements, key=lambda x: x[0], reverse=True):
        normalized = normalized[:start] + placeholder + normalized[end:]
    return normalized


def _find_symbol(text: str, paren_idx: int) -> tuple[int, str]:
    j = paren_idx - 1
    if j < 0 or text[j].isspace():
        return paren_idx, ''
    end = j + 1
    while j >= 0 and (text[j].isalnum() or text[j] == '_'):
        j -= 1
    start = j + 1
    symbol = text[start:end]
    if not symbol:
        return paren_idx, ''
    return start, symbol


def _make_placeholder(segment: str, idx: int) -> str:
    token = re.sub(r'[^A-Za-z0-9_]+', '_', segment).strip('_')
    if not token:
        token = f'NESTED_{idx}'
    return f'{token.upper()}_{idx}'


def _looks_like_formula(text: str) -> bool:
    if not text:
        return False
    if '(' in text and ')' in text:
        return True
    for token in ['∀', '∃', '¬', '→', '↔', '⊕']:
        if token in text:
            return True
    return False


def _line_looks_like_expression(text: str) -> bool:
    if not text:
        return False
    if text[0] in '([¬∀∃':
        return True
    logical_tokens = ['→', '∧', '∨', '⊕', '↔']
    return any(tok in text for tok in logical_tokens)


def _line_can_continue_expression(text: str) -> bool:
    if not text:
        return False
    if _line_looks_like_expression(text):
        return True
    return text[0] in ')]' or text.startswith(('→', '∨', '∧', '⊕', '¬'))

