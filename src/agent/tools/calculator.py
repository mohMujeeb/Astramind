from typing import Dict, Any
import math
import re
import numexpr as ne

# After preprocessing, only these characters should remain (numexpr also accepts sqrt())
ALLOWED_AFTER_PRE = re.compile(r"^[0-9\.\s\+\-\*\/\(\)]+$")

def _expand_factorials(s: str) -> str:
    """
    Replace all occurrences of N! with the computed integer value.
    Repeats until no '!' remains (handles multiple factorials).
    """
    pat = re.compile(r"(\d+)\s*!")
    prev = None

    def repl(m):
        n = int(m.group(1))
        return str(math.factorial(n))

    while prev != s:
        prev = s
        s = pat.sub(repl, s)
    return s

def _preprocess(expr: str) -> str:
    s = (expr or "").strip()

    # Normalize commas and unicode sqrt
    s = s.replace(",", "")
    s = s.replace("√", "sqrt")

    # "square root of N" -> sqrt(N)
    s = re.sub(r"\bsquare\s*root\s*of\s*(\d+)\b", r"sqrt(\1)", s, flags=re.I)

    # caret exponent: 2^10 -> 2**10
    s = re.sub(r"(\d+)\s*\^\s*(\d+)", r"\1**\2", s)

    # Expand all factorials N!
    s = _expand_factorials(s)

    return s.strip()

def _is_pure_math(expr: str) -> bool:
    """
    Validate that the expression is purely mathematical after preprocessing.
    Allow digits, ., whitespace, + - * / ( ) and sqrt().
    """
    tmp = expr
    # Remove 'sqrt(' token for validation only (numexpr supports sqrt at eval time)
    tmp = re.sub(r"sqrt\s*\(", "(", tmp, flags=re.I)
    # If any letters remain, it's not pure math
    if re.search(r"[A-Za-z]", tmp):
        return False
    return bool(ALLOWED_AFTER_PRE.fullmatch(tmp))

def calculate(expression: str) -> Dict[str, Any]:
    """
    Safely evaluate a numeric expression using numexpr with preprocessing.
    Supports: + - * / **, parentheses, sqrt(), integer factorial N! (expanded), and caret ^.
    Rejects sentences or non-math text.
    """
    try:
        expr = _preprocess(expression)
        if not _is_pure_math(expr):
            return {"error": "Not a pure math expression."}
        res = ne.evaluate(expr)
        if hasattr(res, "item"):
            res = res.item()
        return {"result": float(res)}
    except Exception as e:
        return {"error": str(e)}
