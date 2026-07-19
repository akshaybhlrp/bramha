# Development Principles & Anti-Patterns

This document outlines four core principles for coding on this project, with real-world examples showing what to avoid (Anti-Patterns) and what to do instead.

## 0. Architecture Principles (Non-Negotiable)

| Principle | Rule |
|-----------|------|
| **Gate Discipline** | No phase begins until the previous phase's gate is passed. If a gate fails, execute the fallback immediately. |
| **Rust-Only** | Rust only — no Python, no C++ runtimes in core engine. Build tooling may use Python, but end-user binary must not require Python for any operation. `convert.py` is [DEPRECATED]; the canonical tool is `spanda-convert`. |
| **No Retries** | Retries are jitter. All fallbacks are path switches, not re-attempts. |
| **Banker Mode** | When in doubt, ship the conservative option (static 2:4 sparse) that works. |
| **P99 Bound** | Latency must never exceed dense baseline +15% at the 99th percentile. |
| **Mandatory Testing** | Every new functionality, down to the smallest module, must be accompanied by comprehensive unit/integration test cases. Testing is a non-negotiable part of the definition of done. |


---

## 1. Think Before Coding

**Example 1: Hidden Assumptions**
User Request: "Add a feature to export user data"

❌ **What LLMs Do (Wrong Assumptions)**
```python
def export_users(format='json'):
    """Export all users to a file."""
    users = User.query.all()
    
    if format == 'json':
        with open('users.json', 'w') as f:
            json.dump([u.to_dict() for u in users], f)
    elif format == 'csv':
        with open('users.csv', 'w') as f:
            writer = csv.DictWriter(f, fieldnames=['id', 'email', 'name'])
            writer.writeheader()
            writer.writerows([u.to_dict() for u in users])
    
    return f"Exported {len(users)} users"
```
**Problems:**
- Assumed it should export ALL users (what about pagination? privacy?)
- Assumed file location without asking
- Assumed which fields to include
- Assumed CSV fieldnames without checking actual data structure

✅ **What Should Happen (Surface Assumptions)**
Before implementing, clarify:
1. **Scope**: Export all users or filtered subset? (privacy implications)
2. **Format**: Download file in browser? Background job? API endpoint?
3. **Fields**: Which user fields? (some might be sensitive)
4. **Volume**: How many users typically? (affects approach)
*Simplest approach*: Add an API endpoint that returns paginated JSON. Ask the user for their preference.

**Example 2: Multiple Interpretations**
User Request: "Make the search faster"

❌ **What LLMs Do (Pick Silently)**
```python
# Adds caching, database indexes, and async processing
# without asking which "faster" matters
@lru_cache(maxsize=1000)
async def search(query: str) -> List[Result]:
    # ... 200 lines of optimization code
```

✅ **What Should Happen**
"Make the search faster" could mean:
1. **Faster response time** (< 100ms instead of 500ms)
2. **Handle more concurrent searches** (throughput)
3. **Faster perceived speed** (UX)
*Clarify which aspect matters most for the use case before writing code.*

---

## 2. Simplicity First

**Example 1: Over-abstraction**
User Request: "Add a function to calculate discount"

❌ **What LLMs Do (Overengineered)**
```python
from abc import ABC, abstractmethod
from dataclasses import dataclass

class DiscountStrategy(ABC):
    @abstractmethod
    def calculate(self, amount: float) -> float:
        pass

class PercentageDiscount(DiscountStrategy):
    def __init__(self, percentage: float):
        self.percentage = percentage
    
    def calculate(self, amount: float) -> float:
        return amount * (self.percentage / 100)

# (plus fixed discount, config, calculator class... 30+ lines)
```

✅ **What Should Happen (Simple)**
```python
def calculate_discount(amount: float, percent: float) -> float:
    """Calculate discount amount. percent should be 0-100."""
    return amount * (percent / 100)

# Usage
discount = calculate_discount(100.0, 10.0)  # $10 off
```
*When to add complexity*: Only when you actually need multiple discount types. Refactor then.

**Example 2: Speculative Features**
User Request: "Save user preferences to database"

❌ **What LLMs Do (Unnecessary Features)**
Implements caching, validation, merging, and notification features that weren't asked for.

✅ **What Should Happen (Just What Was Asked)**
```python
def save_preferences(db, user_id: int, preferences: dict):
    """Save user preferences to database."""
    db.execute(
        "UPDATE users SET preferences = ? WHERE id = ?",
        (json.dumps(preferences), user_id)
    )
```
*Add later if needed*: Caching, validation, merging, etc.

---

## 3. Surgical Changes

**Example 1: Drive-by Refactoring**
User Request: "Fix the bug where empty emails crash the validator"

❌ **What LLMs Do (Too Much)**
Adds username validation, email format checking, and changes comments while fixing the bug.

✅ **What Should Happen (Surgical)**
```diff
  def validate_user(user_data):
      # Check email format
-     if not user_data.get('email'):
+     email = user_data.get('email', '')
+     if not email or not email.strip():
          raise ValueError("Email required")
```
Only change the specific lines that fix the issue.

**Example 2: Style Drift**
User Request: "Add logging to the upload function"

❌ **What LLMs Do (Reformat Everything)**
Changes quote styles, adds type hints, docstrings, and reformats whitespace.

✅ **What Should Happen (Match Existing Style)**
Only add the `logger.info()` and `logger.error()` lines while preserving single quotes, no type hints, existing boolean patterns, and spacing style.

---

## 4. Goal-Driven Execution

**Example 1: Vague vs. Verifiable**
User Request: "Fix the authentication system"

❌ **What LLMs Do (Vague Approach)**
"I'll review and improve the code" [Proceeds to make changes without clear success criteria]

✅ **What Should Happen (Verifiable Goals)**
Define success criteria first:
1. Write test for bug X
2. Verify test fails
3. Make it pass
4. Verify no regressions

**Example 2: Multi-Step with Verification**
User Request: "Add rate limiting to the API"

❌ **What LLMs Do (All at Once)**
Implements full rate limiting with Redis, config system, and monitoring in one huge commit.

✅ **What Should Happen (Incremental with Verification)**
1. Add basic in-memory rate limiting (verify).
2. Extract to middleware (verify).
3. Add Redis backend (verify).
4. Add configuration (verify).

**Example 3: Test-First Verification**
User Request: "The sorting breaks when there are duplicate scores"

❌ **What LLMs Do (Fix Without Reproducing)**
Immediately changes sort logic without confirming the bug.

✅ **What Should Happen (Reproduce First)**
1. Write a test that reproduces the issue.
2. Verify test fails with inconsistent ordering.
3. Fix the logic.
4. Verify test passes consistently.

---

## 5. Validation-First Engineering *(Inspired by DS4/DwarfStar)*

> Reference: [antirez/ds4](https://github.com/antirez/ds4) — DS4's engineering culture prioritizes validation tooling as first-class features, not afterthoughts.

**Principle**: Every inference change must be testable against known-good reference outputs before it ships.

**Key Practices**:
1. **Reference Logit Vectors**: Capture official API outputs for target models (exact logits at fixed seeds). Use these as regression oracles.
2. **Frontier-Based Benchmarking**: Measure performance at context frontiers (2K, 4K, 8K, 16K...) not single-number averages. Report prefill and generation rates separately.
4. **Deterministic Regression Gate**: Fixed seed + fixed prompt + greedy decode = exact token count match across builds.

**Anti-Pattern (DS4-Informed)**:
- ❌ Adding diagnostic tools only after bugs appear
- ❌ Reporting single averaged tok/s number for entire run
- ❌ Testing inference changes without reference logits
- ✅ Shipping diagnostic flags alongside the feature itself
- ✅ Measuring at multiple context sizes to profile scaling behavior
- ✅ Validating against captured official model outputs

---

## Anti-Patterns Summary

| Principle | Anti-Pattern | Fix |
|---|---|---|
| **Think Before Coding** | Silently assumes file format, fields, scope | List assumptions explicitly, ask for clarification |
| **Simplicity First** | Strategy pattern for single discount calculation | One function until complexity is actually needed |
| **Surgical Changes** | Reformats quotes, adds type hints while fixing bug | Only change lines that fix the reported issue |
| **Goal-Driven** | "I'll review and improve the code" | "Write test for bug X → make it pass → verify no regressions" |
| **Validation-First** | Adding `--trace` only after debugging is needed | Ship diagnostic flags alongside every inference change |

## Key Insight
Good code is code that solves today's problem simply, not tomorrow's problem prematurely. "Overcomplicated" code often follows design patterns but adds complexity before it's needed, making it harder to read, test, and maintain. Start simple and refactor when complexity is required.
