# Package Structure Fix for llm-med v0.2.0

## Current Issue (v0.1.0)

The package was published incorrectly. Modules are installed directly in site-packages instead of under a `llm_med` namespace:

```
site-packages/
├── model/           # ❌ Wrong - should be llm_med/model/
├── inference/       # ❌ Wrong - should be llm_med/inference/
├── configs/         # ❌ Wrong - should be llm_med/configs/
└── ...
```

This means imports like `from llm_med.model.architecture import GPTTransformer` fail.

## Current Workaround (v0.1.0)

Backend uses direct imports:

```python
from model.architecture import GPTTransformer  # Works
from llm_med.model.architecture import GPTTransformer  # Fails
```

## Root Cause

In `pyproject.toml`, the packages are listed directly:

```toml
[tool.setuptools]
packages = ["model", "inference", "training", "utils", "configs", "data", "tokenizer"]
```

## Fix for v0.2.0

### Option 1: Use find_packages() in setup.py

```python
from setuptools import setup, find_packages

setup(
    name="llm-med",
    packages=find_packages(),  # This auto-detects all packages
    ...
)
```

### Option 2: Create proper package structure

Restructure the code:

```
medllm/
├── llm_med/              # Top-level package directory
│   ├── __init__.py
│   ├── model/
│   │   ├── __init__.py
│   │   └── architecture/
│   ├── inference/
│   ├── configs/
│   └── ...
├── setup.py
└── pyproject.toml
```

### Option 3: Use package_dir in setup.py

```python
setup(
    name="llm-med",
    packages=find_packages(),
    package_dir={'llm_med': '.'},  # Map llm_med to current directory
    ...
)
```

## Recommended Solution

For v0.2.0, restructure with proper namespace:

1. Create `llm_med/` directory in medllm repo
2. Move all modules inside it:

   ```bash
   mkdir llm_med
   mv model inference training utils configs data tokenizer llm_med/
   ```

3. Update `pyproject.toml`:

   ```toml
   [tool.setuptools]
   packages = ["llm_med", "llm_med.model", "llm_med.inference", ...]
   # Or use find_packages()
   ```

4. Then imports will work correctly:
   ```python
   from llm_med.model.architecture import GPTTransformer  # ✓ Works
   ```

## Migration Plan

1. **v0.1.0** (current): Use direct imports (`from model.architecture import ...`)
2. **v0.2.0** (next): Fix package structure, use namespaced imports
3. Update backend when v0.2.0 is published

## Files to Update for v0.2.0

In medllm repository:

- [ ] Restructure directories (create llm_med/ folder)
- [ ] Update setup.py packages list
- [ ] Update pyproject.toml
- [ ] Test local install: `pip install -e .`
- [ ] Verify import: `python -c "from llm_med.model.architecture import GPTTransformer"`
- [ ] Update version to 0.2.0
- [ ] Republish to PyPI

In code-llm backend:

- [ ] Update imports back to `from llm_med.model...`
- [ ] Update requirements.txt to `llm-med>=0.2.0`
- [ ] Test backend service

---

**Status**: v0.1.0 works with workaround (direct imports)  
**Next**: Fix structure for v0.2.0
