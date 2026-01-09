# Using llm-med Package in Backend

## Overview

Your backend now uses the `llm-med` PyPI package instead of local imports from the medllm directory. This makes your code cleaner and easier to maintain.

## Package Information

- **Package Name (PyPI)**: `llm-med`
- **Import Name (Python)**: `llm_med` (with underscore)
- **Version**: 0.1.0+

## Installation

### For Local Development (Before Publishing to PyPI)

Since the package isn't published to PyPI yet, install it locally:

```bash
# Option 1: Install in editable mode from medllm directory
cd /path/to/code-llm/medllm
pip install -e .

# Option 2: Build and install the wheel
cd /path/to/code-llm/medllm
python -m build
pip install dist/llm_med-0.1.0-py3-none-any.whl
```

### After Publishing to PyPI

Once you publish to PyPI, simply:

```bash
pip install llm-med
```

## Updated Backend Service

The [`medquad_custom_service.py`](medquad_custom_service.py) now imports from the `llm-med` package:

```python
# OLD (local imports)
from model.architecture import GPTTransformer
from model.configs.model_config import ModelConfig

# NEW (package imports)
from llm_med.model.architecture import GPTTransformer
from llm_med.model.configs.model_config import ModelConfig
from llm_med.inference.generator import TextGenerator
```

## Key Changes

### 1. Imports

All model-related imports now come from `llm_med` package:

- `llm_med.model.architecture.GPTTransformer`
- `llm_med.model.configs.model_config.ModelConfig`
- `llm_med.inference.generator.TextGenerator`

### 2. Generator Usage

Uses `TextGenerator` from the package instead of custom implementation:

```python
# Load model and tokenizer
model = GPTTransformer(model_config)
tokenizer = spm.SentencePieceProcessor()

# Create generator
self._generator = TextGenerator(
    model=model,
    tokenizer=tokenizer,
    device=self._device
)

# Generate text
response = self._generator.generate(
    prompt="What causes diabetes?",
    max_length=150,
    temperature=0.6
)
```

### 3. Dependencies

Updated [`requirements.txt`](requirements.txt) to include:

```
llm-med>=0.1.0
sentencepiece>=0.1.99
```

## Testing the Integration

### 1. Install Dependencies

```bash
cd /path/to/code-llm/backend

# Install llm-med package locally first
cd ../medllm
pip install -e .

# Then install other backend requirements
cd ../backend
pip install -r requirements.txt
```

### 2. Verify Imports

```bash
python -c "from llm_med.model.architecture import GPTTransformer; print('✓ Import successful')"
python -c "from llm_med.inference.generator import TextGenerator; print('✓ Import successful')"
```

### 3. Start Backend Server

```bash
cd /path/to/code-llm/backend
uvicorn main:app --reload
```

### 4. Test API

```bash
# Test the medquad-custom model endpoint
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{"model": "MedQuAD-Custom", "prompt": "What causes diabetes?"}'
```

## Troubleshooting

### Import Error: No module named 'llm_med'

**Solution:** Install the package locally:

```bash
cd /path/to/code-llm/medllm
pip install -e .
```

### Model Files Not Found

**Solution:** Ensure your model files are in the correct location:

```
backend/models/medquad-custom/
├── best_model.pt
├── medquad_tokenizer.model
└── medquad_tokenizer.vocab
```

### Package Version Mismatch

**Solution:** Reinstall the package:

```bash
cd /path/to/code-llm/medllm
pip uninstall llm-med -y
pip install -e .
```

## Directory Structure

```
code-llm/
├── backend/
│   ├── main.py
│   ├── config.py
│   ├── requirements.txt  # Now includes llm-med
│   ├── services/
│   │   ├── medquad_custom_service.py  # Uses llm-med package
│   │   └── ...
│   └── models/
│       └── medquad-custom/
│           ├── best_model.pt
│           └── medquad_tokenizer.model
│
└── medllm/  # Will become separate repo
    ├── setup.py
    ├── pyproject.toml
    ├── model/
    ├── inference/
    └── ...
```

## Migration Benefits

✅ **Cleaner Code**: No more sys.path hacks or relative imports  
✅ **Version Control**: Explicit version dependencies  
✅ **Reusability**: Same package across multiple projects  
✅ **Distribution**: Easy to share and deploy  
✅ **Updates**: Simple `pip install --upgrade llm-med`

## Next Steps

1. **Local Testing**: Test thoroughly with local installation
2. **Publish to Test PyPI**: Verify package works when installed from PyPI
3. **Publish to PyPI**: Make it available to everyone
4. **Update Backend**: In production, just run `pip install llm-med`

## Future Usage in Other Projects

Once published to PyPI, you can use llm-med in any Python project:

```python
from llm_med.model.architecture import GPTTransformer
from llm_med.model.configs.model_config import get_small_config
from llm_med.inference.generator import TextGenerator

# Create model
config = get_small_config()
model = GPTTransformer(config)

# Generate text
generator = TextGenerator(model, tokenizer)
response = generator.generate("What is hypertension?")
```

---

**Status**: ✅ Backend updated to use llm-med package  
**Next**: Test locally, then publish to PyPI
