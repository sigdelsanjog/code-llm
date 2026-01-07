# MedLLM Project Structure

```
medllm/
├── data/
│   ├── parsers/              # XML parsing logic (SRP)
│   │   ├── __init__.py
│   │   ├── medquad_parser.py # Parse MedQuAD XML
│   │   └── text_formatter.py # Format Q&A to causal text
│   ├── processed/            # Cleaned text files
│   └── tokenized/            # Tokenized datasets
│
├── tokenizer/                # Tokenizer training (Phase 1)
│   └── configs/              # Tokenizer configurations
│
├── model/                    # Model definition (Phase 2)
│   ├── architecture/         # Transformer components
│   ├── configs/              # Model configurations
│   └── checkpoints/          # Saved model weights
│
├── training/                 # Training logic (Phase 3)
│   ├── trainer.py            # Training loop
│   ├── loss.py               # Loss functions
│   └── schedulers.py         # LR schedulers
│
├── inference/                # Generation (Phase 4)
│   ├── generator.py          # Text generation
│   └── decoding.py           # Decoding strategies
│
├── evaluation/               # Metrics and evaluation
│   └── metrics.py
│
├── utils/                    # Shared utilities
│   ├── logging.py
│   └── visualization.py
│
├── configs/                  # Global configurations
│   └── config.yaml
│
├── dataset/                  # Raw data
│   └── MedQuAD/              # Downloaded dataset
│
├── preprocess.py             # Main preprocessing script
├── download_medquad.py       # Dataset download script
└── prompts.md                # Your learning roadmap
```

## Design Principles Applied

**Single Responsibility Principle (SRP)**:

- `medquad_parser.py`: Only parses XML → structured data
- `text_formatter.py`: Only formats text (doesn't parse)
- `preprocess.py`: Orchestrates the pipeline

**Open-Closed Principle**:

- Easy to add new formatters without changing existing code
- Easy to add new parsers for different datasets

**Dependency Inversion**:

- Components work with abstractions (QAPair dataclass)
- Parser doesn't know about formatters
- Formatters don't know about parsers

## Phase Mapping

- **Phase 1**: `data/`, `tokenizer/`, `preprocess.py`
- **Phase 2**: `model/architecture/`
- **Phase 3**: `training/`
- **Phase 4**: `inference/`
- **Phase 5**: Optimize components across folders
- **Phase 6-7**: Extend with new data/training strategies
