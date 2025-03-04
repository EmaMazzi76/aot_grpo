# AOT-GRPO Development Guide

## Environment Setup
```
pip install -r requirements.txt
```

## Running Code
- Run AOT decomposition: `python aot.py`
- Run GRPO optimization: `python grpo.py`
- Run integrated solution: `python aot_grpo.py`
- Load dataset samples: `python dataset.py`

## Code Style Guidelines
- **Imports**: Standard library first, then third-party, then local modules
- **Typing**: Use type annotations for function parameters and return values
- **Naming**: 
  - Functions: snake_case
  - Variables: snake_case
  - Constants: UPPER_CASE
- **Documentation**: Use docstrings with Args/Returns sections for complex functions
- **Error Handling**: Use ValueError for invalid inputs, implement retry patterns for operations that may fail

## Project Structure
- `aot.py`: Algorithm of Thoughts implementation for question decomposition
- `grpo.py`: Generative Representational Prompt Optimization
- `aot_grpo.py`: Integration of AOT and GRPO approaches
- `dataset.py`: Dataset handling utilities

## Common Patterns
- Leverage async/await for concurrent operations
- Use torch device detection for cross-platform GPU compatibility
- Employ functional programming where appropriate