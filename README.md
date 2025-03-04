# AOT-GRPO: Atom of Thoughts with Group Relative Policy Optimization

AoT enhances LLM performance by breaking down complex queries into smaller, independent "atomic questions" and organizing them into a directed acyclic graph (DAG) based on their dependencies. This Markov-like process minimizes reliance on lengthy historical context, reducing computational overhead and noise while improving efficiency and accuracy. For tasks requiring multi-step reasoning—such as multi-hop question answering (e.g., HotpotQA) or mathematical problem-solving—AoT’s structured approach is highly effective.

For broader applications where reasoning isn’t the primary focus—such as simple question answering, text generation, or tasks requiring minimal multi-step logic—GRPO’s versatility shines.

By refining prompts, it can improve overall model outputs across a wide range of use cases without needing the structural overhaul that AoT provides.

AoT and GRPO operate on different levels: AoT restructures the reasoning process, while GRPO optimizes the input prompts. 

This complementary nature suggests potential synergy.

For instance, GRPO could refine the phrasing of AoT’s atomic questions or improve how the model transitions between them, enhancing the overall efficiency and accuracy of the reasoning pipeline.
For Reasoning-Intensive Tasks AoT seems the primary approach. Its ability to decompose and manage complex reasoning makes it superior for tasks like multi-hop QA or problem-solving.
For General Improvement GRPO is the go-to choice if you need a versatile, task-agnostic boost in performance across diverse applications.

Combining AoT and GRPO could yield enhanced results, especially if GRPO is tailored to support AoT’s atomic reasoning framework. This hybrid approach could be particularly powerful for applications requiring both robust reasoning and optimized model responses.

This repository implements the combination of two powerful techniques for enhancing LLM performance:

1. **Atom of Thoughts (AoT)**: A technique that decomposes complex problems into simpler sub-problems, solving them individually and then combining their solutions.

2. **Group Relative Policy Optimization (GRPO)**: A test-time optimization technique that enhances model inputs by optimizing learnable tokens to improve the quality of generated outputs.

## Installation

```bash
# Clone the repository
git clone https://github.com/EmaMazzi76/aot_grpo.git
cd aot_grpo

# Install dependencies
pip install -r requirements.txt
```

## Features

- Question decomposition via AoT
- Prompt optimization via GRPO
- Support for multiple dataset formats (GSM8K, MMLU, HotpotQA, etc.)
- Support for different reasoning types (math, multi-choice, multi-hop QA)
- Handling of both independent and dependent sub-questions
- Comprehensive logging and error handling
- Memory-efficient mode for running on consumer hardware

## Usage

### Running with Local Models

The easiest way to run a test with a local model (such as Microsoft Phi-mini) is to use the provided script:

```bash
./run_test.sh [MODEL_PATH] [DATASET] [NUM_SAMPLES] [OPT_STEPS] [USE_CPU]
```

For example:
```bash
# Run with default settings (1 GSM8K sample, 5 optimization steps)
./run_test.sh

# Run with custom settings
./run_test.sh ~/models/phi-mini bbh 2 10 true
```

### Advanced Usage

For more control, you can run the Python script directly with additional arguments:

```bash
python aot_grpo.py --model_path ~/models/phi-mini --dataset gsm8k --samples 3 --opt_steps 10 --no_dependent --low_memory
```

Available arguments:
- `--model_path`: Path to your local model directory
- `--dataset`: Dataset to use (gsm8k, math, bbh, mmlu, hotpotqa, longbench)
- `--samples`: Number of samples to process
- `--opt_steps`: Number of optimization steps for GRPO
- `--no_dependent`: Skip processing dependent sub-questions (faster)
- `--low_memory`: Enable memory-efficient mode for large models

## Project Structure

- `aot.py`: Implementation of Atom of Thoughts
- `grpo.py`: Implementation of Group Relative Policy Optimization
- `dataset.py`: Utilities for loading and processing datasets
- `aot_grpo.py`: Integration of AoT and GRPO approaches
- `run_test.sh`: Helper script for running tests

## Requirements

- Python 3.8+
- PyTorch 1.13+
- Transformers 4.26+
- Datasets library
- A local model (e.g., Microsoft Phi Mini) or internet access for downloading models

## Acknowledgements

This project stands on the shoulders of giants and builds upon several groundbreaking works:

### Atom of Thoughts (AoT)
Implementation based on the research by Fengwei Teng et al. in their paper "Atom of Thoughts for Markov LLM Test-Time Scaling":

```bibtex
@article{teng2024atom,
  title={Atom of Thoughts for Markov LLM Test-Time Scaling},
  author={Teng, Fengwei and Yu, Zhaoyang and Shi, Quan and Zhang, Jiayi and Wu, Chenglin and Luo, Yuyu},
  journal={arXiv preprint arXiv:2502.12018},
  year={2025}
}
```

### Group Relative Policy Optimization (GRPO)
The concept of optimizing prompts through gradient-based methods is inspired by various techniques in the field, with conceptual acknowledgement to Andriy Burkov, author of "The Hundred-Page Machine Learning Book," for his explanations of related optimization techniques.

### Models and Tools
- **Microsoft Phi**: Special thanks to the Microsoft Phi team for creating the Phi models that make this research accessible on consumer hardware
- **Hugging Face**: For their transformers library and datasets that serve as the foundation for this implementation
- **Apple MLX Team**: For their exceptional MLX framework that enables efficient ML computations on Apple Silicon, providing key optimizations for running these models on macOS
- **xAI's Grok**: For assistance during the initial brainstorming and conceptualization phase of this project
- **Anthropic**: For their Claude AI assistant, which provided guidance and assistance throughout the development process
- **Claude Code**: For helping refactor, debug, and optimize the codebase with advanced AI capabilities

## Contributing

Contributions to improve the implementation or extend functionality are welcome! Please open an issue or submit a pull request.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{aot-grpo,
  author = {Emanuele Mazzitelli},
  title = {AOT-GRPO: Atom of Thoughts with Group Relative Policy Optimization},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/EmaMazzi76/aot_grpo}}
}
```

## Acknowledgements

Special thanks to:
- **Anthropic**: For their Claude AI assistant, which provided guidance in refactoring and improving this codebase
- **Claude Code**: For technical assistance in developing, debugging, and optimizing the implementation

## License

MIT License

Copyright (c) 2025 Emanuele Mazzitelli

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.