# Contributing to code-not-text

Thank you for your interest in contributing to this project! This repository contains research code and artifacts for a workshop paper on cross-domain limits of hand-crafted CoT-surface features.

## Development Setup

This repository uses Python 3.8+ and requires several dependencies for running the analysis scripts.

```bash
# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies (if requirements.txt exists)
pip install -r requirements.txt
```

## Project Structure

- `paper/`: LaTeX source for the workshop paper
- `scripts/`: Analysis and figure generation scripts
- `figures/`: Generated figures for the paper
- `results/`: Experimental results and tables
- `temp/`: Temporary files and drafts (not tracked)

## Code Style

This repository follows standard Python conventions (PEP 8). When contributing new code:

1. Use clear, descriptive variable names
2. Add docstrings to functions
3. Keep functions focused and modular
4. Include comments for non-obvious logic

## Submitting Changes

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Make your changes and commit them with clear messages
4. Push to your fork and submit a pull request

## Research Integrity

This repository contains research artifacts. When contributing:

- Ensure reproducibility of any new experiments
- Document data sources and preprocessing steps
- Include version information for dependencies
- Update relevant documentation

## Questions?

For questions about the research or code, please open an issue or contact the maintainer at yhchi25@m.fudan.edu.cn.
