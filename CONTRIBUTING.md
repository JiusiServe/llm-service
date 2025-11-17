# Contributing to LM-Service

Thank you for your interest in contributing to LM-Service! This document
provides guidelines and information for contributors.

## Code of Conduct

By participating in this project, you agree to abide by our Code of
Conduct. Please be respectful and professional in all interactions.

## Getting Started

### Prerequisites

- Python 3.10 or higher
- [uv](https://docs.astral.sh/uv/) package manager
- Git

### Development Setup

1. Fork the repository
2. Clone your fork:

   ```bash
   git clone https://github.com/yourusername/LM-Service.git
   cd LM-Service
   ```

3. Install dependencies:

   ```bash
   uv sync --extra develop
   ```

4. Install pre-commit hooks:

   ```bash
   uv run pre-commit install
   ```

## Development Workflow

### Making Changes

1. Create a new branch for your feature or bugfix:

   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes following the coding standards below

3. Run tests and linting:

   ```bash
   uv run pre-commit run --all-files
   uv run pytest
   ```

4. Commit your changes with a signed-off commit:

   ```bash
   git commit -s -m "Your commit message"
   ```

   The `-s` flag adds a "Signed-off-by" line to your commit, which is
   required for this project (Developer Certificate of Origin).

5. Push to your fork and create a pull request

### Coding Standards

- Follow PEP 8 style guidelines (enforced by ruff)
- Use type hints for all function parameters and return values
- Write docstrings for all public functions and classes
- Keep line length to 80 characters (enforced by ruff)
- Use regex instead of re module (enforced by pre-commit hook)

### Code Quality Tools

We use several tools to maintain code quality:

- **ruff**: Linting and code formatting
- **mypy**: Static type checking
- **pytest**: Testing framework
- **pre-commit**: Git hooks for code quality checks

All tools are configured in `pyproject.toml` and `.pre-commit-config.yaml`.

### Testing

- Write tests for new functionality
- Ensure all tests pass before submitting a PR
- Aim for good test coverage
- Use descriptive test names and docstrings

Run tests with:

```bash
uv run pytest
```

### Documentation

- Update documentation for any new features or API changes
- Add docstrings to new functions and classes
- Update the README if needed
- Check documentation builds correctly

## Pull Request Process

1. Ensure your PR description clearly describes the problem and solution
2. Include relevant issue numbers if applicable
3. Make sure all status checks pass
4. Request review from maintainers
5. Address any feedback promptly

### PR Requirements

- [ ] All tests pass
- [ ] Code follows style guidelines
- [ ] Documentation updated if needed
- [ ] Commits are signed off (DCO)
- [ ] PR template is filled out completely

## Issue Guidelines

When creating issues:

- Use the appropriate issue template
- Provide clear, detailed descriptions
- Include reproduction steps for bugs
- Add relevant labels
- Search for existing issues first

## Security

If you discover a security vulnerability, please report it privately through
GitHub's security advisory feature rather than creating a public issue.

## License

By contributing to this project, you agree that your contributions will be
licensed under the same license as the project.

## Questions?

If you have questions about contributing, feel free to:

- Open a discussion on GitHub
- Ask in the issue comments
- Contact the maintainers

Thank you for contributing to LM-Service!
