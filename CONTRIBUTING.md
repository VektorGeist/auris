# Contributing

## Setup

```bash
pip install .[dev]
```

Pillow is required for the image tests:

```bash
pip install .[images]
```

## Tests

```bash
python -m pytest tests/ -v
```

All tests must pass before a change is submitted.

## Pull requests

- One change per pull request.
- Include tests for new behavior.
- Update CHANGELOG.md under Unreleased.
