# Contributing

## Setup

```bash
pip install .[dev]
```

Pillow is required for the image tests:

```bash
pip install pillow
```

## Tests

```bash
python -m pytest tests/ -v
```

The suite contains 45 tests. All tests must pass before a change is
submitted.

## Pull requests

- One change per pull request.
- Include tests for new behavior.
- Update CHANGELOG.md under Unreleased.
