# PyKokoro Documentation

This directory contains the Sphinx documentation for PyKokoro.

## Building the Documentation

### Prerequisites

Install documentation dependencies:

```bash
pip install -r requirements.txt
```

### Build HTML Documentation

**On Linux/macOS:**

```bash
make html
```

**On Windows:**

```bash
python make.py html
```

**Or using make.bat on Windows:**

```bash
make.bat html
```

The generated HTML documentation will be in `_build/html/`.

### View Documentation

Open `_build/html/index.html` in your browser.

### Live Reload (Development)

For automatic rebuilding during development:

```bash
make livehtml
```

This starts a local server at `http://127.0.0.1:8000` with auto-reload.

### Clean Build

Remove generated files:

```bash
make clean
```

## Documentation Structure

* `index.rst` - Main documentation index
* `quickstart.rst` - Quick start guide for new users
* `installation.rst` - Installation instructions
* `basic_usage.rst` - Basic usage guide
* `advanced_features.rst` - Advanced features and techniques
* `examples.rst` - Practical examples
* `api_reference.rst` - Complete API reference
* `changelog.rst` - Version history and changes
* `conf.py` - Sphinx configuration

## Publishing to Read the Docs

The documentation is configured to work with Read the Docs automatically:

1. Push changes to GitHub
2. Read the Docs will automatically build from the `docs/` directory
3. No additional configuration needed

## Documentation Style Guide

* Use reStructuredText (.rst) format
* Follow NumPy docstring style for API documentation
* Include code examples for all features
* Keep line length to ~80-100 characters
* Use proper Sphinx directives (.. code-block::, .. note::, etc.)

## Updating Documentation

1. Edit the relevant .rst files
2. Build locally to test: `make html`
3. Review changes in browser
4. Commit and push to repository

## Troubleshooting

**Build errors:**

```bash
make clean
make html
```

**Missing dependencies:**

```bash
pip install -r requirements.txt
```

**Auto-documentation not working:**

Ensure PyKokoro is installed:

```bash
cd ..
pip install -e .
```
