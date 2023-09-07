# Sandia Spatial Environmental Assessment Tool QGIS Plugin

This folder contains the documentation source code for SEAT. This README provides instructions on how to build, view, and contribute to the documentation.

## Prerequisites

Ensure you have the following installed:

- Python (>=3.6)
- pip
- Sphinx (>=3.0.0)
- sphinx-rtd-theme (>=0.4.3)

## Building the Documentation

1.Navigate to the docs directory:

```bash
cd docs
make html
```

or in Windows

```bash
./make.bat html
```

The generated documentation can be found in `docs/_build/html`.

## Viewing the Documentation Locally

After building the docs, you can view the documentation in your browser:

```bash
open docs/_build/html/index.html  # For macOS
# OR
xdg-open docs/_build/html/index.html  # For Linux
# OR
start docs/_build/html/index.html  # For Windows
```

## Contributing to the Documentation

1. Making Changes:

- Edit the .rst files in the docs/ directory.
- You can add new .rst files if you want to create new sections/pages.

2. Reviewing Changes Locally:
   After making changes, rebuild the documentation (as mentioned above) and view it in your browser to ensure your changes appear as expected.

3. Pushing Changes:
   Once you're satisfied with your changes, commit and push them to your fork branch, and create a pull request.
