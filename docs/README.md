# Sandia Spatial Environmental Assessment Toolkit QGIS Plugin

This folder contains the documentation source code for SEAT. This README provides instructions on how to build, view, and contribute to the documentation.

## Prerequisites

Please see requirements.txt for a list of required packages. You can install them with:

```bash
pip install -r requirements.txt
```

## Building the Documentation

1.Navigate to the docs directory:

```bash
cd docs
make clean
make html
```

or in Windows

```bash
./make.bat clean && ./make.bat html
```

## Viewing the Documentation Locally

The generated documentation can be found in `docs/_build/html`.

To view in your browser open `docs/_build/html/index.html`.

## Contributing to the Documentation

1. Making Changes:

- Edit the .rst files in the docs/src directory.

2. Reviewing Changes Locally:
   After making changes, rebuild the documentation (as mentioned above) and view it in your browser to ensure your changes appear as expected.

3. Pushing Changes:
   Once you're satisfied with your changes, commit and push the commit to your fork feature branch, and create a pull request.
