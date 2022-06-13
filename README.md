| **Authors**  | **Project** |  **Build Status** | **License** | **Code Quality** | **Coverage** |
|:------------:|:-----------:|:-----------------:|:-----------:|:----------------:|:------------:|
|**R. Biondi** |**IPT**      | **Windows** : [![Windows CI](https://github.com/RiccardoBiondi/ImageProcessingTools/workflows/Windows%20CI/badge.svg)](https://github.com/RiccardoBiondi/ImageProcessingTools/actions/workflows/windows.yml)    <br/> **Ubuntu** : [![Ubuntu CI](https://github.com/RiccardoBiondi/ImageProcessingTools/workflows/Ubuntu%20CI/badge.svg)](https://github.com/RiccardoBiondi/ImageProcessingTools/actions/workflows/ubuntu.yml)            |      [![license](https://img.shields.io/github/license/mashape/apistatus.svg)](https://github.com/RiccardoBiondi/ImageProcessingTools/blob/master/LICENSE.md)       |                  |              |

Put here all the required badges

# Image Processing Tools

A more or less organized collection of script and functions of medical image
analysis tools.

**What is IPT**

IPT is a repository that collects all those function, scripts, etc that I am
implementing during my journey in medical image analysis

**What is not IPT**

IPT is not an organic library for image processing like ITK, PIL or OpenCV.

1. [Contents](#Contents)
2. [Prerequisites](#Prerequisites)
3. [Installation](#Installation)
4. [Usage](#Usage)
5. [Contribute](#Contribute)
6. [License](#License)
7. [Authors](#Authors)
8. [References](#References)
9. [Acknowledgments](#Acknowledgments)
10. [Citation](#Citation)


## Contents

| **Module Name**| **Description**|
|:--------------:|:--------------:|
| io             | Functions to read and write medical images |
| itk_wrapping   | Functional Wrapping of itk filters |
| decorators     | collection of useful decoratos |
| visualization  | inline plotting and rendering of medical image volumes|


## Prerequisites

Supported python versions: ![Python version](https://img.shields.io/badge/python-3.5.*|3.6.*|3.7.*|3.8.*|3.9.*-blue.svg)

To run the tests you need to install ```PyTest``` and ```Hypothesis```.
Installation instructions are available at: [PyTest](https://docs.pytest.org/en/6.2.x/getting-started.html), [Hypothesis](https://docs.pytest.org/en/6.2.x/getting-started.html)


## Installation

Download the project or the latest release:

```console
git clone https://github.com/RiccardoBiondi/segmentation
```

Now build and activate the conda environment

```console
conda env create -f environment.yaml
conda env activate ipt
```

Or, if you are using `pip`, install the required packages:

```console
python -m pip install -r requirements.txt
```

Now you are ready to build the package:

```console
python setup.py develop --user
```

### Testing

We have provide a test routine in [test](./test) directory. This routine use:
  - pytest >= 3.0.7

  - hypothesis >= 4.13.0

Please install these packages to perform the test.
You can run the full set of test with:

```console
  python -m pytest
```

## Usage

## Contribute

## License

Any contribution is more than welcome. Just fill an [issue]() or a [pull request]() and we will check ASAP!

See [here]() for further information about how to contribute with this project.

## Authors

* **Riccardo Biondi** [git](https://github.com/RiccardoBiondi), [unibo](https://www.unibo.it/sitoweb/riccardo.biondi7)


## References

## Acknowledgments


## Citation

If you have found `Image Processing Tools` helpful in your research, please consider citing the project

```tex
@misc{IPT_2022,
  author = {Biondi, Riccardo},
  title = {Image Processing Tools},
  year = {2022},
  publisher = {GitHub},
  howpublished = {\url{}},
}
```
