| **Authors**  | **Project** |  **Build Status** | **License** | **Code Quality** | **Documentation** |
|:------------:|:-----------:|:-----------------:|:-----------:|:----------------:|:---:|
|**R. Biondi** |**IPT**      | **Windows** : [![Windows CI](https://github.com/RiccardoBiondi/ImageProcessingTools/workflows/Windows%20CI/badge.svg)](https://github.com/RiccardoBiondi/ImageProcessingTools/actions/workflows/windows.yml)    <br/> **Ubuntu** : [![Ubuntu CI](https://github.com/RiccardoBiondi/ImageProcessingTools/workflows/Ubuntu%20CI/badge.svg)](https://github.com/RiccardoBiondi/ImageProcessingTools/actions/workflows/ubuntu.yml)            |      [![license](https://img.shields.io/github/license/mashape/apistatus.svg)](https://github.com/RiccardoBiondi/ImageProcessingTools/blob/master/LICENSE.md)       |  **codebeat** [![codebeat badge](https://codebeat.co/badges/6021933b-ccad-4811-b7a4-cf6924956ea7)](https://codebeat.co/projects/github-com-riccardobiondi-imageprocessingtools-master)         <br> **codacy** [![Codacy Badge](https://app.codacy.com/project/badge/Grade/e5f17dafa6654034b605f67f6c8dfce9)](https://www.codacy.com/gh/RiccardoBiondi/ImageProcessingTools/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=RiccardoBiondi/ImageProcessingTools&amp;utm_campaign=Badge_Grade)     | [![Documentation Status](https://readthedocs.org/projects/imageprocessingtools/badge/?version=latest)](https://imageprocessingtools.readthedocs.io/en/latest/?badge=latest)|

[![GitHub pull-requests](https://img.shields.io/github/issues-pr/RiccardoBiondi/ImageProcessingTools.svg?style=plastic)](https://github.com/RiccardoBiondi/ImageProcessingTools/pulls)
[![GitHub issues](https://img.shields.io/github/issues/RiccardoBiondi/ImageProcessingTools.svg?style=plastic)](https://github.com/RiccardoBiondi/ImageProcessingTools/issues)

[![GitHub stars](https://img.shields.io/github/stars/RiccardoBiondi/ImageProcessingTools.svg?label=Stars&style=social)](https://github.com/RiccardoBiondi/ImageProcessingTools/stargazers)
[![GitHub watchers](https://img.shields.io/github/watchers/RiccardoBiondi/ImageProcessingTools.svg?label=Watch&style=social)](https://github.com/RiccardoBiondi/ImageProcessingTools/watchers)


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

Supported python versions: ![Python version](https://img.shields.io/badge/python-3.6.*|3.7.*|3.8.*|3.9.*-blue.svg)

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

```python

import itk
import IPT.io as io
import IPT.itk_wrapping as itkf

# Load a volume
image = io.itk_image_file_reader('/path/to/image.nrrd', itk.Image[itk.F, 3])

# apply a median filter
median = itkf.itk_median(image.GetOutput(), radius=2)

# and save the image
_ = io.itk_image_file_writer('/path/to/output.nii', median.GetOutput())
```


## Contribute

Any contribution is welcome.  You can fill a issue or a pull request!


## License

Any contribution is more than welcome. Just fill an [issue]() or a [pull request]() and we will check ASAP!

See [here]() for further information about how to contribute with this project.

## Authors

* **Riccardo Biondi** [git](https://github.com/RiccardoBiondi), [unibo](https://www.unibo.it/sitoweb/riccardo.biondi7)


## References


<blockquote>1- McCormick M, Liu X, Jomier J, Marion C, Ibanez L. ITK: enabling reproducible research and open science. Front Neuroinform. 2014;8:13. Published 2014 Feb 20. doi:10.3389/fninf.2014.00013</blockquote>

<blockquote> 2- Yoo TS, Ackerman MJ, Lorensen WE, Schroeder W, Chalana V, Aylward S, Metaxas D, Whitaker R. Engineering and Algorithm Design for an Image Processing API: A Technical Report on ITK – The Insight Toolkit. In Proc. of Medicine Meets Virtual Reality, J. Westwood, ed., IOS Press Amsterdam pp 586-592 (2002). </blockquote>


## Citation

If you have found `Image Processing Tools` helpful in your research, please consider citing the project

```tex
@misc{IPT_2022,
  author = {Biondi, Riccardo},
  title = {Image Processing Tools},
  year = {2022},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/RiccardoBiondi/ImageProcessingTools}},
}
```
