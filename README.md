# Transformer

<p align="center">
  <img src="https://github.com/andrew-m-holmes/transformer/blob/main/architecture.jpg" alt="Transformer Architecture">
</p>

---

## Description

This repository contains my PyTorch implementation of the Transformer model presented in the _Attention Is All You Need_[^1] paper. Included are python scripts that enable users to create, train, and evaluate a custom Transformer model for language translation tasks. Alongside this functionality, are jupyters notebooks that walk through a step by step process for training, retraining, and prompting a Transformer model using the English to German Multi30k[^2] dataset.

For a comprehensive understanding of the Transformer model and its significance in deep learning, refer to my article: _What is a Transformer?_[^3]

__Disclaimer__: _This repository is not optimized for Windows operatoring systems and is targeted towards Unix-based systems (e.g. macOS, Ubuntu, etc.). Incompatability issues may arise when operating Windows on the programs and files inside this repository._

---

## Prerequisites

To use this repository, ensure that your system meets the following requirements:

- **Python**: Version 3.9 or higher. Confirm your python version by running: `python --version`. Visit the [official Python download page](https://www.python.org/downloads/) for installation or upgrade.

- **pip**: Version 22.3.1 or higher. Confirm your pip version by running: `pip --version`. Follow the [pip installation guide](https://pip.pypa.io/en/stable/installation/) for installation or upgrade.

- **Git LFS**: Version 3.3.0 or higher. Confirm your git-lfs version by running: `git lfs --version`. Follow the [installation guide for Git Large File Storage](https://git-lfs.com/) for installation or upgrade.

- **GNU Make**: Version 4.3 or higher. Confirm your GNU Make version by running: `make --version`. Visit the [official GNU Make webpage](https://www.gnu.org/software/make/) for installation or upgrade.

> **Note**: _GNU Make is technically not required, although it's recommended._
  
---

## Installation

1. Open your terminal and navigate to your desired directory using the command below:
   
    ```
    cd path/to/directory
    ```

2. Clone the repository using the following command:
   
    ```
    git clone https://github.com/andrew-m-holmes/Transformer
    ```

---

## Dependencies

To install the required dependencies for running python files and jupyter notebooks, use one of the two commands below:

1. This command creates a virtual environment and installs the dependencies:
   
    ```
    make venv
    ```
    
  > **Note**: _This command will not activate the created virtual environment._.

2. This installs dependencies into a specified virtual environment:
   
    ```
    path/to/venv/bin/pip install -r dependencies.txt
    ```

---

## Running

To train, retrain, or prompt a model, use the [main.py](https://github.com/andrew-m-holmes/Transformer/blob/master/main.py) script. This script allows you to pass command line arguments through the `argaprse` library which alters the execution of the program. Different arguments can change what parts of the program is executed as well as alter some values passed to parameters in functions (e.g. changing the path to datasets).   

**Important**: _If you're not using GNU Make to run main.py, the virtual environment with the required dependencies (installed) must be active! You can activate the environment by running: `source path/to/venv/bin/activate`. Follow the [official virtual environment guide](https://docs.python.org/3/library/venv.html) for more information._

> **Note**: _If you're not familiar with argparse, use `python main.py -h` for help. For more details refer to [argparse documentation](https://docs.python.org/3/library/argparse.html)._

---

### Examples

1. This command displays verbose output while training the `Transformer` and saves the metrics graph as _"metrics.jpg_" in the folder named _"saves"_:

   ```
    python main.py -v train --metrics saves/metrics.jpg
     ```

2. This executes the command found in example `1.`, but uses `make` to activate the virtual environment and pass arguments:
   
   ```
    make train args="-v" subargs="--metrics saves/metrics.jpg"
    ```

3. This command is similar to example `2.`, although it specifies a virtual environment path:
   
   ```
    make train venv="path/to/venv" args="-v" subargs="--metrics saves/metrics.jpg"
    ```
   
> **Note**: _Values passed to args and subargs must be in quotes for proper functionality of `make` commands._

---

## Notebooks

Outlined in the description section, this repository contains three Jupyter notebooks for training, testing, and prompting:

- **en-de-train**: This notebook demonstrates the step by step process of creating a pipeline for training a `Transformer` for English to German language translation. Take a look at the notebook here: [en-de-train.ipynb](https://github.com/andrew-m-holmes/Transformer/blob/master/en-de-train.ipynb).

- **en-de-retrain**: This notebook shows the pipeline process to retrain a `Transformer` model from a `Checkpoint` for the same task. Start with retraining here: [en-de-retrain.ipynb](https://github.com/andrew-m-holmes/Transformer/blob/master/en-de-retrain.ipynb).

- **en-de-prompt**: This notebook shows the pipeline process of loading a `Transformer` model from model weights and prompting it for English to German language translation. Learn how to prompt here: [en-de-prompt.ipynb](https://github.com/andrew-m-holmes/Transformer/blob/master/en-de-prompt.ipynb).

> **Note**: Its highly recommended you start with the notebooks to understand how the tools and modules work within this repository. From there, they're useful guidelines for building other notebooks or python files for your own language translation tasks. 

---

## Contributing

- **Issues**: Issues are great way of to keep track of different topics within this repository. Open issues when you want to keep documentation on code ehancements, bugs, or other topics relevant to this repository. Refer to the [Issues page](https://github.com/andrew-m-holmes/Transformer/issues) to get started.
  
- **Pull Request**: Please use the pull request feature to implement your own code that can improve different parts of this repository. Pull request can be as simple as a bug fix or as complex as refactoring this repository. When applicable, make sure to link pull requests to related open issues and I'll make sure to review it for merging. Go to the [Pull request page](https://github.com/andrew-m-holmes/Transformer/pulls) to begin.
  
_I would like to thank you in advance for any contributions made!_

---

## Social

Before I leave you, please connect with me on [LinkedIn](https://www.linkedin.com/in/andrewmicholmes/) and be on the look out for articles on my [Medium](https://medium.com/@andmholm).

--- 

## External Links
[^1]: https://arxiv.org/pdf/1706.03762.pdf  

[^2]: https://arxiv.org/pdf/1605.00459.pdf 

[^3]: https://medium.com/@andmholm/what-is-a-transformer-d68bd1647c57
