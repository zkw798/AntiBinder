# AntiBinder

## Introduction
This project is used for predicting antigen-antibody affinity for protein types. The model can be trained and used based solely on sequence data. You can also stack the modules within, making the model's parameters significantly larger, and train it to achieve a plug-and-play effect.

## Dependencies
List the libraries and tools required to run the project:
- Python              3.10+
- beautifulsoup4      4.12.3
- biopython           1.83
- biotite             0.40.0
- bs4                 0.0.2
- fair-esm            2.0.0
- idna                3.4
- igfold              0.4.0
- lmdb                1.4.1
- logomaker           0.8
- ml-collections      0.1.1
- scikit-learn        1.4.2
- seaborn             0.13.2
- six                 1.16.0
- tqdm                4.66.2

## Installation Guide
Detailed instructions on how to install and set up the project:

# Clone the repository
git clone https://github.com/shiipl/AntiBinder.git

# Install dependencies
pip install -r requirements.txt

## Usage Instructions
### Training
Prepare labeled sequence data for antigens and antibody heavy chains. Name the columns according to the names specified in the `antigen_antibody_emb.py`. If the heavy chain sequences of the antibodies have not been split, first use `heavy_chain_split.py` to split the sequences. Then, use the command: `python main_trainer.py` to start the model training.

```python
# Example code for starting training
# Start training
!python main_trainer.py
```
### Testing
Prepare labeled sequence data for antigens and antibodies. Name the columns according to the names specified in the `antigen_antibody_emb.py`. Then, use the command: `python main_test.py` to start the test.
```python
# Example code for starting tesing

# Start training
!python main_test.py
```