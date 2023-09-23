# NLP_Fewshot_project
## Description
The goal of this project is to utilize few shot learning for NER on the ChEMU dataset.
## Running instructions
### Step 1
Create a python 3.10 or 3.11 virtual environment. There are many ways to do this, depending on your OS.
### Step 2
Install requirements. If you are using pip, run:
```pip install -r requirements.txt```
from the project root directory
### Step 3
You need to convert your BRAT data into CONLL (BIO) format for this program. There is a program in the preprocess folder that can do this. For each directory of files that you have, run with your virtual environment activated in the project root:
```python preprocess/brat_to_bio.py [path to input dir] [path to output dir]```
Note that you will need to create output directories. 
### Step 4
Run the code! With your virtual environment activated, run:
```python main.py``` with the appropriate flags. A quick rundown: if you are training, use --train, --train_path, --dev_path, --epochs. If you are testing use --test, --test_path, --model_path. If you want to train and test in one run, use
--train --train_path --dev_path --epochs --test --test_path (note that you do not have to secify model path if you are training and testing). If you are ever confused, run ```python main.py -h```. Also the code will print out errors if the arguments are wrong.
