# Training T5-type of Language Models for Organic Reaction Prediction
This short tutorial provides a quick overview of the basic code functionality related to single-task fine-tuning and inference of T5-style models: this includes the original T5, FlanT5, as well as ByT5, and other T5-based models such as molT5 or nach0.

## Installation
The requirement.txt file list all depending Python libraries and is provided to create a conda environment.

## Usage

### Training

The running bash script is run_t5_single-task_train.sh which calls the actual Python code/script t5_single-task_train.py. You can run the script by simply executing bash run_t5_single-task_train.sh.

Some hyperparameters are hard-coded in the python file under TrainingArguments. Most of these parameters are directly associated with the standard TrainingArguments from HuggingFace, see the following links for further guidance:
https://huggingface.co/docs/transformers/en/main_classes/trainer
https://huggingface.co/docs/transformers/v4.41.1/en/main_classes/trainer#transformers.Seq2SeqTrainingArguments

### Inference

The running bash script is run_t5_infer.sh which calls the actual Python code/script t5_infer.py. You can run the script by simply executing bash run_t5_single-task_train.sh.


## Data Format

An example dataset is provided under data/. The format is quite simple:

•	It is a csv file with two columns, where “,” is the delimiter

•	First column is the “Input” column (the original sequence before any preprocessing)

•	Second column is the “Output” column (also the original gold sequence).

## License
MIT license

## Reference
Jiayun Pang and Ivan Vulić. Specialising and Analysing Instruction-Tuned and Byte-Level Language Models for Organic Reaction Prediction. (2024) arXiv preprint arXiv:2405.10625
https://arxiv.org/abs/2405.10625
