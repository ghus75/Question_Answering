# Question Answering

Implementation of the BiDAF model (https://arxiv.org/abs/1611.01603) for Question Answering using the SQuAD dataset (https://rajpurkar.github.io/SQuAD-explorer/). Given a context and a question, the model highlights the beginning and end of the the sequence within the context that answers the question. 
The current implementation evaluated on the dev set yields EM = 65.7, F1 = 76.

The code is written in python 2.7 and uses starter files from Stanford CS224n Deep learning & NLP course, available at https://github.com/abisee/cs224n-win18-squad . It includes a script which will create the environment and install all the necessary dependencies.

# Installing
Run get_started.sh

# Training
source activate squad
$ python main.py --experiment_name=<YOUR EXPERIMENT NAME> --mode=train

Lots other options are available for the command line (see flags defined in main.py)
Training takes 10-20 hours using a 12Gb GPU.

# Inspecting output
Show some examples:

$ python main.py --experiment_name=<YOUR EXPERIMENT NAME> --mode=show_examples

# Run official eval locally
$ python main.py <OTHER FLAGS> --mode=official_eval \
--json_in_path=data/tiny-dev.json \
--ckpt_load_dir=experiments/<YOUR EXPERIMENT NAME>/best_checkpoint
