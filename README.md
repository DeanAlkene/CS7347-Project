# Course Project of NLU 2022, SJTU

In this project, we finetune some pretrained language models on X-QQP dataset, which is a semantic similarity task.

## Environment

To install package dependencies:

```shell
pip install -r requirements.txt
```

If you want to train on multiple GPUs with mixed precision, please refer to the documentation of the HuggingFace Accelerate package.

## Dataset

X-QQP dataset is a collection of paired questions from Quora. Each question pair is annotated with a binary label indicating if they have similar intents. It is similar to the QQP tasks in the GLUE benchmark.

The training set and testing set are stored in `data/`. The training set consists of 375832 question pairs with labels, and the test set consists of 755 question pairs.

## Train

You can finetune PLMs using the shell script:

```shell
./train.sh
```

The arguments are listed as follows:

- `model_type`: `concat` for BERT/RoBERTa/ELECTRA, `paired` for Sentence-BERT
- `backbone`: specifies the PLM, please select among `bert`, `roberta`, and `electra`
- `embedding_method`: determines the embeddings that Sentence-BERT used for classification
  - `cls_with_pooler`: the pooler output of `[CLS]` (default)
  - `first_last_avg`: average of the hidden states of the first layer and the last layer
  - `last_avg`: average of the hidden states of the last layer
  - `last_2_avg`: average of the hidden states of the last two layers
  - `cls`: the hidden state of `[CLS]` in the last layer
- `train_data_path`: path to the training data
- `dev_data_path`: : path to the dev data (optional)
- `max_length`: the maximum length of the input token sequence
- `pad_to_max_length`: if added, the tokenizer pads the input sequence to `max_length`
- `per_device_train_batch_size`: the training batch size per GPU
- `per_device_dev_batch_size`: the dev batch size per GPU (optional)
- `learning_rate`: the learning rate
- `weight_decay`: the weight decay
- `llrd_factor`: the rate of the layer-wise learning rate decay technique
- `num_train_epochs`: the epochs to train
- `gradient_accumulation_steps`: the steps to accumulate gradients
- `lr_scheduler_type`: the type of learning rate scheduler
- `num_warmup_ratio`: the ratio of warm-up steps to the total training steps
- `classifier_dropout_rate`: the dropout rate of the classifier
- `reinit_layers`: the number of layers that need to re-initialize (from the top down)
- `noise_lambda`: the relative noise intensity to perturb the parameters before finetuning
- `lr_scale_factor`: the scale factor of the classifier learning rate based on `learning_rate`
- `seed`: the random seed
- `output_dir`: the directory to store the models
- `log_dir`: the directory to store log files

## Test

You can test finetuned PLMs using the shell script:

```shell
./test.sh
```

The arguments are listed as follows:

- `model_type`: same as in the `train.sh`, select according to the model you want to test
- `backbone`: same as in the `train.sh`, select according to the model you want to test
- `embedding_method`: same as in the `train.sh`, select according to the model you want to test
- `model_path`: the path to the model file
- `test_data_path`: the path to the testing data
- `max_length`: keep the same with the `train.sh`
- `pad_to_max_length`: keep the same with the `train.sh`
- `batch_size`: the testing batch size
- `output_dir`: the directory of the output file (named `submission.csv`)

You need to modify the script by specifying the path to the finetuned model and the corresponding arguments (i.e., `model_type`, `backbone`, and `embedding_method`).

We provide five finetuned models, which are 2 BERT, 1 RoBERTa, and 2 ELECTRA. You can reproduce the testing results by setting `model_type` as `concat`, `embedding_method` as `cls_with_pooler`, and `backbone` accordingly. You can download them from [here](https://jbox.sjtu.edu.cn/l/e186j3).