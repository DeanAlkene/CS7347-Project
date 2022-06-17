import functools
import torch
import pandas as pd
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, default_data_collator

def get_tokenizer_by_model(backbone):
    if backbone == "bert":
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    elif backbone == "roberta":
        tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    elif backbone == "electra":
        tokenizer = AutoTokenizer.from_pretrained("google/electra-base-discriminator")
    else:
        raise ValueError("Unsupported backbone model {}".format(backbone))
    return tokenizer

def get_concat_data_column_name_by_model(backbone):
    if backbone == "bert" or backbone == "electra":
        col_name = ['input_ids', 'token_type_ids', 'attention_mask', 'labels']
    elif backbone == "roberta":
        col_name = ['input_ids', 'attention_mask', 'labels']
    else:
        raise ValueError("Unsupported backbone model {}".format(backbone))
    return col_name

def get_paired_data_column_name_by_model(backbone):
    if backbone == "bert" or backbone == "electra":
        col_name = ['input_ids_a', 'token_type_ids_a', 'attention_mask_a', 'input_ids_b', 'token_type_ids_b', 'attention_mask_b', 'labels']
    elif backbone == "roberta":
        col_name = ['input_ids_a', 'attention_mask_a', 'input_ids_b', 'attention_mask_b', 'labels']
    else:
        raise ValueError("Unsupported backbone model {}".format(backbone))
    return col_name

def get_concat_tokenize_function(backbone, padding, max_length):
    tokenizer = get_tokenizer_by_model(backbone)
    def tokenize_function(instance):
        return tokenizer(
                            instance['Question1'], 
                            instance['Question2'], 
                            padding=padding, 
                            max_length=max_length, 
                            truncation=True
                        )
    return tokenize_function

def get_paired_tokenize_function(backbone, padding, max_length):
    tokenizer = get_tokenizer_by_model(backbone)
    def tokenize_function(instance):
        res_a = tokenizer(
            instance['Question1'],
            padding=padding,
            max_length=max_length,
            truncation=True
        )
        res_b = tokenizer(
            instance['Question2'],
            padding=padding,
            max_length=max_length,
            truncation=True
        )
        ret = {
            'input_ids_a': res_a['input_ids'],
            'token_type_ids_a': res_a['token_type_ids'],
            'attention_mask_a': res_a['attention_mask'],
            'input_ids_b': res_b['input_ids'],
            'token_type_ids_b': res_b['token_type_ids'],
            'attention_mask_b': res_b['attention_mask'],
        }
        return ret
    return tokenize_function

def load_concat_train_data(backbone, path, batch_size, padding, max_length):
    df = pd.read_csv(path, sep='\t')
    dataset = Dataset.from_pandas(df)
    # dataset = dataset.shuffle(seed=42).select(range(1000))
    dataset = dataset.map(get_concat_tokenize_function(backbone, padding, max_length), batched=True)
    dataset = dataset.rename_column('Label', 'labels')
    dataset.set_format(type='torch', columns=get_concat_data_column_name_by_model(backbone))
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=batch_size)
    return dataloader

def load_concat_dev_data(backbone, path, batch_size, padding, max_length):
    df = pd.read_csv(path, sep='\t')
    dataset = Dataset.from_pandas(df)
    # dataset = dataset.shuffle(seed=42).select(range(1000))
    dataset = dataset.map(get_concat_tokenize_function(backbone, padding, max_length), batched=True)
    dataset = dataset.rename_column('Label', 'labels')
    dataset.set_format(type='torch', columns=get_concat_data_column_name_by_model(backbone))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    return dataloader

def load_concat_test_data(backbone, path, batch_size, padding, max_length):
    df = pd.read_csv(path, sep='\t')
    dataset = Dataset.from_pandas(df)
    # dataset = dataset.shuffle(seed=42).select(range(1000))
    dataset = dataset.map(get_concat_tokenize_function(backbone, padding, max_length), batched=True)
    column_name = get_concat_data_column_name_by_model(backbone)
    column_name.remove('labels')
    dataset.set_format(type='torch', columns=column_name)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    return dataloader

def load_paired_train_data(backbone, path, batch_size, padding, max_length):
    df = pd.read_csv(path, sep='\t')
    dataset = Dataset.from_pandas(df)
    # dataset = dataset.shuffle(seed=42).select(range(1000))
    dataset = dataset.map(get_paired_tokenize_function(backbone, padding, max_length), batched=True)
    dataset = dataset.rename_column('Label', 'labels')
    dataset.set_format(type='torch', columns=get_paired_data_column_name_by_model(backbone))
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=batch_size)
    return dataloader

def load_paired_dev_data(backbone, path, batch_size, padding, max_length):
    df = pd.read_csv(path, sep='\t')
    dataset = Dataset.from_pandas(df)
    # dataset = dataset.shuffle(seed=42).select(range(1000))
    dataset = dataset.map(get_paired_tokenize_function(backbone, padding, max_length), batched=True)
    dataset = dataset.rename_column('Label', 'labels')
    dataset.set_format(type='torch', columns=get_paired_data_column_name_by_model(backbone))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    return dataloader

def load_paired_test_data(backbone, path, batch_size, padding, max_length):
    df = pd.read_csv(path, sep='\t')
    dataset = Dataset.from_pandas(df)
    # dataset = dataset.shuffle(seed=42).select(range(1000))
    dataset = dataset.map(get_paired_tokenize_function(backbone, padding, max_length), batched=True)
    column_name = get_paired_data_column_name_by_model(backbone)
    column_name.remove('labels')
    dataset.set_format(type='torch', columns=column_name)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    return dataloader