import os
import argparse
import pandas as pd

import torch

from model import XQBert, XQSBert
from dataset import load_concat_test_data, load_paired_test_data

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type", type=str, default="concat"
    )
    parser.add_argument(
        "--backbone", type=str, default="bert"
    )
    parser.add_argument(
        "--embedding_method", type=str, default="cls_with_pooler"
    )
    parser.add_argument(
        "--model_path", type=str, default=None
    )
    parser.add_argument(
        "--test_data_path", type=str, default=None
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8
    )
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()
    return args

def test():
    args = parse_args()
    padding = "max_length" if args.pad_to_max_length else False
    if args.model_type == 'concat':
        test_dataloader = load_concat_test_data(args.backbone, args.test_data_path, args.batch_size, padding, args.max_length)
    elif args.model_type == 'paired':
        test_dataloader = load_paired_test_data(args.backbone, args.test_data_path, args.batch_size, padding, args.max_length)
    else:
        raise ValueError('Unknown model type {}'.format(args.model_type))
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if args.model_type == 'concat':
        model = XQBert(args.backbone, num_labels=2, dropout=0.0, embedding_method=args.embedding_method).to(device)
    else:
        model = XQSBert(args.backbone, num_labels=2, dropout=0.0, embedding_method=args.embedding_method).to(device)
    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    predictions = []
    model.eval()
    for batch in test_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(batch)
            logits = outputs.logits
            predictions.extend(torch.argmax(logits, dim=-1).to('cpu').tolist())
    res = pd.DataFrame({'Id': [i for i in range(len(predictions))], 'Category': predictions}, columns=['Id', 'Category'])

    if args.output_dir is not None:
        output_path = os.path.join(args.output_dir, 'submission.csv')
        res.to_csv(output_path, index=False)
    else:
        raise(ValueError('Please provide output dir!'))

if __name__ == '__main__':
    test()