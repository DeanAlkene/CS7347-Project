import os
import pandas as pd

def split_train_dev(path : str, frac, seed):
    if frac < 0 or frac > 0.4:
        raise(ValueError('0% < test ratio <= 40% !'))
    old_train_file = os.path.join(path, "train.tsv")
    new_train_file = os.path.join(path, "train_full.tsv")
    new_dev_file = os.path.join(path, "dev.tsv")
    os.rename(old_train_file, new_train_file)
    df = pd.read_csv(new_train_file, sep='\t')
    test_df = df.sample(frac=frac, random_state=seed)
    train_df = df.drop(test_df.index)
    train_df.to_csv(old_train_file, sep='\t', index=False)
    test_df.to_csv(new_dev_file, sep='\t', index=False)