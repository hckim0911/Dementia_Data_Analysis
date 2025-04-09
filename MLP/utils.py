import numpy as np
import tensorflow as tf
import pandas as pd
import random

def set_seed(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)

def preprocessing(df, drop_cols):
    df = df.drop(columns=drop_cols, axis=1)
    df['DIAG_NM'] = df['DIAG_NM'].apply(lambda x: 0 if x == 'CN' else 1)
    return df

def get_Xy(df):
    X = df.drop(columns=['DIAG_NM'])
    y = df['DIAG_NM']
    return X, y

def print_cv_results(acc_scores, f1_scores, conf_matrices):
    print("\nCross-Validation Results")
    print(f"avrg Accuracy: {np.mean(acc_scores):.4f}")
    print(f"avrg F1 Score : {np.mean(f1_scores):.4f}")

    # 혼동 행렬 누적
    total_cm = np.sum(conf_matrices, axis=0)
    print(f"\n Cumulative Confusion Matrix \n{total_cm}")

def load_and_prepare_data(root_dir, drop_cols):
    train_df = pd.read_csv(f'{root_dir}/train/merged.csv')
    test_df = pd.read_csv(f'{root_dir}/test/merged.csv')

    train_df = preprocessing(train_df, drop_cols)
    test_df = preprocessing(test_df, drop_cols)

    df = pd.concat([train_df, test_df], axis=0)
    X, y = get_Xy(df)

    return X, y