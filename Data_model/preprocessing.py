import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocessing(df, drop_cols):
    df = df.drop(columns=drop_cols, axis=1)
    df['DIAG_NM'] = df['DIAG_NM'].apply(lambda x: 0 if x == 'CN' else 1)
    
    return df

def get_Xy(df):
    X = df.drop(columns=['DIAG_NM'])
    y = df['DIAG_NM']
    
    return X, y

def random_split(train_df, test_df, random_state=42):
    df = pd.concat([train_df, test_df], axis=0)
    X, y = get_Xy(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state)
    
    return X_train, X_test, y_train, y_test

def normalized(X_train, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test