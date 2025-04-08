import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import  GridSearchCV, cross_val_predict, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from preprocessing import preprocessing, get_Xy
from modeling import get_model_params, get_models

root_dir = 'dataset/dataset'
train_df = pd.read_csv(f'{root_dir}/train/merged.csv')
test_df = pd.read_csv(f'{root_dir}/test/merged.csv')

drop_cols = ['activity_inactivity_alerts', 'activity_non_wear', 'date', 'EMAIL']
train_df = preprocessing(train_df, drop_cols)
test_df = preprocessing(test_df, drop_cols)

df = pd.concat([train_df, test_df], axis=0)
X, y = get_Xy(df)

# 교차 검증 설정
random_state = 42
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

# 모델, 파라미터 불러오기
models = get_models(random_state)
params = get_model_params()

# 하이퍼 파라미터 튜닝
result = {}

for name, model in models.items():
    print('-' * 50)
    print(f'{name} model training')
    result[name] = {}
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', model)
        ])
    
    
    # 파이프라인 쓸거면 그리디서치 파라미터 키 값이 clf__가 접두사로 들어가야함
    #params[name] = {f'clf__{k}': v for k, v in params[name].items()}
    grid_model = GridSearchCV(estimator=pipe, param_grid={f'clf__{k}': v for k, v in params[name].items()}, cv=cv, scoring='f1_macro', n_jobs=-1)
    grid_model.fit(X, y)
    
    result[name]['best_params'] = grid_model.best_params_
    result[name]['best_score'] = grid_model.best_score_
    
    print(f'best params     : {result[name]["best_params"]}')
    print(f'best f1 - score : {result[name]["best_score"]}')
    
    model_ = grid_model.best_estimator_
    y_pred = cross_val_predict(pipe, X, y, cv=cv)
    
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, average='macro')
    cm = confusion_matrix(y, y_pred)
    
    result[name]['accuracy'] = acc
    result[name]['f1'] = f1
    result[name]['confusion_matrix'] = cm
    
    print(f'accuracy : {result[name]["accuracy"]}')
    print(f'f1       : {result[name]["f1"]}')
    print(f'confusion matrix : \n{result[name]["confusion_matrix"]}')