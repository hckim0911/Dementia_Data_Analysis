from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

def train(X_train, y_train, X_test, y_test, models):
    result = {}

    for name, model in models.items():
        result[name] = {}
        model_ = model.fit(X_train, y_train)
        pred = model_.predict(X_test)
        
        result[name]['accuracy'] = accuracy_score(y_test, pred)
        result[name]['f1'] = f1_score(y_test, pred, average='macro')
        print(f'{name} model trained')
        print(f'accuracy : {result[name]["accuracy"]}')
        print(f'f1       : {result[name]["f1"]}\n')
    
    return result

def get_model_params():
    param_XGB = {
        "max_depth": [10,30,50],
        "min_child_weight" : [1,3,6,10],
        "n_estimators": [200,300,500,1000]
    }    
    # LGB                        
    param_LGBM = {
        "learning_rate" : [0.01,0.1,0.2,0.3,0.4,0.5],
        "max_depth" : [25, 50, 75],
        "num_leaves" : [100,300,500,900,1200],
        "n_estimators" : [100, 200, 300,500,800,1000],
        "learning_rate" : [0.01,0.1,0.2,0.3,0.4,0.5]
    }

    # Logistic Regression
    param_LR = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],         # 정규화 세기
        'penalty': ['l1', 'l2'],                     # 정규화 종류 (solver에 따라 다름)
    }

    # Extra Trees
    param_RF = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2', None]
    }

    # SVM
    param_svm = {
        'C': [0.01, 0.1, 1, 10, 100], 
        'gamma' : [0.001, 0.01, 0.1, 1]
    }

    # Gradient Boosting
    param_GB = {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0]
    }

    # Extra Trees
    param_ET = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }

    # Decision Tree
    param_DT = {
        'max_depth': [None, 5, 10, 15],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'criterion': ['gini', 'entropy']
    }
    # KNN
    param_KNN = {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'p': [1, 2]
    }

    params = {}
    params['XGB'] = param_XGB
    params['LGBM'] = param_LGBM
    params['GB'] = param_GB
    params['DT'] = param_DT
    params['RF'] = param_RF
    params['ET'] = param_ET
    params['LR'] = param_LR
    params['SVM'] = param_svm
    params['KNN'] = param_KNN

    return params

def get_models(random_state):
    models = {
        # 본인이 돌리기로 한 모델을 제외한 나머지 모델은 주석처리
        # "XGB":XGBClassifier(verbosity=0, random_state=random_state),
        # "LGBM": LGBMClassifier(verbosity=-1, random_state=random_state),
        # "GB": GradientBoostingClassifier(random_state=random_state),
        # "DT": DecisionTreeClassifier(random_state=random_state, criterion='entropy'),
        # "RF": RandomForestClassifier(random_state=random_state),
        # "ET": ExtraTreesClassifier(random_state=random_state),
        # "LR": LogisticRegression(random_state=random_state),
        # "SVM": SVC(random_state=random_state),
        "KNN": KNeighborsClassifier(n_neighbors=9),
    }
    return models