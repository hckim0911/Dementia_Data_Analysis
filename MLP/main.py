import os
import pandas as pd
import model
from utils import set_seed, print_cv_results, load_and_prepare_data
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold 
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from tensorflow.keras.callbacks import TensorBoard

root_dir = '/content/drive/MyDrive/Colab/TeamPrj/datasets/w_label/'
drop_cols = ['activity_inactivity_alerts', 'activity_non_wear', 'date', 'EMAIL']

X, y = load_and_prepare_data(root_dir, drop_cols)

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

epochs = 30
# hyperparameter tuning
learning_rate = 0.01
batch_size = 32

f1_scores = []
acc_scores = []
conf_matrices = []

set_seed(42)

# rm -rf에서 윈도우 명령어 rmdir로 변경
os.system('rmdir logs/*')  # 이전 로그 강제 삭제

for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):

    log_dir = f"logs/exp_lr0.0001/fold_{fold}"
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    # X_train, X_val = X[train_idx], X[val_idx]
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # 데이터 누수를 막기 위한 스케일링
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    model_instance = model.build_model(input_shape=(X.shape[1],), learning_rate=learning_rate)

    model_instance.fit(X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data = (X_val, y_val),
                verbose=0,
                callbacks=[tensorboard_callback])

    y_pred = model_instance.predict(X_val)
    y_pred_bin = (y_pred > 0.5).astype(int)

    acc = accuracy_score(y_val, y_pred_bin)
    f1 = f1_score(y_val, y_pred_bin, average='macro')
    cm = confusion_matrix(y_val, y_pred_bin)

    acc_scores.append(acc)
    f1_scores.append(f1)
    conf_matrices.append(cm)

print_cv_results(acc_scores, f1_scores, conf_matrices)

# %load_ext tensorboard
# %tensorboard --logdir logs/fit
# %tensorboard --logdir logs/exp_lr0.0001