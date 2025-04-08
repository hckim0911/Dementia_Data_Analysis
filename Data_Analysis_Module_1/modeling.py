# @title Modeling
class Modeling: # 머신 모델링.
    @staticmethod
    def Help():
        print(textwrap.dedent(
            '''
            ### Modeling (머신 모델링) ###
            

            # 모델 리스트 반환 : Get_InitModels(_PersonalModelDict: Dict[str, Any] = {}) -> dict[str, object]
                설명) 기본 제공되는 분류/군집 모델들과 사용자 정의 모델을 합쳐 딕셔너리 형태로 반환합니다.
                인자)
                    _PersonalModelDict: 사용자 정의 모델 딕셔너리 (예: {"MyModel": 모델객체})
                반환) {"모델명": 모델객체} 형태의 딕셔너리

            # 자동 결정 트리 모델링 : Set_auto_tree_model(_df, _targetColms='DIAG_NM', test_df=None, visualize=True, max_depth=10, min_samples_split=5, min_samples_leaf=2, class_balance=True)
                설명) 주어진 데이터로 분류 또는 회귀 여부를 자동 판별한 후 결정 트리 모델을 학습하고 평가합니다.
                    테스트 데이터가 있으면 평가에 사용되며, 시각화 옵션에 따라 결과도 출력됩니다.
                인자)
                    _targetColms: 타겟 컬럼 이름
                    test_df: 테스트용 데이터프레임 (기본값: 없음 → 훈련 데이터에서 분할)
                    visualize: 트리 및 결과 시각화 여부 (기본: True)
                    max_depth: 트리 최대 깊이
                    min_samples_split: 내부 노드를 분할하기 위한 최소 샘플 수
                    min_samples_leaf: 리프 노드가 되기 위한 최소 샘플 수
                    class_balance: 클래스 불균형 자동 처리 여부 (분류일 경우에만 적용)
                반환) 
                    model: 학습된 결정 트리 모델 객체
                    feature_importance_df: 피처 중요도 DataFrame (중요도 기준 내림차순 정렬)
            '''
        ))

    @staticmethod# 사용할 모델 리스트
    def Get_InitModels( _PersonalModelDict:Dict[str, Any] = {} ) ->  dict[str, object]:
        models = {
            "Logistic Regression": LogisticRegression(),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(),
            "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
            #"LightGBM": LGBMClassifier(),
            "SVM": SVC(),
            "KNN": KNeighborsClassifier(),
            "K-Means": KMeans(n_clusters=2, random_state=42)
        }
        models.update(_PersonalModelDict)
        return models


    @staticmethod# 사용할 모델 리스트
    def Set_auto_tree_model(_df: pd.DataFrame, _targetColms: str = 'DIAG_NM', 
                    test_df: pd.DataFrame = None, visualize: bool = True,
                    max_depth: int = 10, min_samples_split: int = 5, 
                    min_samples_leaf: int = 2, class_balance: bool = True):
        """
        DataFrame과 타겟 컬럼 이름만 넣으면 분류/회귀 판단 후 트리 모델을 훈련하고 성능 평가를 출력합니다.
        test_df가 제공되면 별도의 테스트 데이터로 평가합니다.

        Parameters:
            _df (pd.DataFrame): 입력 데이터 (피처 + 타겟 포함)
            _targetColms (str): 타겟 컬럼명
            test_df (pd.DataFrame, optional): 테스트용 데이터프레임 (피처 + 타겟 포함)
            visualize (bool): 트리 시각화 여부 (기본값: True)
            max_depth (int): 트리 최대 깊이 (기본값: 10)
            min_samples_split (int): 노드 분할을 위한 최소 샘플 수 (기본값: 5)
            min_samples_leaf (int): 리프 노드가 되기 위한 최소 샘플 수 (기본값: 2)
            class_balance (bool): 클래스 불균형 처리 여부 (기본값: True)

        Returns:
            tuple: (model, feature_importance_df) - 학습된 트리 모델 객체와 피처 중요도 데이터프레임
        """
        try:
            # 데이터 유효성 검사
            if not isinstance(_df, pd.DataFrame):
                raise TypeError("_df must be a pandas DataFrame")
            
            if _targetColms not in _df.columns:
                raise ValueError(f"Target column '{_targetColms}' not found in the dataframe")
            
            # 1. 피처, 타겟 분리
            X_train = _df.drop(columns=[_targetColms])
            y_train = _df[_targetColms]
            
            # 2. 수치형 인코딩 처리 (범주형만 원핫 인코딩)
            X_train = pd.get_dummies(X_train)
            
            # 3. 분류/회귀 판단
            if y_train.dtype == 'object' or y_train.nunique() < 10:
                task_type = 'classification'
                
                # 라벨 인코딩 (유니크한 클래스 값 보존)
                classes, y_train_encoded = np.unique(y_train, return_inverse=True)
                
                # 클래스 불균형 처리
                if class_balance and len(classes) > 1:
                    class_weights = compute_class_weight('balanced', classes=np.unique(y_train_encoded), y=y_train_encoded)
                    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
                else:
                    class_weight_dict = None
                    
            else:
                task_type = 'regression'
                y_train_encoded = y_train
                class_weight_dict = None
                classes = None  # 회귀는 클래스가 없음
            
            # 4. 테스트 데이터 처리
            if test_df is None:
                # 테스트 데이터가 없는 경우 - 훈련 데이터 분할
                X_train_final, X_test, y_train_final, y_test = train_test_split(
                    X_train, y_train_encoded, test_size=0.2, random_state=42
                )
                
                # 교차 검증 수행 (분류인 경우)
                if task_type == 'classification':
                    # 클래스 수가 적을 경우 fold 수 조정
                    n_classes = len(np.unique(y_train_encoded))
                    cv_folds = min(5, n_classes)
                    if cv_folds >= 2:
                        try:
                            cv_scores = cross_val_score(
                                DecisionTreeClassifier(
                                    max_depth=max_depth, 
                                    min_samples_split=min_samples_split,
                                    min_samples_leaf=min_samples_leaf,
                                    class_weight=class_weight_dict,
                                    random_state=42
                                ), 
                                X_train, y_train_encoded, cv=cv_folds
                            )
                            print(f"{cv_folds}-Fold Cross-Validation Accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
                        except Exception as e:
                            print(f"Warning: Cross-validation failed with error: {str(e)}")
            else:
                # 테스트 데이터가 있는 경우
                if not isinstance(test_df, pd.DataFrame):
                    raise TypeError("test_df must be a pandas DataFrame")
                
                if _targetColms not in test_df.columns:
                    raise ValueError(f"Target column '{_targetColms}' not found in the test dataframe")
                
                X_train_final = X_train  # 전체 훈련 데이터 사용
                y_train_final = y_train_encoded
                
                X_test_raw = test_df.drop(columns=[_targetColms])
                y_test_raw = test_df[_targetColms]
                
                # 테스트 데이터 인코딩
                X_test = pd.get_dummies(X_test_raw)
                
                # 훈련 데이터와 테스트 데이터의 컬럼 일치시키기
                for col in X_train.columns:
                    if col not in X_test.columns:
                        X_test[col] = 0
                
                # 훈련에 없는 컬럼 제거
                X_test = X_test[X_train.columns]
                
                # 테스트 데이터 라벨 인코딩
                if task_type == 'classification':
                    # 알려진 클래스만 사용하도록 매핑
                    y_test_list = y_test_raw.tolist()
                    
                    # 클래스가 존재하는지 확인
                    valid_indices = []
                    y_test_values = []
                    
                    for i, val in enumerate(y_test_list):
                        if val in classes:
                            class_idx = np.where(classes == val)[0][0]
                            valid_indices.append(i)
                            y_test_values.append(class_idx)
                    
                    if not valid_indices:
                        raise ValueError("No valid test samples: all test classes not seen in training data")
                    
                    # 유효한 샘플만 선택
                    if len(valid_indices) < len(y_test_list):
                        print(f"Warning: {len(y_test_list) - len(valid_indices)} samples in test data have classes not seen in training data.")
                        X_test = X_test.iloc[valid_indices]
                        y_test = np.array(y_test_values)
                    else:
                        y_test = np.array(y_test_values)
                else:
                    y_test = y_test_raw
            
            # 5. 모델 정의 및 학습
            if task_type == 'classification':
                model = DecisionTreeClassifier(
                    max_depth=max_depth, 
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    class_weight=class_weight_dict,
                    random_state=42
                )
            else:
                model = DecisionTreeRegressor(
                    max_depth=max_depth, 
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    random_state=42
                )
            
            model.fit(X_train_final, y_train_final)
            
            # 6. 평가
            y_pred = model.predict(X_test)
            
            print(f"Task type: {task_type}")
            if task_type == 'classification':
                acc = accuracy_score(y_test, y_pred)
                print(f"Accuracy: {acc:.4f}")
                
                # 분류 보고서
                if classes is not None and len(classes) <= 20:  # 클래스가 너무 많으면 보고서가 너무 길어짐
                    print("\nClassification Report:")
                    # 클래스 이름으로 레이블 복원
                    target_names = [str(c) for c in classes]
                    print(classification_report(y_test, y_pred, target_names=target_names))
                
                # 혼동 행렬 (분류)
                if classes is not None and len(classes) <= 10 and visualize:  # 클래스가 너무 많으면 시각화하지 않음
                    plt.figure(figsize=(10, 8))
                    cm = confusion_matrix(y_test, y_pred)
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                                xticklabels=[str(c) for c in classes], 
                                yticklabels=[str(c) for c in classes])
                    plt.xlabel('Predicted')
                    plt.ylabel('True')
                    plt.title('Confusion Matrix')
                    plt.tight_layout()
                    plt.show()
            else:
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                print(f"MSE: {mse:.4f}")
                print(f"R^2 Score: {r2:.4f}")
                
                # 예측 vs 실제 시각화 (회귀)
                if visualize:
                    plt.figure(figsize=(10, 6))
                    plt.scatter(y_test, y_pred, alpha=0.5)
                    min_val = min(y_test.min(), y_pred.min())
                    max_val = max(y_test.max(), y_pred.max())
                    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
                    plt.xlabel('Actual')
                    plt.ylabel('Predicted')
                    plt.title('Actual vs Predicted Values')
                    plt.tight_layout()
                    plt.show()
            
            # 7. 트리 시각화
            if visualize:
                # 기본 트리 시각화
                plt.figure(figsize=(20, 10))
                
                # 트리가 클 경우 글꼴 크기 자동 조정
                if max_depth > 5:
                    fontsize = max(3, int(10 - (max_depth - 5)))
                    print(f"Note: Tree is deep (depth={max_depth}). Using smaller font size for visualization.")
                else:
                    fontsize = 10
                    
                plot_tree(
                    model,
                    filled=True,
                    feature_names=X_train.columns,
                    class_names=[str(c) for c in classes] if task_type == 'classification' and classes is not None else None,
                    rounded=True,
                    precision=2,
                    fontsize=fontsize
                )
                plt.title("Default Decision Tree Visualization", fontsize=16)
                plt.tight_layout()
                plt.show()
                
                # 사용자 친화적인 트리 시각화 (그래프 형태)
                try:
                    from dtreeviz.trees import dtreeviz
                    
                    # 분류 모델인 경우
                    if task_type == 'classification' and classes is not None:
                        class_names = [str(c) for c in classes]
                        viz = dtreeviz(
                            model, 
                            X_train, 
                            y_train_encoded,
                            target_name=_targetColms,
                            feature_names=list(X_train.columns),
                            class_names=class_names,
                            title="Decision Tree Visualization",
                            colors={
                                'classes': ['red', 'green'] if len(class_names) == 2 else None
                            }
                        )
                    # 회귀 모델인 경우
                    else:
                        viz = dtreeviz(
                            model, 
                            X_train, 
                            y_train_encoded,
                            target_name=_targetColms,
                            feature_names=list(X_train.columns),
                            title="Decision Tree Visualization"
                        )
                    
                    viz.view()
                    print("User-friendly decision tree visualization generated.")
                except ImportError:
                    print("Info: For more user-friendly tree visualization, install dtreeviz package.")
                    print("Run: pip install dtreeviz")
                    
                    # 대체 시각화 방법 (NetworkX 사용)
                    try:
                        import networkx as nx
                        
                        # 트리 노드 정보 추출 함수
                        def extract_tree_info(tree, feature_names):
                            n_nodes = tree.tree_.node_count
                            children_left = tree.tree_.children_left
                            children_right = tree.tree_.children_right
                            feature = tree.tree_.feature
                            threshold = tree.tree_.threshold
                            
                            # 분류 모델인 경우 클래스 정보 추출
                            if task_type == 'classification' and classes is not None:
                                n_classes = len(classes)
                                class_names = [str(c) for c in classes]
                                values = tree.tree_.value.reshape(n_nodes, n_classes)
                                
                                # 각 노드의 주요 클래스 결정
                                node_class = np.argmax(values, axis=1)
                                node_class_names = [class_names[i] for i in node_class]
                                node_samples = np.sum(values, axis=1).astype(int)
                            else:
                                # 회귀 모델인 경우
                                values = tree.tree_.value.flatten()
                                node_class_names = None
                                node_samples = tree.tree_.n_node_samples
                            
                            # 노드 유형 결정 (내부 노드 vs 리프 노드)
                            is_leaf = np.zeros(n_nodes, dtype=bool)
                            node_depth = np.zeros(n_nodes, dtype=int)
                            
                            for i in range(n_nodes):
                                if children_left[i] == children_right[i]:  # 자식 노드가 없으면 리프
                                    is_leaf[i] = True
                            
                            # 노드 깊이 계산
                            stack = [(0, 0)]  # (노드 인덱스, 깊이)
                            while stack:
                                node_idx, depth = stack.pop()
                                node_depth[node_idx] = depth
                                
                                # 비-리프 노드는 자식을 스택에 추가
                                if not is_leaf[node_idx]:
                                    stack.append((children_left[node_idx], depth + 1))
                                    stack.append((children_right[node_idx], depth + 1))
                            
                            # 노드 정보 저장
                            node_info = []
                            for i in range(n_nodes):
                                # 비-리프 노드: 피처 조건 표시
                                if not is_leaf[i]:
                                    label = f"{feature_names[feature[i]]}\n≤ {threshold[i]:.2f}"
                                else:
                                    # 리프 노드: 클래스 또는 값과 샘플 수 표시
                                    if task_type == 'classification':
                                        label = f"{node_class_names[i]}\n({node_samples[i]} samples)"
                                        label += f"\nProbability: {values[i][node_class[i]]/node_samples[i]:.2f}"
                                    else:
                                        label = f"Value: {values[i]:.2f}\n({node_samples[i]} samples)"
                                
                                node_info.append({
                                    'idx': i,
                                    'label': label,
                                    'is_leaf': is_leaf[i],
                                    'depth': node_depth[i],
                                    'class': node_class_names[i] if task_type == 'classification' and node_class_names else None
                                })
                            
                            return node_info, children_left, children_right
                        
                        # 트리 정보 추출
                        node_info, children_left, children_right = extract_tree_info(model, X_train.columns)
                        
                        # 그래프 생성
                        G = nx.DiGraph()
                        
                        # 노드 추가
                        for node in node_info:
                            # 클래스에 따른 노드 색상 설정
                            if task_type == 'classification' and node['is_leaf']:
                                if node['class'] in ['1', 'Yes', 'True', 'Positive']:
                                    color = 'lightgreen'
                                elif node['class'] in ['0', 'No', 'False', 'Negative']:
                                    color = 'lightcoral'
                                else:
                                    color = 'lightblue'
                            else:
                                color = 'lightblue'
                            
                            G.add_node(node['idx'], label=node['label'], is_leaf=node['is_leaf'], depth=node['depth'], color=color)
                        
                        # 엣지 추가
                        for i, (left, right) in enumerate(zip(children_left, children_right)):
                            if left != -1:  # 왼쪽 자식이 있음
                                G.add_edge(i, left, decision='Yes/True')
                            if right != -1:  # 오른쪽 자식이 있음
                                G.add_edge(i, right, decision='No/False')
                        
                        # 시각화
                        plt.figure(figsize=(30, 20))  # 훨씬 더 넓은 캔버스 사용
                        
                        # 균형 트리 레이아웃 계산
                        def hierarchy_pos(G, root=None, width=1., vert_gap=0.4, vert_loc=0, xcenter=0.5):
                            """
                            계층적 그래프 레이아웃을 생성합니다.
                            
                            G: 그래프 객체
                            root: 루트 노드 (None인 경우 자동 검색)
                            width: 수평 공간 크기
                            vert_gap: 수직 간격
                            vert_loc: 루트 노드의 y 좌표
                            xcenter: 루트 노드의 상대적 x 위치
                            """
                            if root is None:
                                # 들어오는 간선이 없는 노드를 루트로 찾음
                                roots = [v for v, d in G.in_degree() if d == 0]
                                if len(roots) > 1:
                                    # 여러 루트가 있는 경우 가상 루트 생성
                                    root = -1
                                    G.add_node(root)
                                    for r in roots:
                                        G.add_edge(root, r)
                                elif len(roots) == 1:
                                    root = roots[0]
                                else:
                                    # 사이클이 있는 경우 임의 노드를 루트로 선택
                                    root = list(G.nodes())[0]
                            
                            def _hierarchy_pos(G, root, width=1., vert_gap=0.4, vert_loc=0, xcenter=0.5, pos=None, parent=None, parsed=[]):
                                if pos is None:
                                    pos = {root: (xcenter, vert_loc)}
                                else:
                                    pos[root] = (xcenter, vert_loc)
                                children = list(G.neighbors(root))
                                if parent is not None:  # 부모 노드 제외
                                    children.remove(parent)
                                if len(children) != 0:
                                    dx = width / len(children)
                                    nextx = xcenter - width/2 - dx/2
                                    for child in children:
                                        nextx += dx
                                        pos = _hierarchy_pos(G, child, width=dx, vert_gap=vert_gap,
                                                            vert_loc=vert_loc-vert_gap, xcenter=nextx,
                                                            pos=pos, parent=root, parsed=parsed)
                                return pos
                            
                            return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)
                        
                        # 방향 그래프를 트리로 변환 (각 노드가 유일한 부모를 가지도록)
                        def to_tree(G):
                            """방향 그래프를 트리로 변환 (각 노드는 하나의 부모만 가짐)"""
                            T = nx.DiGraph()
                            for node in G.nodes(data=True):
                                T.add_node(node[0], **node[1])
                            
                            # 각 노드의 부모-자식 관계를 유지
                            for u in G.nodes():
                                predecessors = list(G.predecessors(u))
                                if predecessors:  # 부모가 있다면
                                    parent = predecessors[0]  # 첫 번째 부모 선택
                                    T.add_edge(parent, u, **G.get_edge_data(parent, u))
                            
                            return T
                        
                        # 루트 노드 식별
                        roots = [n for n, d in G.in_degree() if d == 0]
                        if roots:
                            root = roots[0]
                        else:
                            # 루트 없으면 0번 노드를 루트로 간주
                            root = 0
                        
                        # 노드 위치 계산 (계층적 레이아웃)
                        T = to_tree(G)  # 그래프를 트리로 변환
                        pos = hierarchy_pos(T, root=root, width=8., vert_gap=1)
                        
                        # 결정 경로에 따른 엣지 색상 설정
                        edge_colors = []
                        for u, v, data in G.edges(data=True):
                            if 'Yes' in data.get('decision', ''):
                                edge_colors.append('green')
                            else:
                                edge_colors.append('red')
                        
                        # 노드별 크기 계산 (조건 복잡도에 따라)
                        node_sizes = []
                        for node in G.nodes():
                            label = G.nodes[node]['label']
                            lines = label.split('\n')
                            # 텍스트 길이와 줄 수에 비례하는 노드 크기
                            max_line_len = max(len(line) for line in lines)
                            node_size = max(3500, 200 * max_line_len)
                            node_sizes.append(node_size)
                        
                        # 노드 그리기 (경계선 추가)
                        node_colors = [data['color'] for _, data in G.nodes(data=True)]
                        nx.draw_networkx_nodes(G, pos, 
                                            node_size=node_sizes,
                                            node_color=node_colors, 
                                            edgecolors='black',
                                            linewidths=2,
                                            alpha=0.9)
                        
                        # 노드 레이블 수동 그리기 (더 세밀한 제어)
                        for node, (x, y) in pos.items():
                            label = G.nodes[node]['label']
                            lines = label.split('\n')
                            
                            # 줄 높이 계산
                            bbox = dict(boxstyle="round,pad=0.5", facecolor=G.nodes[node]['color'], 
                                    edgecolor='black', alpha=0.9)
                            
                            # 텍스트 위치 조정
                            line_height = 0.05
                            y_offset = (len(lines) - 1) * line_height / 2
                            
                            # 모든 줄을 하나의 텍스트 상자에 표시
                            label_text = '\n'.join(lines)
                            plt.text(x, y, label_text,
                                    horizontalalignment='center',
                                    verticalalignment='center',
                                    fontsize=10, fontweight='bold',
                                    bbox=bbox, wrap=True)
                        
                        # 엣지 그리기 (곡선 방식)
                        for i, (u, v) in enumerate(G.edges()):
                            # 곡선 정도 랜덤화
                            rad = 0.15
                            
                            # 부모/자식 위치
                            x1, y1 = pos[u]
                            x2, y2 = pos[v]
                            
                            # 피처 조건 엣지 레이블
                            edge_label = G.edges[u, v].get('decision', '')
                            
                            # 엣지 그리기
                            arrow = patches.FancyArrowPatch((x1, y1), (x2, y2),
                                                        connectionstyle=f'arc3,rad={rad}',
                                                        arrowstyle='-|>',
                                                        mutation_scale=20,
                                                        lw=2,
                                                        color=edge_colors[i],
                                                        alpha=0.8)
                            plt.gca().add_patch(arrow)
                            
                            # 엣지 레이블 위치 계산
                            # 곡선 경로의 중간점 계산 (라디안 보정)
                            middle_x = (x1 + x2) / 2
                            middle_y = (y1 + y2) / 2
                            
                            # 곡선 효과 추가
                            middle_x += rad * (y2 - y1)
                            middle_y += rad * (x1 - x2)
                            
                            # 레이블 배경 추가
                            plt.text(middle_x, middle_y, edge_label,
                                    horizontalalignment='center',
                                    verticalalignment='center',
                                    fontsize=9, fontweight='bold',
                                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round'),
                                    color=edge_colors[i])
                        
                        plt.axis('off')  # 축 제거
                        plt.title("Decision Tree Visualization", fontsize=20, pad=20)
                        plt.tight_layout()
                        plt.show()
                        
                    except ImportError:
                        print("Info: For better tree visualization, install networkx package.")
                        print("Run: pip install networkx")
            
            # 8. 피처 중요도 분석
            feature_importance = model.feature_importances_
            feature_names = X_train.columns
            
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': feature_importance
            }).sort_values(by='Importance', ascending=False)
            
            # 상위 10개 피처 중요도 시각화
            if visualize:
                plt.figure(figsize=(12, 6))
                top_features = importance_df.head(min(10, len(importance_df)))
                sns.barplot(x='Importance', y='Feature', data=top_features)
                plt.title('Top Feature Importance')
                plt.tight_layout()
                plt.show()
            
            print("\nTop 5 Important Features:")
            print(importance_df.head(5))
            
            return model, importance_df
            
        except Exception as e:
            print(f"Error in auto_tree_model: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None