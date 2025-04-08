#@title Feature_Selection
class Feature_Selection: # 피쳐 선택 혹은 제거
    @staticmethod
    def Help():
        print(textwrap.dedent(
            '''
            ### Feature_Selection (피처 선택 및 제거) ###

            # 컬럼 제거 : Get_DelColm(df, delCol: list = []) -> pd.DataFrame
                설명) 지정된 컬럼들을 제거한 DataFrame을 반환합니다.
                인자)
                    delCol: 제거할 컬럼 리스트
                반환) 컬럼이 제거된 DataFrame

            # 다중공선성(VIF) 계산 : Get_Calc_VIF(df, targetColm=[], axisNum=1, feature='Feature', VIFValue='VIF Value') -> pd.DataFrame
                설명) VIF(분산팽창계수)를 계산하여 다중공선성이 높은 피처를 탐지합니다.
                인자)
                    targetColm: 제거할 대상(타겟) 컬럼 리스트
                    axisNum: 제거할 축 번호 (기본: 1 → 컬럼)
                    feature: 결과에 표시될 컬럼 이름
                    VIFValue: 결과에 표시될 VIF 수치 컬럼명
                반환) 컬럼별 VIF 값이 포함된 DataFrame (내림차순 정렬)

            # 상관계수 기반 변수쌍 추출 : Get_HighCorrPairs(_df, _threshold=0.8, _positive=None) -> pd.DataFrame
                설명) 상관계수가 임계값 이상인 변수쌍만 추출합니다.
                인자)
                    _threshold: 상관계수 기준값 (기본: 0.8)
                    _positive:
                        True → 양의 상관관계만
                        False → 음의 상관관계만
                        None → 양/음 모두 포함
                반환) MultiIndex로 (변수1, 변수2), 값은 상관계수인 DataFrame

            # 순방향 선택 (Forward Selection) : Get_ForwardSelection(_dfX, _y=None, _model=SVC(), _targetColms='DIAG_NM') -> Tuple[List[str], float]
                설명) F1-score를 기준으로 가장 성능이 좋은 피처들을 순차적으로 추가합니다.
                인자)
                    _targetColms: 타겟 컬럼명
                반환) 선택된 피처 리스트, 최고 F1-score

            # 후방 제거 (Backward Elimination) : Get_BackwardElimination(_dfX, _y=None, _model=SVC(), _targetColms='DIAG_NM') -> Tuple[List[str], float]
                설명) 모든 피처에서 시작하여 F1-score가 개선되는 방향으로 피처를 제거합니다.
                인자)
                    _targetColms: 타겟 컬럼명
                반환) 선택된 피처 리스트, 최고 F1-score
            '''
        ))


    @staticmethod
    def Get_DelColm(df,delCol:list = [])->pd.DataFrame:
		# df = 어떤 df에 대한 , delCol = 컬럼 이름 리스트
		# 반환 : 어떤 df에서 delCol에 들어있는 컬럼 제거 후 반환
	    return df.drop(columns= delCol, errors='ignore')

    @staticmethod
    def Get_Calc_VIF(df,targetColm = [], axisNum:int = 1, feature:str = 'Feature', VIFValue:str = 'VIF Value') -> pd.DataFrame:
        #설명 : 다중공선성을 계산하는 코드
        from statsmodels.stats.outliers_influence import variance_inflation_factor

        df_x = df.drop(targetColm, axis = axisNum)

        # VIF 계산
        vif_df = pd.DataFrame()
        vif_df[feature] = df_x.columns
        vif_df[VIFValue] = [variance_inflation_factor(df_x.values, i) for i in range(df_x.shape[1])]
        vif_df[VIFValue] = vif_df[VIFValue].apply(lambda x: f"{x:.4f}")
        vif_df[VIFValue] = vif_df[VIFValue].astype(float)
        vif_df = vif_df.sort_values(by=VIFValue, ascending=False)
        return vif_df

    @staticmethod
    def Get_HighCorrPairs(
        _df: pd.DataFrame, # 데이터 타입
        _threshold: float = 0.8, # 임계값
        _positive: bool | None = None # Noen 이면 _threshold 기준 이상, _threshol -1 이하, True 면 _threshold 이상만 반환, False 면 _threshold * -1 이하만 반환
    ) -> pd.DataFrame:
        """
        주어진 DataFrame에서 상관계수가 조건을 만족하는 컬럼 쌍만 반환하는 함수.
        자기 자신과의 상관관계(1.0)는 제외됩니다.

        Parameters:
        - _df: pandas DataFrame
        - _threshold: float, 상관계수 임계값 (기본값은 0.8)
        - _positive: bool or None
            - True: 양의 상관관계만 (_threshold 이상)
            - False: 음의 상관관계만 (-_threshold 이하)
            - None: 양/음 둘 다 포함 (_threshold 이상 또는 -_threshold 이하)

        Returns:
        - result_df: (Feature1, Feature2)를 MultiIndex로 갖고 Correlation 값을 가진 DataFrame
        """
        # 음수 threshold가 들어오는 경우를 대비해 절대값으로 보정
        _threshold = abs(_threshold)

        # 숫자형 컬럼만 필터링
        numeric_df = _df.select_dtypes(include=[np.number])

        # 상관행렬 계산
        corr_matrix = numeric_df.corr()

        # 상삼각 행렬만 추출하여 중복 제거 (대각선 제외)
        upper_tri = corr_matrix.where(~np.tril(np.ones(corr_matrix.shape)).astype(bool))

        # 조건을 만족하는 값만 추출
        high_corr = upper_tri.stack().reset_index()
        high_corr.columns = ['Feature1', 'Feature2', 'Correlation']

        # 방향성 필터링
        if _positive is True:
            filtered = high_corr[high_corr['Correlation'] >= _threshold]
        elif _positive is False:
            filtered = high_corr[high_corr['Correlation'] <= -_threshold]
        else:
            filtered = high_corr[
                (high_corr['Correlation'] >= _threshold) |
                (high_corr['Correlation'] <= -_threshold)
            ]

        # MultiIndex 설정
        filtered.set_index(['Feature1', 'Feature2'], inplace=True)

        return filtered

    @staticmethod
    def Get_ForwardSelection( # 이진 분류함.
                _dfX: pd.DataFrame,
                _y: pd.Series | np.ndarray = None,
                _model = SVC(),
                _targetColms: str = 'DIAG_NM',
            ) -> Tuple[List[str], float]: 

        _dfX = Data_Prep.Get_BinaryTargetDataFrame(_dfX, _targetColms)
        if _y is None :
            _y = EasyPanda.Get_TargetToSeries(_dfX,_targetColms)

        _dfX = EasyPanda.Get_AllNumType(_dfX)
        _dfX = _dfX.drop(columns=[_targetColms], errors='ignore')

        selected_features = []
        remaining_features = list(_dfX.columns)
        best_score = 0  # F1-score는 높을수록 좋음

        while remaining_features:
            best_feature = None
            for feature in remaining_features:
                _model.fit(_dfX[selected_features + [feature]], _y)
                score = f1_score(_y, _model.predict(_dfX[selected_features + [feature]]))

                if score > best_score:  # F1-score가 높아지는 방향으로 선택
                    best_score = score
                    best_feature = feature

            if best_feature:
                selected_features.append(best_feature)
                remaining_features.remove(best_feature)
            else:
                break

        return selected_features, best_score
        
    @staticmethod
    def Get_BackwardElimination( # 이진 분류.
                _dfX: pd.DataFrame,
                _y: pd.Series | np.ndarray =None,
                _model = SVC(),
                _targetColms: str = 'DIAG_NM',
            ) -> Tuple[List[str], float]: 
        
            _dfX = Data_Prep.Get_BinaryTargetDataFrame(_dfX, _targetColms)
            if _y is None :
                _y = EasyPanda.Get_TargetToSeries(_dfX,_targetColms)

            _dfX = EasyPanda.Get_AllNumType(_dfX)
            _dfX = _dfX.drop(columns=[_targetColms], errors='ignore')

            selected_features = list(_dfX.columns)
            _model.fit(_dfX[selected_features], _y)
            best_score = f1_score(_y, _model.predict(_dfX[selected_features]))
            
            while len(selected_features) > 0:
                worst_feature = None
                for feature in selected_features:
                    temp_features = selected_features.copy()
                    temp_features.remove(feature)
                    _model.fit(_dfX[temp_features], _y)
                    score = f1_score(_y, _model.predict(_dfX[temp_features]))
                    
                    if score > best_score:  # F1-score가 높아지는 방향으로 제거
                        best_score = score
                        worst_feature = feature
                
                if worst_feature:
                    selected_features.remove(worst_feature)
                else:
                    break
            
            return selected_features, best_score