#@title Feature_Engineering
class Feature_Engineering: # 피쳐 엔지니어링.
    def Help():
        print(textwrap.dedent(
            '''
            ### Feature_Engineering (피처 엔지니어링) ###
            
            # PCA 수행 및 결과 반환 : Get_PCA(_df, _targetColm='DIAG_NM', _selectColms=[], _n_components=2, newNamePC1='PC1', newNamePC2='PC2') -> pd.DataFrame
                설명) 수치형 데이터를 대상으로 PCA를 수행하고, 지정된 주성분 이름으로 결과 컬럼명을 설정하여 반환합니다.
                인자)
                    _targetColm: 타겟 컬럼명
                    _selectColms: PCA 적용할 컬럼 리스트 (비워두면 전체 수치형 사용)
                    _n_components: 추출할 주성분 개수
                    newNamePC1: 첫 번째 주성분의 컬럼명 (2차원 PCA일 경우)
                    newNamePC2: 두 번째 주성분의 컬럼명 (2차원 PCA일 경우)
                반환) PCA 결과 + 타겟 컬럼이 포함된 DataFrame

            # One-Hot Encoding : Get_OneHotEncoding(_df, _columns=None, _numType=True) -> pd.DataFrame
                설명) 지정된 또는 자동 감지된 범주형 컬럼에 대해 One-Hot Encoding을 적용합니다.
                인자)
                    _columns: 원-핫 인코딩할 컬럼 리스트 (None이면 자동 감지)
                    _numType: True → 숫자 타입 유지, False → 불리언 타입으로 변환
                반환) 원-핫 인코딩된 DataFrame

            # 다양한 스케일링 결과 반환 : Get_ScaledDict(_df, _targetColms='DIAG_NM') -> dict[str, pd.DataFrame]
                설명) 여러 종류의 스케일링 기법(Z-score, MinMax, Robust, Yeo-Johnson)을 적용한 결과를 딕셔너리로 반환합니다.
                인자)
                    _targetColms: 타겟 컬럼명
                반환) {스케일링 방식 이름: 정규화된 DataFrame} 형식의 딕셔너리

            # Z-Score 정규화 : Get_ZScoreScaled(_df, _targetColms='DIAG_NM') -> pd.DataFrame
                설명) Z-score(표준화) 정규화를 적용한 데이터프레임을 반환합니다.
                인자)
                    _targetColms: 타겟 컬럼명
                반환) 정규화된 수치형 컬럼 + 타겟 컬럼이 포함된 DataFrame
            '''
        ))

    @staticmethod
    def Get_PCA(_df: pd.DataFrame, _targetColm: str = 'DIAG_NM', _selectColms : list = [],  _n_components = 2 ,newNamePC1:str = 'PC1', newNamePC2:str = 'PC2') -> pd.DataFrame:
        '''
        pca1, pca2 , (있다면)target  합친 df 반환.
\
        '''
        num_df = _df.select_dtypes(include=['int64', 'float64']).drop(columns=[_targetColm], errors='ignore')

        df_scaled = pd.DataFrame()
        if len(_selectColms) > 0: # 리스트에 요소가 존재한다면. 리스트의 요소만 PCA수행 아니면 모두 수행.
            df_scaled = num_df[_selectColms]
        else :
            df_scaled = num_df

        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df_scaled)

        #PCA 수행
        pca_2d = PCA(n_components = _n_components)
        df_pca_2d = pca_2d.fit_transform(df_scaled)
        #df_pca_2d = pd.DataFrame(df_pca_2d, columns=[newNamePC1, newNamePC2])

        if _n_components == 2:
            df_pca_2d = pd.DataFrame(df_pca_2d, columns=[newNamePC1, newNamePC2])
        else:
            col_names = [f'PC{i+1}' for i in range(_n_components)]
            df_pca_2d = pd.DataFrame(df_pca_2d, columns=col_names)

        #타겟 컬럼 붙이기
        if _targetColm in _df.columns: #타깃 컬럼이 있는지 확인
            df_pca_2d[_targetColm] = _df[_targetColm]

        return df_pca_2d#

    @staticmethod # df에 남아있는 모든 범주형 또는 선택한 피쳐에 one hot 인코딩 적용
    def Get_OneHotEncoding(_df, _columns:list = None, _numType:bool = True):  # 범주형 컬럼들을 one-hot encoding
        if _columns is None or len(_columns) == 0:
            # 범주형 컬럼 자동 감지 (object, category 타입)
            categorical_columns = _df.select_dtypes(include=['object', 'category']).columns.tolist()
        else:
            categorical_columns = _columns

        # 범주형 컬럼이 없으면 원본 데이터프레임 반환
        if len(categorical_columns) == 0:
            return _df

        # 원-핫 인코딩 수행
        encoded_df = pd.get_dummies(_df, columns=categorical_columns, drop_first=True)

        # Boolean 타입으로 변환 (_use_numeric이 False인 경우)
        if not _numType:
            for col in encoded_df.columns:
                if col not in _df.columns:  # 원-핫 인코딩으로 추가된 컬럼만 처리
                    encoded_df[col] = encoded_df[col].astype(bool)

        return encoded_df
    
    @staticmethod
    def Get_ScaledDict(_df :pd.DataFrame, _targetColms: str = 'DIAG_NM' ) -> dict[str, pd.DataFrame]:
        scalers = {
            'Z-score': StandardScaler(),
            'MinMax': MinMaxScaler(),
            'Robust': RobustScaler(),
            'Yeo-Johnson': PowerTransformer(method='yeo-johnson')
        }
        #df_x = _df.drop(columns=[_targetColms])
        df_x = EasyPanda.Get_AllNumType(_df)
        df_scaled = {}

        for name, scaler in scalers.items():
            scaled = scaler.fit_transform(df_x)
            df_temp = pd.DataFrame(scaled, columns=df_x.columns, index=_df.index)
            df_temp[_targetColms] = _df[_targetColms]
            df_scaled[name] = df_temp

        return df_scaled

    @staticmethod
    def Get_ZScoreScaled(_df: pd.DataFrame, _targetColms: str = 'DIAG_NM') -> pd.DataFrame:
        # 수치형 데이터만 추출 (사용자 정의 함수)
        df_x = EasyPanda.Get_AllNumType(_df)

        # Z-score 정규화 수행
        scaler = StandardScaler()
        scaled = scaler.fit_transform(df_x)

        # 정규화된 데이터프레임 생성
        df_scaled = pd.DataFrame(scaled, columns=df_x.columns, index=_df.index)

        # 타겟 컬럼 다시 합치기
        df_scaled[_targetColms] = _df[_targetColms]

        return df_scaled