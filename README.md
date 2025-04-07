##EasyPand (판다스 랩핑)

# 세이브 : Set_SaveCSV(_df: pd.DataFrame, _name: str = 'saveData.csv', _path: str = '')
    설명) 인자로 받은 DataFrame을 지정한 경로와 이름으로 CSV 파일로 저장합니다.
    인자)
        _name: str = 'saveData.csv' <- 저장할 파일 이름
        _path: str = '' <- 저장 경로 (기본값: 현재 경로)
    반환) 없음

# 컬럼 드랍 : Get_DropColms(_df: pd.DataFrame, _dropColms: list = []) -> pd.DataFrame
    설명) 지정된 컬럼 리스트 중 존재하는 컬럼만 제거하여 DataFrame을 안전하게 반환합니다.
    인자)
        _dropColms: list = [] <- 제거할 컬럼 이름 리스트 (예: ['이름', '나이'])
    반환) 지정된 컬럼이 제거된 새로운 DataFrame

# Object 타입 컬럼만 반환 : Get_AllObjType(_df: pd.DataFrame) -> pd.DataFrame
    설명) 문자열(object) 타입의 컬럼만 추출합니다.
    인자)
    반환) 문자열 컬럼만 포함된 DataFrame

# 수치형 컬럼만 반환 : Get_AllNumType(_df: pd.DataFrame) -> pd.DataFrame
    설명) 숫자(int, float 등) 타입의 컬럼만 추출합니다.
    인자)
    반환) 수치형 컬럼만 포함된 DataFrame

# 수치형 컬럼만 반환 (타겟 제외) : Get_AllNumTypeWithTarget(_df: pd.DataFrame, _targetColms: str = 'DIAG_NM') -> pd.DataFrame
    설명) 타겟 컬럼을 제외하고 수치형 컬럼만 추출합니다.
    인자)
        _targetColms: str = 'DIAG_NM' <- 제외할 타겟 컬럼 이름
    반환) 타겟 컬럼을 제외한 수치형 컬럼 DataFrame

# 날짜형으로 변환 : Get_ConvertToDatetime(_df: pd.DataFrame, _selectColms: list = []) -> pd.DataFrame
    설명) 선택한 컬럼들을 datetime 타입으로 변환합니다.
    인자)
        _selectColms: list = [] <- datetime으로 변환할 컬럼명 리스트
    반환) 변환된 DataFrame

# 결측치를 지정한 값으로 채움 : Get_Fillna(_df: pd.DataFrame, _value) -> pd.DataFrame
    설명) 모든 결측치를 지정한 값으로 채웁니다.
    인자)
        _value: object <- 결측치를 대체할 값
    반환) 결측치가 채워진 DataFrame

# 결측치가 있는 행 제거 : Get_Dropna(_df: pd.DataFrame, _how: str = 'any') -> pd.DataFrame
    설명) 결측치가 포함된 행을 제거합니다.
    인자)
        _how: str = 'any' <- 'any': 하나라도 NaN이면 제거, 'all': 모두 NaN일 경우만 제거
    반환) 결측치가 제거된 DataFrame

# 컬럼명 변경 : Get_RenameColumns(_df: pd.DataFrame, _renameDict: dict) -> pd.DataFrame
    설명) 컬럼명을 딕셔너리의 매핑에 따라 변경합니다.
    인자)
        _renameDict: dict <- {'기존컬럼명': '새컬럼명'} 형태의 딕셔너리
    반환) 컬럼명이 변경된 DataFrame

# 타겟 컬럼을 시리즈로 반환 : Get_TargetToSeries(_df: pd.DataFrame, _targetColms: str = 'DIAG_NM') -> pd.Series
    설명) 지정한 컬럼을 Series 형태로 반환합니다.
    인자)
        _targetColms: str = 'DIAG_NM' <- 추출할 컬럼명
    반환) pd.Series 형태의 타겟 컬럼

# 타겟 컬럼을 리스트로 반환 : Get_TargetToList(_df: pd.DataFrame, _targetColms: str = 'DIAG_NM') -> list
    설명) 지정한 컬럼을 리스트(list) 형태로 반환합니다.
    인자)
        _targetColms: str = 'DIAG_NM' <- 추출할 컬럼명
    반환) list 형태의 타겟 컬럼

# 타겟 컬럼을 배열로 반환 : Get_TargetToArray(_df: pd.DataFrame, _targetColms: str = 'DIAG_NM') -> np.ndarray
    설명) 지정한 컬럼을 NumPy 배열(ndarray) 형태로 반환합니다.
    인자)
        _targetColms: str = 'DIAG_NM' <- 추출할 컬럼명
    반환) NumPy 배열(ndarray) 형태의 타겟 컬럼


==================================================================================================================================================


###InitPrj (프로젝트 초기 세팅)###

# 데이터 로드 및 병합 : Get_LoadDataNMerge(_rootDir='', _fileName1='activity.csv', _fileName2='sleep.csv') -> pd.DataFrame
    설명) root 디렉토리 하위의 train/test 폴더에서 두 개의 csv(activity, sleep)를 읽고,
        날짜 기준으로 병합하여 merged.csv로 저장한 뒤 반환합니다.
    인자)
        _rootDir: 데이터가 저장된 상위 폴더 경로
        _fileName1: 활동(activity) 데이터 파일명 (기본값: 'activity.csv')
        _fileName2: 수면(sleep) 데이터 파일명 (기본값: 'sleep.csv')
    반환) 병합된 DataFrame

# 데이터 로드 및 컬럼 드랍 : load_data(_df_path: str, _drop_cols: list, _target: str = None) -> pd.DataFrame
    설명) CSV 파일을 로드하고, 지정된 컬럼들을 제거하여 DataFrame으로 반환합니다.
    인자)
        _df_path: 불러올 CSV 파일 경로
        _drop_cols: 제거할 컬럼명 리스트
        _target: (옵션) 타겟 컬럼 이름 (현재 로직에서 사용하지 않음)
    반환) 컬럼이 제거된 DataFrame


==================================================================================================================================================


###Data_Prep(데이터 전처리)###

# 이진 타겟 변환 : Get_BinaryTargetDataFrame(_df, _targetColms: str = 'DIAG_NM', _zeroTarget: str = 'CN') -> pd.DataFrame
    설명) 다중 클래스 타겟 컬럼을 이진 분류용으로 변환합니다.
        zeroTarget 값이면 0, 그 외는 1로 변환합니다.
    인자)
        _targetColms: 이진 분류로 변환할 대상 컬럼 이름
        _zeroTarget: 기준이 되는 클래스 이름 (예: 'CN')
    반환) 이진 변환된 타겟 컬럼이 포함된 DataFrame


==================================================================================================================================================


 ###Draw (시각화)###

# 테스트 출력 : test()
    설명) 시각화 모듈이 정상 연결되었는지 확인용 테스트 함수입니다.
    반환) 없음

# 상관관계 히트맵 : Draw_HeatMap(_df, _title='', _size=10, _dfName='')
    설명) 수치형 컬럼 간의 상관계수를 히트맵으로 시각화합니다.
    인자)
        _title: 그래프 제목
        _size: 히트맵 크기
        _dfName: 데이터셋 이름 (제목에 활용됨)
    반환) 없음

# 바 차트 : Draw_Bar(df, colm='DIAG_NM', labeX='', lableY='', title='')
    설명) 범주형 변수(기본: DIAG_NM)의 빈도수를 바 차트로 시각화합니다.
    인자)
        colm: 카운트할 컬럼 이름
        labeX: x축 라벨
        lableY: y축 라벨
        title: 그래프 제목
    반환) 없음

# 밀도 기반 히스토그램 : Draw_DensityHistogram(_df, _xFeature='activity_average_met', _hueTarget='DIAG_NM', _bins=50)
    설명) 지정한 feature의 밀도 기반 히스토그램을 그립니다.
    인자)
        _xFeature: 시각화할 수치형 컬럼
        _hueTarget: 색상 기준 범주 컬럼
        _bins: 히스토그램 구간 수
    반환) 없음

# 모든 수치형 컬럼에 대한 히스토그램 : Draw_AllDensityHistogram(_df, _bins_list=[], _targetCol='DIAG_NM')
    설명) 수치형 변수들에 대해 다양한 bin 크기로 밀도 기반 히스토그램을 서브플롯으로 시각화합니다.
    인자)
        _bins_list: bin 개수 리스트
        _targetCol: 범주형 기준 컬럼
    반환) 없음

# 2차원 산점도 : Draw_Scatter(_df, _targetColm='DIAG_NM', _SelectColm=[], _labelX='', _labelY='', _title='', _legend='', _size=None)
    설명) 기본적으로 PCA 결과(PC1, PC2)를 기준으로 타겟에 따른 산점도를 그립니다.
    인자)
        _targetColm: 색상 기준 컬럼
        _labelX: x축 라벨
        _labelY: y축 라벨
        _title: 그래프 제목
        _legend: 범례 제목
        _size: 그래프 크기
    반환) 없음

# PCA 시각화 : Draw_PCA(_df, _targetColm='DIAG_NM', _selectColms=[], _n_components=2)
    설명) PCA 수행 후 2D 산점도로 시각화합니다. 내부적으로 Draw_Scatter를 호출합니다.
    인자)
        _targetColm: 범주 기준 컬럼
        _selectColms: PCA 적용할 컬럼 리스트
        _n_components: 주성분 개수
    반환) 없음

# 상위 상관관계 히트맵 : Draw_HeatMapGang(_df, _targetCol='DIAG_NM', size=20)
    설명) 상위 상관관계를 중심으로 하는 대형 히트맵 시각화
    인자)
        _targetCol: 타겟 컬럼명
        size: 그래프 크기
    반환) 없음

# 전체 컬럼 박스플롯 : Draw_BoxplotAllColms(_df, _targetColms='DIAG_NM')
    설명) 수치형 컬럼에 대한 박스 플롯을 타겟 기준으로 여러 개 시각화합니다.
    인자)
        _targetColms: 기준이 되는 타겟 컬럼
    반환) DataFrame (타겟 포함된 수치형 데이터프레임)

# 전체 컬럼 산점도 : Draw_ScatterPlotAllColms(_df, _itSoLong_Are_you_sure: bool, _targetColms='DIAG_NM')
    설명) 모든 수치형 컬럼 조합에 대해 타겟 기준 산점도를 그립니다. 수가 많을 수 있습니다.
    인자)
        _itSoLong_Are_you_sure: 실행 전 확인용 안전장치
        _targetColms: 색상 기준 컬럼
    반환) 없음

# 상관 높은 변수 간 산점도 : Draw_ScatterPlotAboutCorr(_df, _highCorr_df, _targetColms='DIAG_NM')
    설명) 상관계수가 높은 변수 쌍을 대상으로 타겟 기준 산점도를 그립니다.
    인자)
        _highCorr_df: 변수쌍 데이터프레임 (없으면 자동 추출)
        _targetColms: 색상 기준 컬럼
    반환) 없음


==================================================================================================================================================


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


==================================================================================================================================================


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


==================================================================================================================================================


### Sampling (샘플링 / 파이프라인 전체 흐름 제어) ###

# 테스트 출력 : test()
    설명) 샘플링 또는 전체 파이프라인 흐름 관련 클래스가 정상적으로 연결되었는지 확인하는 테스트 함수입니다.
    반환) 없음


==================================================================================================================================================


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


==================================================================================================================================================
