# EasyPand (판다스 랩핑)

## Set_SaveCSV(_df, _name, _path)
**Describe**
* 인자로 받은 DataFrame을 지정한 경로와 이름으로 CSV 파일로 저장.

**Parameter**
| Parameter | Type | Description | Default Value |
|---|---|---|---|
| `_df` | `pd.DataFrame` | CSV 파일로 저장할 pandas DataFrame | (필수) |
| `_name` | `str` | 저장할 CSV 파일의 이름 | 'saveData.csv' |
| `_path` | `str` | 파일을 저장할 경로 | 현재 작업 디렉토리 |

**Return**
* None

## Get_DropColms(_df, _dropColms)
**Describe**
* 지정된 컬럼 리스트 중 존재하는 컬럼만 제거하여 DataFrame을 안전하게 반환.

**Parameter**
| Parameter | Type | Description | Default Value |
|---|---|---|---|
| `_df`    | `pd.DataFrame` | 열을 제거할 pandas DataFrame 객체 | (필수) |
| `_dropColms` | `list` | 제거할 열 이름들의 리스트 | `[]` (빈 리스트) |

**Return**
* 지정된 컬럼이 제거된 새로운 DataFrame

## Get_AllObjType(_df)
**Describe**
* 문자열(object) 타입의 컬럼만 추출.

**Parameter**
| Parameter | Type | Description | Default Value |
|---|---|---|---|
| `_df`    | `pd.DataFrame` | 열을 제거할 pandas DataFrame 객체 | (필수) |

**Return**
* 문자열(Object) 컬럼만 포함된 DataFrame

## Get_AllNumType(_df)
**Describe**
* 숫자(int, float 등) 타입의 컬럼만 추출

**Parameter**
| Parameter | Type | Description | Default Value |
|---|---|---|---|
| `_df`    | `pd.DataFrame` | 열을 제거할 pandas DataFrame 객체 | (필수) |

**Return**
* 수치형(numeric) 컬럼만 포함된 DataFrame

## Get_AllNumTypeWithTarget(_df, _targetColms)
**Describe**
* 타겟 컬럼을 제외하고 수치형 컬럼만 추출

**Parameter**
| Parameter | Type | Description | Default Value |
|---|---|---|---|
| `_df` | `pd.DataFrame` | 처리할 pandas DataFrame 객체 | (필수) |
| `_targetColms` | `str` | 처리할 대상 열의 이름| Parme |

**Return**
* 타겟 컬럼을 제외한 수치형 컬럼 DataFrame

## Get_ConvertToDatetime(_df, _selectColms)
**Describe**
* 선택한 컬럼들을 datetime 타입으로 변환
 
**Parameter**
| Parameter | Type | Description | Default Value |
|---|---|---|---|
| `_df` | `pd.DataFrame` | 날짜형으로 변환할 pandas DataFrame | (필수) |
| `_selectColms` | `list` | datetime으로 변환할 컬럼명 리스트 | `[]` |

**Return**
* 변환된 DataFrame

## Get_Fillna(_df, _value)
**Describe**
* 모든 결측치를 지정한 값으로 채움

**Parameter**
| Parameter | Type | Description | Default Value |
|---|---|---|---|
| `_df` | `pd.DataFrame` | 결측치를 채울 pandas DataFrame | (필수) |
| `_value` | `object` | 결측치를 대체할 값 | 없음 |

**Return**
* 결측치가 채워진 DataFrame
 
## Get_Dropna(_df, _how)
**Describe**
* 결측치가 포함된 행을 제거
  
**Parameter**
| Parameter | Type | Description | Default Value |
|---|---|---|---|
| `_df` | `pd.DataFrame` | 결측치를 제거할 pandas DataFrame | (필수) |
| `_how` | `str` | 'any': 하나라도 NaN이면 제거, 'all': 모두 NaN일 경우만 제거 | 'any' |

**Return**
* 결측치가 제거된 DataFrame

## Get_RenameColumns(_df, _renameDict)
**Describe**
* 컬럼명을 딕셔너리의 매핑에 따라 변경
  
**Parameter**
| Parameter | Type | Description | Default Value |
|---|---|---|---|
| `_df` | `pd.DataFrame` | 컬럼명을 변경할 pandas DataFrame | (필수) |
| `_renameDict` | `dict` | {'기존컬럼명': '새컬럼명'} 형태의 딕셔너리 | (필수) |

**Return**
* 컬럼명이 변경된 DataFrame

## Get_TargetToSeries(_df, _targetColms)
**Describe**
* 지정한 컬럼을 Series 형태로 반환

**Parameter**
| Parameter | Type | Description | Default Value |
|---|---|---|---|
| `_df` | `pd.DataFrame` | 대상 pandas DataFrame | (필수) |
| `_targetColms` | `str` | 추출할 컬럼명 | 'DIAG_NM' |

**Return**
* pd.Series 형태의 타겟 컬럼

## Get_TargetToList(_df, _targetColms)
**Describe**
* 지정한 컬럼을 리스트(list) 형태로 반환

**Parameter**
| Parameter | Type | Description | Default Value |
|---|---|---|---|
| `_df` | `pd.DataFrame` | 대상 pandas DataFrame | (필수) |
| `_targetColms` | `str` | 추출할 컬럼명 | 'DIAG_NM' |

**Return**
* list 형태의 타겟 컬럼

## Get_TargetToArray(_df, _targetColms)
**Describe**
* 지정한 컬럼을 NumPy 배열(ndarray) 형태로 반환

**Parameter**
| Parameter | Type | Description | Default Value |
|---|---|---|---|
| `_df` | `pd.DataFrame` | 대상 pandas DataFrame | (필수) |
| `_targetColms` | `str` | 추출할 컬럼명 | 'DIAG_NM' |

**Return**
* NumPy 배열(ndarray) 형태의 타겟 컬럼

# InitPrj (프로젝트 초기 세팅)

## Get_LoadDataNMerge(_rootDir, _fileName1, _fileName2)
**Describe**
* root 디렉토리 하위의 train/test 폴더에서 두 개의 csv(activity, sleep)를 읽고, 날짜 기준으로 병합하여 merged.csv로 저장한 뒤 반환

**Parameter**
| Parameter | Type | Description | Default Value |
|---|---|---|---|
| `_rootDir` | `str` | 데이터가 저장된 상위 폴더 경로 | (필수) |
| `_fileName1` | `str` | 활동(activity) 데이터 파일명 | 'activity.csv' |
| `_fileName2` | `str` | 수면(sleep) 데이터 파일명 | 'sleep.csv' |

**Return**
* 병합된 DataFrame

## load_data(_df_path, _drop_cols, _target)
**Describe**

* CSV 파일을 로드하고, 지정된 컬럼들을 제거하여 DataFrame으로 반환

**Parameter**
| Parameter | Type | Description | Default Value |
|---|---|---|---|
| `_df_path` | `str` | 불러올 CSV 파일 경로 | (필수) |
| `_drop_cols` | `list` | 제거할 컬럼명 리스트 | (필수) |
| `_target` | `str` | (옵션) 타겟 컬럼 이름 (현재 로직에서 사용하지 않음) | None |

**Return**
* 컬럼이 제거된 DataFrame

# Data_Prep(데이터 전처리)

## Get_BinaryTargetDataFrame(_df, _targetColms, _zeroTarget)
**Describe**
* 다중 클래스 타겟 컬럼을 이진 분류용으로 변환합니다. zeroTarget 값이면 0, 그 외는 1로 변환

**Parameter**
| Parameter | Type | Description | Default Value |
|---|---|---|---|
| `_df` | `pd.DataFrame` | 변환할 pandas DataFrame | (필수) |
| `_targetColms` | `str` | 이진 분류로 변환할 대상 컬럼 이름 | 'DIAG_NM' |
| `_zeroTarget` | `str` | 기준이 되는 클래스 이름 | 'CN' |

**Return**
* 이진 변환된 타겟 컬럼이 포함된 DataFrame

# Draw (시각화)

## test()
**Describe**
* 시각화 모듈이 정상 연결되었는지 확인용 테스트 함수

**Return**
* 없음

## Draw_HeatMap(_df, _title, _size, _dfName)
**Describe**
* 수치형 컬럼 간의 상관계수를 히트맵으로 시각화

**Parameter**
| Parameter | Type | Description | Default Value |
|---|---|---|---|
| `_df` | `pd.DataFrame` | 상관관계를 분석할 pandas DataFrame | (필수) |
| `_title` | `str` | 그래프 제목 | '' |
| `_size` | `int` | 히트맵 크기 | 10 |
| `_dfName` | `str` | 데이터셋 이름 (제목에 활용됨) | '' |

**Return**
* 없음

## Draw_Bar(df, colm, labeX, lableY, title)
**Describe**
* 범주형 변수(기본: DIAG_NM)의 빈도수를 바 차트로 시각화

**Parameter**
| Parameter | Type | Description | Default Value |
|---|---|---|---|
| `df` | `pd.DataFrame` | 시각화할 pandas DataFrame | (필수) |
| `colm` | `str` | 카운트할 컬럼 이름 | 'DIAG_NM' |
| `labeX` | `str` | x축 라벨 | '' |
| `lableY` | `str` | y축 라벨 | '' |
| `title` | `str` | 그래프 제목 | '' |

**Return**
* 없음

## Draw_DensityHistogram(_df, _xFeature, _hueTarget, _bins)
**Describe**
* 지정한 feature의 밀도 기반 히스토그램을 그림

**Parameter**
| Parameter | Type | Description | Default Value |
|---|---|---|---|
| `_df` | `pd.DataFrame` | 시각화할 pandas DataFrame | (필수) |
| `_xFeature` | `str` | 시각화할 수치형 컬럼 | 'activity_average_met' |
| `_hueTarget` | `str` | 색상 기준 범주 컬럼 | 'DIAG_NM' |
| `_bins` | `int` | 히스토그램 구간 수 | 50 |

**Return**
* 없음

## Draw_AllDensityHistogram(_df, _bins_list, _targetCol)
**Describe**
* 수치형 변수들에 대해 다양한 bin 크기로 밀도 기반 히스토그램을 서브플롯으로 시각화

**Parameter**
| Parameter | Type | Description | Default Value |
|---|---|---|---|
| `_df` | `pd.DataFrame` | 시각화할 pandas DataFrame | (필수) |
| `_bins_list` | `list` | bin 개수 리스트 | `[]` |
| `_targetCol` | `str` | 범주형 기준 컬럼 | 'DIAG_NM' |

**Return**
* 없음

## Draw_Scatter(_df, _targetColm, _SelectColm, _labelX, _labelY, _title, _legend, _size)
**Describe**
* 기본적으로 PCA 결과(PC1, PC2)를 기준으로 타겟에 따른 산점도를 그림

**Parameter**
| Parameter | Type | Description | Default Value |
|---|---|---|---|
| `_df` | `pd.DataFrame` | 시각화할 pandas DataFrame | (필수) |
| `_targetColm` | `str` | 색상 기준 컬럼 | 'DIAG_NM' |
| `_SelectColm` | `list` | 사용할 컬럼 리스트 | `[]` |
| `_labelX` | `str` | x축 라벨 | '' |
| `_labelY` | `str` | y축 라벨 | '' |
| `_title` | `str` | 그래프 제목 | '' |
| `_legend` | `str` | 범례 제목 | '' |
| `_size` | `int` | 그래프 크기 | None |

**Return**
* 없음

## Draw_PCA(_df, _targetColm, _selectColms, _n_components)
**Describe**
* PCA 수행 후 2D 산점도로 시각화합니다. 내부적으로 Draw_Scatte림

**Parameter**
| Parameter | Type | Description | Default Value |
|---|---|---|---|
| `_df` | `pd.DataFrame` | 시각화할 pandas DataFrame | (필수) |
| `_highCorr_df` | `pd.DataFrame` | 변수쌍 데이터프레임 (없으면 자동 추출) | (필수) |
| `_targetColms` | `str` | 색상 기준 컬럼 | 'DIAG_NM' |

**Return**
* 없음

# Feature_Selection (피처 선택 및 제거)

## Get_DelColm(df, delCol)
**Describe**
* 지정된 컬럼들을 제거한 DataFrame을 반환

**Parameter**
| Parameter | Type | Description | Default Value |
|---|---|---|---|
| `df` | `pd.DataFrame` | 컬럼을 제거할 pandas DataFrame | (필수) |
| `delCol` | `list` | 제거할 컬럼 리스트 | `[]` |

**Return**
* 컬럼이 제거된 DataFrame

## Get_Calc_VIF(df, targetColm, axisNum, feature, VIFValue)
**Describe**
* VIF(분산팽창계수)를 계산하여 다중공선성이 높은 피처를 탐지

**Parameter**
| Parameter | Type | Description | Default Value |
|---|---|---|---|
| `df` | `pd.DataFrame` | VIF를 계산할 pandas DataFrame | (필수) |
| `targetColm` | `list` | 제거할 대상(타겟) 컬럼 리스트 | `[]` |
| `axisNum` | `int` | 제거할 축 번호 | 1 |
| `feature` | `str` | 결과에 표시될 컬럼 이름 | 'Feature' |
| `VIFValue` | `str` | 결과에 표시될 VIF 수치 컬럼명 | 'VIF Value' |

**Return**
* 컬럼별 VIF 값이 포함된 DataFrame (내림차순 정렬)

## Get_HighCorrPairs(_df, _threshold, _positive)
**Describe**
* 상관계수가 임계값 이상인 변수쌍만 추출

**Parameter**
| Parameter | Type | Description | Default Value |
|---|---|---|---|
| `_df` | `pd.DataFrame` | 상관계수를 계산할 pandas DataFrame | (필수) |
| `_threshold` | `float` | 상관계수 기준값 | 0.8 |
| `_positive` | `bool` or `None` | True: 양의 상관관계만, False: 음의 상관관계만, None: 양/음 모두 포함 | None |

**Return**
* MultiIndex로 (변수1, 변수2), 값은 상관계수인 DataFrame

## Get_ForwardSelection(_dfX, _y, _model, _targetColms)
**Describe**
* F1-score를 기준으로 가장 성능이 좋은 피처들을 순차적으로 추가

**Parameter**
| Parameter | Type | Description | Default Value |
|---|---|---|---|
| `_dfX` | `pd.DataFrame` | 피처 선택을 수행할 pandas DataFrame | (필수) |
| `_y` | `pd.Series` | 타겟 변수 | (필수) |
| `_model` | `object` | 사용할 모델 | `SVC()` |
| `_targetColms` | `str` | 타겟 컬럼명 | 'DIAG_NM' |

**Return**
* 선택된 피처 리스트, 최고 F1-score

## Get_BackwardElimination(_dfX, _y, _model, _targetColms)
**Describe**
* 모든 피처에서 시작하여 F1-score가 개선되는 방향으로 피처를 제거

**Parameter**
| Parameter | Type | Description | Default Value |
|---|---|---|---|
| `_dfX` | `pd.DataFrame` | 피처 선택을 수행할 pandas DataFrame | (필수) |
| `_y` | `pd.Series` | 타겟 변수 | (필수) |
| `_model` | `object` | 사용할 모델 | `SVC()` |
| `_targetColms` | `str` | 타겟 컬럼명 | 'DIAG_NM' |

**Return**
* 선택된 피처 리스트, 최고 F1-score

# Feature_Engineering (피처 엔지니어링)

## Get_PCA(_df, _targetColm, _selectColms, _n_components, newNamePC1, newNamePC2)
**Describe**
* 수치형 데이터를 대상으로 PCA를 수행하고, 지정된 주성분 이름으로 결과 컬럼명을 설정하여 반환

**Parameter**
| Parameter | Type | Description | Default Value |
|---|---|---|---|
| `_df` | `pd.DataFrame` | PCA를 수행할 pandas DataFrame | (필수) |
| `_targetColm` | `str` | 타겟 컬럼명 | 'DIAG_NM' |
| `_selectColms` | `list` | PCA 적용할 컬럼 리스트 (비워두면 전체 수치형 사용) | `[]` |
| `_n_components` | `int` | 추출할 주성분 개수 | 2 |
| `newNamePC1` | `str` | 첫 번째 주성분의 컬럼명 (2차원 PCA일 경우) | 'PC1' |
| `newNamePC2` | `str` | 두 번째 주성분의 컬럼명 (2차원 PCA일 경우) | 'PC2' |

**Return**
* PCA 결과 + 타겟 컬럼이 포함된 DataFrame

## Get_OneHotEncoding(_df, _columns, _numType)
**Describe**
* 지정된 또는 자동 감지된 범주형 컬럼에 대해 One-Hot Encoding을 적용

**Parameter**
| Parameter | Type | Description | Default Value |
|---|---|---|---|
| `_df` | `pd.DataFrame` | One-Hot Encoding을 적용할 pandas DataFrame | (필수) |
| `_columns` | `list` | 원-핫 인코딩할 컬럼 리스트 (None이면 자동 감지) | None |
| `_numType` | `bool` | True → 숫자 타입 유지, False → 불리언 타입으로 변환 | True |

**Return**
* 원-핫 인코딩된 DataFrame

## Get_ScaledDict(_df, _targetColms)
**Describe**
* 여러 종류의 스케일링 기법(Z-score, MinMax, Robust, Yeo-Johnson)을 적용한 결과를 딕셔너리로 반환

**Parameter**
| Parameter | Type | Description | Default Value |
|---|---|---|---|
| `_df` | `pd.DataFrame` | 스케일링을 적용할 pandas DataFrame | (필수) |
| `_targetColms` | `str` | 타겟 컬럼명 | 'DIAG_NM' |

**Return**
* {스케일링 방식 이름: 정규화된 DataFrame} 형식의 딕셔너리

## Get_ZScoreScaled(_df, _targetColms)
**Describe**
* Z-score(표준화) 정규화를 적용한 데이터프레임을 반환

**Parameter**
| Parameter | Type | Description | Default Value |
|---|---|---|---|
| `_df` | `pd.DataFrame` | Z-score 스케일링을 적용할 pandas DataFrame | (필수) |
| `_targetColms` | `str` | 타겟 컬럼명 | 'DIAG_NM' |

**Return**
* 정규화된 수치형 컬럼 + 타겟 컬럼이 포함된 DataFrame

## test()
**Describe**
* 샘플링 또는 전체 파이프라인 흐름 관련 클래스가 정상적으로 연결되었는지 확인하는 테스트 함수

**Return**
* 없음

# Modeling (모델링)

## Get_InitModels(_PersonalModelDict)
**Describe**
* 기본 제공되는 분류/군집 모델들과 사용자 정의 모델을 합쳐 딕셔너리 형태로 반환

**Parameter**
| Parameter | Type | Description | Default Value |
|---|---|---|---|
| `_PersonalModelDict` | `Dict[str, Any]` | 사용자 정의 모델 딕셔너리 (예: {"MyModel": 모델객체}) | `{}` |

**Return**
* {"모델명": 모델객체} 형태의 딕셔너리

## Set_auto_tree_model(_df, _targetColms, test_df, visualize, max_depth, min_samples_split, min_samples_leaf, class_balance)
**Describe**
* 주어진 데이터로 분류 또는 회귀 여부를 자동 판별한 후 결정 트리 모델을 학습하고 평가합니다. 테스트 데이터가 있으면 평가에 사용되며, 시각화 옵션에 따라 결과도 출력

**Parameter**
| Parameter | Type | Description | Default Value |
|---|---|---|---|
| `_df` | `pd.DataFrame` | 학습 데이터프레임 | (필수) |
| `_targetColms` | `str` | 타겟 컬럼 이름 | 'DIAG_NM' |
| `test_df` | `pd.DataFrame` | 테스트용 데이터프레임 (None: 훈련 데이터에서 분할) | None |
| `visualize` | `bool` | 트리 및 결과 시각화 여부 | True |
| `max_depth` | `int` | 트리 최대 깊이 | 10 |
| `min_samples_split` | `int` | 내부 노드를 분할하기 위한 최소 샘플 수 | 5 |
| `min_samples_leaf` | `int` | 리프 노드가 되기 위한 최소 샘플 수 | 2 |
| `class_balance` | `bool` | 클래스 불균형 자동 처리 여부 (분류일 경우에만 적용) | True |

**Return**
* model: 학습된 결정 트리 모델 객체
* feature_importance_df: 피처 중요도 DataFrame (중요도 기준 내림차순 정렬)
