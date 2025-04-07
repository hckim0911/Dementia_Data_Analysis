# @title 이지 판다스
class EasyPanda: # 판다스 문법 랩핑
    @staticmethod
    def Help():
        print(textwrap.dedent(
            '''
            ###EasyPand (판다스 랩핑)###

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
            '''
        ))

    @staticmethod
    def Set_SaveCSV(_df:pd.DataFrame, _name:str = 'saveData.csv', _path : str = ''):
        _df.to_csv( _path +_name,index = False)

    @staticmethod
    def Get_DropColms(_df: pd.DataFrame, _dropColms: list = [])->pd.DataFrame: 
        return _df.drop(columns=_dropColms, errors='ignore')

    @staticmethod
    def Get_AllObjType(_df: pd.DataFrame)->pd.DataFrame: # object 탑입 컬럼만 df 형태로 반환
        return _df.select_dtypes(include=['object'],)

    @staticmethod
    def Get_AllNumType(_df: pd.DataFrame)->pd.DataFrame: # num 탑입 컬럼만 df 형태로 반환
        return _df.select_dtypes(include=['number'])

    @staticmethod
    def Get_AllNumTypeWithTarget(_df: pd.DataFrame, _targetColms: str = 'DIAG_NM') -> pd.DataFrame:
        """
        타겟 컬럼을 제외하고, 수치형 컬럼만 남긴 DataFrame을 반환합니다.
        """
        _features = _df.drop(columns=[_targetColms], errors='ignore')
        return _features.select_dtypes(include=['number'])
        
    @staticmethod
    def Get_AllObjTypeWithTarget(_df: pd.DataFrame, _targetColms: str = 'DIAG_NM') -> pd.DataFrame:
        """
        타겟 컬럼을 제외하고, 범주형 컬럼만 남긴 DataFrame을 반환합니다.
        """
        _features = _df.drop(columns=[_targetColms], errors='ignore')
        return _features.select_dtypes(include=['object'])

    @staticmethod
    def Get_ConvertToDatetime(_df: pd.DataFrame, _selectColms: list = []) -> pd.DataFrame:
        for col in _selectColms:
            if col in _df.columns:
                _df[col] = pd.to_datetime(_df[col], errors='coerce')
        return _df

    @staticmethod# 결측치를 지정한 값으로 채움
    def Get_Fillna(_df, _value)-> pd.DataFrame:
        return _df.fillna(_value)

    @staticmethod# 결측치가 있는 행 제거
    def Get_Dropna(_df, _how='any')-> pd.DataFrame:
        return _df.dropna(how=_how)

    @staticmethod
    def Get_RenameColumns(_df, _renameDict)-> pd.DataFrame:
        # 컬럼명 변경
        return _df.rename(columns=_renameDict)

    @staticmethod
    def Get_ColmsStats(_df, _colms : list = None) :
        # 지정된 컬럼들 또는 모든 컬럼에 대한 통계 정보 반환
        # _colms가 None이면 모든 컬럼 처리
        if _colms is None:
            _colms = _df.columns.tolist()
    
    @staticmethod
    def Get_TargetToSeries(_df: pd.DataFrame, _targetColms: str ='DIAG_NM' ) -> pd.Series:
        if _targetColms not in _df.columns:
            raise ValueError(f"타겟 컬럼 '{_targetColms}'이(가) 데이터프레임에 존재하지 않습니다.")
        _target_series = _df[_targetColms]
        # Series 보장 (보통은 이미 Series지만, 혹시 모르니 방어적)
        if not isinstance(_target_series, pd.Series):
            _target_series = _target_series.squeeze()
        return _target_series

    @staticmethod
    def Get_TargetToList(_df: pd.DataFrame, _targetColms: str = 'DIAG_NM') -> list:
        """
        지정한 타겟 컬럼을 리스트(list) 형태로 반환합니다.
        """
        if _targetColms not in _df.columns:
            raise ValueError(f"타겟 컬럼 '{_targetColms}'이(가) 데이터프레임에 없습니다.")
        return _df[_targetColms].tolist()
        
    @staticmethod
    def Get_TargetToArray(_df: pd.DataFrame, _targetColms: str = 'DIAG_NM') -> np.ndarray:
        """
        지정한 타겟 컬럼을 NumPy 배열(ndarray) 형태로 반환합니다.
        """
        if _targetColms not in _df.columns:
            raise ValueError(f"타겟 컬럼 '{_targetColms}'이(가) 데이터프레임에 존재하지 않습니다.")
        return _df[_targetColms].to_numpy()
