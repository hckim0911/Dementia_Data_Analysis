# @title Data_Prep
#데이터 전처리.
class Data_Prep:  # 데이터 전처리
    @staticmethod
    def Help():
        print(textwrap.dedent(
            '''
            ###Data_Prep(데이터 전처리)###

            # 이진 타겟 변환 : Get_BinaryTargetDataFrame(_df, _targetColms: str = 'DIAG_NM', _zeroTarget: str = 'CN') -> pd.DataFrame
                설명) 다중 클래스 타겟 컬럼을 이진 분류용으로 변환합니다.
                    zeroTarget 값이면 0, 그 외는 1로 변환합니다.
                인자)
                    _targetColms: 이진 분류로 변환할 대상 컬럼 이름
                    _zeroTarget: 기준이 되는 클래스 이름 (예: 'CN')
                반환) 이진 변환된 타겟 컬럼이 포함된 DataFrame
            '''
        ))



    @staticmethod
    def Get_BinaryTargetDataFrame(_df, _targetColms:str = 'DIAG_NM', _zeroTarget:str = 'CN')->pd.DataFrame: #3클래스 분류를 2클래스 분류로 변환
        #출력 merge되어 0과 1로 출력
        #return _df[_targetColms] = np.where(_df[_targetColms] == _zeroTarget, 0, 1)
        _df[_targetColms] = np.where(_df[_targetColms] == _zeroTarget, 0, 1)
        return _df
