# @title InitPrj
# 프로젝트 초기 셋팅
class InitPrj:
    @staticmethod
    def Help():
        print(textwrap.dedent(
            '''
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
            '''
        ))

        
    @staticmethod
    def Get_LoadDataNMerge(_rootDir = '', _fileName1 = 'activity.csv', _fileName2 = 'sleep.csv')->pd.DataFrame:
        for dataset_type in ['train', 'test']: 
            data_dir = f'{_rootDir}/{dataset_type}'
            
            df = {}
            df['activity'] = pd.read_csv(f'{data_dir}/'+_fileName1)
            df['sleep'] = pd.read_csv(f'{data_dir}/'+_fileName2)
            
            # 수면 끝 - 활동 시작
            df['activity']['date'] = df['activity']['activity_day_start']
            df['activity']['date'] = pd.to_datetime(df['activity']['date'])
            df['activity']['date'] = df['activity']['date'].dt.date
            df['activity'] = df['activity'].drop(columns=['activity_day_start', 'activity_day_end'])

            df['sleep']['date'] = df['sleep']['sleep_bedtime_end']
            df['sleep']['date'] = pd.to_datetime(df['sleep']['date'])
            df['sleep']['date'] = df['sleep']['date'].dt.date
            df['sleep'] = df['sleep'].drop(columns=['sleep_bedtime_start', 'sleep_bedtime_end'])
            
            merged_df = pd.merge(df['activity'], df['sleep'], how='inner', on=['date', 'EMAIL', 'DIAG_NM'])
            
            save_path = f'{data_dir}/merged.csv'
            merged_df.to_csv(save_path, index=False)
            return merged_df

    @staticmethod
    def load_data(_df_path: str, _drop_cols: list, _target: str = None) -> pd.DataFrame:
        '''
        - load_data
	        - 설명 : dataframe을 load하는 함수
	        - input
			        - df_path : 파일경로
			        - drop_cols : 삭제할 columns 이름이 담긴 리스트
		      - output
	        - pd.Dataframe
        '''
        df = pd.read_csv(_df_path)
        df = df.drop(_drop_cols, axis=1)
        return df