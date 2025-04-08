#@title 시각화
# 시각화
class Draw:
    @staticmethod
    def Help():
        print(textwrap.dedent(
            '''
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
            '''
        ))


    @staticmethod
    def Draw_HeatMap(_df, _title: str = '', _size: int = 10, _dfName: str = ""):
        numeric_df = _df.select_dtypes(include=['int64', 'float64'])  # 숫자형 선별
        corr = numeric_df.corr()

        if _size is not None:
            plt.figure(figsize=(_size, _size))

        sns.heatmap(corr, annot=True, fmt=".2f", cmap='viridis', square=True, cbar=True)
        plt.title('Correlation Heatmap of ' + _dfName + ' Features', fontsize=16)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def Draw_Bar(df: pd.DataFrame, colm: str = 'DIAG_NM' ,
        labeX: str = "", lableY: str = "",title: str = ""):
        '''
        타깃의 바 차트를 그립니다.
        기본 타깃 = DIAG_NM
        #사용법
            Draw_Bar(df)
        '''
        value_counts = df[colm].value_counts()

        value_counts.plot(kind='bar')
        if title != None:
            plt.title(title)

        plt.xlabel(labeX)
        plt.ylabel(lableY)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def Draw_DensityHistogram(_df, _xFeature = 'activity_average_met', _hueTarget = 'DIAG_NM', _bins:int = 50  ):
        #밀도 기반 히스토그램 1개 그리기
        feature = _xFeature
        target = _hueTarget
        bins = _bins
        sns.histplot(x=feature, data=df, hue=target, bins=bins, kde=True, stat="density", common_norm=False)

    @staticmethod
    def Draw_AllDensityHistogram(_df, _bins_list:list = [], _targetCol = 'DIAG_NM' ):
        #밀도 기반 히스토그램 subplots로 여러개 그리기

        bins_list = _bins_list

        if not bins_list:
            bins_list = [10, 30, 50, 100]

        feat_cnt = len(_df.columns) - 1 # target에 대한것 제외
        ncols = len(bins_list)
        target = _targetCol

        fig, axes = plt.subplots(nrows=feat_cnt, ncols=ncols, figsize=(ncols*5,feat_cnt*3))
        axes = axes.flatten()  # 2D 배열을 1D 배열로 변환

        # 각 컬럼별로 히스토그램 그리기
        for i, feature in enumerate(_df.drop(target, axis=1).columns):
            for j, bins in enumerate(bins_list):
                axis_idx = i*ncols+j
                sns.histplot(x=feature, data=_df, hue=target, bins=bins, kde=True, ax=axes[axis_idx], stat="density", common_norm=False, palette='Accent')
                axes[axis_idx].set_title(f'{feature} at bins = {bins}')


        plt.tight_layout()  # 레이아웃 조정
        plt.show()

    @staticmethod
    def Draw_Scatter(_df, _targetColm:str = 'DIAG_NM', _SelectColm : list = []
                    , _labelX:str = '', _labelY:str = '',  _title:str = "" , _legend = '', _size:int = None):

        if _size == None:
            plt.figure(figsize=(8,6))
        else:
            plt.figure(figsize=(_size,_size))

        sns.scatterplot(x=_df['PC1'], y=_df['PC2'], hue=_df[_targetColm], palette='Set1', alpha=0.7)
        plt.xlabel(_labelX)
        plt.ylabel(_labelY)
        plt.title(_targetColm)
        plt.legend(title=_legend)
        plt.grid(True)
        plt.show()


    @staticmethod
    def Draw_PCA(_df, _targetColm='DIAG_NM', _selectColms = [], _n_components=2):
        """
        PCA를 수행 후 2D 시각화 진행
        """
        df_pca_2d = Feature_Engineering.Get_PCA(
            _df=_df,
            _targetColm=_targetColm,
            _selectColms=_selectColms,
            _n_components=_n_components
        )

        # 시각화 수행
        Draw.Draw_Scatter(
            df_pca_2d,
            _targetColm,
            ['PC1', 'PC2'],
            _labelX='Principal Component 1',
            _labelY='Principal Component 2',
            _title='PCA Visualization (2D) with Labels',
            _legend='Class Label'
        )

    @staticmethod
    def Draw_HeatMapGang(_df, _targetCol = 'DIAG_NM', size = 20 ):

        corr_matrix = _numeric_df = _df.select_dtypes(include=['int64', 'float64']).corr()
        print(corr_matrix.shape)
        #corr_matrix

        # 히트맵 그리기
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        mask = np.zeros_like(corr_matrix, dtype=np.bool_)
        mask[np.triu_indices_from(mask)] = True

        plt.figure(figsize=(size, size))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap=cmap, center=0, fmt='.2f',
                    square=True, linewidths=.5, cbar_kws={"shrink": .5})
        plt.tick_params(axis='both', labelsize=12)  # 축 레이블 크기를 8로 설정

    @staticmethod
    def Draw_BoxplotAllColms(_df:pd.DataFrame, _targetColms:str = 'DIAG_NM'): #작동확인.
        '''
        범주형을 제외한 컬럼에 대한 박스 플롯 수행.

        '''
        target_df = pd.DataFrame()
        target_df[_targetColms] = data[_targetColms]
        _df = EasyPanda.Get_AllNumType(data)
        _df[target] = target_df[target]
        num_features = len(_df.drop(target, axis=1, errors='ignore').columns)

        print(num_features)

        nrows  = int(math.ceil(math.sqrt(num_features)))
        ncols = int(math.ceil( num_features / nrows ))

        width_per_subplot = 4  # 각 서브플롯의 가로 크기
        height_per_subplot = 4  # 각 서브플롯의 세로 크기
        figsize = (ncols * width_per_subplot, nrows * height_per_subplot)

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        axes = axes.flatten()  # 2D 배열을 1D 배열로 변환

        # 각 컬럼별로 boxplot 그리기
        for i, feature in enumerate(_df.drop(target, axis=1).columns):
            sns.boxplot(x=feature, data= _df, hue=target, ax=axes[i], palette='Accent')
            axes[i].set_title(f'{feature}')


        plt.tight_layout()  # 레이아웃 조정
        plt.show()
        return _df

    @staticmethod
    def Draw_ScatterPlotAllColms(_df:pd.DataFrame,_itSoLong_Are_you_sure : bool,  _targetColms:str = 'DIAG_NM',  ): #작동확인.
            '''
            범주형을 제외한 컬럼에 대한 타깃에 대한 산점도를 수행.
            최대 연결 횟수를 구해 두 변수간에 중복이 없게 모든 수치형 변수에서 산점도를 수행합니다.

            '''
            target_df = pd.DataFrame()
            target_df[_targetColms] = data[_targetColms]

            num_df = EasyPanda.Get_AllNumType(data)
            _df = EasyPanda.Get_AllNumType(data)
            num_features = len(_df.drop(target, axis=1, errors='ignore').columns)
            _df[target] = target_df[target]

            num_features = num_features * (num_features - 1 ) / 2
            nrows  = int(math.ceil(math.sqrt(num_features)))
            ncols = int(math.ceil( num_features / nrows ))

            width_per_subplot = 4  # 각 서브플롯의 가로 크기
            height_per_subplot = 4  # 각 서브플롯의 세로 크기
            figsize = (ncols * width_per_subplot, nrows * height_per_subplot)

            # 서브플롯 생성 (num_plots행, 1열)
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
            axes = axes.flatten()

            num_df_1 = num_df
            num_df_2 = num_df

            count : int = 0
            for featureA in num_df_1:
                for featureB in num_df_2:
                    if featureA != _targetColms and featureB != _targetColms and featureA != featureB:
                        #sns.scatterplot(x=var1, y=var2, data=data, ax=axes[i], hue=target, s=10)
                        sns.scatterplot(x=featureA, y=featureB, data=_df, ax=axes[count], hue=_targetColms, s=10)
                        axes[count].set_xlabel(featureA)#@title 시각화
# 시각화
class Draw:
    @staticmethod
    def Help():
        print(textwrap.dedent(
            '''
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
            '''
        ))


    @staticmethod
    def Draw_HeatMap(_df, _title: str = '', _size: int = 10, _dfName: str = ""):
        numeric_df = _df.select_dtypes(include=['int64', 'float64'])  # 숫자형 선별
        corr = numeric_df.corr()

        if _size is not None:
            plt.figure(figsize=(_size, _size))

        sns.heatmap(corr, annot=True, fmt=".2f", cmap='viridis', square=True, cbar=True)
        plt.title('Correlation Heatmap of ' + _dfName + ' Features', fontsize=16)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def Draw_Bar(df: pd.DataFrame, colm: str = 'DIAG_NM' ,
        labeX: str = "", lableY: str = "",title: str = ""):
        '''
        타깃의 바 차트를 그립니다.
        기본 타깃 = DIAG_NM
        #사용법
            Draw_Bar(df)
        '''
        value_counts = df[colm].value_counts()

        value_counts.plot(kind='bar')
        if title != None:
            plt.title(title)

        plt.xlabel(labeX)
        plt.ylabel(lableY)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def Draw_DensityHistogram(_df, _xFeature = 'activity_average_met', _hueTarget = 'DIAG_NM', _bins:int = 50  ):
        #밀도 기반 히스토그램 1개 그리기
        feature = _xFeature
        target = _hueTarget
        bins = _bins
        sns.histplot(x=feature, data=df, hue=target, bins=bins, kde=True, stat="density", common_norm=False)

    @staticmethod
    def Draw_AllDensityHistogram(_df, _bins_list:list = [], _targetCol = 'DIAG_NM' ):
        #밀도 기반 히스토그램 subplots로 여러개 그리기

        bins_list = _bins_list

        if not bins_list:
            bins_list = [10, 30, 50, 100]

        feat_cnt = len(_df.columns) - 1 # target에 대한것 제외
        ncols = len(bins_list)
        target = _targetCol

        fig, axes = plt.subplots(nrows=feat_cnt, ncols=ncols, figsize=(ncols*5,feat_cnt*3))
        axes = axes.flatten()  # 2D 배열을 1D 배열로 변환

        # 각 컬럼별로 히스토그램 그리기
        for i, feature in enumerate(_df.drop(target, axis=1).columns):
            for j, bins in enumerate(bins_list):
                axis_idx = i*ncols+j
                sns.histplot(x=feature, data=_df, hue=target, bins=bins, kde=True, ax=axes[axis_idx], stat="density", common_norm=False, palette='Accent')
                axes[axis_idx].set_title(f'{feature} at bins = {bins}')


        plt.tight_layout()  # 레이아웃 조정
        plt.show()

    @staticmethod
    def Draw_Scatter(_df, _targetColm:str = 'DIAG_NM', _SelectColm : list = []
                    , _labelX:str = '', _labelY:str = '',  _title:str = "" , _legend = '', _size:int = None):

        if _size == None:
            plt.figure(figsize=(8,6))
        else:
            plt.figure(figsize=(_size,_size))

        sns.scatterplot(x=_df['PC1'], y=_df['PC2'], hue=_df[_targetColm], palette='Set1', alpha=0.7)
        plt.xlabel(_labelX)
        plt.ylabel(_labelY)
        plt.title(_targetColm)
        plt.legend(title=_legend)
        plt.grid(True)
        plt.show()


    @staticmethod
    def Draw_PCA(_df, _targetColm='DIAG_NM', _selectColms = [], _n_components=2):
        """
        PCA를 수행 후 2D 시각화 진행
        """
        df_pca_2d = Feature_Engineering.Get_PCA(
            _df=_df,
            _targetColm=_targetColm,
            _selectColms=_selectColms,
            _n_components=_n_components
        )

        # 시각화 수행
        Draw.Draw_Scatter(
            df_pca_2d,
            _targetColm,
            ['PC1', 'PC2'],
            _labelX='Principal Component 1',
            _labelY='Principal Component 2',
            _title='PCA Visualization (2D) with Labels',
            _legend='Class Label'
        )

    @staticmethod
    def Draw_HeatMapGang(_df, _targetCol = 'DIAG_NM', size = 20 ):

        corr_matrix = _numeric_df = _df.select_dtypes(include=['int64', 'float64']).corr()
        print(corr_matrix.shape)
        #corr_matrix

        # 히트맵 그리기
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        mask = np.zeros_like(corr_matrix, dtype=np.bool_)
        mask[np.triu_indices_from(mask)] = True

        plt.figure(figsize=(size, size))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap=cmap, center=0, fmt='.2f',
                    square=True, linewidths=.5, cbar_kws={"shrink": .5})
        plt.tick_params(axis='both', labelsize=12)  # 축 레이블 크기를 8로 설정

    @staticmethod
    def Draw_BoxplotAllColms(_df:pd.DataFrame, _targetColms:str = 'DIAG_NM'): #작동확인.
        '''
        범주형을 제외한 컬럼에 대한 박스 플롯 수행.

        '''
        target_df = pd.DataFrame()
        target_df[_targetColms] = data[_targetColms]
        _df = EasyPanda.Get_AllNumType(data)
        _df[target] = target_df[target]
        num_features = len(_df.drop(target, axis=1, errors='ignore').columns)

        print(num_features)

        nrows  = int(math.ceil(math.sqrt(num_features)))
        ncols = int(math.ceil( num_features / nrows ))

        width_per_subplot = 4  # 각 서브플롯의 가로 크기
        height_per_subplot = 4  # 각 서브플롯의 세로 크기
        figsize = (ncols * width_per_subplot, nrows * height_per_subplot)

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        axes = axes.flatten()  # 2D 배열을 1D 배열로 변환

        # 각 컬럼별로 boxplot 그리기
        for i, feature in enumerate(_df.drop(target, axis=1).columns):
            sns.boxplot(x=feature, data= _df, hue=target, ax=axes[i], palette='Accent')
            axes[i].set_title(f'{feature}')


        plt.tight_layout()  # 레이아웃 조정
        plt.show()
        return _df

    @staticmethod
    def Draw_ScatterPlotAllColms(_df:pd.DataFrame,_itSoLong_Are_you_sure : bool,  _targetColms:str = 'DIAG_NM',  ): #작동확인.
            '''
            범주형을 제외한 컬럼에 대한 타깃에 대한 산점도를 수행.
            최대 연결 횟수를 구해 두 변수간에 중복이 없게 모든 수치형 변수에서 산점도를 수행합니다.

            '''
            target_df = pd.DataFrame()
            target_df[_targetColms] = data[_targetColms]

            num_df = EasyPanda.Get_AllNumType(data)
            _df = EasyPanda.Get_AllNumType(data)
            num_features = len(_df.drop(target, axis=1, errors='ignore').columns)
            _df[target] = target_df[target]

            num_features = num_features * (num_features - 1 ) / 2
            nrows  = int(math.ceil(math.sqrt(num_features)))
            ncols = int(math.ceil( num_features / nrows ))

            width_per_subplot = 4  # 각 서브플롯의 가로 크기
            height_per_subplot = 4  # 각 서브플롯의 세로 크기
            figsize = (ncols * width_per_subplot, nrows * height_per_subplot)

            # 서브플롯 생성 (num_plots행, 1열)
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
            axes = axes.flatten()

            num_df_1 = num_df
            num_df_2 = num_df

            count : int = 0
            for featureA in num_df_1:
                for featureB in num_df_2:
                    if featureA != _targetColms and featureB != _targetColms and featureA != featureB:
                        #sns.scatterplot(x=var1, y=var2, data=data, ax=axes[i], hue=target, s=10)
                        sns.scatterplot(x=featureA, y=featureB, data=_df, ax=axes[count], hue=_targetColms, s=10)
                        axes[count].set_xlabel(featureA)
                        axes[count].set_ylabel(featureB)
                        axes[count].set_title(f"{featureA} vs {featureB}")
                        count += 1
                num_df_2.drop(columns=featureA, inplace=True)

            plt.tight_layout()  # 레이아웃 조정
            plt.show()

    @staticmethod
    def Draw_ScatterPlotAboutCorr(_df: pd.DataFrame, _highCorr_df: pd.DataFrame, _targetColms = 'DIAG_NM'):

        if _highCorr_df is None:
            _highCorr_df = Feature_Selection.Get_HighCorrPairs(_df)

        # 자동으로 서브플롯 그리드 계산
        num_plots = len(_highCorr_df)
        ncols = 4
        nrows = (num_plots + ncols - 1) // ncols  # 올림 처리

        fig_width = ncols * 5
        fig_height = nrows * 4

        # 서브플롯 생성
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_width, fig_height))
        axes = axes.flatten()

        # 변수 쌍별로 산점도 그리기
        for i, (var1, var2) in enumerate(_highCorr_df.index):
            sns.scatterplot(x=var1, y=var2, data=_df, ax=axes[i], hue=target, s=10)
            axes[i].set_xlabel(var1)
            axes[i].set_ylabel(var2)
            axes[i].set_title(f"Scatter Plot: {var1} vs {var2}")
            #wrapped_title = "\n".join(textwrap.wrap(title, width=30))

        plt.tight_layout()  # 레이아웃 조정
        plt.show()
                        axes[count].set_ylabel(featureB)
                        axes[count].set_title(f"{featureA} vs {featureB}")
                        count += 1
                num_df_2.drop(columns=featureA, inplace=True)

            plt.tight_layout()  # 레이아웃 조정
            plt.show()

    @staticmethod
    def Draw_ScatterPlotAboutCorr(_df: pd.DataFrame, _highCorr_df: pd.DataFrame, _targetColms = 'DIAG_NM'):

        if _highCorr_df is None:
            _highCorr_df = Feature_Selection.Get_HighCorrPairs(_df)

        # 자동으로 서브플롯 그리드 계산
        num_plots = len(_highCorr_df)
        ncols = 4
        nrows = (num_plots + ncols - 1) // ncols  # 올림 처리

        fig_width = ncols * 5
        fig_height = nrows * 4

        # 서브플롯 생성
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_width, fig_height))
        axes = axes.flatten()

        # 변수 쌍별로 산점도 그리기
        for i, (var1, var2) in enumerate(_highCorr_df.index):
            sns.scatterplot(x=var1, y=var2, data=_df, ax=axes[i], hue=target, s=10)
            axes[i].set_xlabel(var1)
            axes[i].set_ylabel(var2)
            axes[i].set_title(f"Scatter Plot: {var1} vs {var2}")
            #wrapped_title = "\n".join(textwrap.wrap(title, width=30))

        plt.tight_layout()  # 레이아웃 조정
        plt.show()