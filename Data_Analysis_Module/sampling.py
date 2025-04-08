# @title Sampling
class Sampling: # Feature_Selection->Feature_Engineering-> 모델 평가까지.
    @staticmethod
    def Help():
        print(textwrap.dedent(
            '''
            ### Sampling (샘플링 / 파이프라인 전체 흐름 제어) ###

            # 테스트 출력 : test()
                설명) 샘플링 또는 전체 파이프라인 흐름 관련 클래스가 정상적으로 연결되었는지 확인하는 테스트 함수입니다.
                반환) 없음
            '''
        ))

    @staticmethod
    def test():
        print('Sampling - Hello')