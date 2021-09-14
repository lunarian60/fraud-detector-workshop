# Fraud-Detector-Workshop

# 1. 워크샵 필수 사항
- 세이지 메이커 노트북 인스턴스는 메모리가 많은 ml.m5.4xlarge (64 GiB) 의 사용을 권장 합니다.
    - 기본 샘플링된 데이터 세트 200,000 개 로딩시에 약 9.6 GB 의 메모리를 사용합니다.



# 2. 워크샵 업데이트 사항

## <li> Update: TBD</li>

## <li> Update: 09-13-2021</li>
Amazon Fraud Detector (AFD) 코드 업로드
- 자세한 내역은 여기를 클릭 하세요. --> [AFD ReadMe](code/phase0/afd/README.md)

## <li> Update: 09-06-2021 </li>
    
    
### 파일 추가 목록
아래의 파일 세가지를 순서대로 실행하면, prepare_data/data/train/train.csv , prepare_data/data/test/test.csv 생성되고, 또한 S3에 디폴트 버킷에 업로딩이 됩니다. 이것을 바탕으로 afd, sm 에서 훈련, 테스트 데이터를 로딩하여 사용할 수 있습니다.

- How_to_downlaod_kaggle_data/
    - 0.download_kaggle_dataset.ipynb
        - AdTalking 데이터 세트 다운로드 방법


- prepare_data/
    - 0.1.Skim_Dataset.ipynb
        - 간략하게 캐클에서 다운로드 받은 데이터 확인
    - 1.1.Prepare_Dataset.ipynb
        - 데이터 세트 샘플링하여 훈련, 테스트 데이터 세트 저장



- 참고: 파일 폴더 생성 명령어
`find . | sed -e "s/[^-][^\/]*\// |/g" -e "s/|\([^ ]\)/|-\1/"`

```
 |-code
 | |-phase0
 | | |-afd
 | | | |-src
 | | |-prepare_data
 | | | |-1.1.Prepare_Dataset.ipynb
 | | | |-0.1.Skim_Dataset.ipynb
 | | | |-src
 | | | | |-p_utils.py
 | | | |-How_to_downlaod_kaggle_data
 | | | | |-0.download_kaggle_dataset.ipynb
 | | |-sm
 | | | |-xgboost
 | | | | |-src
 | | | | | |-p_utils.py
 | | | | |-fraud-detector-xgboost.ipynb
```






    
