# Fraud-Detector-Workshop



## <li> Update: TBD</li>

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
```

## 유동님 이제 작업 하셔도 됩니다. 




    