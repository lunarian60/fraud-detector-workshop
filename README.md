# Fraud-Detector-Workshop

# 1. 워크샵 필수 사항
- [중요] 세이지 메이커 노트북 인스턴스는 메모리가 많은 `ml.m5.4xlarge (64 GiB)` 의 사용을 권장 합니다.
    - 기본 샘플링된 데이터 세트 200,000 개 로딩시에 약 9.6 GB 의 메모리를 사용합니다.
- [중요] 세이지 메이커 노트북 인스턴스 생성시에 EBS 볼륨을 `20 GB` 로 설정 해주세요. 
    - 디폴트로 사용하면 욜량이 부족합니다.

# 2. 워크샵 실행 순서

## (1) 데이터 준비
   
- A. 캐글 데이터 다운로드
    - code/phase0/prepare_data/How_to_downlaod_kaggle_data/0.download_kaggle_dataset.ipynb
- B. 데이터 스키밍
    - code/phase0/prepare_data/0.1.Skim_Dataset.ipynb
- C. 훈련, 테스트 데이터 세트 준비
    - code/phase0/prepare_data/1.1.Prepare_Dataset.ipynb
    - <font color="red">[신규 및 권장] Transaction fraud insights 모델 타입으로 진행시에 아래 노트북 추가 실행 필요</font>
        - code/phase0/prepare_data/1.2.Prepare_Datase_AFD_TRANt.ipynb


- 준비된 데이터 컬럼에서 모델 훈련을 위한 피쳐 선택 가이드 입니다.
    - [모델 훈련을 위한 피쳐 선택 가이드](code/phase0/prepare_data/Feature_Selection_Guide.md)

## (2) Amazon Fraud Detector 

- [신규] Transaction fraud insights 모델 타입 실행시
    - 5.1.Trans-Create-Model.ipynb
    - 5.2.Trans-Create-Detector.ipynb
    - 5.3.Trans-Inference-Scratch.ipynb
    - 5.4.Trans-Inference-Parallel.ipynb
    - 7.1.CleanUp.ipynb 


- ONLINE_FRAUD_INSIGHTS insights 모델 타입 실행시
    - 2.1.Create-Model.ipynb
    - 3.1.Create-Detector.ipynb
    - 4.1.Inference-Scratch.ipynb
    - 4.2.Inference-Parallel.ipynb
    - 7.1.CleanUp.ipynb 


## (3) Amazon SageMaker

    - 아래 노트북 실행
    - code/phase0/sm/xgboost/fraud-detector-xgboost.ipynb

        

# 9. 참조 자료

##  Amazon Fraud Detector

- 처음 시작: Build, train, and deploy a fraud detection modelwith Amazon Fraud Detector
    - https://aws.amazon.com/getting-started/hands-on/build-train-deploy-fraud-detection-model-amazon-fraud-detector/
    
    
- Transaction fraud insights 모델 타입을 위한 개발자 가이드 
    - 블로그 --> ['Detect online transaction fraud with new Amazon Fraud Detector features'](https://aws.amazon.com/blogs/machine-learning/detect-online-transaction-fraud-with-new-amazon-fraud-detector-features/)
    - [ransaction fraud insights](https://docs.aws.amazon.com/frauddetector/latest/ug/transaction-fraud-insights.html)
    - [공식 AFD 샘플 코드 예시](https://github.com/aws-samples/aws-fraud-detector-samples)
        - 아래는 위 리파지토리 하위에 있는 TRANSACTION_FRAUD_INSIGHTS 의 예시 입니다.
        - https://github.com/aws-samples/aws-fraud-detector-samples/blob/master/Fraud_Detector_End_to_End_Stored_Data.ipynb

