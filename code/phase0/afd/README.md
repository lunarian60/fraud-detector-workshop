
# 폴더 및 파일 설명

```
|-README.md
 |-src
 | |-p_utils.py
     - 유티리티 함수 모음
 |-1.1.Setup-Environment.ipynb 
     - AFD role 권한 가이드
 |-2.1.Create-Model.ipynb
     - 모델 생성 노트북
 |-3.1.Create-Detector.ipynb 
     - 디텍터 생성 노트북
 |-4.1.Inference-Scratch.ipynb
     - 추론 생성 노트북
 |-4.2..Inference-Parallel.ipynb     
     - 추론을 병렬로 실행
 |-5.1.Trans-Create-Model.ipynb
     - Transaction fraud insights 모델 타입으로 훈련
 |-5.2.Trans-Create-Detector.ipynb
     - Transaction fraud insights 모델의 디텍터 생성
 |-5.3.Inference-Scratch.ipynb
     - Transaction fraud insights 디텍터의 추론 실행
 |-5.4.Inference-Parallel.ipynb
     - Transaction fraud insights 디텍터의 추론 병렬 실행 및 결과 분석
 |-7.1.CleanUp.ipynb 
     - 리소스 제거 노트북
 |-9.1.Error.ipynb
     - 트러블 슈팅 노트북
```

# 참고 사이트:
-  파일 폴더 생성 명령어
`find . | sed -e "s/[^-][^\/]*\// |/g" -e "s/|\([^ ]\)/|-\1/"`


- 처음 시작: Build, train, and deploy a fraud detection modelwith Amazon Fraud Detector
    - https://aws.amazon.com/getting-started/hands-on/build-train-deploy-fraud-detection-model-amazon-fraud-detector/
    
    
- Transaction fraud insights 모델 타입을 위한 개발자 가이드 
    - [ransaction fraud insights](https://docs.aws.amazon.com/frauddetector/latest/ug/transaction-fraud-insights.html)
    - [공식 AFD 샘플 코드 예시](https://github.com/aws-samples/aws-fraud-detector-samples)
        - 아래는 위 리파지토리 하위에 있는 TRANSACTION_FRAUD_INSIGHTS 의 예시 입니다.
        - https://github.com/aws-samples/aws-fraud-detector-samples/blob/master/Fraud_Detector_End_to_End_Stored_Data.ipynb

    