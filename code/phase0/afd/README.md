
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
 |-7.1.CleanUp.ipynb 
     - 리소스 제거 노트북
 |-9.1.Error.ipynb
     - 트러블 슈팅 노트북
```

# 참고 사이트:
-  파일 폴더 생성 명령어
`find . | sed -e "s/[^-][^\/]*\// |/g" -e "s/|\([^ ]\)/|-\1/"`


- Build, train, and deploy a fraud detection modelwith Amazon Fraud Detector
    - https://aws.amazon.com/getting-started/hands-on/build-train-deploy-fraud-detection-model-amazon-fraud-detector/