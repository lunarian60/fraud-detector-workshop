

def get_payload(sample, label_col = 'fraud', verbose=False):    
    '''
    아래왁 같이 ',' 형태의 문자형을 리턴함.
    0,0,750,2,0,2,0,2,1,16596.0,1,18,0,59,1,0,0,4500.0,0,0,1,1,0,52,2020,3,0,0,0,2,1,0,0,0,0,0,0,10,12096.0,1,3000,1,0,1,0    
    '''

    sample = sample.drop(columns=[label_col]) # 레이블 제거
    payload = sample.to_csv(header=None, index=None)
    
    if verbose:
        #dp(sample)
        # print("payload length: \n", len(payload))    
        print("pay load type: ", type(payload))
        print("payload: \n", payload)
    
    return payload




import sagemaker

def get_predictor(endpoint_name, session, csv_deserializer):
    '''
    predictor = get_predictor(endpoint_name, session, csv_deserializer)    
    '''
    predictor = sagemaker.predictor.Predictor(
        endpoint_name=endpoint_name,
        sagemaker_session= session, # 세션 할당: 로컬 세션 혹은 세이지 메이커 호스트 세션
        deserializer = csv_deserializer, # byte stream을 csv 형태로 변환하여 제공        
    )
    return predictor

