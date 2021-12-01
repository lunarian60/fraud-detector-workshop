import boto3, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn


#########################
# 전처리
#########################

def upload_s3(bucket, file_path, prefix):
    '''
    bucket = sagemaker.Session().default_bucket()
    prefix = 'comprehend'
    train_file_name = 'test/train/train.csv'
    s3_train_path = upload_s3(bucket, train_file_name, prefix)
    '''
    
    prefix_path = os.path.join(prefix, file_path)
    # prefix_test_path = os.path.join(prefix, 'infer/test.csv')

    boto3.Session().resource('s3').Bucket(bucket).Object(prefix_path).upload_file(file_path)
    s3_path = "s3://{}/{}".format(bucket, prefix_path)
    # print("s3_path: ", s3_path)

    return s3_path

def filter_df(raw_df, cols):
    '''
    cols = ['label','userno', 'ipaddr','try_cnt','paytool_cnt','total_amount','EVENT_DATE']    
    df = filter_df(df, cols)
    '''
    df = raw_df.copy()
    df = df[cols]
    return df


def save_local(raw_df, preproc_folder, label, file_name):
    df = raw_df.copy()
    df = pd.concat([df[label], df.drop([label], axis=1)], axis=1)
    file_path = os.path.join(preproc_folder, file_name)
#    df.to_csv(file_path, index=False, )
    df.to_csv(file_path)
    print(f'{file_path} is saved')

    
    return file_path



def split_data_date(raw_df, sort_col,data1_end, data2_start):
    '''
    train, test 데이터 분리
    train_end = '2020-01-31'
    test_start = '2020-02-01'
    train_df, test_df = split_data_date(df, sort_col='EVENT_DATE',
                                        data1_end = train_end, 
                                    data2_start = test_start)

    '''
    df = raw_df.copy()
    
    df = df.sort_values(by= sort_col) # 시간 순으로 정렬
    # One-Hot-Encoding
    data1 = df[df[sort_col] <= data1_end]
    data2 = df[df[sort_col] >= data2_start]    
        
    print(f"data1, data2 shape: {data1.shape},{data2.shape}")
    print(f"data1 min, max date: {data1[sort_col].min()}, {data1[sort_col].max()}")
    print(f"data2 min, max date: {data2[sort_col].min()}, {data2[sort_col].max()}")        
    
    return data1, data2


########################
# 훈련
########################
def get_pos_scale_weight(df, label):
    '''
    1, 0 의 레이블 분포를 계산하여 클래스 가중치 리턴
    예: 1: 10, 0: 90 이면 90/10 = 9 를 제공함. 
    '''
    fraud_sum = df[df[label] == 1].shape[0]
    non_fraud_sum = df[df[label] == 0].shape[0]
    class_weight = int(non_fraud_sum / fraud_sum)
    print(f"fraud_sum: {fraud_sum} , non_fraud_sum: {non_fraud_sum}, class_weight: {class_weight}")
    return class_weight






#########################
# 평가
#########################
import itertools


def plot_conf_mat(cm, classes, title, cmap = plt.cm.Greens):
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
        horizontalalignment="center",
        color="black" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        
        
import numpy as np

def combine_pred(test_true, prediction_df):
    '''
    예측 데이터와 실제 참값을 합쳐서 제공 함.
    '''
    df = pd.concat([prediction_df, test_true], axis=1)
    
    return df

def get_prediction_set(prediction,prediction_prob ):
    '''
    예측 결과(0 혹은 1) 와 스코어를 제공함.
    스코어의 경우는 모델이 두개의 확률값(0 과 1)을 제공하는 것 중에서 1의 확률값을 제공함.
    '''
    df = prediction_prob.copy()
    df.insert(len(df.columns), column='pred', value=prediction.values)
    # df['score'] = np.where( df[0] > df[1], df[0], df[1]) * 1000
    df['score'] = df[1] * 1000
    df['score'] = df['score'].astype('int')
    df.drop(columns=[0,1], inplace=True)
    cols = ['score','pred']
    df = df[cols]
    
    return df

from sklearn.metrics import classification_report, roc_auc_score
from IPython.display import display as dp
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
# %matplotlib inline
# %config InlineBackend.figure_format='retina'

import seaborn as sns
sns.set(style="whitegrid")


def compute_f1(y_true, y_pred):
    '''
    classification_report 를 출력 해주고, 혼돈 팽령을 출력 함.
    '''
    print("- ROC_AUC SCORE")
    print(f'\t-{round(roc_auc_score(y_true, y_pred),3)}')
    print("\n- F1 SCORE")    
    print(classification_report(y_true = y_true, y_pred = y_pred))    
    
    cm = confusion_matrix(y_true= y_true, y_pred= y_pred)
    print(cm)



def evaluate_model(predictor,y_test, test_df, test_data_nolab, single_model_name, single_model=False ):
    '''
    앙상블 모델 및 싱글 모델의 평가를 함.
    '''
    if single_model:
        print(f"single model {single_model_name} is used")
        prediction = predictor.predict(test_data_nolab, model=single_model_name)
        prediction_prob = predictor.predict_proba(test_data_nolab, model = single_model_name)
    else:
        print("ensemble model is used")
        prediction = predictor.predict(test_data_nolab)
        prediction_prob = predictor.predict_proba(test_data_nolab)

        
    # 오토글루온의 평가 함수 실행하여 평가 지표 확인
    perf = predictor.evaluate_predictions(y_true=y_test, y_pred=prediction_prob, auxiliary_metrics=True)
    # 예측 레이블 및 스코어를 추출
    df_pred = get_prediction_set(prediction, prediction_prob )        
    # 테스트 데이터와 예측 결과를 결합하여 추후 분석에 사용
    test_result = combine_pred(test_df, df_pred )
    # classification_report 출력
    compute_f1(test_result.abusing.values, test_result.pred.values)    
    
    return test_result
    

def show_feature_imp(fea_importance):
    '''
    치쳐 중요성을 그래표로 그림
    '''
    f, ax = plt.subplots(figsize=(10,5))
    plot = sns.barplot(x=fea_importance.index, y = fea_importance.values)
    # plot = sns.barplot(x=features, y= fea_importance)
    ax.set_title('Feature Importance')
    plot.set_xticklabels(plot.get_xticklabels(),rotation='vertical')
    plt.show()


    
####################
# 배포 함수
####################
import os
import pickle 

def create_column_list(raw, code_dir, columns_file_name):
    '''
    훈련에 사용된 피쳐 컬럼 생성하여 pickle 파일로 저장
    '''
    df = raw.copy()
    os.makedirs(code_dir, exist_ok=True)
    columns = df.columns.tolist()
    column_dict = {"columns": columns}
    columns_file_path = os.path.join(code_dir, columns_file_name)
    with open(columns_file_path, "wb") as f:
        pickle.dump(column_dict, f)
        print(f"{columns_file_path} is saved")
        
    return columns_file_path

def create_bestmodel_file(best_model, code_dir, best_model_file_name):
    '''
    best model 을 pickle 파일로 저장
    '''
    best_model_dict = {"best_model": best_model}
    print("best_model_dict: ", best_model_dict)
    best_model_file_path = os.path.join(code_dir, best_model_file_name)
    with open(best_model_file_path, "wb") as f:
        pickle.dump(best_model_dict, f)
        print(f"{best_model_file_path} is saved")
        
    return best_model_file_path


    

####################
# 추론 함수
####################

def invoke_endpoint(runtime_client, endpoint_name, payload):
    '''
    로컬 엔드 포인트 호출
    '''
    # runtime_client = sagemaker.local.LocalSagemakerRuntimeClient()


    response = runtime_client.invoke_endpoint(
        EndpointName=endpoint_name, 
        ContentType='text/csv', 
        # Accept='application/json',
        Body=payload,
        )

    result = response['Body'].read().decode().splitlines()    
    
    return result

def invoke_host_endpoint(endpoint_name, payload):
    '''
    로컬 엔드 포인트 호출
    '''
    runtime_client = boto3.Session().client('sagemaker-runtime')


    response = runtime_client.invoke_endpoint(
        EndpointName=endpoint_name, 
        ContentType='text/csv', 
        # Accept='application/json',
        Body=payload,
        )

    result = response['Body'].read().decode().splitlines()    
    
    return result



def get_score_label(result, threshold=0.5):
    '''
    레이블과 확률값 리턴
    '''
    result = result[0].split(",")
    if result[0] > result[1]:
        label = 0
        prob = round(float(result[0]),3)
    else:
        label = 1
        prob = round(float(result[1]),3)
    
    return label, prob


#################################
# 리소스 정리
#################################

def delete_endpoint(client, endpoint_name ,is_delete=False, is_del_model=True, is_del_endconfig=True,is_del_endpoint=True):
    '''
    model, EndpointConfig, Endpoint 삭제
    '''
    try:
        response = client.describe_endpoint(EndpointName=endpoint_name)
        EndpointConfigName = response['EndpointConfigName']

        response = client.describe_endpoint_config(EndpointConfigName=EndpointConfigName)
        model_name = response['ProductionVariants'][0]['ModelName']    

        print("model_name: \n", model_name)        
        print("EndpointConfigName: \n", EndpointConfigName)
        print("endpoint_name: \n", endpoint_name)    

        if is_delete: # is_delete가 True 이면 삭제
            if is_del_endconfig:
                client.delete_endpoint_config(EndpointConfigName=EndpointConfigName)    
                print(f'--- Deleted endpoint: {endpoint_name}')                

            
            if is_del_model: # 모델도 삭제 여부 임.
                client.delete_model(ModelName=model_name)    
                print(f'--- Deleted model: {model_name}')                

            if is_del_endpoint:
                client.delete_endpoint(EndpointName=endpoint_name)
                print(f'--- Deleted endpoint_config: {EndpointConfigName}')                




    except:
            print("There is no avaliable endpoint")