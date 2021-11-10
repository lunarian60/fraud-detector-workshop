
import argparse
import logging
import os
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
import numpy as np
import pandas as pd
import joblib

print("joblib version: ", joblib.__version__)

if __name__ =='__main__':

    print('extracting arguments')
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--train-file', type=str, default='train.csv')
    parser.add_argument('--cat-features', type=str)  # in this script we ask user to explicitly name features
    parser.add_argument('--target', type=str) # in this script we ask user to explicitly name the target
    

    args, _ = parser.parse_known_args()

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    logging.info('reading data')
    train_df = pd.read_csv(os.path.join(args.train, args.train_file))

    logging.info('building training and testing datasets')
    X = train_df.drop(args.target, axis=1)
    y = train_df[args.target]
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
    
        
    # define and train model
    clf = CatBoostClassifier(
        iterations=10,
        learning_rate=0.1, 
    )

    clf.fit(X_train, y_train, cat_features=args.cat_features.split(), eval_set=(X_val, y_val),)
    
    # persist model
    path = os.path.join(args.model_dir, "model.joblib")
    logging.info('saving to {}'.format(path))
    joblib.dump(clf, path)    
#     clf.save_model(path)
    
# inference functions ---------------
def model_fn(model_dir):
    print("############### Model_FN() ###################")
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    print("### Model loaded ")
    return clf

def predict_fn(input_data, model):
    print("############### predict_FN() ###################")    

    print("type: ", type(input_data))
    print("input: \n", input_data)    
#     predictions = model.predict(input_data)
    predictions = model.predict_proba(input_data)    
    
    return predictions


import numpy as np
from io import StringIO

def input_fn(input_data, content_type):
    '''
    content_type == 'application/x-npy' 일시에 토치 텐서의 변환 작업 수행
    '''
    print("#### input_fn starting ######")
    print(f"data type: {type(input_data)}")    
    print("input is \n", input_data)
    
# 데이터가 String 이면 바로 데이터 프레임으로 로딩, 그렇지 않고 바이트스트림이면 디코딩 이후에 로딩
    if type(input_data) == str:
        # Load dataset
        input_data = pd.read_csv(StringIO(input_data), header=None)
    else:
        # df = pd.read_csv(StringIO(input_data.decode()), header=None)
        pass

    input_data = input_data.values
    
#    input_data = pd.DataFrame(data=input_data)
    
    print(f"After tansformation, data type: {type(input_data)}")        
    print("input is \n", input_data)   

    return input_data



