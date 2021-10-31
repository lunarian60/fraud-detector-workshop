import pandas as pd
import os

#########################
# 데이터 프로파일링 함수
#########################

import boto3
client = boto3.client('frauddetector')

# --- no changes; just run this code block ---
def guess_type_based_on_name(name):
    """
    Guess feature type based on its name. This is a help function used in summary_stats. It does not cover all cases. Suggest map the variable types in cofig
    """
    name = name.replace("_","").lower()
    guess_type = []
    guess_map = {
        'IP_ADDRESS': ['ipaddr'],
        'EMAIL_ADDRESS': ['email'],
        'CARD_BIN': ['cardbin','cardnum'],
        'PHONE_NUMBER': ['phone'],
        'USERAGENT': ['useragent', 'ua'],
        'PRICE': ['price'],
        'PRODUCT_CATEGORY':['productcategory','prodcategory'],
        'CURRENCY_CODE': ['currency'],
        'BILLING_ADDRESS_L1': ['billingstreet','billstreet'],
        'BILLING_CITY': ['billingcity','billcity'],
        'BILLING_STATE': ['billingstate','billstate'],
        'BILLING_ZIP': ['billingzip','billzip','billingpostal','billpostal'],
        'EVENT_ID': ['eventid'],
        'ENTITY_ID': ['entityid', 'customerid'],
        'ENTITY_TYPE': ['entitytype']
    }
    
    for var in guess_map:
        if any(x in name for x in guess_map[var]):
            guess_type.append(var) 
            
    return guess_type


def summary_stats(df, variables_map):
    """
    Generate summary statistics for a pandas data frame  
    """
    rowcnt = len(df)
    
    # -- calculating data statistics and data types -- 
    df_s1  = df.agg(['count', 'nunique']).transpose().reset_index().rename(columns={"index":"feature_name"})
    df_s1["null"] = (rowcnt - df_s1["count"]).astype('int64')
    df_s1["not_null"] = rowcnt - df_s1["null"]
    df_s1["null_pct"] = df_s1["null"] / rowcnt
    df_s1["nunique_pct"] = df_s1['nunique']/ rowcnt

    dt = pd.DataFrame(df.dtypes).reset_index().rename(columns={"index":"feature_name", 0:"dtype"})
    df_stats = pd.merge(dt, df_s1, on='feature_name', how='inner').round(4)
    df_stats['nunique'] = df_stats['nunique'].astype('int64')
    df_stats['count'] = df_stats['count'].astype('int64')
    
    
    # -- variable type mapper: map mandatory variables and variables_map  -- 
    flatten_var_maps = []
    for vartype in variables_map.keys():
        if isinstance(variables_map[vartype], list):
            for var in variables_map[vartype]:
                flatten_var_maps.append([vartype, var])
        else:
            flatten_var_maps.append([vartype, variables_map[vartype]])
            
    for vartype in ['ENTITY_TYPE','ENTITY_ID','EVENT_ID','EVENT_TIMESTAMP','EVENT_LABEL','LABEL_TIMESTAMP']:
        flatten_var_maps.append([vartype, vartype])
    df_schema = pd.DataFrame(flatten_var_maps, columns = ['feature_type', 'feature_name'])
    df_stats = pd.merge(df_stats, df_schema, how = 'left', on = 'feature_name')
    
    # -- variable type mapper: guess types based on feature names -- 
    guess_types = df_stats.loc[df_stats['feature_type'].isna(),'feature_name'].apply(guess_type_based_on_name)
    df_stats.loc[df_stats['feature_type'].isna(),'feature_type'] = guess_types[guess_types.apply(len) == 1].apply(lambda x: x[0])
    
    # -- variable type mapper: map the rest types based on data type -- 
    df_stats.loc[(df_stats['feature_type'].isna())&(df_stats["dtype"] == object), 'feature_type'] = "CATEGORICAL"
    df_stats.loc[(df_stats['feature_type'].isna())&((df_stats["dtype"] == "int64") | (df_stats["dtype"] == "float64")), 'feature_type'] = "NUMERIC"
    
    # -- variable validation -- 
    df_stats['feature_warning'] = "NO WARNING"
    df_stats.loc[(df_stats["nunique"] != 2) & (df_stats["feature_name"] == "EVENT_LABEL"),'feature_warning' ] = "LABEL WARNING, NON-BINARY EVENT LABEL"
    df_stats.loc[(df_stats["nunique_pct"] > 0.9) & (df_stats['feature_type'] == "CATEGORICAL") ,'feature_warning' ] = "EXCLUDE, GT 90% UNIQUE"
    df_stats.loc[(df_stats["null_pct"] > 0.2) & (df_stats["null_pct"] <= 0.75), 'feature_warning' ] = "NULL WARNING, GT 20% MISSING"
    df_stats.loc[df_stats["null_pct"] > 0.75,'feature_warning' ] = "EXCLUDE, GT 75% MISSING"
    df_stats.loc[((df_stats['dtype'] == "int64" ) | (df_stats['dtype'] == "float64" ) ) & (df_stats['nunique'] < 0.2), 'feature_warning' ] = "LIKELY CATEGORICAL, NUMERIC w. LOW CARDINALITY"
    return df_stats


def prepare_schema(df, df_stats, variables_map):
    """
    Prepare schema for following steps
    """
    # -- prepare event variables --
    exclude_list = ['ENTITY_TYPE','ENTITY_ID','EVENT_ID','EVENT_TIMESTAMP','EVENT_LABEL','LABEL_TIMESTAMP','UNKNOWN']
    event_variables = df_stats.loc[(~df_stats['feature_type'].isin(exclude_list))]['feature_name'].to_list()
    
    # -- target -- 
    label_value_count = df['EVENT_LABEL'].dropna().astype('str', errors='ignore').value_counts()
    event_labels      = label_value_count.index.unique().tolist()  
    
    # -- define training_data_schema, Stored events need to specify unlabeledEventsTreatment --
    training_data_schema = {
        'modelVariables' : df_stats.loc[~(df_stats['feature_type'].isin(exclude_list))]['feature_name'].to_list(),
        'labelSchema'    : {
            # we assume the rare event as fraud, and the rest as not-fraud. 
            # if you have more than 2 labels in the data or want to map them in a different way, you can manually modify the training data schema
            'labelMapper' : {
                'FRAUD' : [str(label_value_count.idxmin())],
                'LEGIT' : [i for i in event_labels if i not in [str(label_value_count.idxmin())]]
            },
            # there are there options for unlabeledEventsTreatment: 
            #   'IGNORE' - ignore unlabeled events; 
            #   'FRAUD'  - use unlabeled events as fraud 
            #   'LEGIT'  - use unlabeled events as legit
            'unlabeledEventsTreatment': 'IGNORE'
        }
    }
    return training_data_schema, event_variables, event_labels


def profiling(df, variables_map):
    """
    profiling the input pandas data frame and prepare schema for following steps  
    
    Arguments:
        df (DataFrame)             - panda's dataframe to create summary statistics for
        variables_map (dictionary) - variables map dictionary - key is the variable type and value is the list of variable name
    
    Returns:
        DataFrame of summary statistics, training data schema, event variables and event labels  
    """
    df = df.copy()
    
    # -- check required variables --
    missing_required_vars = [i for i in ['ENTITY_TYPE','ENTITY_ID','EVENT_ID','EVENT_TIMESTAMP','EVENT_LABEL','LABEL_TIMESTAMP'] if i not in set(df.columns)]
    if len(missing_required_vars) != 0:
        raise ValueError(f'Required columns {missing_required_vars} are not included in the training data.')
    
    # -- check if entity types only contains one value --
    entity_types = list(df['ENTITY_TYPE'].unique())
    if len(entity_types)> 1:
        raise ValueError('Currently, Amazon Fraud Detector only support one ENTITY_TYPE per EVENT_TYPE.')
    
    # -- get data summary --
    df_stats = summary_stats(df, variables_map)
    
    # -- prepare schema for following steps -- 
    training_data_schema, event_variables, event_labels = prepare_schema(df, df_stats, variables_map)
    
    print("--- summary stats ---")
    print(df_stats)
    print("\n")
    print("--- event variables ---")
    print(event_variables)
    print("\n")
    print("--- event labels ---")
    print(event_labels)
    print("\n")
    print("--- training data schema ---")
    print(training_data_schema)
    
    return df_stats, training_data_schema, event_variables, event_labels

# -- function to create all your variables --- 
def create_variables(features_dict):
    """
    Check if variables exist, if not, adds the variable to Fraud Detector 
    
    Arguments: 
        features_dict  -  a dictionary maps your variables to variable type
    """
    for feature in features_dict.keys(): 
        DEFAULT_VALUE = '0.0' if features_dict[feature] in ['NUMERIC','PRICE'] else '<null>'
        DATA_TYPE = 'FLOAT' if features_dict[feature] in ['NUMERIC','PRICE'] else 'STRING'
        
        try:
            resp = client.get_variables(name = feature)
            features_dict[feature] = resp['variables'][0]['dataType']
            print("{0} has been defined, data type: {1}".format(feature, features_dict[feature]))
        except:
            print("Creating variable: {0}".format(feature))
            resp = client.create_variable(
                    name         = feature,
                    dataType     = DATA_TYPE,
                    dataSource   ='EVENT',
                    defaultValue = DEFAULT_VALUE, 
                    description  = feature,
                    variableType = features_dict[feature])
    return features_dict


# -- function to create all your labels --- 
def create_label(label_mapper):
    """
    Add labels to Fraud Detector
    
    Arguments:
        label_mapper   - a dictionary maps Fraud/Legit to your labels in data
    """
    for label in label_mapper['FRAUD']:
        response = client.put_label(
            name = label,
            description = "FRAUD")
    
    for label in label_mapper['LEGIT']:
        response = client.put_label(
            name = label,
            description = "LEGIT")



#########################
# 모생 생성 함수
#########################





#########################
# 피쳐 엔지니얼링 관련 함수
#########################


def save_csv_local(raw_df, preproc_folder, label, file_name):
    '''
    주어진 파일을 저장
    '''
    os.makedirs(preproc_folder, exist_ok=True)
    
    df = raw_df.copy()
    df = pd.concat([df[label], df.drop([label], axis=1)], axis=1)
    file_path = os.path.join(preproc_folder, file_name)
    df.to_csv(file_path, index=False, )

    print(f'{file_path} is saved')

    
    return file_path


from IPython.display import display as dp

def change_code_to_string(raw, col, new_col, verbose=False):
    '''
    숫자값에 'str' 를 넣어서 명시적으로 스트링으로 타입 변환
    '''
    df = raw.copy()
    col_val = df[col].unique()
    str_code = df[col].apply(lambda x: 'str_' + str(x))    
    
    index = [ i for i, e in enumerate(df.columns) if e == col]    # 생성 컬럼의 위치를 알기 위해서 임. 해당 컬럼의 옆에 삽입하기 위함.
    df.insert(index[0], column=new_col, value=str_code)
    if verbose:
        dp(col_val)
    
    return df


#########################
# 평가
#########################
import itertools
import matplotlib.pyplot as plt
import numpy as np

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
