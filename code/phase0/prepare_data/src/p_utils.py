import pandas as pd
import os




#########################
# 데이터 탐색
#########################

import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style(style="whitegrid")


def drop_column(raw_df, col):
    '''
    해당 컬럼 제거
    '''
    df = raw_df.drop(columns=[col])
    return df


def create_datetype_cols(ldf, source_col, target_col):
    '''
    YYYY-MM-DD HH:00:00 을 가지는 EVENT_TIMESTAMP_SIMPLE 추가
    '''
    df = ldf.copy()
    
    def convert_time(str):
        try:
            date = datetime.strptime(str, '%Y-%m-%d %H:%M:%S')

        except:
            print(f"Error: Incorrect form: {str}")
            str = '1988-12-31' 
            date = datetime.strptime(str, '%Y-%m-%d')        
        return date

    df['temp'] = pd.to_datetime(df[source_col], format= '%Y.%m.%d %H:%M:%S')        
    df[target_col] = df['temp'].apply(lambda x: x.strftime("%Y-%m-%d %H")) # yyyy-mm-dd 로 컬럼 생성    
    df[target_col] = pd.to_datetime(df[target_col], format= '%Y-%m-%d %H:%M:%S')    
    df.drop(columns=['temp'], inplace=True)
            
    return df



def show_classes_date(ldf,params):
    '''
    시간의 흐름에 따라 프로드의 시간별 비율을 보여줌. 
    '''
    df = ldf.copy()
    
    frac = params['frac']
    target_col = params['target_col']    
    label = params['label_col']        
    xticks = params['xticks']            
    
    df = df.sample(frac=frac, random_state= 100)
    g_df = df[[target_col, label]].groupby(target_col)[label].mean()
    # dp(g_df)
    plt.figure(figsize=(params['FigSizeW'],params['FigSizeH']))
    plt.plot(g_df)
    plt.ylabel('average fraud_mean')
    plt.title(params['title'])
    plt.xlabel(target_col)
    plt.xticks(g_df.index[::xticks], rotation='vertical')    
    print('')



def display_category_dist(fdf, cols, top_num = 20, verbose=False):
    '''
    카테고리 변수의 값을 큰 값 기준으로 정령하여 보기
    '''
    cols_num = len(cols)
    fig, axes = plt.subplots(nrows=1, ncols=cols_num)
    fig.set_size_inches(20,5)

    
    for idx, col in enumerate(cols):
#        df_unique = fdf[col].value_counts()[0:top_num]
        df_unique = fdf[col].value_counts().sort_values(ascending=False)[0:top_num]        

        axes[idx].set_title(col)
        sns.barplot(ax=axes[idx], x=df_unique.index, y=df_unique.values)
        
        if verbose:
            print(f'\n{df_unique}')



def plot_cor_feature_label(df, params):
    '''
    컬럼의 갯수와 평균값을 보여줌
    '''
    num = params['num_x_items']
    col = params['target_col']
    label_col = params['label_col']    

    stats = df[[col, label_col]].groupby(col).agg(['count', 'mean'])
    stats = stats.reset_index()
    stats.columns = [col, 'count', 'mean']
    stats = stats.sort_values('mean', ascending=False)
    # dp(stats)
    fig, ax1 = plt.subplots(figsize=(params['FigSizeW'],params['FigSizeH']))
    
    ax2 = ax1.twinx()
    ax1.bar(stats[col].astype(str).values[0:num], stats['count'].values[0:num])
    ax1.set_xticklabels(stats[col].astype(str).values[0:num], rotation='vertical')
    ax2.plot(stats['mean'].values[0:num], color='red')
    ax2.set_ylim(0,1.5)
    ax2.set_ylabel('Mean Target')
    ax1.set_ylabel('Frequency')
    ax1.set_xlabel(col)
    ax1.set_title('TopN ' + col + 's based on frequency')



#########################
# 데이터 준비 관련 함수
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

import os
def save_pickle_local(raw_df, preproc_folder, label, file_name):
    '''
    데이터 프레임을 pickle 파일로 저장
    '''
    df = raw_df.copy()
    
    os.makedirs(preproc_folder, exist_ok=True)        
    df = pd.concat([df[label], df.drop([label], axis=1)], axis=1)
    file_path = os.path.join(preproc_folder, file_name)

    df.to_pickle(file_path)
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


def split_data_by_time(df, target_col, label_col, total_samples, split_rate, train_end, test_start, verbose=False):
    '''
    시간 관점으로 번반부튼 훈련, 후반부는 테스트 데이터로 해서 샘블링 함.
    '''
    
    # 훈련 데이터 셋
    train_df = df[df[target_col] <= train_end]   
    train_num = int(total_samples * (1 - split_rate))    # 훈련 샘플 데이터 수
    train_sample = train_df.sample(n = train_num, random_state=100)    # 샘플링    

    print("train sample shape: ", train_sample.shape)
    print("train min time: ", train_sample[target_col].min())
    print("train max time: ", train_sample[target_col].max())
    print("Train fraud ratio: ", round(train_sample[label_col].value_counts()[1] / train_sample.shape[0],5))
    print("# of Train frauds: ", train_sample[label_col].value_counts()[1])     


    # 테스트 데이터 셋    
    test_df = df[df[target_col] >= test_start]    
    test_num = int(total_samples * (split_rate))    # 테스트 샘플 데이터 수
    test_sample = test_df.sample(n = test_num, random_state=100)    
    

    print("\ntest sample shape: ", test_sample.shape)    
    print("test min time: ", test_sample[target_col].min())
    print("test max time: ", test_sample[target_col].max())
    print("Test fraud ratio: ", round(test_sample[label_col].value_counts()[1] / test_sample.shape[0],5))    
    print("# of test frauds: ", test_sample[label_col].value_counts()[1])         
    
    
    return train_sample, test_sample

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

