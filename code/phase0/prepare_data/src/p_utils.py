import pandas as pd
import os


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

