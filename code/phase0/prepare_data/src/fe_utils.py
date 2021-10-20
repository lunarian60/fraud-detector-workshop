

#####################################
# RFM 피쳐 함수
#####################################

def run_user_recency_data(df, params, recent_threshold, verbose=False):
    '''
    # recent_data = [recent_time, recent_duration, recent_freq
    '''
    # 유저 데이터의  사전 생성
    user_dic = {
                #'userno': {'timestamp': '2021-05-15 20:10:10', 'recent_duration': 350, 'recent_freq': 0, 'recent_threshold': 600,'money_threshold': 3600,   'money_amt': 0}
               }
    
    # 최근 데이타를 담을 리스트
    recent_data = []    
    df2 = df.copy()
    
    if verbose:
        print("df shape: ",df2.shape)
    
    for i, row in enumerate(df2.iterrows()):
        user_no = row[1][params['userno']]
        regdate = row[1][params['event_tiemstamp']]
#         pay_amt = row[1]['totalpayamt']
        
        if verbose:
            print("################# run_user_recency_data() ###############")
            print("regdate: ", regdate)
            print("userno: ", user_no)

        
        user_recent_record = get_userno(user_dic, user_no= user_no ) # userno 에 대해서 조회
        
        if user_recent_record == None:  # 유저 사전에 데이터가 없는 경우. 처음 빌랭 트랜잭션을 의미 함.
            updated_user_dic, user_recent_record = add_user_recency(user_no, regdate, user_dic, recent_threshold,verbose=False)     
        else:
            updated_user_dic, user_recent_record, recent_time = update_user_recency(user_no, regdate, user_recent_record, user_dic, verbose=False )
            regdate = recent_time # 바로 과거 트랜잭션으로 할당 함. 

            
        recent_duration = user_recent_record['recent_duration']


        recent_freq = user_recent_record['recent_freq']                
        

        recent_data.append([])    
        recent_data[i].append(regdate)
        recent_data[i].append(recent_duration)
        recent_data[i].append(recent_freq)        
        recent_data[i].append(recent_threshold)            

    return recent_data, updated_user_dic

def get_userno(user_dic, user_no):
    '''
    user_no 를 입력하면 최근 레코드를 제공
    get_userno(user_dic, user_no='meta' )
    '''
    user_data = user_dic.get(user_no)
    return user_data

def add_user_recency(user_no, regdate,user_dic, recent_threshold, verbose=False):
    '''
    유저가 없기에 새로운 유저를 유저 사전에 입력 함.
    '''
    #recent_duration = 9999999 # 처음 트랜잭션은 큰 값으로 지정
    recent_duration = 0 # 처음 트랜잭션은 0 으로 지정            
    recent_freq = 0 # 처음 트랜잭션이기에 0 으로 설정
    user_data  = {'timestamp': regdate, 'past_timestamp':None , 'recent_duration': recent_duration,'recent_freq': recent_freq, 'recent_threshold': recent_threshold,} # 유저 사전에 담을 레코드


    if verbose:
        print("add_user_recency()")
        print("user_no: ", user_no)
        print("user_data: \n", user_data)        

    
    added_user_dic = insert_userno(user_dic, user_no= user_no, user_data = user_data) # 유저 사전에 입력     
    if verbose:
        print("added_user_dic: \n", added_user_dic )
    
    return added_user_dic, user_data



def insert_userno(user_dic, user_no, user_data, verbose=False):
    '''
    user_no 가 없으면 유저 사전에 입력 함.
    user_no 가 있으면 기존 것을 제거하고, 새로이 레코드 생성하여 입력
    # user_data = {'timestamp': '2021-05-15 20:10:10','recent_duration': 350,'recent_freq': 100, 'threshold': 600}
    # insert_userno(user_dic, user_no='user_1', user_data = user_data)
    '''

    if user_dic.get(user_no) == None:        
        user_dic[user_no] = user_data
        output = user_dic.get(user_no)    
        if verbose:
            print(f'{output} is inserted')
    else:
        user_dic.pop(user_no)
        user_dic[user_no] = user_data        
        if verbose:
            print("userno info is updated")
            print(user_dic.get(user_no))
            
    return user_dic


    

def update_user_recency(user_no, regdate, user_recent_record, user_dic, verbose=False ):
    '''
    유저의 레코드를 업데이트를 함.
    '''
    recent_time = user_recent_record['timestamp']

    recent_duration = regdate - recent_time # 현재 트랜잭션 시간 - 최근 마지막 트랜잭션 시간
    recent_duration = int(recent_duration.seconds) # 초로 변환

    recent_threshold = user_recent_record['recent_threshold']    

    
    
    # recent_freq 계산
    if recent_duration < recent_threshold: # 지정 threshold 보다 적으면 +1, 그렇지 않으면 0으로 리셋
        recent_freq = int(user_recent_record['recent_freq'] + 1) # 현재 트랜잭션을 더함.
    else:
        recent_freq = 0 # 리셋은 현재 트랜잭션이므로 0
        

    user_data  = {'timestamp': regdate, 'past_timestamp': recent_time, 'recent_duration': recent_duration, 'recent_freq': recent_freq, 
                  'recent_threshold': recent_threshold, }  # 현재 값을 기반으로 갱신하여 입력

    
            
    insert_userno(user_dic, user_no= user_no, user_data = user_data)                

    regdate = recent_time # 바로 과거 트랜잭션으로 할당 함. 

    if verbose:
        print(recent_time)            
        print(recent_duration)                
        print(recent_freq)
        
    return user_dic, user_data, recent_time


            