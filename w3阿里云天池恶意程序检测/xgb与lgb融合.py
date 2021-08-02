import time
import pickle
import os
import pandas as pd
import xgboost as xgb
import lightgbm as lgb


def get_data(file_name):
    result = []
    chunk_index = 0
    for df in pd.read_csv(open(file_name, 'r'), chunksize=1000000):
        result.append(df)
        #print('chunk', chunk_index)
        chunk_index += 1
    result = pd.concat(result, ignore_index=True, axis=0)
    return result


# 获取全量数据
train = get_data('./security_train.csv')
test = get_data('./security_test.csv')
print("原始数据读取成功")



with open('./df_train.pkl', 'rb') as file:
    df_train = pickle.load(file)

with open('./df_test.pkl', 'rb') as file:
    df_test = pickle.load(file)
print("pkl数据读取成功")

def get_apis(df):
    # 按照file_id进行分组
    group_fileid = df.groupby('file_id')

    # 统计file_id 和对应的 api_sequence
    file_api = {}

    # 计算每个file_id的api_sequence
    for file_id, file_group in group_fileid:
        # 针对file_id 按照线程tid 和 顺序index进行排序
        result = file_group.sort_values(['tid', 'index'], ascending=True)
        # 得到api的调用序列
        api_sequence = ' '.join(result['api'])
        # print(api_sequence)
        file_api[file_id] = api_sequence
    return file_api


test_apis=get_apis(test)
train_apis = get_apis(train)

df_train.drop(['api', 'tid', 'index'], axis=1, inplace=True)
df_test.drop(['api', 'tid', 'index'], axis=1, inplace=True)

temp = pd.DataFrame.from_dict(train_apis, orient='index', columns=['api'])
temp = temp.reset_index().rename(columns={'index': 'file_id'})
df_train = df_train.merge(temp, on='file_id', how='left')

temp = pd.DataFrame.from_dict(test_apis, orient='index', columns=['api'])
temp = temp.reset_index().rename(columns={'index': 'file_id'})
df_test = df_test.merge(temp, on='file_id', how='left')


df_all = pd.concat([df_train, df_test], axis=0)

from sklearn.feature_extraction.text import TfidfVectorizer
#使用1-3元语法（1元语法 + 2元语法 + 3 元语法）
vec=TfidfVectorizer(ngram_range=(1,3),min_df=0.1)
api_features=vec.fit_transform(df_all['api'])

df_apis = pd.DataFrame(api_features.toarray(), columns=vec.get_feature_names())

df_train_apis = df_apis[df_apis.index <= 13886]
df_test_apis = df_apis[df_apis.index > 13886]
df_test_apis.index = range(len(df_test_apis))
df_train = df_train.merge(df_train_apis, left_index=True, right_index=True)
df_test = df_test.merge(df_test_apis, left_index=True, right_index=True)
df_train.drop('api', axis=1, inplace=True)
df_test.drop('api', axis=1, inplace=True)

start_time=time.time()
print("开始训练...")
clf = lgb.LGBMClassifier(num_leaves=2 ** 5 - 1, reg_alpha=0.25, reg_lambda=0.25, objective='multiclass', max_depth=5,
                         learning_rate=0.005, min_child_sample=3, random_state=2021,
                         n_estimators=2000, subsample=1, colsample_bytree=1)
clf.fit(df_train.drop(['label'], axis=1), df_train['label'])
result = clf.predict_proba(df_test)
result_lgb = pd.DataFrame(result, columns=['prob0', 'prob1', 'prob2', 'prob3', 'prob4', 'prob5', 'prob6', 'prob7'])
result_lgb['file_id'] = df_test['file_id'].values
model_xgb = xgb.XGBClassifier(
    max_depth=5, learning_rate=0.005, n_estimators=2000,
    objective='multi:softprob', tree_method='auto',
    subsample=0.8, colsample_bytree=0.8,
    min_child_samples=3, eval_metric='logloss', reg_lambda=0.5)
model_xgb.fit(df_train.drop('label', axis=1), df_train['label'])
result_xgb = model_xgb.predict_proba(df_test)
result_xgb = pd.DataFrame(result_xgb, columns=['prob0', 'prob1', 'prob2', 'prob3', 'prob4', 'prob5', 'prob6', 'prob7'])
result_xgb['file_id'] = df_test['file_id'].values
# 对两个模型的结果模型融合 进行加权平均
result = result_lgb.copy()
weight_lgb, weight_xgb = 0.5, 0.5
result['prob0'] = result['prob0'] * weight_lgb + result_xgb['prob0'] * weight_xgb
result['prob1'] = result['prob1'] * weight_lgb + result_xgb['prob1'] * weight_xgb
result['prob2'] = result['prob2'] * weight_lgb + result_xgb['prob2'] * weight_xgb
result['prob3'] = result['prob3'] * weight_lgb + result_xgb['prob3'] * weight_xgb
result['prob4'] = result['prob4'] * weight_lgb + result_xgb['prob4'] * weight_xgb
result['prob5'] = result['prob5'] * weight_lgb + result_xgb['prob5'] * weight_xgb
result['prob6'] = result['prob6'] * weight_lgb + result_xgb['prob6'] * weight_xgb
result['prob7'] = result['prob7'] * weight_lgb + result_xgb['prob7'] * weight_xgb

columns = ['file_id', 'prob0', 'prob1', 'prob2', 'prob3', 'prob4', 'prob5', 'prob6', 'prob7']
result.to_csv('./baseline_2.csv', index=False, columns=columns)
end_time=time.time()
use_time=end_time-start_time
print("花费的时间:",use_time,'秒\t',use_time/60,"分\t",use_time/3600,"时")