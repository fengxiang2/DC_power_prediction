import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, TimeSeriesSplit
import lightgbm as lgb

from sklearn.preprocessing import PolynomialFeatures
import glob
import datetime
import time
import scipy.signal as signal

train_files = glob.glob('train/*.csv') 
test_files = glob.glob('test/*.csv')
print(test_files)

test_files = pd.Series(test_files)
train_files = pd.Series(train_files)

def read(file):
    d = pd.read_csv(file)
    #记录气象站编号
    d['index'] = int(file[-5])
    return d

#将十个电站数据合并
test = pd.concat(test_files.map(read).values, axis = 0)
train = pd.concat(train_files.map(read).values, axis = 0)

#删除
train.drop(['实际辐照度'],axis=1,inplace=True)


# 数据修正
def data_corection(data,test):
    l=len(data)
    for feature_num, feature in enumerate(data.keys()):
        if feature in ['时间','实际功率','实际辐照度', 'id'] :
            1+1
        else:
            print('正在处理的特征：',feature)
            Q1 = np.percentile(test[feature], 25) # test中25%分位的数值
            Q3 = np.percentile(test[feature], 75) # test中75%分位的数值
            step = (Q3-Q1)
            # 删除掉data中不在【(Q1 - step)和(Q3 + step)】范围内的异常值
            feature_index=data[~((data[feature] >= (Q1 - step)) & (data[feature] <= (Q3 + step)))].index
            data=data.drop(index=feature_index, axis= 0)
    return data


train =  data_corection(train,test)

#修改列名
train.columns = ['time','fzd','fs','fx','wd','sd','yq','label','index']
test.columns = ['id','time','fzd','fs','fx','wd','sd','yq','index']

#提取时间
train['year'] = train['time'].map(lambda x:int(datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').strftime('%Y')))
train['month'] = train['time'].map(lambda x:int(datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').strftime('%m')))
train['day'] = train['time'].map(lambda x:int(datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').strftime('%d')))
train['hour'] = train['time'].map(lambda x:int(datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').strftime('%H')))
train['minute'] = train['time'].map(lambda x:int(datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').strftime('%M')))

test['year'] = test['time'].map(lambda x:int(datetime.datetime.strptime(x, '%Y/%m/%d %H:%M').strftime('%Y')))
test['month'] = test['time'].map(lambda x:int(datetime.datetime.strptime(x, '%Y/%m/%d %H:%M').strftime('%m')))
test['day'] = test['time'].map(lambda x:int(datetime.datetime.strptime(x, '%Y/%m/%d %H:%M').strftime('%d')))
test['hour'] = test['time'].map(lambda x:int(datetime.datetime.strptime(x, '%Y/%m/%d %H:%M').strftime('%H')))
test['minute'] = test['time'].map(lambda x:int(datetime.datetime.strptime(x, '%Y/%m/%d %H:%M').strftime('%M')))

#记录每个电站的娤机功率
cp = [20,20,30,10,20,21,10,40,30,50]
train['cp'] = train['index'].map(lambda x:cp[x])
test['cp'] = test['index'].map(lambda x:cp[x])

#消除装机功率带来的影响
train['label'] = train['label']/train['cp']
#测试集提交时需要再乘回去

#平滑数据
train = train.reset_index(drop=True)
test = test.reset_index(drop=True)

def smooth(sig, method):
    if method == 0:
        return sig
    elif method == 1:
        return signal.medfilt(volume=sig, kernel_size=3)
    elif method == 2:
        return signal.savgol_filter(sig, 5, 3, 0)
    elif method == 3:
        kernel = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
        return np.convolve(sig, kernel, mode='same')
    
    
def smooth_cols(group):
    cols = ['fzd','fs','fx','wd','sd','yq']
    for col in cols:
        sig = group[col]
        group[col] = smooth(sig,1)
    return group

train = train.groupby('index').apply(smooth_cols)
test = test.groupby('index').apply(smooth_cols)

#添加多项式特征,为二阶特征

def add_poly_features(data, column_names):
    '''
    进行特征的构造，构造的方式就是特征与特征相乘（自己与自己，自己与其他人）
    例如：有 a、b 两个特征，那么它的 2 次多项式的次数为 (1,a,b,a^2,ab, b^2)。

    PolynomialFeatures 这个类有 3 个参数：
        degree：控制多项式的次数；
        interaction_only：默认为 False，如果指定为 True，那么就不会有特征自己和自己结合的项，组合的特征中没有 a2a2 和 b2b2；
        include_bias：默认为 True 。如果为 True 的话，那么结果中就会有 0 次幂项，即全为 1 这一列。

    :param data: df
    :param column_names: 字段名
    :return:
    '''
    features = data[column_names]
    rest_features = data.drop(column_names, axis=1)
    poly_transformer = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
    poly_features = pd.DataFrame(poly_transformer.fit_transform(features),
                                 columns=poly_transformer.get_feature_names(column_names))

    for col in poly_features.columns:
        if col in rest_features.columns.tolist():
            continue
        rest_features.insert(1, col, poly_features[col])
    return rest_features

train = add_poly_features(train, ['fzd','fs','fx','wd','sd','yq'])
test = add_poly_features(test, ['fzd','fs','fx','wd','sd','yq'])

base_fea = ['fzd','fs','fx','wd','sd','yq']
#将连续特征离散化
def get_bin(data,colmun_names):
    for col in colmun_names:
        data[col+'_bin'] = pd.cut(data[col],10,labels=False)

get_bin(train,base_fea)
get_bin(test,base_fea)

#获取单个和交互特征点击率特征
def ratio(data, feats0):
    label_feature = feats0
    data_temp = data[label_feature]
    df_feature = pd.DataFrame()
    data_temp['cnt'] = 1

    print('Begin ratio clcik...')
    col_type = label_feature.copy()
    n = len(col_type)
    for i in range(n):
        col_name = "ratio_click_of_" + col_type[i]
        df_feature[col_name] = (data_temp[col_type[i]].map(data_temp[col_type[i]].value_counts()) / len(data) * 100).astype(int)
    n = len(col_type)
    for i in range(n):
        for j in range(n):
            if i != j:
                col_name = "ratio_click_of_" + col_type[j] + "_in_" + col_type[i]
                se = data_temp.groupby([col_type[i], col_type[j]])['cnt'].sum()
                dt = data_temp[[col_type[i], col_type[j]]]
                cnt = data_temp[col_type[i]].map(data[col_type[i]].value_counts())
                a1 = pd.merge(dt, se.reset_index(), how='left', on=[col_type[i], col_type[j]])
                a = a1.sort_index()['cnt'].fillna(value=0) / cnt
                df_feature[col_name] = (a * 100).astype(int).values
    data = pd.concat([data, df_feature], axis=1)
    print('The end')

    return data


lbl_fea = [f+'_bin' for f in base_fea ]
train = ratio(train,lbl_fea)
test = ratio(test,lbl_fea)

#点击计数统计
def count(data, feats0):
    label_feature = feats0
    data_temp = data[label_feature]
    df_feature = pd.DataFrame()
    data_temp['cnt'] = 1
    print('Begin click stat...')
    col_type = label_feature.copy()
    n = len(col_type)
    
    for i in range(n):
        col_name = "cnt_click_of_" + col_type[i]
        se = (data[col_type[i]].map(data[col_type[i]].value_counts())).astype(int)
        semax = se.max()
        semin = se.min()
        df_feature[col_name] = ((se - se.min()) / (se.max() - se.min()) * 100).astype(int).values
    n = len(col_type)
    
    for i in range(n):
        for j in range(n - i - 1):
            col_name = "cnt_click_of_" + col_type[i + j + 1] + "_and_" + col_type[i]
            se = data_temp.groupby([col_type[i], col_type[i + j + 1]])['cnt'].sum()
            dt = data_temp[[col_type[i], col_type[i + j + 1]]]
            se = (pd.merge(dt, se.reset_index(), how='left',
                           on=[col_type[i], col_type[j + i + 1]]).sort_index()['cnt'].fillna(value=0)).astype(int)
            df_feature[col_name] = ((se - se.min()) / (se.max() - se.min()) * 100).fillna(value=0).astype(int).values
    data = pd.concat([data, df_feature], axis=1)
    print('The end')

    return data

train = count(train,lbl_fea)
test = count(test,lbl_fea)

#利用时间提取特征，光伏发电存在峰值
def dis_2peak(data):
    data['dis-2peak'] = data['hour'].map(lambda x: (13-abs(13-x))/13)

dis_2peak(train)
dis_2peak(test)

#将一天分为不同的时区
def time_zone(h):
    if h < 7:
        return 1
    elif h < 10:
        return 2
    elif h < 13:
        return 3
    elif h < 16:
        return 4
    elif h < 19:
        return 5
    else:
        return 1


def get_timezone(data):
    data['timezone'] = data['hour'].map(lambda x: time_zone(x))

get_timezone(train)
get_timezone(test)

#获取月份信息
def get_absmonth(data):
    data['abs-month'] = abs(data['month']-6)

get_absmonth(train)
get_absmonth(test)

def getRatioFeatures(con_data):
    # 比例特征
    feat1 = ['fzd', 'fs', 'fx']
    feat0 = ['fzd', 'fs', 'fx', 'wd', 'yq', 'sd']
    for fea1 in feat1:
        for fea2 in feat0:
            if (fea1 != fea2):
                con_data[fea1 + '/' + fea2] = con_data[fea1] / con_data[fea2]
    return con_data

train = getRatioFeatures(train)
test = getRatioFeatures(test)


dff_fea = [f+'_diff' for f in base_fea]
train[dff_fea] = train.groupby('index')[base_fea].diff()
test[dff_fea] = test.groupby('index')[base_fea].diff()

fea = [f for f in train.columns if f not in ['id','label','time','index','cp']]
X = train[fea]
y = train['label']
X_test = test[fea]

#lgb五折交叉验证
kf = KFold(n_splits=5, random_state=2019, shuffle=True)

lgb_param = {    'boosting_type': 'gbdt', 
    'objective': 'regression', 
    'learning_rate': 0.1, 
         
    'max_depth':-1,   
     'subsample': 0.7, 
    'colsample_bytree': 0.7,
    'metric':'mae'
    }


imp = pd.DataFrame()
oof_lgb = np.zeros(len(X))
pred_lgb = np.zeros(len(X_test))

for index, (trn_idx, val_idx) in enumerate(kf.split(X,y)):
    print('{}折交叉验证开始'.format(index+1))
    trn_data = lgb.Dataset(X.values[trn_idx], y.values[trn_idx])
    val_data = lgb.Dataset(X.values[val_idx], y.values[val_idx])
    num_round = 2000
    clf = lgb.train(lgb_param, trn_data, num_round, valid_sets = [val_data],verbose_eval = 100, 
                    early_stopping_rounds = 200)
    oof_lgb[val_idx] = clf.predict(X.values[val_idx], num_iteration = clf.best_iteration)
    #f1s = f1_score(y.values[val_idx],np.argmax(oof_lgb[val_idx], axis=1),average='macro')
    imp['imp_'+str(index+1)] = clf.feature_importance()
    pred_lgb += clf.predict(X_test.values, num_iteration=clf.best_iteration)/5

imp.index = fea

sub = pd.DataFrame()
sub['id'] = test['id']
sub['prediction'] = pred_lgb
sub['prediction'] = sub['prediction']*test['cp']
sub.to_csv('sub.csv',index=None)
