import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from category_encoders import TargetEncoder
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
from sklearn.svm import SVR
import lightgbm as lgb


pd.set_option('display.max_columns', None)

PATH = './data/'

train_data = pd.read_csv(PATH + 'train.csv')
test_data = pd.read_csv(PATH + 'test.csv')

train_data_ids = train_data.id

# print(train_data.head())
# print(train_data.info())
# print(train_data.shape)
# print(train_data.describe())
# print(train_data.columns)

# label_1 = train_data[train_data['label'] == 1]
# label_0 = train_data[train_data['label'] == 0]
# print(label_0.head())
# print(label_1.head())
# label_1.to_csv('label_1.csv',index=False,sep=',')
# label_0.to_csv('label_0.csv',index=False,sep=',')
#
# exit(0)

#[个人缴存/个人基数，个人加单位/个人基数，已还款额，还款余额/发放额，还款余额*利率=（年利息），还款余额*利率/12=（月利息）
# 月利息/（个人+单位缴存），（个人+单位）*12（年缴存）
# 年利息/去年结存  年结转/年缴存
# 月结存，月还款，    去年结存/12（月结存） （个人加单位）-月结存 = 月还款
# 月利息 / 月还款
# ]

# exit(0)


del_columns = ['label','id']




categoricals_1 = ['XUELI','ZHIWU','HYZK','ZHIYE','XINGBIE','GRZHZT','ZHICHEN']
categoricals_2 = ['DWJJLX','DWSSHY']
categoricals = categoricals_1 + categoricals_2

pd_data = pd.concat([train_data,test_data],axis= 0).reset_index(drop=True)


scaler_temp = LabelEncoder()
pd_data['XUELI'] = scaler_temp.fit_transform(pd_data['XUELI'])
pd_data['GRZHZT'] = scaler_temp.fit_transform(pd_data['GRZHZT'])
pd_data['ZHICHEN'] = scaler_temp.fit_transform(pd_data['ZHICHEN'])




#个人存缴/个人基数
pd_data['person_GJJ_ratio'] = pd_data['GRYJCE'] / pd_data['GRJCJS']
pd_data['sum_GJJ_ratio'] = (pd_data['GRYJCE'] + pd_data['DWYJCE']) / pd_data['GRJCJS']
pd_data['sum_GJJ_month'] = pd_data['GRYJCE'] *2
pd_data['sum_GJJ_year'] = pd_data['sum_GJJ_month'] *12
pd_data['ratio_of_GJJ_jiezhuan'] = pd_data['GRZHSNJZYE'] / pd_data['sum_GJJ_year']

#已还贷款
pd_data['already_loan'] = pd_data['DKFFE'] - pd_data['DKYE']
pd_data['loan_ratio'] = pd_data['already_loan'] / pd_data['DKFFE']
pd_data['already_year'] = pd_data['already_loan'] / (pd_data['sum_GJJ_year'])
pd_data['DKFFE_DKYE_multi_DKLL'] = (pd_data['DKFFE'] + pd_data['DKYE']) * pd_data['DKLL']
pd_data['DKFFE_multi_DKLL_ratio'] = pd_data['DKFFE'] * pd_data['DKLL'] / pd_data['DKFFE_DKYE_multi_DKLL']
pd_data['DKYE_multi_DKLL'] = pd_data['DKYE'] * pd_data['DKLL'] / pd_data['DKFFE_DKYE_multi_DKLL']


#利息
pd_data['DK_lixi_year'] = pd_data['DKFFE'] * pd_data['DKLL'] / 100
pd_data['DK_lixi_month'] = pd_data['DK_lixi_year'] / 12
pd_data['ratio_of_lixi_GJJ'] = pd_data['DK_lixi_year'] / pd_data['sum_GJJ_year']
pd_data['ratio_of_lixi_jiezhuan'] = pd_data['DK_lixi_year'] / pd_data['GRZHSNJZYE']

#结转
pd_data['jiezhuan_month'] = pd_data['GRZHSNJZYE'] / 12
pd_data['huankuan_month'] = pd_data['GRYJCE'] *2 - pd_data['jiezhuan_month']
pd_data['ratio_of_lixi_huankuan'] = pd_data['DK_lixi_month'] / pd_data['huankuan_month']
pd_data['ratio_of_jiezhuan_DKYE'] = pd_data['GRZHSNJZYE'] / pd_data['DKYE']

#余额
pd_data['grzhye'] = pd_data['GRZHSNJZYE'] + pd_data['GRZHDNGJYE']

#unknown
pd_data['unknown'] = pd_data['GRYJCE'] * 24 *(1+(pd_data['DKLL']/100))
#结息
pd_data['SNJZ_without_jiexi'] = pd_data['GRZHSNJZYE'] / 1.015
pd_data['SN_jiexi'] = pd_data['SNJZ_without_jiexi'] * 0.015

#DK_year>5 DK_num #2.75 3.25
pd_data['DK_5year'] = np.where(pd_data['DKLL']<=3.025,1,0)
pd_data['DK_num'] = 1
def cal(x):
    if x['DKLL'] >2.75 and x['DKLL'] <=3.025:
        num = 2
    else:
        num = 1
    if x['DKLL'] == 3.575:
        num = 2
    return num

pd_data['DK_num'] = pd_data.apply(cal,axis =1)

pd_data['age'] = ((1609430399 - pd_data['CSNY']) / (365 * 24 * 3600)).astype(int)
pd_data.loc[pd_data['age']<0,'age'] = 35
bin_1 = [i*10 for i in range(10)]
pd_data['age'] = pd.cut(pd_data['age'], bin_1, labels=False)


bin_2 = [i*100000 for i in range(5)]
pd_data['DKFFE_bin'] = pd.cut(pd_data['DKFFE'], bin_2, labels=False)
pd_data['DKYE_bin'] = pd.cut(pd_data['DKYE'], bin_2, labels=False)

bin_3 = [i*300 for i in range(7)]
pd_data['GRYJCE_bin'] = pd.cut(pd_data['GRYJCE'], bin_3, labels=False)

# print(pd_data['GRYJCE_bin'].value_counts())
# exit(0)

#group1
grps1 = pd_data.groupby(['DWJJLX','DWSSHY']).agg({'GRYJCE':'median','DKYE':'median','DKFFE':'median',
                                                  'GRZHYE':'median','GRZHSNJZYE':'median','GRZHDNGJYE':'median','grzhye':'median',
                                                  'unknown':'median'}).reset_index()
pd_data = pd.merge(pd_data,grps1,on=['DWJJLX','DWSSHY'],how='inner')
pd_data['dis_GRYJCE'] = pd_data['GRYJCE_x'] - pd_data['GRYJCE_y']
pd_data['dis_DKYE'] = pd_data['DKYE_x'] - pd_data['DKYE_y']
pd_data['dis_DKFFE'] = pd_data['DKFFE_x'] - pd_data['DKFFE_y']
pd_data['dis_GRZHYE'] = pd_data['GRZHYE_x'] - pd_data['GRZHYE_y']
pd_data['dis_GRZHSNJZYE'] = pd_data['GRZHSNJZYE_x'] - pd_data['GRZHSNJZYE_y']
pd_data['dis_GRZHDNGJYE'] = pd_data['GRZHDNGJYE_x'] - pd_data['GRZHDNGJYE_y']
pd_data['dis_grzhye'] = pd_data['grzhye_x'] - pd_data['grzhye_y']
pd_data['dis_unknown'] = pd_data['unknown_x'] - pd_data['unknown_y']

pd_data = pd_data.drop(columns=['GRYJCE_y','DKYE_y','DKFFE_y','GRZHYE_y','GRZHSNJZYE_y','GRZHDNGJYE_y','grzhye_y','unknown_y','DWYJCE'])

categoricals = categoricals + ['DK_5year','DK_num']
num_cols = [col for col in pd_data.columns if col not in categoricals and col not in del_columns]
# 特征交叉
# for f1 in tqdm(categoricals):
#     g = pd_data.groupby(f1)
#     stats = ['mean','sum','std','max','min']
#     for f2 in num_cols:
#         for stat in stats:
#             pd_data['{}_{}_{}'.format(f1,f2,stat)] = g[f2].transform(stat)

drop_feats = [f for f in train_data.columns if train_data[f].nunique() == 1 or train_data[f].nunique() == 0]


scaler_2 = OneHotEncoder()
pd_onehot = scaler_2.fit_transform(pd_data[['XINGBIE','HYZK']]).toarray()

pd_onehot = pd.DataFrame(pd_onehot)

pd_data = pd.concat([pd_data,pd_onehot],axis= 1)

pd_data = pd_data.drop(columns=['XINGBIE','HYZK'])

scaler_3 = TargetEncoder(cols=['ZHIYE','DWJJLX','DWSSHY'])

# print(pd_data.head())
#
# print(pd_data.isnull().sum())
#
# exit(0)

train_data = pd_data[pd_data['id'].isin(train_data_ids) ]
test_data = pd_data[~pd_data['id'].isin(train_data_ids)]


# exit(0)



train_data = scaler_3.fit_transform(train_data,train_data['label'])
test_data = scaler_3.transform(test_data)


# print(train_data.columns)
features = [col for col in train_data.columns if col not in del_columns]


# exit(0)
x_train = np.array(train_data[features])
y_train = np.array(train_data['label'])
x_test = np.array(test_data[features])

folds = StratifiedKFold(n_splits=6,shuffle=True,random_state=111)
kfolds = folds.split(x_train,y_train)

oof_lgb = np.zeros((len(train_data), 2))
predictions_lgb = np.zeros((len(test_data),2))

oof_cat = np.zeros((len(train_data), 2))
predictions_cat = np.zeros((len(test_data),2))


params_lgb = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    # 'num_class': 2,
    'metric': 'binary_logloss',
    'learning_rate': 0.01,
    'num_leaves': 40,
    'max_depth': 6,
    'subsample': .9,
    'colsample_bytree': .8,
    # 'reg_alpha': .01,
    # 'reg_lambda': .01,
    # 'min_split_gain': 0.01,
    # 'min_child_weight': 10,
    # 'silent':True,
    'verbosity': -1,
    # 'nthread':-1
}

params_cat = {
    'objective': 'Logloss',
    # 'num_class': 2,
    'eval_metric': 'Logloss',
    'learning_rate': 0.01,
    # 'num_leaves': 40,
    # 'max_depth': 6,
    'subsample': .9,
    # 'colsample_bytree': .8,
    # 'reg_alpha': .01,
    # 'reg_lambda': .01,
    # 'min_split_gain': 0.01,
    # 'min_child_weight': 10,
    # 'silent':True,
    'verbose': False,
    # 'nthread':-1
}

for i,(tra_id,val_id) in enumerate(kfolds):
    print("{}/6 folds".format(i+1))
    tra_x, tra_y = x_train[tra_id], y_train[tra_id]
    val_x, val_y = x_train[val_id], y_train[val_id]

    cls_lgb = LGBMClassifier(**params_lgb,n_estimators=5000)
    cls_cat = CatBoostClassifier(**params_cat,n_estimators=5000)



    cls_lgb.fit(tra_x, tra_y, eval_set=(val_x,val_y), early_stopping_rounds=50,verbose=50)
    cls_cat.fit(tra_x, tra_y, eval_set=(val_x,val_y), early_stopping_rounds=50,verbose=50)

    oof_lgb[tra_id] = cls_lgb.predict_proba(tra_x,num_iteration=cls_lgb.best_iteration_)
    predictions_lgb += cls_lgb.predict_proba(x_test, num_iteration=cls_lgb.best_iteration_)/folds.n_splits

    oof_cat[tra_id] = cls_cat.predict_proba(tra_x)
    predictions_cat += cls_cat.predict_proba(x_test) / folds.n_splits

    #print feature_importance
    features_weight = pd.DataFrame(data=cls_lgb.feature_importances_,
                                   index=features,columns=['weight'])
    features_weight.sort_values(by='weight',ascending=False,inplace=True)
    print(features_weight.head(10))


def tpr_weight_funtion(y_true,y_predict):

    d = pd.DataFrame()

    d['prob'] = list(y_predict)

    d['y'] = list(y_true)

    d = d.sort_values(['prob'], ascending=[0])

    y = d.y

    PosAll = pd.Series(y).value_counts()[1]

    NegAll = pd.Series(y).value_counts()[0]

    pCumsum = d['y'].cumsum()

    nCumsum = np.arange(len(y)) - pCumsum + 1

    pCumsumPer = pCumsum / PosAll

    nCumsumPer = nCumsum / NegAll

    TR1 = pCumsumPer[abs(nCumsumPer-0.001).idxmin()]

    TR2 = pCumsumPer[abs(nCumsumPer-0.005).idxmin()]

    TR3 = pCumsumPer[abs(nCumsumPer-0.01).idxmin()]

    return 0.4 * TR1 + 0.3 * TR2 + 0.3 * TR3


score_lgb = tpr_weight_funtion(train_data['label'].values, oof_lgb[:,1])
print("oof_lgb_score: {:<8.8f}".format(score_lgb))

score_cat = tpr_weight_funtion(train_data['label'].values, oof_cat[:,1])
print("oof_cat_score: {:<8.8f}".format(score_cat))

score = (score_lgb + score_cat)/2

#stacking

oof_stack = np.hstack((oof_lgb,oof_cat))
predictions_stack = np.hstack((predictions_lgb,predictions_cat))
ps_label = pd.DataFrame()
ps_label['id'] = test_data.id
ps_label['label_1'] = predictions_lgb[:,1]
ps_label['label_2'] = predictions_cat[:,1]
ps_label.to_csv(PATH + 'ps_label.csv',index=False,header=True)


lr = LinearRegression()
lr.fit(oof_stack,train_data['label'].values)
predictions_lr = lr.predict(predictions_stack)


sub = pd.DataFrame()
sub['id'] = test_data.id
sub['label'] = predictions_lr
sub['label'] = np.where(sub['label'] < 0, 0, sub['label'])
sub['label'] = np.where(sub['label'] >=1, 1, sub['label'])
sub.to_csv(PATH + 'stacking_featurev3_{:<8.8f}'.format(score),index=False,header=True)
