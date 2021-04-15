import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from category_encoders import TargetEncoder
from catboost import CatBoostClassifier


pd.set_option('display.max_columns', None)

PATH = './data/'

train_data = pd.read_csv(PATH + 'train.csv')
test_data = pd.read_csv(PATH + 'test.csv')

# print(train_data.head())
# print(train_data.info())
# print(train_data.shape)
# print(train_data.describe())

del_columns = ['label','id']
features = [col for col in train_data.columns if col not in del_columns]

# print(features)

categoricals_1 = ['XUELI','ZHIWU','HYZK','ZHIYE','XINGBIE','GRZHZT','ZHICHEN']
categoricals_2 = ['DWJJLX','DWSSHY']

categoricals_indices = list(train_data[features].columns.get_indexer(categoricals_1))
categoricals_indices = categoricals_indices + list(train_data[features].columns.get_indexer(categoricals_2))
print(categoricals_indices)



#[个人缴存/个人基数，个人加单位/个人基数，已还款额，还款余额/发放额，还款余额*利率=（年利息），还款余额*利率/12=（月利息）
# 月利息/（个人+单位缴存），（个人+单位）/12（年缴存）
# 年利息/去年结存
# 月结存，月还款，    去年结存/12（月结存） （个人加单位）-月结存 = 月还款
# 月利息 / 月还款
# ]

pd_data = pd.concat([train_data,test_data],axis= 0).reset_index(drop=True)

#个人存缴/个人基数
pd_data['person_GJJ_ratio'] = pd_data['GRYJCE'] / pd_data['GRJCJS']
pd_data['sum_GJJ_ratio'] = (pd_data['GRYJCE'] + pd_data['DWYJCE']) / pd_data['GRJCJS']
pd_data['sum_GJJ_month'] = pd_data['GRYJCE'] *2
pd_data['sum_GJJ_year'] = pd_data['sum_GJJ_month'] *12
pd_data['ratio_of_GJJ_jiezhuan'] = pd_data['GRZHSNJZYE'] / pd_data['sum_GJJ_year']

#已还贷款
pd_data['already_loan'] = pd_data['DKFFE'] - pd_data['DKYE']
pd_data['loan_ratio'] = pd_data['already_loan'] / pd_data['DKFFE']

#利息
pd_data['DK_lixi_year'] = pd_data['DKYE'] * pd_data['DKLL']
pd_data['DK_lixi_month'] = pd_data['DK_lixi_year'] / 12
pd_data['ratio_of_lixi_GJJ'] = pd_data['DK_lixi_year'] / pd_data['sum_GJJ_year']
pd_data['ratio_of_lixi_jiezhuan'] = pd_data['DK_lixi_year'] / pd_data['GRZHSNJZYE']



train_data = pd_data[:40000]
test_data = pd_data[40000:]

# print(train_data)

x_train = np.array(train_data[features])
y_train = np.array(train_data['label'])
x_test = np.array(test_data[features])

folds = StratifiedKFold(n_splits=6,shuffle=True,random_state=111)
kfolds = folds.split(x_train,y_train)
oof_cat = np.zeros((len(train_data), 2))
predictions_cat = np.zeros((len(test_data),2))


params = {
    'objective': 'Logloss',
    # 'num_class': 2,
    'eval_metric': 'Logloss',
    'learning_rate': 0.01,
    # 'num_leaves': 40,
    # 'max_depth': 6,
    # 'subsample': .9,
    # 'colsample_bytree': .7,
    # 'reg_alpha': .01,
    # 'reg_lambda': .01,
    # 'min_split_gain': 0.01,
    # 'min_child_weight': 10,
    # 'silent':True,
    'verbose': False,
    # 'nthread':-1
}

for i,(tra_id,val_id) in enumerate(kfolds):
    print("{}/5 folds".format(i+1))
    tra_x, tra_y = x_train[tra_id], y_train[tra_id]
    val_x, val_y = x_train[val_id], y_train[val_id]

    cls_cat = CatBoostClassifier(**params,n_estimators=5000)

    cls_cat.fit(tra_x, tra_y, eval_set=(val_x,val_y), early_stopping_rounds=50,verbose=50)

    oof_cat[tra_id] = cls_cat.predict_proba(tra_x)
    predictions_cat += cls_cat.predict_proba(x_test)/folds.n_splits

    #print feature_importance
    features_weight = pd.DataFrame(data=cls_cat.feature_importances_,
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

oof_cat = oof_cat[:,1]
score = tpr_weight_funtion(train_data['label'].values, oof_cat)
print("oof_cat_score: {:<8.8f}".format(score))



oof_cat_pd = pd.DataFrame()
oof_cat_pd['id'] = train_data.id
oof_cat_pd['label'] = oof_cat
oof_cat_pd.to_csv(PATH + 'oof_catfeaturev1{:.4f}.csv'.format(score),index=False,header=True)

predictions_cat = predictions_cat[:,1]
cat_pred = pd.DataFrame()
cat_pred['id'] = test_data.id
cat_pred['label'] = predictions_cat
cat_pred.to_csv(PATH+'cat_scorefeaturev1{:.4f}.csv'.format(score),index=False,header=True)
