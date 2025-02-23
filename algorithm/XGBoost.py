import os
import re
import sys, getopt
import json
import pandas as pd
import numpy as np

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats import pointbiserialr, spearmanr

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

from collections import Counter

def read_clean_file(file):
    df = pd.read_csv(file, dtype={'订单首次配足款时间': str, '末次生效时间': str})
    string_columns = df.select_dtypes(include=['object', 'string']).columns  # 注意：在pandas较新版本中，'string'类型已经被引入

    # 然后对这些列应用str.strip()
    df[string_columns] = df[string_columns].apply(lambda x: x.str.strip() if x.dtype == 'object' else x)
    # 去除空数据
    df_cleaned = df[~(df[['强制生效时间']]
                      .applymap(lambda x: pd.isnull(x) or (isinstance(x, str) and x.strip() == ''))).any(axis=1)]

    # 再次去空数据
    df_cleaned = df_cleaned.dropna(subset=['订单首次配足款时间', '营销员代码'
        , '运输方式代码', '强制生效时间', '强制生效-首付款差值综合（客户）'])

    df_cleaned['date1'] = pd.to_datetime(df_cleaned['订单首次配足款时间'].astype(str).str[:12], format='%Y%m%d%H%M')
    df_cleaned['date2'] = pd.to_datetime(df_cleaned['强制生效时间'].astype(str).str[:12], format='%Y%m%d%H%M')
    df_cleaned.loc[:, '回款时间'] = (df_cleaned['date1'] - df_cleaned['date2']).dt.total_seconds() / 3600 / 24
    df_cleaned['渠道委托控货标记'] = df_cleaned['渠道委托控货标记'].fillna("N")
    df_cleaned['协议价格类型'] = df_cleaned['协议价格类型'].fillna("N")
    df_cleaned['合同是否厂商银（F-融资）'] = df_cleaned['合同是否厂商银（F-融资）'].fillna("N")
    df_cleaned['订单是否免息'] = df_cleaned['订单是否免息'].fillna("N")
    df_cleaned['收条是否有免息政策'] = df_cleaned['收条是否有免息政策'].fillna("N")
    df_cleaned['收条上做买方付息，做报批件，做过折现的'] = df_cleaned['收条上做买方付息，做报批件，做过折现的'].fillna("N")
    df_cleaned['备货转销售'] = df_cleaned['备货转销售'].fillna("N")
    df_cleaned['渠道强制生效&延付罚息白名单维护（Q01-延付罚息白名单，Q03-强制生效管控名单）'] = df_cleaned[
        '渠道强制生效&延付罚息白名单维护（Q01-延付罚息白名单，Q03-强制生效管控名单）'].fillna("N")

    df_cleaned.loc[df_cleaned['备货转销售'] == '(null)', '备货转销售'] = 'N'
    df_cleaned.loc[df_cleaned['备货转销售'] != 'N', '备货转销售'] = 'Y'
    df_cleaned = df_cleaned.drop_duplicates(subset=['订单号'])
    columns_to_drop = ['发布的期货价格政策', '客商信息中的经销商（客户分级）', '是否为协议用户（需方）需找客商'
        , '账套', '客户名称', '最终用户名称', '运输方式名称', '营销员姓名', '订单价格(不含税）', '订单价格(结算）'
        , 'date1', 'date2', '订单首次配足款时间', '强制生效时间', '订单子项号']
    # 使用drop方法丢弃指定的列
    df_cleaned = df_cleaned.drop(columns=columns_to_drop)
    df_cleaned = df_cleaned[df_cleaned['回款时间'] <= 90]
    # 重新调整index
    df_cleaned.reset_index(drop=True, inplace=True)
    return df_cleaned

def preprocess_data(df_cleaned):
    # 指定需要归一化的列
    # df_cleaned = pd.read_csv("./trainData/cleaned_day.csv")

    columns_to_normalize = ['订货量', '订单价格(含税）', '总重量', '总金额(含税)'
        , '本客户的平均回款天数:本客户的所有订单sum（订单首次配足款时间-合同强制生效时间）/订单数'
        , '本单该客户的平均晚回款比例：(本单客户的本次配足款时间-合同强制生效时间)/本客户的平均回款天数'
        , '本品种的平均回款天数:本品种的所有订单sum（订单首次配足款时间-合同强制生效时间）/订单数'
        , '本单该品种的平均晚回款比例：(本单品种的本次配足款时间-合同强制生效时间)/本品种的平均回款天数'
        , '总件数（按件计）', '本客户历史订货总量', '本客户历史订单总数（合约数）', '本品种历史订单总数（合约数）'
        , '强制生效-首付款差值综合（客户）', '强制生效-首付款差值综合（品种）']
    # 初始化MinMaxScaler
    scaler = StandardScaler()

    # 拟合并转换数据
    normalized_data = scaler.fit_transform(df_cleaned[columns_to_normalize])

    # 将归一化后的数据转换为DataFrame，并保留原始DataFrame的列名
    normalized_df = pd.DataFrame(normalized_data, columns=columns_to_normalize)

    # 将归一化后的列与原始DataFrame中的其他列合并
    standarded_df = pd.concat([df_cleaned.drop(columns=columns_to_normalize), normalized_df], axis=1)

    # 处理日期
    standarded_df['最早交货月'] = pd.to_datetime(standarded_df['最早交货月'].astype(str).str[:6], format='%Y%m')
    standarded_df['订单交货月'] = pd.to_datetime(standarded_df['订单交货月'].astype(str).str[:12], format='%Y%m%d%H%M')
    standarded_df['最早交货月_year'] = standarded_df['最早交货月'].dt.year
    standarded_df['最早交货月_month'] = standarded_df['最早交货月'].dt.month
    standarded_df['订单交货月_year'] = standarded_df['订单交货月'].dt.year
    standarded_df['订单交货月_month'] = standarded_df['订单交货月'].dt.month
    standarded_df['订单交货月_day'] = standarded_df['订单交货月'].dt.day

    # 指定需要转换为类别类型的列
    columns_to_category = ['客户代码（需方）', '最终用户', '部门代码', '保产合同/非保产合同', '渠道委托控货标记'
        , '制造单元', '股份合同性质', '品种代码', '订单是否变更过(价格变更)', '订单是否变更过(重量变更)'
        , '订单是否变更过 (交货信息变更(订单是否变更物流模板))', '营销员代码', '协议价格类型'
        , '渠道强制生效&延付罚息白名单维护（Q01-延付罚息白名单，Q03-强制生效管控名单）'
        , '合同是否厂商银（F-融资）', '交货方式', '运输方式代码', '备货转销售'
        , '订单是否免息', '收条是否有免息政策', '收条上做买方付息，做报批件，做过折现的'
        , '最早交货月_year', '最早交货月_month', '订单交货月_year', '订单交货月_month', '订单交货月_day']

    standarded_df[columns_to_category] = standarded_df[columns_to_category].astype('category')
    df_model = standarded_df.drop(columns=['最早交货月', '订单交货月'])

    # 计算每个条目的出现次数
    item_counts = df_model['客户代码（需方）'].value_counts()

    # 筛选出现次数不少于5次的条目
    items_to_keep = item_counts[item_counts >= 5].index

    # 应用筛选条件，只保留column_name中条目出现次数不少于5次的行
    df_model = df_model[df_model['客户代码（需方）'].isin(items_to_keep)]

    return df_model

def train_model(df_model):
    X = df_model.drop(columns=['回款时间', '订单号'])
    y = df_model[['回款时间', '订单号']]
    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    # # 将数据转换为DMatrix格式，这是XGBoost的输入格式
    # dtrain = xgb.DMatrix(X_train, label=y_train['回款时间'], enable_categorical=True)
    # dval = xgb.DMatrix(X_val, label=y_val['回款时间'], enable_categorical=True)
    #
    # # 设置XGBoost参数（回归问题）
    # params = {
    #     'booster': 'gbtree',
    #     'objective': 'reg:squarederror',  # 回归问题使用平方误差损失
    #     'eta': 0.1,  # 学习率
    #     'max_depth': 6,  # 树的最大深度
    #     'eval_metric': 'rmse'  # 评估指标使用均方根误差
    # }
    #
    # watchlist = [(dtrain, 'train'), (dval, 'val')]
    #
    # # 训练模型
    # num_round = 800  # 迭代次数
    # bst = xgb.train(params, dtrain, num_round, evals=watchlist, early_stopping_rounds=10)
    #
    # bst.save_model('./model/xgboost_model1017_day.json')
    # # 预测
    # y_pred = bst.predict(dval)

    other_params = {'learning_rate': 0.05, 'n_estimators': 800, 'max_depth': 5, 'min_child_weight': 10, 'seed': 0
        , 'subsample': 0.8, 'colsample_bytree': 0.66, 'colsample_bylevel': 0.66, 'colsample_bynode': 0.66
        , 'gamma': 2620, 'reg_alpha': 0, 'reg_lambda': 1}

    model = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse'
                             , enable_categorical=True, **other_params)
    model.fit(X_train, y_train['回款时间'])

    # 预测
    y_pred = model.predict(X_val)

    # 计算均方根误差（RMSE）
    rmse = np.sqrt(mean_squared_error(y_val['回款时间'], y_pred))

    model.save_model('./model/xgboost_model1026_day.json')

    # 保存结果
    output = pd.DataFrame(y_val)
    output['预测回款时间'] = y_pred
    output.to_csv("./trainData/训练_XGBoost回款时间_90天范围内_1026.csv", index=False)

    return rmse

def predict_model(df_model):
    train_data = pd.read_csv("./trainData/train_data_latest.csv")
    customer_list = list(df_model['客户代码（需方）']) + list(train_data['客户代码（需方）'])
    # 使用Counter计算每个元素的出现频次
    count = Counter(customer_list)
    # 筛选出出现次数大于等于5的元素
    result = [item for item, freq in count.items() if freq >= 6]
    # 应用筛选条件，只保留column_name中条目出现次数不少于5次的行
    df_model = df_model[df_model['客户代码（需方）'].isin(result)]

    X = df_model.drop(columns=['回款时间', '订单号'])
    y = df_model[['回款时间', '订单号']]

    model = xgb.Booster()
    model.load_model('./model/xgboost_model1026_day.json')

    dtest = xgb.DMatrix(X, label=y['回款时间'], enable_categorical=True)
    y_pred = model.predict(dtest)

    rmse = np.sqrt(mean_squared_error(y['回款时间'], y_pred))

    # 保存结果
    output = pd.DataFrame(y)
    output['预测回款时间'] = y_pred
    output.to_csv("./testData/预测_XGBoost回款时间_90天范围内_1026.csv", index=False)

    return rmse

def build_XGBoost():
    df_cleaned = read_clean_file("./trainData/回款时间原数据1014.csv")
    df_model = preprocess_data(df_cleaned)
    df_model.to_csv("./trainData/train_data_latest.csv", index=False)
    rmse = train_model(df_model)
    return rmse

def test_XGBoost():
    df_cleaned = read_clean_file("./testData/回款时间数据1024.csv")
    df_model = preprocess_data(df_cleaned)
    rmse = predict_model(df_model)
    return rmse


# if __name__ == '__main__':
    # args = sys.argv[1:]
    # try:
    #     opts, args = getopt.getopt(args, "h-m:", ["model="])
    # except getopt.GetoptError:
    #     print('generate_index.py -m <model>')
    #     sys.exit(2)
    #
    # model = ""
    # if opts:
    #     model = opts[0][1]
