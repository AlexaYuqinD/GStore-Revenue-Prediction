
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.naive_bayes import MultinomialNB


def get_values(col_name, df):
    '''
    get all values of a column and restore it into a dictionary
    
        Args: 
            col_name: string variable represents the name of the column
            df: pandas data frame
        
        Returns:
            dic: dictionary of the possible values of a feature 
        
    '''
    
    list_col = df[col_name].tolist()
    col = set(list_col)
    dic = {}
    index = 0
    for item in col:
        dic[item] = index
        index += 1
    
    return dic

def mse(y, yhat):
    n = len(y)
    if len(yhat) != n:
        print("The size of yhat is {}.".format(yhat.shape))
    logy = y
    logy[y>0] = np.log(y[y>0])
    logyhat = yhat
    logyhat[yhat>0] = np.log(yhat[yhat>0])
    diff = logy - logyhat
    mse = (diff.T @ diff)/n
    
    return mse

def trans_Y(Y, dic):
    """
       Transform labeled Y into values
    """
    n = len(Y)
    Y_trans = np.zeros((n,))
    dic2 = {v: k for k, v in dic.items()}
    for i in range(n):
        yi = Y[i]
        Y_trans[i] = dic2[yi]
    
    return Y_trans


#def main():
if 1 == 1:
    df = pd.read_csv('dfTrain.csv',dtype={'fullVisitorId': 'str'})
    var_name = list(df.columns.values)

if 1 == 0:
    # delete rows that won't be used
    col_delete = ['Unnamed: 0','date','visitStartTime','geoNetwork.networkDomain'\
              ,'geoNetwork.metro','geoNetwork.city','geoNetwork.continent'\
              ,'geoNetwork.networkDomain','geoNetwork.region'\
              ,'geoNetwork.subContinent','trafficSource.adContent',\
              'trafficSource.adwordsClickInfo.adNetworkType',\
              'trafficSource.adwordsClickInfo.gclId',\
              'trafficSource.adwordsClickInfo.isVideoAd',\
              'trafficSource.adwordsClickInfo.page',\
              'trafficSource.adwordsClickInfo.slot',\
              'trafficSource.campaign','trafficSource.referralPath','fullVisitorId']
    data = df.drop(columns = col_delete)
    data.info()
    # Transform the discrete features into dictionary form
    col_trans = ['channelGrouping','device.browser','visitNumber','device.deviceCategory',\
                 'device.isMobile','device.operatingSystem', 'geoNetwork.country',\
                 'totals.bounces','totals.hits','totals.newVisits','totals.pageviews','totals.transactionRevenue',\
                 'trafficSource.isTrueDirect', 'trafficSource.keyword',\
                 'trafficSource.medium','trafficSource.source','hasRevenue']
    
    col_trans = ['visitStartTime','geoNetwork.networkDomain'\
              ,'geoNetwork.metro','geoNetwork.city','geoNetwork.continent'\
              ,'geoNetwork.networkDomain','geoNetwork.region'\
              ,'geoNetwork.subContinent','trafficSource.adContent',\
              'trafficSource.adwordsClickInfo.adNetworkType',\
              'trafficSource.adwordsClickInfo.gclId',\
              'trafficSource.adwordsClickInfo.isVideoAd',\
              'trafficSource.adwordsClickInfo.slot',\
              'trafficSource.campaign','trafficSource.referralPath',\
              'channelGrouping','device.browser','visitNumber','device.deviceCategory',\
                 'device.isMobile','device.operatingSystem', 'geoNetwork.country',\
                 'totals.bounces','totals.hits','totals.newVisits','totals.pageviews','totals.transactionRevenue',\
                 'trafficSource.isTrueDirect', 'trafficSource.keyword',\
                 'trafficSource.medium','trafficSource.source','hasRevenue']
   
    col_trans = ['trafficSource.adwordsClickInfo.adNetworkType','trafficSource.campaign','trafficSource.referralPath',\
              'channelGrouping','device.browser','visitNumber','device.deviceCategory',\
                 'device.isMobile','device.operatingSystem', 'geoNetwork.country',\
                 'totals.bounces','totals.hits','totals.newVisits','totals.pageviews','totals.transactionRevenue',\
                 'trafficSource.isTrueDirect', 'trafficSource.keyword',\
                 'trafficSource.medium','trafficSource.source','hasRevenue']
    # create a dic to store all the dics of possible values
    dic_trans = {}
    e = 0
    for item in col_trans:
        e += 1
        dic_trans[item] = get_values(item, df)
        print("Transforming data into discrete: {} of {} finished!".format(e,len(col_trans)))
    
    # get y variable as array
    y_data = data['totals.transactionRevenue'].values
    n = y_data.shape[0]
    k = len(col_trans)
    X = np.zeros((n,k-1))
    Y = np.zeros((n,))
    ro = 0
    for j in range(k):
        item = col_trans[j]
        if item != 'totals.transactionRevenue':
            dic = dic_trans[item]
            x = df[item].values
            for i in range(n):
                X[i,ro] = dic[x[i]]
            ro += 1
        else:
            dic = dic_trans[item]
            y = data[item].values
            for i in range(n):
                Y[i] = dic[y[i]]
            
        print(j,ro)
    
    
    weight = np.zeros((n,))
    weight[y_data>0] = 1
    weight[y_data==0] = 0.0001
    print("Start estimation.")
    clf = MultinomialNB()
    print("Fit the model")
    clf.fit(X, Y, weight)
    print("Predict")
    y_predict = clf.predict(X[10000:20000,:])
    Y_data = Y[10000:20000]
    accuracy = clf.score(X[10000:20000,:], Y[10000:20000])
    #y_predict = trans_Y(Y_labeled, dic_trans['totals.transactionRevenue'])
    print("MSE of predicted y is: {}.".format(mse(Y_data, y_predict)))
    print("Accuracy of the model is : {}.".format(accuracy))
    for i in range(200):
        print(Y_data[i],y_predict[i])
#    return y_predict
    




#y_predict = main()

