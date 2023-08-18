import pandas as pd
import statistics
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn import metrics
import pickle
import joblib

def main():
    main_df = pd.read_csv("Train_keystroke.csv") # original dataset from csv file
    print(main_df.iloc[0,2])

    df = pd.DataFrame({'user': pd.Series(dtype='int'), # creating empty dataframe for new dataset
                   'mean_HT': pd.Series(dtype='float'),
                   'mean_PPT': pd.Series(dtype='float'),
                   'mean_RRT': pd.Series(dtype='float'),
                   'mean_RPT': pd.Series(dtype='float'),
                   'sd_HT': pd.Series(dtype='float'),
                   'sd_PPT': pd.Series(dtype='float'),
                   'sd_RRT': pd.Series(dtype='float'),
                   'sd_RPT': pd.Series(dtype='float')})
    print(df)

    for i in range(len(main_df)): # filling new dataframe as training set with mean and stdev values
        j = 0
        HT_values_list = []
        PPT_values_list = []
        RRT_values_list = []
        RPT_values_list = []
        while j < len(main_df.columns) - 3:
            HT_values_list.append(main_df.iloc[i,j+1] - main_df.iloc[i,j])
            PPT_values_list.append(main_df.iloc[i,j+2] - main_df.iloc[i,j])
            RRT_values_list.append(main_df.iloc[i,j+3] - main_df.iloc[i,j+1])
            RPT_values_list.append(main_df.iloc[i,j+2] - main_df.iloc[i,j+1])
 
            if j == len(main_df.columns) - 4:
                HT_values_list.append(main_df.iloc[i,j+3] - main_df.iloc[i,j+2])
            
            j += 2
    
        mean_HT = statistics.mean(HT_values_list)
        mean_PPT = statistics.mean(PPT_values_list)
        mean_RRT = statistics.mean(RRT_values_list)
        mean_RPT = statistics.mean(RPT_values_list)

        sd_HT = statistics.stdev(HT_values_list)
        sd_PPT = statistics.stdev(PPT_values_list)
        sd_RRT = statistics.stdev(RRT_values_list)
        sd_RPT = statistics.stdev(RPT_values_list)

        df = df.append({'user':main_df.iloc[i,0],
                        'mean_HT':mean_HT,
                        'mean_PPT':mean_PPT,
                        'mean_RRT':mean_RRT,
                        'mean_RPT':mean_RPT,
                        'sd_HT':sd_HT,
                        'sd_PPT':sd_PPT,
                        'sd_RRT':sd_RRT,
                        'sd_RPT':sd_RPT},ignore_index=True)

    print(df)
    
    # shuffling or sampling dataset
    
    df = df.sample(frac=1)

    #df['user'] = df['user'] / max(df['user'])
    '''df['mean_HT'] = df['mean_HT'] / max(df['mean_HT']) 
    df['mean_PPT'] = df['mean_PPT'] / max(df['mean_PPT']) 
    df['mean_RRT'] = df['mean_RRT'] / max(df['mean_RRT']) 
    df['mean_RPT'] = df['mean_RPT'] / max(df['mean_RPT']) 
    df['sd_HT'] = df['sd_HT'] / max(df['sd_HT']) 
    df['sd_PPT'] = df['sd_PPT'] / max(df['sd_PPT']) 
    df['sd_RRT'] = df['sd_RRT'] / max(df['sd_RRT']) 
    df['sd_RPT'] = df['sd_RPT'] / max(df['sd_RPT']) 
    '''
    # splitting X and y from training set 
    
    X_train = df.drop(["user"], axis=1)
    y_train = df["user"]
    #X_train, X_test, y_train, y_test = train_test_split(df.drop(["user"], axis=1), df["user"], train_size=1.00, test_size=0.00, random_state=0)
    #quit()
    # creating and training svm model

    clf = svm.SVC(kernel='linear', C=1, decision_function_shape='ovo')
    clf.fit(X_train, y_train)
    y_pred_train = clf.predict(X_train)

    print(y_pred_train)
    print("Accuracy:",metrics.accuracy_score(y_train, y_pred_train))

    # saving the svm model

    joblib.dump(clf, open('SVM.pkl', 'wb'))
    
    # loading the svm model

    clf = joblib.load(open('SVM.pkl', 'rb'))
    y_pred_test = clf.predict(X_train)
    print(y_pred_test)
    print("Accuracy:",metrics.accuracy_score(y_train, y_pred_test))            
            
    # creating and training RF model

    clf_rf = RandomForestClassifier(n_estimators=100)
    clf_rf.fit(X_train,y_train)
    y_pred_train = clf_rf.predict(X_train)

    print(y_pred_train)
    print("Accuracy:",metrics.accuracy_score(y_train, y_pred_train))      

    # saving the RF model

    joblib.dump(clf_rf, open('RF.pkl', 'wb'))

    # loading the RF model

    clf_rf = joblib.load(open('RF.pkl', 'rb'))
    y_pred_test = clf_rf.predict(X_train)
    print(y_pred_test)
    print("Accuracy:",metrics.accuracy_score(y_train, y_pred_test))

    # creating and training XGBoost model

    clf_xg = XGBClassifier()
    for i in range(len(y_train)): y_train.iloc[i] -= 1
    clf_xg.fit(X_train,y_train)
    y_pred_train = clf_xg.predict(X_train)
    for i in range(len(y_pred_train)): y_pred_train[i] += 1
    for i in range(len(y_train)): y_train.iloc[i] += 1    

    print(y_pred_train)
    print("Accuracy:",metrics.accuracy_score(y_train, y_pred_train))      

    # saving the XGBoost model

    joblib.dump(clf_xg, open('XGB.pkl', 'wb'))

    # loading the XGBoost model

    clf_xg = joblib.load(open('XGB.pkl', 'rb'))
    y_pred_test = clf_xg.predict(X_train)
    for i in range(len(y_pred_test)): y_pred_test[i] += 1
    print(y_pred_test)
    print("Accuracy:",metrics.accuracy_score(y_train, y_pred_test))


if __name__ == "__main__":
    main()
