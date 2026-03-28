# -*- coding: utf-8 -*-
"""
Created on Sat Jul 17 23:12:28 2021

@author: prasa
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, make_scorer, f1_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')


def hamming_distance(s1, s2):
    return sum(el1 != el2 for el1, el2 in zip(s1, s2))

def fitness_func(y_actual, y_pred, selected_features, f1_score_flag=False):
    acc = accuracy_score(y_actual, y_pred)/ (1 + 1 * selected_features)
    f1_sc = f1_score(y_actual, y_pred, average='micro') / (1 + 1 * selected_features)
    return (f1_sc if f1_score_flag else acc)
    
def final_accuracy_func(X_train, X_test, y_train, y_test, f1_score_flag=False):
    
    n_estimators = [50, 100, 200, 300, 500]
    max_features = [0.1, 0.2, 0.4, 0.6, 0.8]
    max_features.append('auto')
    max_depth = [int(x) for x in np.linspace(10, 100, num = 10)]
    max_depth.append(None)

    "Grid"
    random_grid = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth}
    
    clf = RandomForestClassifier()
    scorer = make_scorer(accuracy_score)
    model = RandomizedSearchCV(estimator = clf, 
                               param_distributions = random_grid, 
                               n_iter = 100, 
                               cv = 5, 
                               verbose=1, 
                               random_state=42, 
                               n_jobs = -1,
                               scoring = scorer)

    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1_sc = f1_score(y_test, y_pred, average='micro')
    return f1_sc if f1_score_flag else acc 

def CrossValidation(X_train, X_test, y_train, y_test, f1_score_flag=False):
#    print('f1_score_flag: ', f1_score_flag)
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    selected_features = X_train.shape[1]
    accs = fitness_func(y_test, y_pred, selected_features, f1_score_flag)
    return accs

def CheckforNullPopulation(population):
    i = 0
    #newPop = []
    while(i < len(population)):
        if sum(population[i]) == 0:
            population[i] = np.random.randint(low = 0,high = 2,size = (len(population[i])))
            if sum(population[i]) == 0:
                population[i] = np.random.randint(low = 0,high = 2,size = (len(population[i])))
        i += 1
    return population

def preprocess(pre_data):
    X = pre_data.iloc[:,:-1]
    y = pre_data.iloc[:,-1]
    X.fillna(0,inplace=True)
    
    if y.dtype == 'object':
        encoder = LabelEncoder()
        y = pd.Series(encoder.fit_transform(y))
    
    scalar = MinMaxScaler()
    X = pd.DataFrame(scalar.fit_transform(X))
    return X,y

def data_read(dataset_name):
    try:
        if dataset_name != './datsets/HeartEW.csv':
            pre_data = pd.read_csv(dataset_name,sep=',',header=None)
            pre_data.drop(0,axis=0,inplace=True)
        else:
            pre_data = pd.read_csv(dataset_name,sep=' ',header=None)
            pre_data.drop(0,axis=0,inplace=True)
    except:
        pre_data =  pd.read_excel(dataset_name)
        pre_data.reset_index(drop=True,inplace=True)
        pre_data.drop(0,axis=0,inplace=True)
    
    return pre_data

def Separte(population, PopulationSize, f1_score_flag=False):
    fitness = []
    for pop in range(PopulationSize):
        X_train_sample = X_train.iloc[:,population[pop]==1].copy()
        X_test_sample = X_test.iloc[:,population[pop]==1].copy()
#        print(X_train_sample.shape)
        fitness.append(CrossValidation(X_train_sample, X_test_sample, y_train, y_test, f1_score_flag))
    
    
    BH_fitness = max(fitness)
    BH = population[np.argmax(fitness)]
    
    stars = np.delete(population,np.argmax(fitness),axis=0)
    stars_fitness = np.delete(fitness,np.argmax(fitness),axis=0)
    
    return BH, BH_fitness, stars, stars_fitness

def BH_feature_selection(population, PopulationSize, bitSize, f1_score_flag=False):
    
    for gen in range(maxiter + 1):
        print(gen)
        population = CheckforNullPopulation(population)
        BH, BH_fitness, stars, stars_fitness = Separte(population, PopulationSize, f1_score_flag)
        print('Select Features Size :', sum(BH))
        if gen > maxiter:
            break
        
        event_horizon = BH_fitness / sum(stars_fitness)
            
        for moving in range(len(stars)):
            if (BH_fitness - stars_fitness[moving]) <= event_horizon:
                stars[moving] = np.random.randint(low = 0,high = 2,size = bitSize)
            else:
                index_to_replace = np.random.choice(np.where(BH != stars[moving])[0],
                                                    int(0.25 * sum(BH != stars[moving])),
                                                    replace=False)
            
                stars[moving][index_to_replace] = abs(stars[moving][index_to_replace] - 1)
        
        population = np.append(stars,[BH],axis=0)
        
    return BH, BH_fitness, stars, stars_fitness

PopulationSize = 30
maxiter = 20
dfff = []

filename = [ 
#            'biodeg.csv',
#             'BreastEW.csv',
#             'Cardiotocography.xls',
#             'colon.csv',
#             'derm.csv',
#             'HeartEW.csv',
#             'IonosphereEW.csv',
#             'leukemia.csv',
#             'spambase.csv',
#             'steel-plates-fault_csv.csv',
#             'WaveformEW.csv',
#             'Network_intrusion_detection_clean.csv'
             'KDDTrain_firefly_paper_2.csv'
             ]

#data_set = filename[0]
for data_set in filename:
    print(data_set)
    dfff = []
    for runs in range(5):
#        pre_data = data_read('./datsets/'+data_set)
        data = pd.read_csv('./datsets/KDDTrain_firefly_paper_2.csv', header = None)
        data_test = pd.read_csv('./datsets/KDDTest_firefly_paper_2.csv', header = None)
        data.info()
        data_test.info()
        
        pre_data_1 = pd.concat([data, data_test], axis=0)
        pre_data_1.reset_index(inplace=True, drop=True)
        
        le1 = LabelEncoder()
        le1.fit(pre_data_1[1])
      
        le3 = LabelEncoder()
        le3.fit(pre_data_1[2])
        
        le4 = LabelEncoder()
        le4.fit(pre_data_1[3])
        
        data[1] = le1.transform(data[1])
        data_test[1] = le1.transform(data_test[1])
        


        data[2] = le3.transform(data[2])
        data_test[2] = le3.transform(data_test[2])
        
        data[3] = le4.transform(data[3])
        data_test[3] = le4.transform(data_test[3])
        
        
        grp = {i[0]:i[1] for i in data.groupby(by=[43])}
        grp_test = {i[0]:i[1] for i in data_test.groupby(by=[43])}
        grp['dos'].iloc[:,-3].value_counts()
        
        len(grp_test)
        print(grp.keys())
        print(grp_test.keys())
        
        grp['normal'].reset_index(inplace=True)
        grp_test['normal'].reset_index(inplace=True)
        grp_normal_train = grp.pop('normal')
        grp_normal_test = grp_test.pop('normal')
#        grp['dos'].reset_index(inplace=True)
#        grp['probe'].reset_index(inplace=True)
#        grp['r2l'].reset_index(inplace=True)
#        grp['u2r'].reset_index(inplace=True)
        print(grp.keys())
        print(grp_test.keys())
        for grp_id, grp_jd in zip(grp.keys(),grp_test.keys()):
            print('grp_train_id: ', grp_id)
            print('grp_test_id: ', grp_jd)
            
            grp[grp_id].reset_index(inplace=True)
            grp_test[grp_jd].reset_index(inplace=True)
            
            data_grp_train = pd.concat([grp_normal_train, grp[grp_id]])
            data_grp_train.reset_index(inplace=True)
            
            data_grp_test = pd.concat([grp_normal_test, grp_test[grp_jd]])        
            data_grp_test.reset_index(inplace=True)
                
                
    #        data_test.drop(columns=['Unnamed: 0'], inplace=True)
    
            pre_data = pd.concat([data_grp_train, data_grp_test], axis=0)
            pre_data.reset_index(inplace=True, drop=True)
            
            le2 = LabelEncoder()
            le2.fit(pre_data[41])
            data_grp_train[41] = le2.transform(data_grp_train[41])
            data_grp_test[41] = le2.transform(data_grp_test[41])
            data_grp_train.drop(columns=[42, 43], inplace = True)
            data_grp_test.drop(columns=[42, 43], inplace = True)
#            rf = RandomForestClassifier()
            
#            X_train, y_train = preprocess(data_grp_train)
#            X_test, y_test = preprocess(data_grp_test)
            
            pre_data = pd.concat([data_grp_train, data_grp_test], axis=0)
            
            X, y = preprocess(pre_data)
            
            X_train, y_train = X.iloc[:data_grp_train.shape[0],:], y.iloc[:data_grp_train.shape[0]]
            X_test, y_test = X.iloc[data_grp_test.shape[0]:,:], y.iloc[data_grp_test.shape[0]:]
    #        bitSize = X.shape[1]
            bitSize = X_train.shape[1]
            
    #        X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size = 0.3, random_state = 42)
            
            population = np.random.randint(low = 0,high = 2,size = (PopulationSize, bitSize))
            population = CheckforNullPopulation(population)
            
            BH, BH_fitness, stars, stars_fitness = BH_feature_selection(population, PopulationSize, bitSize, f1_score_flag=True)
            
            X_train_sample = X_train.iloc[:,BH==1].copy()
            X_test_sample = X_test.iloc[:,BH==1].copy()
                
            final_BH_accuracy = final_accuracy_func(X_train_sample, X_test_sample, y_train, y_test, f1_score_flag=True)
        
            dfff.append([X_train_sample.columns, sum(BH), BH_fitness, final_BH_accuracy, grp_id])
        
        dataset_df = pd.DataFrame(dfff)
        dataset_df.columns = ['subset','subset_size','fitness','f1_micro', 'attack_type']
        dataset_df['dataset'] = data_set
        dataset_df.to_csv('./Results/BH/results_lambda{}_'.format(1)+ data_set, encoding='utf-8', index=False)