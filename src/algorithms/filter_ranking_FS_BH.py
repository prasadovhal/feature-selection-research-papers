# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 22:28:15 2021

@author: prasa
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance

import warnings
warnings.filterwarnings('ignore')

def hamming_distance(s1, s2):
    return sum(el1 != el2 for el1, el2 in zip(s1, s2))

def fitness_func(y_actual, y_pred, selected_features):
    acc = accuracy_score(y_actual, y_pred)
    return (acc / (1 + 0.01 * selected_features))
    
def final_accuracy_func(X_train, X_test, y_train, y_test):
    
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
    return acc

def CrossValidation(X_train, X_test, y_train, y_test, w0, w1):
    model = RandomForestClassifier()
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    selected_features = X_train.shape[1]
    accs = fitness_func(y_test, y_pred, selected_features)
    
    matrix = train_corr_mat.loc[X_train.columns, X_train.columns]
    
    corr_wt = np.triu(matrix).sum() - np.trace(matrix)
    
    sc1 = score1[X_train.columns].sum()
    #sc2 = score2[X_train.columns].sum()
    #sc3 = score3[X_train.columns].sum()
    
    #score_to_add = (sc1 + sc2 + sc3) / 3
    
    #if np.random.uniform() < 0.5:
    #    return accs
    #else:
    return accs + w0 * sc1 - w1 * corr_wt

def CheckforNullPopulation(population):
    i = 0
    #newPop = []
    while(i < len(population)):
        if sum(population[i]) == 0:
            population[i] = np.random.randint(low = 0, high = 2,size = (len(population[i])))
            continue
        i += 1
    return population

def Seprate(population,numblackHoles,X_train, X_test, y_train, y_test, w0, w1):
    objective = max
    newPopulationSize = len(population)
    "Find Fitness"
    fitness = []
    for i in range(newPopulationSize):	
        X_train_sample = X_train.drop((np.where(population[i]==0)[0]),axis=1)
        X_test_sample = X_test.drop((np.where(population[i]==0)[0]),axis=1)
        fitness.append(CrossValidation(X_train_sample, X_test_sample, y_train, y_test, w0, w1))
     
    "finding blackhole and stars"
    blackHoleFitnessList = []
    blackHoleList = []
    for j in range(numblackHoles):
        BlackHole_fitness = objective(fitness)
        blackHoleFitnessList.append(BlackHole_fitness)
        BlackHole = population[np.where(fitness == BlackHole_fitness)[0][0]]
        blackHoleList.append(BlackHole)
        fitness.remove(BlackHole_fitness)
        stars_fitness = [x for x in fitness if x != BlackHole_fitness]
        
    stars = []
    for i in range(len(stars_fitness)):
        stars.append(population[np.where(fitness != BlackHole_fitness)[0][i]])
        
    return blackHoleList, blackHoleFitnessList, stars, stars_fitness

def Distance(stars,BlackHole):
    distance = []
    for i in stars:
        dis = []
        for j in BlackHole:
            dis.append(hamming_distance(i,j))
        distance.append(np.argmin(dis))
    return distance

def BH_selection_function(population, X_train, X_test, y_train, y_test, bitSize, w0, w1):
            
    for gen in range(maxiter):
        print(gen)
        population = CheckforNullPopulation(population)
        BlackHole, BlackHole_fitness, stars, stars_fitness = Seprate(population, numblackHoles, 
                                                                     X_train, X_test, y_train, y_test, w0, w1)
        print('BH subset size',sum(BlackHole[0]))
        if len(stars) == 0:
            break
        R_eventHorizon = BlackHole_fitness / sum(stars_fitness)
        
        distance = Distance(stars,BlackHole)
        
        for i in range(len(stars)):
            if abs(BlackHole_fitness[distance[i]] - stars_fitness[i]) <= R_eventHorizon[distance[i]]:
                stars[i] = np.random.randint(low = 0,high = 2,size = bitSize)
                
        for i in range(len(stars)):
            if sum(BlackHole[distance[i]] != stars[i]) == 0:
                continue
            else:
                num = int(0.25 * np.random.random() * sum(BlackHole[distance[i]] != stars[i]))
                tt = np.random.choice(np.where(BlackHole[distance[i]] != stars[i])[0],num)
                stars[i][tt] = abs(stars[i][tt] - 1)
                
        for j in range(numblackHoles):
            stars.append(BlackHole[j])
    
        population = stars.copy()
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
    
    for i in X.columns:
        if X[i].nunique() <= 1:
            X.drop(i,axis=1,inplace=True)
    
    X.columns = range(X.shape[1])
    return X,y

def data_read(datasetss):
    try:
        if datasetss != './datsets/HeartEW.csv':
            pre_data = pd.read_csv(datasetss,sep=',',header=None)
            pre_data.drop(0,axis=0,inplace=True)
        else:
            pre_data = pd.read_csv(datasetss,sep=' ',header=None)
            pre_data.drop(0,axis=0,inplace=True)
    except:
        pre_data = pd.read_excel(datasetss)
        pre_data.reset_index(drop=True,inplace=True)
        pre_data.drop(0,axis=0,inplace=True)
    
    return pre_data


PopulationSize = 20
maxiter = 30
numblackHoles = 1

weight_list = [['biodeg.csv', 0.1, 0.025],
             ['BreastEW.csv' , 0.01, 0.05],
             ['Cardiotocography.xls', 0.05, 0.1],
             ['colon.csv', 0.05, 0.01],
             ['derm.csv', 0.075, 0.05],
             ['HeartEW.csv', 0.05, 0.01],
             ['IonosphereEW.csv', 0.05, 0.01],
             ['leukemia.csv', 0.01, 0.025],
             ['spambase.csv', 0.025, 0.05],
             ['steel-plates-fault_csv.csv', 0.075, 0.01],
             ['WaveformEW.csv', 0.025, 0.075],
             ['WineEW.csv', 0.01, 0.01]]

#weight_list = [['CIDDS-001-external-week1.csv', 0.01, 0.001]]


seperator = []
final_dict = dict()


import itertools
def expandgrid(*itrs):
   product = list(itertools.product(*itrs))
   return {'var{}'.format(i+1):[x[i] for x in product] for i in range(len(itrs))}


for dataset_num in range(len(weight_list)):
    dfff = []
    filename, w0, w1 = weight_list[dataset_num]
    datasetss = "./datsets/"+ filename
    print(datasetss, w0, w1)
    pre_data = data_read(datasetss)        
    X, y = preprocess(pre_data)
    #df = pd.read_csv('./datsets/CIDDS-001-external-week1.csv')

    #y = df['class']
    #X = df.iloc[:,1:-4]
    
    #to_drop = ['Src IP Addr','Flows','Dst IP Addr','Tos','Flags']
    
    #X.drop(to_drop,axis=1,inplace=True)
    
    #X['Bytes'] = X['Bytes'].apply(lambda x: (x.split(' ')[-2]))
    #X['Bytes'] = X['Bytes'].apply(lambda x: float(x) if x != '' else 0)
    
    #encoder = LabelEncoder()
    #X['Proto'] = encoder.fit_transform(X['Proto'])
    #X.columns = range(len(X.columns))
    #encoder = LabelEncoder()
    #y = pd.Series(encoder.fit_transform(y))
    
    bitSize = X.shape[1]
    
    for runs in range(5):
        print('run :', runs)
        X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size = 0.3)
        
        model = RandomForestClassifier()
        model.fit(X_train,y_train)
        
        #results = permutation_importance(model, X_train, y_train, scoring='accuracy')
        score1 = model.feature_importances_
        #score2 = results.importances_mean
        #score3 = mutual_info_classif(X_train, y_train)
        
        train_corr_mat =  X_train.corr()
        
        population = np.random.randint(low = 0,high = 2,size = (PopulationSize, bitSize))
    
        population = BH_selection_function(population, X_train, X_test, y_train, y_test, bitSize, w0, w1)
        population = CheckforNullPopulation(population)
        BlackHole, BlackHole_fitness, stars, stars_fitness = Seprate(population,numblackHoles,
                                                                     X_train, X_test, y_train, y_test, w0, w1)
        
        final_BH_subset = np.where(BlackHole[0]==1)[0]
        final_subset_size = sum(BlackHole[0])
        final_BH_BlackHole_fitness = BlackHole_fitness[0]
        
        X_train_sample = X_train.iloc[:,final_BH_subset]
        X_test_sample = X_test.iloc[:,final_BH_subset]
        
        final_BH_accuracy = final_accuracy_func(X_train_sample, X_test_sample, y_train, y_test)
        
        matrix = train_corr_mat.loc[X_train_sample.columns, X_train_sample.columns]
        
        corr_wt = np.triu(matrix).sum() - np.trace(matrix)
        
        sc1 = score1[X_train_sample.columns].sum()
        #sc2 = score2[X_train_sample.columns].sum()
        #sc3 = score3[X_train_sample.columns].sum()
        #score_to_add  = (sc1 + sc2 + sc3) / 3
    
        dfff.append([final_BH_subset, final_subset_size, final_BH_BlackHole_fitness, final_BH_accuracy, corr_wt, sc1, w0, w1])
    
    final_dict[datasetss] = dfff
    

final_df = pd.DataFrame()
for k in final_dict.keys():
    print(k)
    a = pd.DataFrame(final_dict[k])
    a['dataset'] = k
    
    final_df = pd.concat([final_df, a],axis=0)
    
final_df.columns = ['subset','subset_size','fitness','accuracy', 'sum_of_corrr', 'sum_of_score', 'w0','w1','dataset_name']
final_df.to_csv('filter_ranking_BH_FS_no_prob.csv',encoding='utf-8',index=False)

