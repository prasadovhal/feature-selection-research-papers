#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 30 12:26:04 2018

@author: prasad
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

def hamming_distance(s1, s2):
    return sum(el1 != el2 for el1, el2 in zip(s1, s2))

def fitness_func(y_actual, y_pred, selected_features):
    acc = accuracy_score(y_actual, y_pred)
    return (acc / (1 + 0.01 * selected_features))
    
def final_accuracy_func(X_train, X_test, y_train, y_test):
    model = SVC()
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return acc

#def CrossValidation(X_train, X_test, y_train, y_test):
#    model = RandomForestClassifier()
#    model.fit(X_train,y_train)
#    y_pred = model.predict(X_test)
#    selected_features = X_train.shape[1]
#    accs = fitness_func(y_test, y_pred, selected_features)
#    
#    train_corr_mat =  X_train.corr()
#    corr_wt = np.triu(train_corr_mat).sum() - np.trace(train_corr_mat)
#    
#    return accs - 0.001 * corr_wt

def CrossValidation(X_train, X_test, y_train, y_test):
    model = SVC()
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    selected_features = X_train.shape[1]
    accs = fitness_func(y_test, y_pred, selected_features)
    return accs

def CheckforNullPopulation(population):
    i = 0
    newPop = []
    while(i < len(population)):
        if sum(population[i]) != 0:
            newPop.append(population[i])
        i += 1
    return newPop

def Seprate(population,numblackHoles,X_train, X_test, y_train, y_test):
    objective = max
    newPopulationSize = len(population)
    "Find Fitness"
    fitness = []
    for i in range(newPopulationSize):	
        X_train_sample = X_train.drop((np.where(population[i]==0)[0]),axis=1)
        X_test_sample = X_test.drop((np.where(population[i]==0)[0]),axis=1)
        fitness.append(CrossValidation(X_train_sample, X_test_sample, y_train, y_test))
     
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

#def Mutation(stars,bit,eps):
#    mutatePopulation = []
#    for j in np.arange(len(stars)):
#        mutationExample = stars[j]
#        flip = []
#        for i in np.arange(bit):
#            if np.random.uniform(0,(eps+0.01)) < eps:
#                flip.append(abs(mutationExample[i] - 1))
#            else:
#                flip.append(mutationExample[i])
#        mutatePopulation.append(np.array(flip))
#    return mutatePopulation

def Distance(stars,BlackHole):
    distance = []
    for i in stars:
        dis = []
        for j in BlackHole:
            dis.append(hamming_distance(i,j))
        distance.append(np.argmin(dis))
    return distance

PopulationSize = 20
maxiter = 30
eps = 0.001
numblackHoles = 1
C = 1.5
T = 1.05
dfff = pd.DataFrame()
total_features = np.array([41, 30, 30, 2000, 34, 13, 33, 7129, 57, 33, 40, 13])
subset_size_list = np.round(total_features * 0.2).astype(int)

filename = ['biodeg.csv',
             'BreastEW.csv',
             'Cardiotocography.xls',
             'colon.csv',
             'derm.csv',
             'HeartEW.csv',
             'IonosphereEW.csv',
             'leukemia.csv',
             'spambase.csv',
             'steel-plates-fault_csv.csv',
             'WaveformEW.csv',
             'WineEW.csv']

seperator = []
datasetss = ["./datsets/"+fname_i for fname_i in filename]

def BH_selection_function(population, X_train, X_test, y_train, y_test, bitSize):
            
    for gen in range(maxiter):
        print(gen)
        population = CheckforNullPopulation(population)
        BlackHole, BlackHole_fitness, stars, stars_fitness = Seprate(population, numblackHoles, 
                                                                     X_train, X_test, y_train, y_test)

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
    
        #muateatedExamples = Mutation(stars,bitSize,eps)
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
    return X,y

def data_read(dataset_num):
    try:
        if datasetss[dataset_num] != './datsets/HeartEW.csv':
            pre_data = pd.read_csv(datasetss[dataset_num],sep=',',header=None)
            pre_data.drop(0,axis=0,inplace=True)
        else:
            pre_data = pd.read_csv(datasetss[dataset_num],sep=' ',header=None)
            pre_data.drop(0,axis=0,inplace=True)
    except:
        pre_data =  pd.read_excel(datasetss[dataset_num])
        pre_data.reset_index(drop=True,inplace=True)
        pre_data.drop(0,axis=0,inplace=True)
    
    return pre_data

for run in range(5):
    for dataset_num in range(len(filename)):
        print(datasetss[dataset_num])
        subsetSize = subset_size_list[dataset_num]
        pre_data = data_read(dataset_num)        
        X, y = preprocess(pre_data)
        bitSize = X.shape[1]
        X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size = 0.3, random_state = 42)
        
        population = []
        
        for i in np.arange(PopulationSize):
            population.append(np.random.randint(low = 0,high = 2,size = bitSize))

        population = BH_selection_function(population, X_train, X_test, y_train, y_test, bitSize)
        population = CheckforNullPopulation(population)
        BlackHole, BlackHole_fitness, stars, stars_fitness = Seprate(population,numblackHoles,
                                                                     X_train, X_test, y_train, y_test)
        
        final_BH_subset = np.where(BlackHole[0]==1)[0]
        final_subset_size = sum(BlackHole[0])
        final_BH_BlackHole_fitness = BlackHole_fitness[0]
        
        X_train_sample = X_train.iloc[:,final_BH_subset]
        X_test_sample = X_test.iloc[:,final_BH_subset]
        
        final_BH_accuracy = final_accuracy_func(X_train_sample, X_test_sample, y_train, y_test)
        dfff[datasetss[dataset_num]] = [final_BH_subset, final_subset_size, final_BH_BlackHole_fitness, final_BH_accuracy]
    
    output_df = dfff.T.reset_index()
    output_df.columns = ['dataset','subset','subset size', 'fitness', 'accuracy']
    output_df.to_csv('BBH_results_{}_RF.csv'.format(run),encoding='utf-8')
