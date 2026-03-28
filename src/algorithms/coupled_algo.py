# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 21:13:06 2021

@author: prasa
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

def Distance(stars,BlackHole):
    distance = []
    for i in stars:
        dis = []
        for j in BlackHole:
            dis.append(hamming_distance(i,j))
        distance.append(np.argmin(dis))
    return distance

"Tournament Selection function"
def TournamentSelection(fitness,population,newPopulationSize):
    fitterSolutions = []
    while True:
        p1 = fitness[np.random.randint(newPopulationSize)]
        p2 = fitness[np.random.randint(newPopulationSize)]
        
        if p1 >= p2:
            fitterSolutions.append(p1)
        
        if len(fitterSolutions) == len(fitness):
            break
    
    fitterSolutionsIndex = []
    for i in np.arange(len(fitterSolutions)):
        fitterSolutionsIndex.append(np.where(fitterSolutions[i] == fitness)[0][0])
    
    newSolution = []
    for i in np.arange(len(fitterSolutionsIndex)):
        newSolution.append(population[fitterSolutionsIndex[i]])
    return fitterSolutions , newSolution

"Crossover function"
def Crossover(newSolution,bitSize,newPopulationSize):
    CrossOveredExamples = []
    while True:        
        splitJunction = np.random.randint(bitSize-1)
        p1 = newSolution[np.random.randint(newPopulationSize)]
        p2 = newSolution[np.random.randint(newPopulationSize)]
        
        if splitJunction >= bitSize:
            CrossOveredExamples.append(np.append(p1[:splitJunction],p2[splitJunction:]))
        else:  
            CrossOveredExamples.append(np.append(p1[splitJunction:],p2[:splitJunction]))
        
        CrossOveredExamples = CheckforNullPopulation(CrossOveredExamples)
        if len(CrossOveredExamples) == len(newSolution):
            break
    return CrossOveredExamples

"Mutation function"
def Mutation(CrossOveredExamples,newPopulationSize,bitSize,mutationProbability,newSolution):
    mutatePopulation = []
    while True:
        mutationExample = CrossOveredExamples[np.random.randint(newPopulationSize)]
        flip = []
        for i in np.arange(bitSize):
            if np.random.uniform(0,(mutationProbability+0.01)) < mutationProbability:
                flip.append(abs(mutationExample[i] - 1))
            else:
                flip.append(mutationExample[i])
        mutatePopulation.append(np.array(flip))
        mutatePopulation = CheckforNullPopulation(mutatePopulation)
        if len(mutatePopulation) == len(newSolution):
            break     
    return mutatePopulation

def BH_selection_function(population, X_train, X_test, y_train, y_test, bitSize):
            
    #for gen in range(maxiter+1):
    #print(gen)
    population = CheckforNullPopulation(population)
    BlackHole, BlackHole_fitness, stars, stars_fitness = Seprate(population, 
                                                                 numblackHoles, 
                                                                 X_train, X_test, y_train, y_test)
    #if len(stars) == 0:
    #    break
    
    #if gen > maxiter:
    #    break

    
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


def GA_feature_selection(population, X_train, X_test, y_train, y_test, bitSize):
    #itr = 0
    "Creating generations"
    #for gen in range(maxiter + 1):
    #itr += 1
    #print(itr)
    "Finding fitness ; here it is cv-accuracy using SVM"
    fitness = []
    for i in range(len(population)):
        X_train_sample = X_train.drop((np.where(population[i]==0)[0]),axis=1)
        X_test_sample = X_test.drop((np.where(population[i]==0)[0]),axis=1)
        
        fitness.append(CrossValidation(X_train_sample, X_test_sample, y_train, y_test))

    #if gen > maxiter:
    #    break

    "Tournament Selection"    
    fitterSolutions , newSolution = TournamentSelection(fitness,population, 
                                                        PopulationSize)
    
    "Crossover"
    CrossOveredExamples = Crossover(newSolution,bitSize, PopulationSize)
    
    "Mutation"
    mutatePopulation = Mutation(CrossOveredExamples, 
                                PopulationSize, 
                                bitSize, 
                                mutationProbability, 
                                newSolution)
        
    population = mutatePopulation.copy()
    
    return population

def Feature_selection_method(X, y):
    bitSize = X.shape[1]
    population = np.random.randint(low = 0,high = 2,size = (PopulationSize,bitSize))
    X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size = 0.3, random_state = 42)
    
    for gen in range(maxiter + 1):
        print(gen)
        
        if r <= np.random.uniform():
            population = BH_selection_function(population, X_train, X_test, y_train, y_test, bitSize)
        else:
            population_BH = BH_selection_function(population, X_train, X_test, y_train, y_test, bitSize)
            population_GA = GA_feature_selection(population, X_train, X_test, y_train, y_test, bitSize)
            
            population_mixed = population_BH + population_GA
            
            fitness = []
            for i in range(len(population_mixed)):	
                X_train_sample = X_train.drop((np.where(population_mixed[i]==0)[0]),axis=1)
                X_test_sample = X_test.drop((np.where(population_mixed[i]==0)[0]),axis=1)
                fitness.append(CrossValidation(X_train_sample, X_test_sample, y_train, y_test))
        
            population = np.array(population_mixed)[np.argsort(fitness)[::-1][:PopulationSize]]
    
    fitness_final = []
    for i in range(len(population)):	
        X_train_sample = X_train.drop((np.where(population[i]==0)[0]),axis=1)
        X_test_sample = X_test.drop((np.where(population[i]==0)[0]),axis=1)
        fitness_final.append(CrossValidation(X_train_sample, X_test_sample, y_train, y_test))
    
    subsets = population[np.argmax(fitness_final)]
    final_BH_subset = np.where(subsets==1)[0]
    final_subset_size = sum(subsets)
    final_BH_BlackHole_fitness = max(fitness)
    
    X_train_sample = X_train.iloc[:,final_BH_subset]
    X_test_sample = X_test.iloc[:,final_BH_subset]
            
    final_BH_accuracy = final_accuracy_func(X_train_sample, X_test_sample, y_train, y_test)
    
    return final_BH_subset, final_subset_size, final_BH_BlackHole_fitness, final_BH_accuracy

def preprocess(pre_data):
    X = pre_data.iloc[:,:-1]
    y = pre_data.iloc[:,-1]
    X.fillna(0,inplace=True)
    
    if y.dtype == 'object':
        encoder = LabelEncoder()
        y = pd.Series(encoder.fit_transform(y))
    
    scalar = MinMaxScaler()
    X = pd.DataFrame(scalar.fit_transform(X))
    return X, y

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


r = 0.4
PopulationSize = 10
maxiter = 5
numblackHoles = 1
crossoverProbability = 0.7
mutationProbability = 0.01

dfff = pd.DataFrame()
total_features = np.array([41, 30, 30, 2000, 34, 13, 33, 7129, 57, 33, 40, 13])
subset_size_list = np.round(total_features * 1.0).astype(int)

#from sklearn.datasets import load_breast_cancer
#bc = load_breast_cancer()
#X = pd.DataFrame(bc.data)
#y = pd.Series(bc.target)

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

for dataset_num in range(len(filename)):
    print(dataset_num)
    pre_data = data_read(dataset_num)
    X, y = preprocess(pre_data)
    final_subset, final_subset_size, final_fitness, final_accuracy = Feature_selection_method(X, y)
    dfff[datasetss[dataset_num]] = [final_subset, final_subset_size, final_fitness, final_accuracy]

output_df = dfff.T.reset_index()
output_df.columns = ['dataset','subset','subset size', 'fitness', 'accuracy']
output_df.to_csv('coupled_algo_GA_BH.csv'.format(0),encoding='utf-8')




#BH
#r < U~(0,1)
#GA

