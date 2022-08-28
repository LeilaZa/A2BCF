#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 13:47:37 2021

@author: Leila Zahedi
"""

import numpy as np
import random
import time
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.svm import SVC, SVR
from sklearn.cluster import AgglomerativeClustering
import math


class ABCFSO:
    Model = None 
    CV = None
    folds = None
    FOOD_NUMBER = None 
    CLUSTERS = None 
    Limit = None
    iter = None
    Min_Acc = None
    Stop_Acc = None 
    MaxEval = None
    cat = None
    def __init__(self, model='RF', cv=True, n_folds=3, n_food=500, 
               limit=3, iterations=5, stop_accuracy=0.9,
               max_evaluation=5000, cat="classification", min_accuracy=None):
        
        """
        Parameters
        ----------
        model : TYPE, optional
            DESCRIPTION. The default is 'RF'.
        cv : TYPE, optional
            DESCRIPTION. The default is True.
        n_folds : TYPE, optional
            DESCRIPTION. The default is 3.
        n_food : TYPE, optional
            DESCRIPTION. The default is 1000.
        limit : TYPE, optional
            DESCRIPTION. The default is 3.
        iterations : TYPE, optional
            DESCRIPTION. The default is 5.
        min_accuracy : TYPE, optional
            DESCRIPTION. The default is 0.5.
        stop_accuracy : TYPE, optional
            DESCRIPTION. The default is 0.9.
        max_evaluation : TYPE, optional
            DESCRIPTION. The default is 5000.

        Returns
        -------
        None.

        """
        self.Model = model
        self.CV = cv
        self.folds = n_folds
        self.FOOD_NUMBER = n_food
        self.CLUSTERS = int(self.FOOD_NUMBER/100)
        self.Limit = limit
        self.iter = iterations
        self.Min_Acc = min_accuracy
        self.Stop_Acc = stop_accuracy
        self.MaxEval = max_evaluation
        self.cat=cat

    def fso(self, X, y):
        """
        Parameters
        ----------
        X : TYPE
            Features
        y : TYPE
            Target variable

        Returns
        -------
        TYPE
            Optimal set of features
        """
        DIMENSION = len(X.columns)
        self.EvalNum = 0
        self.solution = np.zeros(DIMENSION)
        self.f = np.ones(self.CLUSTERS)
        self.fitness = np.ones(self.CLUSTERS) * np.iinfo(int).max
        self.trial = np.zeros(self.CLUSTERS)
        self.globalOpt = -math.inf
        self.globalFeatures = [0 for x in range(DIMENSION)]
        self.globalOpts=list()
        self.rnd = 1
        self.p_foods = np.zeros((self.FOOD_NUMBER, DIMENSION))
        self.foods = np.zeros((self.CLUSTERS, DIMENSION))
        self.foods_OBL = np.zeros((2, DIMENSION))
        #food_centroid = np.zeros((self.CLUSTERS))
        self.X_train = None
        self.X_test = None
        self.start_time = time.time() 
        
        # #------------------------------------------------
        # #***Split data***
        # #------------------------------------------------
        
        if not self.CV:
            self.X_train, self.X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=100)
        else:
            cv= KFold(n_splits = self.folds, random_state=100, shuffle=True)
            
        #------------------------------------------------
        #***Update variables according to k-means filtering***
        #------------------------------------------------
        def update_variables(index_to_drop):
            """
            Updating variables sizes after clustering
            Filtering the food sources with rich solutions changes the size of food
            sources, objective function, fitness and trial matrixes.
            
            Args: the indexs of the food source that is not rich enough
            """
            self.CLUSTERS -= 1
            self.f = np.delete(self.f, index_to_drop, axis=0) #obj func
            self.fitness = np.delete(self.fitness, index_to_drop, axis=0) #fitness
            self.trial= np.delete(self.trial, index_to_drop, axis=0) # trial
            self.foods = np.delete(self.foods, index_to_drop, axis=0) # foods
        
        #------------------------------------------------
        #***filter population based on the defined threshold Min_Acc***
        #------------------------------------------------
        def filter_pop():
            """
            Filtering the new population (centroids) with rich food sources 
            
            Args: the minimum accuracy threshold for eliminating food sources
            """
            temp_f = self.f 
            i = self.CLUSTERS-1
            if self.Min_Acc is not None:
                while i >= 0:
                    if temp_f[i] < self.Min_Acc:
                       update_variables(i)
                    i -= 1
            print("New population after filtering:\n" + str(self.f)) 

        
        #------------------------------------------------
        #***Objective Function ***
        #------------------------------------------------
        def calculate_function(sol):
            """
            Calculate the objective function for each solution in the population 
            
            Args: solution
            
            Returns: Accuracy of the solution
            """
            self.EvalNum += 1
        
            list_index = []
            for i in range(len(sol)):
                if(sol[i] == 1):
                    list_index.append(i)
        
            if self.cat=="classification":
                if self.Model == "RF":
                    print("Random Forest")
                    model= RandomForestClassifier (random_state=100)
                elif self.Model == "SVM":
                    print("Support Vector Machine")
                    model= SVC(gamma='scale',random_state=100)
                elif self.Model == "XGB":
                    print("XGBoost")
                    model = XGBClassifier(objective='binary:logistic', use_label_encoder=False,eval_metric='mlogloss', random_state=100)
                if not self.CV:
                    X_train_FS = self.X_train.iloc[:, list_index]
                    X_test_FS = self.X_test.iloc[:, list_index]
                    model.fit(X_train_FS,y_train)
                    predictions = model.predict(X_test_FS)
                    acc = accuracy_score(y_test, predictions)
                else:
                    X_FS = X.iloc[:, list_index]
                    model.fit(X_FS,y)
                    scores = cross_val_score(model, X_FS,y, cv=cv,scoring='accuracy')
                    acc=scores.mean()
            
            
            elif self.cat=="regression":
                if self.Model == "RF":
                        print("Random Forest")
                        model= RandomForestRegressor (random_state=100)
                elif self.Model == "SVR":
                    print("Support Vector Regressor")
                    model= SVR()
                elif self.Model == "XGB":
                    print("XGBoost")
                    model = XGBRegressor(use_label_encoder=False,eval_metric='mlogloss', random_state=100)
                
                if not self.CV:
                    X_train_FS = self.X_train.iloc[:, list_index]
                    X_test_FS = self.X_test.iloc[:, list_index]
                    model.fit(X_train_FS,y_train)
                    predictions = model.predict(X_test_FS)
                    acc = accuracy_score(y_test, predictions)
                else:
                    X_FS = X.iloc[:, list_index]      
                    model.fit(X_FS,y)
                    scores = cross_val_score(model, X_FS,y, cv=cv,scoring='neg_mean_squared_error')
                    acc=scores.mean()
            
            print("F(x):" + str(acc))
            print("Evaluation number: "+ str(self.EvalNum))
            return (acc),
        
        #------------------------------------------------
        #***Fitness Function ***
        #------------------------------------------------
        def calculate_fitness(fun):
            """
            Calculate the fitness function from objective function
            the formula is : 1/(1+acc)
            
            Args: results from objective function
            
            Returns: fitness
            """
            try:
                result = 1 / (fun + 1)
                #print("Fitness:" + str(result))
                print('Duration: {} seconds'.format(time.time() - self.start_time))
                return result
            except ValueError as err:
                print("An Error occured: " + str(err))
                exit()
        
        #------------------------------------------------
        #***Stopping condition***
        #------------------------------------------------
        def stop_condition():
            """
            Define the conditions where ABC should stop and return the results
            """
            stp = (self.EvalNum >= self.MaxEval or self.iter<self.rnd or (self.Stop_Acc is not None and self.globalOpt >= self.Stop_Acc))
            return stp
        
        #------------------------------------------------
        #***Init food source for scout***
        #------------------------------------------------
        def init_scout(i):
            """
            Generate two different food sources based on Random and OBL strategies
            
            Args: index of the exhausted food source
            
            Returns: the food source with better quality (acc)
            """
            print("Food #"+ str(i))
            print(self.foods[i][:])
            
            j=0
            for i in range(len(self.foods)):
                for j in range(len(self.foods[0])):
                    self.foods_OBL[0][j]=1-self.foods[i][j]
        
            j=0
            for i in range(len(self.foods)):
                for j in range(len(self.foods[0])):
                    self.foods_OBL[1][j]=np.random.randint(0,2)
                    
            print("OBL based Food #"+ str(i))
            print(self.foods_OBL[0])
            print("***")
            print("Random food #"+ str(i))
            print(self.foods_OBL[1])
        
            first= calculate_function(self.foods_OBL[0])[0]
            second= calculate_function(self.foods_OBL[1])[0]
            if first>second:
                print(str(first)+" is better than "+  str(second) +"(OBL Selected)")
                self.foods[i]=self.foods_OBL[0]
                r_food = np.copy(self.foods[i][:])
                print(str(r_food))
                self.f[i] = first
                self.fitness[i] = calculate_fitness(self.f[i])
                self.trial[i] = 0
                
            else:
                print(str(second)+" is better than "+  str(first) + "(RANDOM Selected)")
                self.foods[i]=self.foods_OBL[1]
                r_food = np.copy(self.foods[i][:])
                print(str(r_food))
                self.f[i] = second
                self.fitness[i] = calculate_fitness(self.f[i])
                self.trial[i] = 0
        #------------------------------------------------
        #***clustering initial population***
        #------------------------------------------------
        def init_pop(i):
            """
            Generate the very first population (food sources)
            For each feature one random number 0 or 1 is generated 
            This population will go through hierarchical kmeans clustering later on
            
            Args: index of the food source
            """
            for j in range(len(self.p_foods[0])):
                self.p_foods[i][j]=np.random.randint(0,2)
        
        #------------------------------------------------
        #***K-Means Clustering***
        #------------------------------------------------
        def init_kmeans(j):
            """
            Operate Hierarchichal K-Means clustering on the initial large population
            and take a representative from each cluster for the new population
            
            Args: 
            """
            
            clustering = AgglomerativeClustering(n_clusters=j).fit(self.p_foods)
              
            print("\nClustering Done!")
            
            clust = []
            indices = []
            for i in range(len(clustering.labels_)):
                if clustering.labels_[i] not in clust:
                    indices.append(i)
                    clust.append(clustering.labels_[i])
        
            new_centroids= np.array(self.p_foods)[indices] 
            
            print("\nTaking representatives as the new population and start training!\n")
            self.foods[:j] = new_centroids
            print(new_centroids)
            self.foods[0][:]=1
            for i in range(j):
                c_food = np.copy(self.foods[i][:])
                self.f[i] = calculate_function(c_food)[0]
                self.fitness[i] = calculate_fitness(self.f[i])
                self.trial[i] = 0
                print("---")
        
        #------------------------------------------------
        #***Generate all food sources/population***
        #------------------------------------------------        
        if (not (stop_condition())):
            for k in range(self.FOOD_NUMBER):
                init_pop(k)
            print(self.p_foods)
            print("Initial population with the size of " + str(self.FOOD_NUMBER) + " generated...\n")
            init_kmeans(self.CLUSTERS)
            filter_pop()
        else:
            print("Stopping condition is already met!")  
                  
        
        #Best food source of population
        for i in range(self.CLUSTERS):
                if (self.f[i] > self.globalOpt):
                    print("\nUpdating optimal solution and parameters...")
                    self.globalOpt = np.copy(self.f[i])
                    self.globalFeatures = np.copy(self.foods[i][:])
        print("Best found food source so far: "+ str(self.globalOpt)+ "\nWith parameters: "+str(self.globalFeatures))    
            
        
        while (not(stop_condition())): 
            print("\n\nCycle #"+ str(self.rnd))
            
            print("\n\t***Employed Phase***\n")
            i = 0
            while (i < self.CLUSTERS) and (not(stop_condition())):
                r = random.random()
                print("------------------------")
                print("Employed Bee #"+ str(i)+":")
                param2change = (int)(r * DIMENSION)
                print("Feature to change: F" + str(param2change))
                self.solution = np.copy(self.foods[i][:])
                print ("Current Food Source:\n" + str(self.solution))
                print ("F(x): " + str(self.f[i]))
                
                self.solution[param2change] = 1-self.foods[i][param2change] 
        
                print ("Updated Food Source:\n" + str(self.solution))
                ObjValSol = calculate_function(self.solution)[0]
                FitnessSol = calculate_fitness(ObjValSol)
        
                #Replace the results if better and reset trial    
                if  (FitnessSol <= self.fitness[i]):
                        print("The solution improved! Updating the results & resetting trial.... ")
                        self.trial[i] = 0
                        self.foods[i][:] = np.copy(self.solution)
                        self.f[i] = ObjValSol
                        self.fitness[i] = FitnessSol
                else:
                        print("The solution didn't improve! Incrementing trial.... ")
                        self.trial[i] = self.trial[i] + 1
                i += 1
                
            if (stop_condition()):
                print("Stopping condition is met!")
            
            print("\n\t***Onlooker Phase***\n")    
            maxfit = np.copy(max(self.fitness))
            minfit = np.copy(min(self.fitness))
            prob=[]
            for i in range(self.CLUSTERS):
                #prob.append(0.9 *(fitness[i] / maxfit)+0.1)
                prob.append((self.fitness[i]-minfit)/(maxfit-minfit))
            i = 0
            t = 0
            while (t < self.CLUSTERS) and (not(stop_condition())):
                r = random.random()
                if (r > prob[i]):
                    print("Onlooker Bee #"+ str(t)+" on Food Source #" +str(i))
                    t+=1
                    param2change = (int)(r * DIMENSION)
                    print("Feature to change: P" + str(param2change))
                    r = random.random()
                    self.solution = np.copy(self.foods[i][:])
                    print ("Current Food Source:\n" + str(self.solution))
                    print ("F(x):" + str(self.f[i]))
                    
                    r = random.random()
                    self.solution[param2change] = 1-self.foods[i][param2change] 
                    
                    
                    print ("Final updated Food Source:\n" + str(self.solution))
                    ObjValSol = calculate_function(self.solution)[0]
                    FitnessSol = calculate_fitness(ObjValSol)
        
                   #replace the results if better
                    if  (FitnessSol <= self.fitness[i]):
                        print("The solution improved! Updating the results & resetting trial.... ")
                        self.trial[i] = 0
                        self.foods[i][:] = np.copy(self.solution)
                        self.f[i] = ObjValSol
                        self.fitness[i] = FitnessSol
                    else:
                        print("The solution didn't improve! Incrementing trial by one.... ")
                        self.trial[i] = self.trial[i] + 1  
                        
                #else:
                    #print ("r="+str(r)+" is smaller than " +str(prob[i]))
                    #print ("Onlooker bee goes to the next food source")
        
                i += 1
                i = i % (self.CLUSTERS)
                print("------------------------")
            #prob.clear()
            if (stop_condition()):
                print("Stopping condition is met!")
        
            
            print("\n***Best Result So Far***")
            print("\nUpdating optimal solution and parameters...")
            for i in range(self.CLUSTERS):
                    if (self.f[i] > self.globalOpt):
                        self.globalOpt = np.copy(self.f[i])
                        self.globalFeatures = np.copy(self.foods[i][:])
            print("Best food source so far: "+ str(self.globalOpt)+ "\nWith Features Below (ones):\n" +str(self.globalFeatures))    
                  
            print("\n***Scout Phase OBL***")
            if (np.amax(self.trial) >= self.Limit):
                   print("trial" + str(self.trial))
                   print("Max Trial >= Limit, occurs at row " + str(self.trial.argmax(axis = 0)))
                   print("Scout explores a random food source...")
                   init_scout(self.trial.argmax(axis = 0))
                   if self.f[self.trial.argmax(axis = 0)]> self.globalOpt:
                        self.globalOpt = np.copy(self.f[self.trial.argmax(axis = 0)])
                        self.globalFeatures = np.copy(self.foods[self.trial.argmax(axis = 0)][:])
            else:
                print ("Trials < Limit \n=> No scouts are required!")    
            self.rnd=self.rnd+1
            
        print("------------------------------------------------")
        print("\t***Results***")
        print("------------------------------------------------")
        self.globalOpts.append(self.globalOpt)
        print("Global Optimum: " + str(abs(max(self.globalOpts))))
        print("Global Features: " + str(self.globalFeatures))
        #duration= format(end_time-start_time)
        print('Duration: {} seconds'.format(time.time() - self.start_time))
        print("Number of evaluations:" +str(self.EvalNum))
        print("Found optimal features after "+ str(self.rnd-1) + " rounds!")
        
        list_index = []
        for i in range(len(self.globalFeatures)):
            if(self.globalFeatures[i] == 1):
                list_index.append(i)
        X = X.iloc[:, list_index]
        print("Optimal features: " + str(X.columns))
        return self.globalFeatures
