#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 14:17:37 2022

@author: Leila Zahedi. Needs permision upon usage
"""


    def fso_hpo(self, X, y):

        
        if not self.CV:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=100)
        else:
            cv= KFold(n_splits = self.folds, random_state=100, shuffle=True)
            
        #------------------------------------------------
        #***Update variables according to k-means filtering***
        #------------------------------------------------
        def update_variables(self, index_to_drop, f=f, fitness=fitness, 
                             foods=foods, trial=trial):
            """
            Updating variables sizes after clustering
            Filtering the food sources with rich solutions changes the size of food
            sources, objective function, fitness and trial matrixes.
            
            Args: the indexs of the food source that is not rich enough
            """
            self.CLUSTERS -= 1
            f = np.delete(f, index_to_drop, axis=0) #obj func
            fitness = np.delete(fitness, index_to_drop, axis=0) #fitness
            trial= np.delete(trial, index_to_drop, axis=0) # trial
            foods = np.delete(foods, index_to_drop, axis=0) # foods
        
        #------------------------------------------------
        #***filter population based on the defined threshold Min_Acc***
        #------------------------------------------------
        def filter_pop():
            """
            Filtering the new population (centroids) with rich food sources 
            
            Args: the minimum accuracy threshold for eliminating food sources
            """
            temp_f = f 
            i = self.CLUSTERS-1
            while i >= 0:
                if temp_f[i] < self.Min_Acc:
                   update_variables(i)
                i -= 1
            print("New population after filtering:\n" + str(f)) 

        
        #------------------------------------------------
        #***Objective Function ***
        #------------------------------------------------
        def calculate_function(sol, X=X, y=y, X_train=X_train, X_test=X_test):
            """
            Calculate the objective function for each solution in the population 
            
            Args: solution
            
            Returns: Accuracy of the solution
            """
            nonlocal EvalNum
            EvalNum += 1
        
            list_index = []
            for i in range(len(sol)):
                if(sol[i] == 1):
                    list_index.append(i)
        
            
            if not self.CV:
                X_train_FS = X_train.iloc[:, list_index]
                X_test_FS = X_test.iloc[:, list_index]
                if self.Model == "RF":
                    print("Random Forest")
                    model= RandomForestClassifier (random_state=100)
                elif self.Model == "SVM":
                    print("Support Vector Machine")
                    model= SVC(gamma='scale',random_state=100)
                elif self.Model == "XGBoost":
                    print("XGBoost")
                    model = XGBClassifier(objective='binary:logistic', use_label_encoder=False,eval_metric='mlogloss', random_state=100)
            
                model.fit(X_train_FS,y_train)
                predictions = model.predict(X_test_FS)
                acc = accuracy_score(y_test, predictions)
            else:
                X_FS = X.iloc[:, list_index]
                if self.Model == "RF":
                #for 80:20
                    print("Random Forest")
                    model= RandomForestClassifier (random_state=100)
                elif self.Model == "SVM":
                    print("Support Vector Machine")
                    model = SVC(gamma='scale',random_state=100)
                elif self.Model == "XGBoost":
                    print("XGBoost")
                    model = XGBClassifier(objective='binary:logistic', use_label_encoder=False,eval_metric='mlogloss', random_state=100)      
                model.fit(X_FS,y)
                scores = cross_val_score(model, X_FS,y, cv=cv,scoring='accuracy')
                acc=scores.mean()
            
            print("F(x):" + str(acc))
            print("Evaluation number: "+ str(EvalNum))
            return (acc),
        
        #------------------------------------------------
        #***Fitness Function ***
        #------------------------------------------------
        def calculate_fitness(fun, start_time=start_time):
            """
            Calculate the fitness function from objective function
            the formula is : 1/(1+acc)
            
            Args: results from objective function
            
            Returns: fitness
            """
            try:
                result = 1 / (fun + 1)
                #print("Fitness:" + str(result))
                print('Duration: {} seconds'.format(time.time() - start_time))
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
            stp = (EvalNum >= self.MaxEval or self.iter<rnd or globalOpt >= self.Stop_Acc)
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
            print(foods[i][:])
            
            j=0
            for i in range(len(foods)):
                for j in range(len(foods[0])):
                    foods_OBL[0][j]=1-foods[i][j] # 1-1=0 , 1-0=1
        
            j=0
            for i in range(len(foods)):
                for j in range(len(foods[0])):
                    foods_OBL[1][j]=np.random.randint(0,2)
                    
            print("OBL based Food #"+ str(i))
            print(foods_OBL[0])
            print("***")
            print("Random food #"+ str(i))
            print(foods_OBL[1])
        
            first= calculate_function(foods_OBL[0])[0]
            second= calculate_function(foods_OBL[1])[0]
            if first>second:
                print(str(first)+" is better than "+  str(second) +"(OBL Selected)")
                foods[i]=foods_OBL[0]
                r_food = np.copy(foods[i][:])
                print(str(r_food))
                f[i] = first
                fitness[i] = calculate_fitness(f[i])
                trial[i] = 0
                
            else:
                print(str(second)+" is better than "+  str(first) + "(RANDOM Selected)")
                foods[i]=foods_OBL[1]
                r_food = np.copy(foods[i][:])
                print(str(r_food))
                f[i] = second
                fitness[i] = calculate_fitness(f[i])
                trial[i] = 0
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
            for j in range(len(p_foods[0])):
                p_foods[i][j]=np.random.randint(0,2)
        
        #------------------------------------------------
        #***K-Means Clustering***
        #------------------------------------------------
        def init_kmeans(j):
            """
            Operate Hierarchichal K-Means clustering on the initial large population
            and take a representative from each cluster for the new population
            
            Args: 
            """
            
            
            clustering = AgglomerativeClustering(n_clusters=j,affinity='cosine').fit(p_foods)
              
            print("\nK-Means Clustering Done!")
            
            clust = []
            indices = []
            for i in range(len(clustering.labels_)):
                if clustering.labels_[i] not in clust:
                    indices.append(i)
                    clust.append(clustering.labels_[i])
        
            new_centroids= np.array(p_foods)[indices] 
            
            print("\nTaking representatives as the new population and start training!\n")
            foods[:j] = new_centroids
            print(new_centroids)
            #foods[:0]=1
            #print(foods)
            #print(kmeans.labels_)
            #centroids = kmeans.cluster_centers_
        
            #labelss=kmeans.labels_
            #centroid_labels = [centroids[i] for i in labelss]
            #print(centroid_labels)
        
            #foods[:,1]=foods[:,1].round(0).astype(int)
            foods[0][:]=1
            for i in range(j):
                c_food = np.copy(foods[i][:])
                #print(str(c_food))
                f[i] = calculate_function(c_food)[0]
                fitness[i] = calculate_fitness(f[i])
                trial[i] = 0
                print("---")
        
        #------------------------------------------------
        #***Generate all food sources/population***
        #------------------------------------------------
        #start_time = time.time() 
        
        if (not (stop_condition())):
            for k in range(self.FOOD_NUMBER):
                init_pop(k)
            print(p_foods)
            print("Initial population with the size of " + str(self.FOOD_NUMBER) + " generated...\n")
            init_kmeans(self.CLUSTERS)
            filter_pop()
        else:
            print("Stopping condition is already met!")  
                  
        
        #Best food source of population
        for i in range(self.CLUSTERS):
                if (f[i] > globalOpt):
                    #print(str(f[i]) +">=" + str(globalOpt) + "\t->
                    print("\nUpdating optimal solution and parameters...")
                    globalOpt = np.copy(f[i])
                    globalFeatures = np.copy(foods[i][:])
        print("Best found food source so far: "+ str(globalOpt)+ "\nWith parameters: "+str(globalFeatures))    
            
        
        while (not(stop_condition())): 
            print("\n\nCycle #"+ str(rnd))
            
            print("\n\t***Employed Phase***\n")
            i = 0
            while (i < self.CLUSTERS) and (not(stop_condition())):
                r = random.random()
                print("------------------------")
                print("Employed Bee #"+ str(i)+":")
                param2change = (int)(r * DIMENSION)
                print("Feature to change: F" + str(param2change))
                solution = np.copy(foods[i][:])
                print ("Current Food Source:\n" + str(solution))
                print ("F(x): " + str(f[i]))
                #print ("Neighbor:" + str(foods[neighbour]))
                
                solution[param2change] = 1-foods[i][param2change] 
        
                print ("Updated Food Source:\n" + str(solution))
                ObjValSol = calculate_function(solution)[0]
                FitnessSol = calculate_fitness(ObjValSol)
        
                #Replace the results if better and reset trial    
                if  (FitnessSol <= fitness[i]):
                        print("The solution improved! Updating the results & resetting trial.... ")
                        trial[i] = 0
                        foods[i][:] = np.copy(solution)
                        f[i] = ObjValSol
                        fitness[i] = FitnessSol
                else:
                        print("The solution didn't improve! Incrementing trial.... ")
                        trial[i] = trial[i] + 1
                i += 1
                
            if (stop_condition()):
                print("Stopping condition is met!")
            
            print("\n\t***Onlooker Phase***\n")    
            maxfit = np.copy(max(fitness))
            minfit = np.copy(min(fitness))
            prob=[]
            for i in range(self.CLUSTERS):
                #prob.append(fitness[i] / sum(fitness))
                #prob.append(0.9 *(fitness[i] / maxfit)+0.1)
                prob.append((fitness[i]-minfit)/(maxfit-minfit))
            #print(prob)    
            i = 0
            t = 0
            while (t < self.CLUSTERS) and (not(stop_condition())):
                r = random.random()
                if (r > prob[i]):
                    #print ("Generated random number "+str(r)+" is larger than probability " +str(prob[i])+ " =>\n")
                    print("Onlooker Bee #"+ str(t)+" on Food Source #" +str(i))
                    t+=1
                    param2change = (int)(r * DIMENSION)
                    print("Feature to change: P" + str(param2change))
                    r = random.random()
                    solution = np.copy(foods[i][:])
                    print ("Current Food Source:\n" + str(solution))
                    print ("F(x):" + str(f[i]))
                    
                    r = random.random()
                    solution[param2change] = 1-foods[i][param2change] 
                    
                    
                    print ("Final updated Food Source:\n" + str(solution))
                    ObjValSol = calculate_function(solution)[0]
                    FitnessSol = calculate_fitness(ObjValSol)
        
                   #replace the results if better
                    if  (FitnessSol <= fitness[i]):
                        print("The solution improved! Updating the results & resetting trial.... ")
                        trial[i] = 0
                        foods[i][:] = np.copy(solution)
                        f[i] = ObjValSol
                        fitness[i] = FitnessSol
                    else:
                        print("The solution didn't improve! Incrementing trial by one.... ")
                        trial[i] = trial[i] + 1  
                        
                else:
                    #print ("r="+str(r)+" is smaller than " +str(prob[i]))
                    print ("Onlooker bee goes to the next food source")
        
                i += 1
                i = i % (self.CLUSTERS)
                print("------------------------")
            #prob.clear()
            if (stop_condition()):
                print("Stopping condition is met!")
        
            
            print("\n***Best Result So Far***")
            print("\nUpdating optimal solution and parameters...")
            for i in range(self.CLUSTERS):
                    if (f[i] > globalOpt):
                        #print(str(f[i]) +">" + str(globalOpt) + "\t-> 
                        globalOpt = np.copy(f[i])
                        globalFeatures = np.copy(foods[i][:])
            print("Best food source so far: "+ str(globalOpt)+ "\nWith Features Below (ones):\n" +str(globalFeatures))    
                  
            print("\n***Scout Phase OBL***")
            if (np.amax(trial) >= self.Limit):
                   print("trial" + str(trial))
                   print("Max Trial >= Limit, occurs at row " + str(trial.argmax(axis = 0)))
                   print("Scout explores a random food source...")
                   init_scout(trial.argmax(axis = 0))
                   if f[trial.argmax(axis = 0)]> globalOpt:
                        globalOpt = np.copy(f[trial.argmax(axis = 0)])
                        globalFeatures = np.copy(foods[trial.argmax(axis = 0)][:])
            else:
                print ("Trials < Limit \n=> No scouts are required!")    
            rnd=rnd+1
        
            
        #end_time = datetime.datetime.now() #end time
        print("------------------------------------------------")
        print("\t***Results***")
        print("------------------------------------------------")
        globalOpts.append(globalOpt)
        print("Global Optimum: " + str(max(globalOpts)))
        print("Global Features: " + str(globalFeatures))
        #duration= format(end_time-start_time)
        print('Duration: {} seconds'.format(time.time() - start_time))
        print("Number of evaluations:" +str(EvalNum))
        print("Found optimal after "+ str(rnd-1) + " rounds!")