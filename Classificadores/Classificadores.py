import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

for _ in range(20):

    dados = pd.read_csv("../Dataset/studentp.csv")

    dados = shuffle(dados)

    X = dados.iloc[:,:-1]
    Y = dados.iloc[:,-1]

    x_treino,x_temp,y_treino,y_temp=train_test_split(X,Y,test_size=0.5,stratify=Y)
    x_validacao,x_teste,y_validacao,y_teste=train_test_split(x_temp,y_temp,test_size=0.5, stratify = y_temp)

    print("Treino")
    x_treino.info()
    y_treino.info()

    print("\nValidação")
    x_validacao.info()
    y_validacao.info()

    print("\nTeste")
    x_teste.info()
    y_teste.info()

    ###############################################KNN##########################################################

    maior = -1 
    Acc_knn = []

    for j in ("distance","uniform"):
        for i in range (1,20):

            KNN = KNeighborsClassifier(n_neighbors=i,weights=j)
            KNN.fit(x_treino,y_treino)
            opiniao = KNN.predict(x_validacao)
            Acc = accuracy_score(y_validacao, opiniao)
        
            #print("K: ",i," Métrica: ",j," Acc: ",Acc)

            if (Acc > maior):
                maior = Acc
                Melhor_k = i
                Melhor_metrica = j

            KNN = KNeighborsClassifier(n_neighbors=Melhor_k,weights=Melhor_metrica)
            KNN.fit(x_treino,y_treino)

            opiniao = KNN.predict(x_teste)

            Acc = accuracy_score(y_teste, opiniao)
            Acc_knn.append(Acc)

        ###############################################KNN##########################################################

        ###############################################DT###########################################################

    maior = -1
    Acc_DT = []
    
    for j in ("entropy","gini"):  #criterion
        for i in range (1,20):      #max_depth
            for k in range (1,20):    #min_samples_leaf
                for l in range (2,20):  #min_samples_split
                    for m in ('best','random'): #splitter
                        AD = DecisionTreeClassifier(criterion=j,max_depth=i,min_samples_leaf=k,min_samples_split=l,splitter=m)
                        AD.fit(x_treino,y_treino)
                        opiniao = AD.predict(x_validacao)
                        Acc = accuracy_score(y_validacao, opiniao)
                        print("Criterion: ",j," max_depth: ",i," min_samples_leaf: ",k," min_samples_split: ",l," splitter: ",m," Acc: ",Acc)
                        if (Acc > maior):
                            maior = Acc
                            crit = j
                            md = i
                            msl = k
                            mss = l
                            split = m

    # print("\nMelhor configuração para a AD")
    # print("Criterion: ",crit," max_depth: ",md," min_samples_leaf: ",msl," min_samples_split: ",mss," splitter: ",split," Acc: ",maior)

    # print("\n\nDesempenho sobre o conjunto de teste")

    AD = DecisionTreeClassifier(criterion=crit,max_depth=md,min_samples_leaf=msl,min_samples_split=mss,splitter=split)
    AD.fit(x_treino,y_treino)

    opiniao = AD.predict(x_teste)
    Acc = accuracy_score(y_teste, opiniao)
    Acc_DT.append(Acc)

    #print("Acurácia sobre o teste: ",accuracy_score(y_teste, opiniao))

    ###############################################DT###########################################################

    ###############################################SVM##########################################################

    maior = -1
    Acc_SVM = []
    for k in ("linear", "poly", "rbf", "sigmoid"):  #kernel
        for i in (0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0):  #custo
            SVM = SVC(kernel=k,C=i)
            SVM.fit(x_treino,y_treino)
            opiniao = SVM.predict(x_validacao)
            Acc = accuracy_score(y_validacao, opiniao)
            print("Kernel: ",k," C: ",i," Acc: ",Acc)
            if (Acc > maior):
                maior = Acc
                ker = k
                custo = i

    # print("\nMelhor configuração para o SVM")
    # print("Kernel: ",ker," C: ",custo)

    # print("\n\nDesempenho sobre o conjunto de teste")

    SVM = SVC(kernel=ker,C=custo)
    SVM.fit(x_treino,y_treino)

    opiniao = SVM.predict(x_teste)
    Acc = accuracy_score(y_teste, opiniao)

    Acc_SVM.append(Acc)


    ###############################################SVM##########################################################

    ###############################################MLP##########################################################

    maior = -1
    Acc_MLP = []

    for i in (5,6,10,12):
        for j in ('constant','invscaling', 'adaptive'):
            for k in (50,100,150,300,500,1000):
                for l in ('identity', 'logistic', 'tanh', 'relu'):
                    MLP = MLPClassifier(hidden_layer_sizes=(i,i,i), learning_rate=j, max_iter=k, activation=l )
                    MLP.fit(x_treino,y_treino)

                    opiniao = MLP.predict(x_validacao)
                    Acc = accuracy_score(y_validacao, opiniao)
                    #print("Acurácia: ",Acc)

                    if (Acc > maior):
                        maior = Acc
                        Melhor_i = i
                        Melhor_j = j
                        Melhor_k = k
                        Melhor_l = l

    # print("Acc do  MLP sobre o conjunto de teste")
   
    MLP = MLPClassifier(hidden_layer_sizes=(Melhor_i,Melhor_i,Melhor_i), learning_rate=Melhor_j, max_iter=Melhor_k, activation=Melhor_l)
    MLP.fit(x_treino,y_treino)
   
    opiniao = MLP.predict(x_teste)
   
    Acc = accuracy_score(y_teste, opiniao)
    Acc_MLP.append(Acc)
    

    ###############################################MLP##########################################################

    ###############################################NB###########################################################

    Acc_NB = []

    NB = GaussianNB()
    NB.fit(x_treino,y_treino)
    opiniao = NB.predict(x_validacao)
    Acc_validacao = accuracy_score(y_validacao, opiniao)

    # print("Acurácia sobre a validação: ", Acc_validacao)
    # print("\n\nDesempenho sobre o conjunto de teste")

    NB.fit(x_treino,y_treino)
    opiniao = NB.predict(x_teste)
    Acc = accuracy_score(y_teste, opiniao)

    # print("\n\nAcurácia sobre o teste: ", Acc)
    Acc_NB.append(Acc)

print("\n\nMedia de acurácia do Naive Bayes:", media/20)
