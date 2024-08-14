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

for _ in range(1):

    dados = pd.read_csv("../Dataset/Diabetes.csv")

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
        for i in range (1,50):

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

        probabilidades_knn = KNN.predict_proba(x_teste)

        opiniao = KNN.predict(x_teste)
        Acc = accuracy_score(y_teste, opiniao)
        Acc_knn.append(Acc)


# #         ###############################################KNN##########################################################

# #         ###############################################DT###########################################################

    # maior = -1
    # Acc_DT = []
    
    # for j in ("entropy","gini"):  #criterion
    #    for i in (3,4,5,6,7):      #max_depth
    #         for k in (3,4,5,6):    #min_samples_leaf
    #             for l in (5,6,8,10):  #min_samples_split
    #                 for m in ('best','random'): #splitter
                        
    #                     AD = DecisionTreeClassifier(criterion=j,max_depth=i,min_samples_leaf=k,min_samples_split=l,splitter=m)
    #                     AD.fit(x_treino,y_treino)
                        
    #                     opiniao = AD.predict(x_validacao)
    #                     Acc = accuracy_score(y_validacao, opiniao)
                        
    #                     #print("Criterion: ",j," max_depth: ",i," min_samples_leaf: ",k," min_samples_split: ",l," splitter: ",m," Acc: ",Acc)
    #                     if (Acc > maior):
    #                         maior = Acc
    #                         crit = j
    #                         md = i
    #                         msl = k
    #                         mss = l
    #                         split = m

    # # print("\nMelhor configuração para a AD")
    # # print("Criterion: ",crit," max_depth: ",md," min_samples_leaf: ",msl," min_samples_split: ",mss," splitter: ",split," Acc: ",maior)

    # # print("\n\nDesempenho sobre o conjunto de teste")

    # AD = DecisionTreeClassifier(criterion=crit,max_depth=md,min_samples_leaf=msl,min_samples_split=mss,splitter=split)
    # AD.fit(x_treino,y_treino)

    # opiniao = AD.predict(x_teste)
    # probabilidades_dt = AD.predict(x_teste)

    # Acc = accuracy_score(y_teste, opiniao)
    # Acc_DT.append(Acc)

    #print("Acurácia sobre o teste: ",accuracy_score(y_teste, opiniao))
    classe = 2
    c0 = 0
    c1 = 0
    c2 = 0
    c3 = 0
    c4 = 0

    for i in range(len(x_teste)):
        for j in  range(5):
            if j == 0:
                c0 += probabilidades_knn[i][j] + probabilidades_dt[i][j] + probabilidades_mlp[i][j] + probabilidades_nb[i][j]
            if j == 1:
                c1 += probabilidades_knn[i][j] + probabilidades_dt[i][j] + probabilidades_mlp[i][j] + probabilidades_nb[i][j]
            if j == 2:
                c2 += probabilidades_knn[i][j] + probabilidades_dt[i][j] + probabilidades_mlp[i][j] + probabilidades_nb[i][j]
            if j == 3:
                c3 += probabilidades_knn[i][j] + probabilidades_dt[i][j] + probabilidades_mlp[i][j] + probabilidades_nb[i][j]
            if j == 4:
                c4 += probabilidades_knn[i][j] + probabilidades_dt[i][j] + probabilidades_mlp[i][j] + probabilidades_nb[i][j]