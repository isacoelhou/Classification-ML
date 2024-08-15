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


#         ###############################################KNN##########################################################

#         ###############################################DT###########################################################

    maior = -1
    Acc_DT = []
    
    for j in ("entropy","gini"):  #criterion
       for i in (3,4,5,6,7):      #max_depth
            for k in (3,4,5,6):    #min_samples_leaf
                for l in (5,6,8,10):  #min_samples_split
                    for m in ('best','random'): #splitter
                        
                        AD = DecisionTreeClassifier(criterion=j,max_depth=i,min_samples_leaf=k,min_samples_split=l,splitter=m)
                        AD.fit(x_treino,y_treino)
                        
                        opiniao = AD.predict(x_validacao)
                        Acc = accuracy_score(y_validacao, opiniao)
                        
                        #print("Criterion: ",j," max_depth: ",i," min_samples_leaf: ",k," min_samples_split: ",l," splitter: ",m," Acc: ",Acc)
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
    probabilidades_dt = AD.predict_proba(x_teste)

    Acc = accuracy_score(y_teste, opiniao)
    Acc_DT.append(Acc)

    #print("Acurácia sobre o teste: ",accuracy_score(y_teste, opiniao))

#     ###############################################DT###########################################################

#     ###############################################SVM##########################################################

    maior = -1
    Acc_SVM = []

# Ajustar SVM com suporte a probabilidades
    for k in ("linear", "poly", "rbf", "sigmoid"):  # kernel
        for i in (0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 1):  # custo

            SVM = SVC(kernel=k, C=i, probability=True)  # Adicionar probability=True
            SVM.fit(x_treino, y_treino)

            opiniao = SVM.predict(x_validacao)
            Acc = accuracy_score(y_validacao, opiniao)
            
            # Verificar se a acurácia é maior que a anterior
            if Acc > maior:
                maior = Acc
                ker = k
                custo = i

    # Treinar o modelo SVM com a melhor configuração encontrada
    SVM = SVC(kernel=ker, C=custo, probability=True)  # Adicionar probability=True
    SVM.fit(x_treino, y_treino)

    # Obter probabilidades de classificação para o conjunto de teste
    probabilidades_svm = SVM.predict_proba(x_teste)

    # Avaliar a acurácia no conjunto de teste
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
    probabilidades_mlp = MLP.predict_proba(x_teste)

    Acc = accuracy_score(y_teste, opiniao)
    Acc_MLP.append(Acc)
    

#     ###############################################MLP##########################################################

#     ###############################################NB###########################################################

    Acc_NB = []

    NB = GaussianNB()
    NB.fit(x_treino,y_treino)
    opiniao = NB.predict(x_validacao)
    Acc_validacao = accuracy_score(y_validacao, opiniao)

    # print("Acurácia sobre a validação: ", Acc_validacao)
    # print("\n\nDesempenho sobre o conjunto de teste")

    NB.fit(x_treino,y_treino)
    
    opiniao = NB.predict(x_teste)
    probabilidades_nb = NB.predict_proba(x_teste)

    Acc = accuracy_score(y_teste, opiniao)

    # print("\n\nAcurácia sobre o teste: ", Acc)
    Acc_NB.append(Acc)
    regra_da_soma = []

    for i in range(len(x_teste)):
        c0 = 0
        c1 = 0
        c2 = 0
        c3 = 0
        c4 = 0

        c0 += probabilidades_knn[i][0] + probabilidades_dt[i][0] + probabilidades_mlp[i][0] + probabilidades_nb[i][0] + probabilidades_svm[i][0]
        c1 += probabilidades_knn[i][1] + probabilidades_dt[i][1] + probabilidades_mlp[i][1] + probabilidades_nb[i][1] + probabilidades_svm[i][1]
        c2 += probabilidades_knn[i][2] + probabilidades_dt[i][2] + probabilidades_mlp[i][2] + probabilidades_nb[i][2] + probabilidades_svm[i][2]
        c3 += probabilidades_knn[i][3] + probabilidades_dt[i][3] + probabilidades_mlp[i][3] + probabilidades_nb[i][3] + probabilidades_svm[i][3]
        c4 += probabilidades_knn[i][4] + probabilidades_dt[i][4] + probabilidades_mlp[i][4] + probabilidades_nb[i][4] + probabilidades_svm[i][4]

        valores = [c0, c1, c2, c3, c4]
        maior_valor = max(valores)
        posicao = valores.index(maior_valor)
        regra_da_soma.append(posicao)


    Acc = accuracy_score(y_teste, regra_da_soma)
    print(Acc)
