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
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

media = 0 

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

    taxa_de_erro = []
    metrica = []
    maior = -1
    for j in ("distance","uniform"):
    for i in range (1,20):
        KNN = KNeighborsClassifier(n_neighbors=i,weights=j)
        KNN.fit(x_treino,y_treino)
        opiniao = KNN.predict(x_validacao)

        taxa_de_erro.append(np.mean(opiniao!=y_validacao))
        metrica.append(j)
        Acc = accuracy_score(y_validacao, opiniao)

        print("K: ",i," Métrica: ",j," Acc: ",Acc)
        if (Acc > maior):
        maior = Acc
        melhor_k_acc = i
        Melhor_metrica = j


    print("\nMelhor configuração para o KNN com relação a taxa de erro sobre a validação:")
    melhor_k_te=np.argmin(taxa_de_erro)+1
    print("\nMelhor K:", melhor_k_te," metrica", metrica[melhor_k_te], "\n\n")

    print("\nMelhor configuração para o KNN com relação a acurácia sobre a validação")
    print("K: ",melhor_k_acc," Métrica: ",Melhor_metrica," Acc",maior)

    print("\n\nDesempenho sobre o conjunto de teste")

    KNN_TE = KNeighborsClassifier(n_neighbors=melhor_k_te,weights=metrica[melhor_k_te])
    KNN_ACC = KNeighborsClassifier(n_neighbors=melhor_k_acc,weights=Melhor_metrica)

    KNN_TE.fit(x_treino,y_treino)
    KNN_ACC.fit(x_treino,y_treino)

    opiniao_TE = KNN_TE.predict(x_teste)
    opiniao_ACC = KNN_ACC.predict(x_teste)

    print("\nCom base na Taxa de Erro:\nK: ",melhor_k_te,"Métrica: ", metrica[melhor_k_te], " Acurácia sobre o testeA: ",accuracy_score(y_teste, opiniao_TE))
    print("\nCom base na Acurácia:\nK: ",melhor_k_acc,"Métrica: ", Melhor_metrica, " Acurácia sobre o teste: ",accuracy_score(y_teste, opiniao_ACC))

    ###############################################KNN##########################################################

    ###############################################DT###########################################################

    maior = -1
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

    print("\nMelhor configuração para a AD")
    print("Criterion: ",crit," max_depth: ",md," min_samples_leaf: ",msl," min_samples_split: ",mss," splitter: ",split," Acc: ",maior)

    print("\n\nDesempenho sobre o conjunto de teste")
    AD = DecisionTreeClassifier(criterion=crit,max_depth=md,min_samples_leaf=msl,min_samples_split=mss,splitter=split)
    AD.fit(x_treino,y_treino)
    opiniao = AD.predict(x_teste)
    print("Acurácia sobre o teste: ",accuracy_score(y_teste, opiniao))

    ###############################################DT###########################################################

    ###############################################SVM##########################################################

    maior = -1
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

    print("\nMelhor configuração para o SVM")
    print("Kernel: ",ker," C: ",custo)

    print("\n\nDesempenho sobre o conjunto de teste")
    SVM = SVC(kernel=ker,C=custo)
    SVM.fit(x_treino,y_treino)
    opiniao = SVM.predict(x_teste)
    print("Acurácia sobre o teste: ",accuracy_score(y_teste, opiniao))

    ###############################################SVM##########################################################

    ###############################################MLP##########################################################

    maior = -1
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

    print("Acc do  MLP sobre o conjunto de teste")
    MLP = MLPClassifier(hidden_layer_sizes=(Melhor_i,Melhor_i,Melhor_i), learning_rate=Melhor_j, max_iter=Melhor_k, activation=Melhor_l)
    MLP.fit(x_treino,y_treino)
    opiniao = MLP.predict(x_teste)
    Acc = accuracy_score(y_teste, opiniao)
    print(Acc)

    ###############################################MLP##########################################################

    ###############################################NB###########################################################

    media = 0 
    for _ in range(20):
        NB = GaussianNB()
        NB.fit(x_treino,y_treino)
        opiniao = NB.predict(x_validacao)
        Acc_validacao = accuracy_score(y_validacao, opiniao)

        print("Acurácia sobre a validação: ", Acc_validacao)

        print("\n\nDesempenho sobre o conjunto de teste")

        NB = GaussianNB()
        NB.fit(x_treino,y_treino)
        opiniao = NB.predict(x_teste)
        Acc = accuracy_score(y_teste, opiniao)

        print("\n\nAcurácia sobre o teste: ", Acc)
        media += Acc
print("\n\nMedia de acurácia do Naive Bayes:", media/20)
