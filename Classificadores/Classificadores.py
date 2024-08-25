import pandas as pd
from statistics import mode
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

def bordacount(matriz):

    ranqueada = []
    
    for linha in matriz:

        indices_ordenados = sorted(range(len(linha)), key=lambda i: linha[i])
        posicoes = [0] * len(linha)
        
        for ranque, indice in enumerate(indices_ordenados):
            posicoes[indice] = ranque
        
        ranqueada.append(posicoes)
    
    return ranqueada

Acc_DT = []
Acc_knn = []
Acc_SVM = []
Acc_MLP = []
Acc_NB = []

#SMC
Acc_VM = []
Acc_RS = []
Acc_BC = []

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

    opiniao_knn = KNN.predict(x_teste)
    Acc = accuracy_score(y_teste, opiniao_knn)
    Acc_knn.append(Acc)


#         ###############################################KNN##########################################################

#         ###############################################DT###########################################################

    maior = -1
    
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

    opiniao_dt = AD.predict(x_teste)
    probabilidades_dt = AD.predict_proba(x_teste)

    Acc = accuracy_score(y_teste, opiniao_dt)
    Acc_DT.append(Acc)

    #print("Acurácia sobre o teste: ",accuracy_score(y_teste, opiniao))

#     ###############################################DT###########################################################

#     ###############################################SVM##########################################################

    maior = -1

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
    opiniao_svm = SVM.predict(x_teste)
    Acc = accuracy_score(y_teste, opiniao_svm)

    Acc_SVM.append(Acc)


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

    # print("Acc do  MLP sobre o conjunto de teste")
   
    MLP = MLPClassifier(hidden_layer_sizes=(Melhor_i,Melhor_i,Melhor_i), learning_rate=Melhor_j, max_iter=Melhor_k, activation=Melhor_l)
    MLP.fit(x_treino,y_treino)
   
    opiniao_mlp = MLP.predict(x_teste)
    probabilidades_mlp = MLP.predict_proba(x_teste)

    Acc = accuracy_score(y_teste, opiniao_mlp)
    Acc_MLP.append(Acc)
    

#     ###############################################MLP##########################################################

#     ###############################################NB###########################################################

    NB = GaussianNB()
    NB.fit(x_treino,y_treino)
    opiniao = NB.predict(x_validacao)
    Acc_validacao = accuracy_score(y_validacao, opiniao)

    # print("Acurácia sobre a validação: ", Acc_validacao)
    # print("\n\nDesempenho sobre o conjunto de teste")

    NB.fit(x_treino,y_treino)
    
    opiniao_nb = NB.predict(x_teste)
    probabilidades_nb = NB.predict_proba(x_teste)

    Acc = accuracy_score(y_teste, opiniao_nb)

    # print("\n\nAcurácia sobre o teste: ", Acc)
    Acc_NB.append(Acc)
    regra_da_soma = []
    voto_majoritario = []
    borda_count = []

    #SISTEMAS DE MULTIPLOS CLASSIFICADORES 

    ranqueada_knn = bordacount(probabilidades_knn)
    ranqueada_dt = bordacount(probabilidades_dt)
    ranqueada_mlp = bordacount(probabilidades_mlp)
    ranqueada_nb = bordacount(probabilidades_nb)
    ranqueada_svm = bordacount(probabilidades_svm)

    for i in range(len(x_teste)):

        #BORDACCOUNT

        c0_ba = ranqueada_knn[i][0] + ranqueada_dt[i][0] + ranqueada_mlp[i][0] + ranqueada_nb[i][0] + ranqueada_svm[i][0]
        c1_ba = ranqueada_knn[i][1] + ranqueada_dt[i][1] + ranqueada_mlp[i][1] + ranqueada_nb[i][1] + ranqueada_svm[i][1]
        c2_ba = ranqueada_knn[i][2] + ranqueada_dt[i][2] + ranqueada_mlp[i][2] + ranqueada_nb[i][2] + ranqueada_svm[i][2]
        c3_ba = ranqueada_knn[i][3] + ranqueada_dt[i][3] + ranqueada_mlp[i][3] + ranqueada_nb[i][3] + ranqueada_svm[i][3]
        c4_ba = ranqueada_knn[i][4] + ranqueada_dt[i][4] + ranqueada_mlp[i][4] + ranqueada_nb[i][4] + ranqueada_svm[i][4]
        
        valores = [c0_ba, c1_ba, c2_ba, c3_ba, c4_ba]
        maior_valor = max(valores)
        posicao = valores.index(maior_valor)
        borda_count.append(posicao)

        #VOTO MAJORITÁRIO

        c0_vm  = opiniao_knn[i]
        c1_vm = opiniao_dt[i]
        c2_vm = opiniao_mlp[i]
        c3_vm = opiniao_svm[i]
        c4_vm = opiniao_nb[i]


        valores = [c0_vm, c1_vm, c2_vm, c3_vm, c4_vm]
        voto_majoritario.append(mode(valores))

        #REGRA DA SOMA

        c0_rs = probabilidades_knn[i][0] + probabilidades_dt[i][0] + probabilidades_mlp[i][0] + probabilidades_nb[i][0] + probabilidades_svm[i][0]
        c1_rs = probabilidades_knn[i][1] + probabilidades_dt[i][1] + probabilidades_mlp[i][1] + probabilidades_nb[i][1] + probabilidades_svm[i][1]
        c2_rs = probabilidades_knn[i][2] + probabilidades_dt[i][2] + probabilidades_mlp[i][2] + probabilidades_nb[i][2] + probabilidades_svm[i][2]
        c3_rs = probabilidades_knn[i][3] + probabilidades_dt[i][3] + probabilidades_mlp[i][3] + probabilidades_nb[i][3] + probabilidades_svm[i][3]
        c4_rs = probabilidades_knn[i][4] + probabilidades_dt[i][4] + probabilidades_mlp[i][4] + probabilidades_nb[i][4] + probabilidades_svm[i][4]
        
        valores = [c0_rs, c1_rs, c2_rs, c3_rs, c4_rs]
        maior_valor = max(valores)
        posicao = valores.index(maior_valor)
        regra_da_soma.append(posicao)


    Acc = accuracy_score(y_teste, regra_da_soma)
    print(Acc)
    Acc_RS.append(Acc)

    Acc = accuracy_score(y_teste, voto_majoritario)
    print(Acc)
    Acc_VM.append(Acc)

    Acc = accuracy_score(y_teste, borda_count)
    print(Acc)
    Acc_BC.append(Acc)

print('KNN:', Acc_knn)
print('DT:', Acc_DT)
print('MLP:', Acc_MLP)
print('SVM:', Acc_SVM)
print('NB:', Acc_NB)
print('RS:', Acc_RS)
print('VM:', Acc_VM)
print('BC:', Acc_BC)

import pandas as pd

estrategias = ["K Nearest Neighbors", "Decision Tree", "Multilayer Perceptron", "Suport Vector Machine", "Naive Bayes", "Regra da soma", "Borda Count", "Voto Majoritario"]
acuracias = [Acc_knn, Acc_DT, Acc_MLP, Acc_SVM, Acc_NB, Acc_RS, Acc_BC ,Acc_VM]

df = pd.DataFrame({
    'Estrategia': estrategias,
    'Acc': acuracias
})

df.set_index('Estrategia', inplace=True)

df.to_csv('./stats/Acuracias.csv')