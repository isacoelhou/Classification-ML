import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

media = 0 

for _ in range(20):

    dados = pd.read_csv("../Dataset/studentp.csv")
    print(dados.head())
    print("\n\n\n")
    print(dados.info())

    print(dados.head())

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
