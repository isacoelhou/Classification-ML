import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

dados = pd.read_csv("../Dataset/battle.csv")
print(dados.head())
print("\n\n\n")
print(dados.info())

dados = shuffle(dados)
print("\n\n\n")
print(dados.head())

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
Acc = accuracy_score(y_validacao, opiniao)


print("\nMelhor configuração para o NB")
print("Acurácia sobre a validação: ", Acc)

print("\n\nDesempenho sobre o conjunto de teste")

NB = GaussianNB()
NB.fit(x_treino,y_treino)
opiniao = NB.predict(x_teste)

print("\n\nAcurácia sobre o teste: ",accuracy_score(y_teste, opiniao))
