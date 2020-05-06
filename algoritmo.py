import pandas as pd
from collections import Counter
import xlwt

class Modelo:
    def setNome(self, nome):
        self.nome = nome
     
    def setScore(self, score):
        self.score = score
	
    def setResultado(self, resultado):
        self.resultado = resultado
     
    def getNome(self):
        return self.nome
         
    def getScore(self):
        return self.score

    def getResultado(self):
        return self.resultado

modelo1 = Modelo()
modelo2 = Modelo()
modelo3 = Modelo()
modelo4 = Modelo()
modelo5 = Modelo()
modelos = {}

planilha_resultados = xlwt.Workbook()

ws_treino = planilha_resultados.add_sheet("validação")
ws_teste = planilha_resultados.add_sheet("teste")

df = pd.read_csv('dados.csv')

X_df = df[['interessadasDummie', 'grupo', 'eventos', 'clima', 'data_especial', 'temporada','semana']] #88%
#X_df = df[['grupo', 'eventos', 'clima', 'data_especial', 'temporada','semana']] #53%
#X_df = df[['interessadasDummie', 'eventos', 'clima', 'data_especial', 'temporada','semana']] #70%
#X_df = df[['interessadasDummie', 'grupo', 'clima', 'data_especial', 'temporada','semana']] #94%
#X_df = df[['interessadasDummie', 'grupo', 'eventos','data_especial', 'temporada','semana']] #58%
#X_df = df[['interessadasDummie', 'grupo', 'eventos', 'clima', 'temporada','semana']] #70%
#X_df = df[['interessadasDummie', 'grupo', 'eventos', 'clima', 'data_especial', 'semana']] #88%
#X_df = df[['interessadasDummie', 'grupo', 'eventos', 'clima', 'data_especial', 'temporada']] #76%

#X_df = df[['interessadasDummie', 'grupo', 'clima', 'data_especial', 'semana']] #94%

Y_df = df['publico']

Xdummies_df = pd.get_dummies(X_df).astype(int)
Ydummies_df = Y_df

X = Xdummies_df.values
Y = Ydummies_df.values

porcentagem_de_treino = 0.7
porcentagem_de_teste = 0.2

tamanho_de_treino = porcentagem_de_treino * len(Y)
tamanho_de_teste = porcentagem_de_teste * len(Y)
tamanho_de_validacao = len(Y) - tamanho_de_treino - tamanho_de_teste

treino_dados = X[:int(tamanho_de_treino)]
treino_marcacoes = Y[:int(tamanho_de_treino)]

fim_de_treino = tamanho_de_treino + tamanho_de_teste

teste_dados = X[int(tamanho_de_treino):int(fim_de_treino)]
teste_marcacoes = Y[int(tamanho_de_treino):int(fim_de_treino)]

validacao_dados = X[int(fim_de_treino):]
validacao_marcacoes = Y[int(fim_de_treino):]

scores_treino = [0]
scores_teste = [0]

def escrever_resultado_real_treino(marcacoes):
	ws_treino.write(0, 0, "publico real")
	linha = 1
	for index in range(0,len(marcacoes)):	
		ws_treino.write(linha, 0, marcacoes[index])
		linha = linha + 1

def escrever_resultado_real_teste(marcacoes):
	ws_teste.write(0, 0, "publico real")
	linha = 1
	for index in range(0,len(marcacoes)):	
		ws_teste.write(linha, 0, marcacoes[index])
		linha = linha + 1

def escrever_resultado_em_excel_treino(nome_modelo,codModelo,marcacoes,resultado):
	ws_treino.write(0, codModelo, nome_modelo)
	linha = 1
	for index in range(0,len(resultado)):	
		ws_treino.write(linha, codModelo, resultado[index])
		linha = linha + 1	

def escrever_resultado_em_excel_teste(nome_modelo,codModelo,resultado):
	ws_teste.write(0, codModelo, nome_modelo)
	linha = 1
	for index in range(0,len(resultado)):	
		ws_teste.write(linha, codModelo, resultado[index])
		linha = linha + 1

def escrever_scores_em_excel_treino(codModelo):
	ws_treino.write(39, codModelo, scores_treino[codModelo])

def escrever_scores_em_excel_teste(codModelo):
	ws_teste.write(39, codModelo, scores_teste[codModelo])

def fit_and_predict(nome,codModelo, modelo, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes):
	modelo.fit(treino_dados, treino_marcacoes)
	print("Score no treino:", nome)
	print(modelo.score(treino_dados, treino_marcacoes))

	resultado = modelo.predict(teste_dados)
	
	escrever_resultado_em_excel_treino(nome,codModelo, teste_marcacoes, resultado)
	
	
	acertos = resultado == teste_marcacoes
	
	total_de_acertos = sum(acertos)
	total_de_elementos = len(teste_dados)
	
	taxa_de_acerto = 100.0 * total_de_acertos / total_de_elementos

	scores_treino.append(taxa_de_acerto)
	escrever_scores_em_excel_treino(codModelo)
	
	msg = "Taxa de acerto do algoritmo {0} na fase de validação: {1}".format(nome, taxa_de_acerto)
	
	print(msg)
	from sklearn.metrics import classification_report, confusion_matrix
	print(classification_report(teste_marcacoes,resultado))
	print(confusion_matrix(teste_marcacoes,resultado))

	return taxa_de_acerto

def teste_real(nome,codModelo,modelo, validacao_dados, validacao_marcacoes):
	resultado = modelo.predict(validacao_dados)
	model = Modelo()
	model.setNome(nome)
	model.setResultado(resultado)

	escrever_resultado_em_excel_teste(nome,codModelo, resultado)
	
	acertos = resultado == validacao_marcacoes
	
	total_de_acertos = sum(acertos)
	total_de_elementos = len(validacao_marcacoes)
	
	taxa_de_acerto = 100.0 * total_de_acertos / total_de_elementos

	model.setScore(taxa_de_acerto)
	modelos[codModelo] = model

	scores_teste.append(taxa_de_acerto)
	escrever_scores_em_excel_teste(codModelo)
	
	msg = "Taxa de acerto do algoritmo {0} na fase de testes: {1}".format(nome,taxa_de_acerto)
	print(msg)
	
	from sklearn.metrics import classification_report, confusion_matrix
	print(classification_report(validacao_marcacoes,resultado))
	print(confusion_matrix(validacao_marcacoes,resultado))
	#print(pd.crosstab(resultado, modelo.predict(validacao_marcacoes), rownames=['Real'], colnames=['Predito'], margins=True))
	
def predict_votacao(validacao_marcacoes):
	acerto_final = []
	resultados_vencedores = []
	for index in range(0,len(validacao_marcacoes)):
		vencedor = declarar_vencedor(index)
		if (vencedor == validacao_marcacoes[index]):
			acerto_final.append(1)
		else:
			acerto_final.append(0)
		resultados_vencedores.append(vencedor)

	escrever_resultado_em_excel_teste("comite",10,resultados_vencedores)
		
	total_de_acertos = sum(acerto_final)
	total_de_elementos = len(validacao_marcacoes)
	
	taxa_de_acerto = 100.0 * total_de_acertos / total_de_elementos
	ws_teste.write(39, 10, taxa_de_acerto)

	msg = "Taxa de acerto do comite: {0}".format(taxa_de_acerto)
	print(msg)
	
def declarar_vencedor(index_array):
	resultados = []
	scores = []

	for index_modelo in range(1,len(modelos)+1):
		resultado = modelos[index_modelo].getResultado()
		score = modelos[index_modelo].getScore()
		resultados.append(resultado[index_array])
		scores.append(score)

	resultCounter = Counter(resultados).most_common(2)
	primeiroResultado = resultCounter[0][0]
	qtdprimeiroResultado = resultCounter[0][1]
	if(len(resultCounter)>1):
		segundoResultado = resultCounter[1][0]
		qtdsegundoResultado = resultCounter[1][1]
		if(qtdprimeiroResultado == qtdsegundoResultado):
			maiorScore = max(scores)
			for index_model in range(1,len(modelos)+1):
				if(modelos[index_model].getScore() == maiorScore):
					resultado = modelos[index_model].getResultado()
					if(primeiroResultado == resultado[index_array]):
						return primeiroResultado
					elif(segundoResultado == resultado[index_array]):
						return segundoResultado
					else:
						return resultado[index_array]
		elif(qtdprimeiroResultado > qtdsegundoResultado):
			return primeiroResultado
	else:
		return primeiroResultado

#def declarar_vencedor(index_array):
#    import heapq
#    scores = []
#    resultados = []
#
#    for index_modelo in range(1,len(modelos)+1):
#        score = modelos[index_modelo].getScore()
#        scores.append(score)
#     
#    maxscores = heapq.nlargest(3,scores)
#
#    modelos_finais = []
#    for index_modelo in range(1,len(modelos)+1):
#        if(modelos[index_modelo].getScore() in maxscores):
#            modelos_finais.append(modelos[index_modelo])
#            resultado = modelos[index_modelo].getResultado()
#            resultados.append(resultado[index_array])
#    
#    resultCounter = Counter(resultados).most_common(3)
#    primeiroResultado = resultCounter[0][0]
#    qtdprimeiroResultado = resultCounter[0][1]
#
#    if(qtdprimeiroResultado > 1):
#        return primeiroResultado
#    else:
#        segundoResultado = resultCounter[1][0]
#        terceiroResultado = resultCounter[2][0]
#        maiorScore = max(scores)
#        for index_model in range(1,len(modelos)+1):
#            if(modelos[index_model].getScore() == maiorScore):
#                resultado = modelos[index_model].getResultado()
#                if(primeiroResultado == resultado[index_array]):
#                    return primeiroResultado
#                if(segundoResultado == resultado[index_array]):
#                    return segundoResultado
#                if(terceiroResultado == resultado[index_array]):
#                    return terceiroResultado
					

resultados = {}


#from sklearn.multiclass import OneVsRestClassifier
#from sklearn.svm import LinearSVC
#modeloOneVsRest = OneVsRestClassifier(LinearSVC(random_state = 0))
#resultadoOneVsRest = fit_and_predict("OneVsRest", modeloOneVsRest, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes)
#resultados[resultadoOneVsRest] = modeloOneVsRest
#
#from sklearn.multiclass import OneVsOneClassifier
#modeloOneVsOne = OneVsOneClassifier(LinearSVC(random_state = 0))
#resultadoOneVsOne = fit_and_predict("OneVsOne", modeloOneVsOne, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes)
#resultados[resultadoOneVsOne] = modeloOneVsOne
#

#
#
#
escrever_resultado_real_treino(teste_marcacoes)
from sklearn.ensemble import RandomForestClassifier
modeloRandomForrest = RandomForestClassifier(n_estimators=10)
resultadoRandomForrest = fit_and_predict("RandomForestClassifier 10 arvores",1, modeloRandomForrest, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes)
resultados[resultadoRandomForrest] = modeloRandomForrest

from sklearn.ensemble import RandomForestClassifier
modeloRandomForrest200 = RandomForestClassifier(n_estimators=50)
resultadoRandomForrest200 = fit_and_predict("RandomForestClassifier 50 arvores",2, modeloRandomForrest200, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes)
resultados[resultadoRandomForrest200] = modeloRandomForrest200

from sklearn.ensemble import RandomForestClassifier
modeloRandomForrest300 = RandomForestClassifier(n_estimators=100)
resultadoRandomForrest300 = fit_and_predict("RandomForestClassifier 100 arvores",3, modeloRandomForrest300, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes)
resultados[resultadoRandomForrest300] = modeloRandomForrest300

from sklearn.neighbors import KNeighborsClassifier
modeloKnn = KNeighborsClassifier(n_neighbors=1)
resultadoKnn = fit_and_predict("KnnClassifier 1 vizinho",4, modeloKnn, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes)
resultados[resultadoKnn] = modeloKnn

from sklearn.neighbors import KNeighborsClassifier
modeloKnn6 = KNeighborsClassifier(n_neighbors=6)
resultadoKnn6 = fit_and_predict("KnnClassifier 6 vizinhos",5, modeloKnn6, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes)
resultados[resultadoKnn6] = modeloKnn6

from sklearn.neighbors import KNeighborsClassifier
modeloKnn11 = KNeighborsClassifier(n_neighbors=11)
resultadoKnn11 = fit_and_predict("KnnClassifier 11 vizinhos",6, modeloKnn11, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes)
resultados[resultadoKnn11] = modeloKnn11

from sklearn.svm import SVC
modeloSvmlinear = SVC(kernel='linear')
resultadoSvmlinear = fit_and_predict("SvmClassifier Linear",7, modeloSvmlinear, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes)
resultados[resultadoSvmlinear] = modeloSvmlinear

from sklearn.svm import SVC
modeloSvmsigmois = SVC(kernel='sigmoid')
resultadoSvmsigmoid = fit_and_predict("SvmClassifier Sigmoid",8, modeloSvmsigmois, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes)
resultados[resultadoSvmsigmoid] = modeloSvmsigmois

from sklearn.svm import SVC
modeloSvmpoly = SVC(kernel='poly')
resultadoSvmpoly = fit_and_predict("SvmClassifier Polinomial",9, modeloSvmpoly, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes)
resultados[resultadoSvmpoly] = modeloSvmpoly

#from sklearn.ensemble import AdaBoostClassifier
#modeloAdaBoost = AdaBoostClassifier()
#resultadoAdaBoost = fit_and_predict("AdaBoostClassifier",8, modeloAdaBoost, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes)
#resultados[resultadoAdaBoost] = modeloAdaBoost

#from sklearn.naive_bayes import MultinomialNB
#modeloMultinomial = MultinomialNB()
#resultadoMultinomial = fit_and_predict("MultinomialNB",6, modeloMultinomial, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes)
#resultados[resultadoMultinomial] = modeloMultinomial

#from sklearn.tree import DecisionTreeClassifier
#modeloDecisionTree = DecisionTreeClassifier()
#resultadoDecisionTree= fit_and_predict("DecisionTreeClassifier",7, modeloDecisionTree, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes)
#resultados[resultadoDecisionTree] = modeloDecisionTree

#from sklearn.naive_bayes import GaussianNB
#modelo_naives_bayes = GaussianNB()
#resultado_naive_bayes = fit_and_predict("Naive bayes Gaussiana",8, modelo_naives_bayes, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes)
#resultados[resultado_naive_bayes] = modelo_naives_bayes

maximo = max(resultados)
vencedor = resultados[maximo]

print ("Vencedor: ",vencedor)

escrever_resultado_real_teste(validacao_marcacoes)
teste_real("RandomForestClassifier 10",1,modeloRandomForrest, validacao_dados, validacao_marcacoes)
teste_real("RandomForestClassifier 50",2,modeloRandomForrest200, validacao_dados, validacao_marcacoes)
teste_real("RandomForestClassifier 100",3,modeloRandomForrest300, validacao_dados, validacao_marcacoes)
teste_real("KnnClassifier",4,modeloKnn, validacao_dados, validacao_marcacoes)
teste_real("KnnClassifier 6 vizinhos",5,modeloKnn6, validacao_dados, validacao_marcacoes)
teste_real("KnnClassifier 11 vizinhos",6,modeloKnn11, validacao_dados, validacao_marcacoes)
teste_real("SvmClassifier Linear",7,modeloSvmlinear, validacao_dados, validacao_marcacoes)
teste_real("SvmClassifier Sigmoid",8,modeloSvmsigmois, validacao_dados, validacao_marcacoes)
teste_real("SvmClassifier Polinomial",9,modeloSvmpoly, validacao_dados, validacao_marcacoes)
#teste_real("AdaBoostClassifier",8,modeloAdaBoost, validacao_dados, validacao_marcacoes)
#teste_real("Decision Tree",7,modeloDecisionTree, validacao_dados, validacao_marcacoes)
#teste_real("Naive Bayes Gaussiana",8,modelo_naives_bayes, validacao_dados, validacao_marcacoes)

predict_votacao(validacao_marcacoes)

planilha_resultados.save('escrever_resultados_variados.xls')

total_de_elementos = len(validacao_dados)
print("Total de teste: %d" % total_de_elementos)

#from dados import carregar_pessoa_interessadas

#interessadas_facebook = carregar_pessoa_interessadas(tamanho_de_treino)
#msg_base = "Taxa de acerto base: {0}".format(interessadas_facebook)
#print(msg_base)