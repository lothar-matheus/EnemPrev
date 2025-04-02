# Previsão de Notas do Enem

## Descrição

O projeto **Previsão de ENEM** utiliza o algoritmo **K-Nearest Neighbors (KNN)** para prever a nota de um candidato do ENEM com base em fatores socioeconômicos. Os dados utilizados são provenientes do **ENEM 2023** e passam por um processo de tratamento, análise e validação antes da previsão.


## Observações

 Antes de dar início a analise do software desenvolvido e suas principais funcionalidades, é de extrema importância relatar que esse projeto não tem, como foi pedido nas especificações do trabalho, as tabelas de renda familiar(q006) e escolaridade da mãe, devido a problemas de conflitos com alguns dados das mesmas.

## Funcionalidades

* **Carregamento de dados** : Importa os dados do ENEM 2023.
* **Processamento de dados** : Realiza a limpeza e transformação dos dados para adequação ao modelo.
* **Análise exploratória** : Gera estatísticas descritivas e insights sobre os fatores socioeconômicos e suas relações com as notas.
* **Treinamento do modelo KNN** : Implementa o algoritmo KNN para prever a nota com base em padrões identificados nos dados.
* **Validação do modelo** : Avalia a precisão das previsões utilizando métricas apropriadas.
* **Exportação de resultados** : Salva as previsões para futuras análises.

#### **Sobre o Algoritmo KNN**

O KNN é um algoritmo de aprendizado supervisionado que classifica ou faz previsões com base nos "K" vizinhos mais próximos de um ponto de dados. Ele segue os seguintes passos:

1. Calcula a distância entre um novo candidato e todos os outros candidatos do conjunto de treinamento.
2. Seleciona os **K** vizinhos mais próximos.
3. Para regressão, calcula a média das notas dos vizinhos para prever a nota do novo candidato.

#### **Cálculos Utilizados na Função**

* **Métrica de Distância** : Utiliza a **distância Euclidiana** para medir a proximidade entre os candidatos.
* **Normalização** : Os dados são normalizados para garantir que todas as variáveis estejam na mesma escala.
* **Divisão de Dados** : Separa os dados em conjunto de treinamento e teste.
* **Validação** : Mede a precisão do modelo utilizando métricas como **erro absoluto médio (MAE)** e  **erro quadrático médio (MSE)** .

##Principios do algorítimo

### `Armazenamento:`

O algoritmo memoriza todos os exemplos de treinamento (características e valores-alvo).

### `Calculo da distância:`

Quando um novo exemplo é apresentado, o algoritmo calcula a distância entre este novo ponto e todos os pontos do conjunto de treinamento.

### `Seleção dos vizinhos:`

Seleciona os K exemplos mais próximos (vizinhos).

### `Previsão:`

Para classificação: utiliza a classe mais frequente entre os K vizinhos.
Para regressão (nosso caso): calcula a média (ou outra medida de tendência central) dos valores-alvo dos K vizinhos.

## Estrutura do Código

O projeto possui as seguintes funcionalidades para o usuário final:

### `Entrada de dados customisável`

Carrega e organiza os dados do ENEM 2023 para análise.

### `Ajuste do parâmetro para o usuário final`


Visualização textual da previsão das notas.

### `Gráficos comparativos e analise de vizinhança`

Treina o modelo **K-Nearest Neighbors (KNN)** utilizando os dados processados e realiza previsões de notas e gera gráficos comparativos.


Avalia o desempenho do modelo comparando as previsões com os valores reais.

Salva os resultados das previsões para um arquivo de saída (CSV, JSON, etc.).

##Implementação do código
Neste projeto, utilizamos o K-NN para regressão, onde a meta é prever valores numéricos (notas do ENEM)
baseados em características categóricas dos estudantes. Principais aspectos da implementação:

### Pré-processamento de dados:


## 1. Tecnologias Utilizadas

* **Codificação de variáveis categóricas usando LabelEncoder** 
* **Normalização dos dados com StandardScaler** 
* **Normalização dos dados com StandardScaler** 
* ** Tratamento de valores ausentes**

## 2. Parametizaçã

* **O usuário pode definir o valor de K (número de vizinhos a considerar)** 
* **A métrica de distância utilizada é a distância euclidiana (padrão da implementação sklearn)** 

## 3. Modelos independentes

* **Um modelo K-NN separado é treinado para cada área de conhecimento do ENEM** 
* **Esta abordagem permite capturar relações específicas entre as características do estudante e cada componente da prova*

## 4. Analise de Vizinhaça

* **Além da previsão, o sistema mostra a média dos K vizinhos mais próximos** 
* **Permite uma comparação direta entre a previsão e os casos similares no conjunto de dados*

* **Linguagem:** Python
* **Bibliotecas:** Pandas e NumPy: Manipulaçao e processamento de dados,
Scikit-learn:Implementação do algoritmo K-NN e ferramentas de pré-processamento,
Matplotlib:Geração de gráficos e visualizações,
Tkinter: Construção da interface gráfica

## Preparação dos Dados
O sistema requer um arquivo CSV com dados do ENEM no formato especificado:

*O arquivo deve ser nomeado "MICRODADOS_ENEM_2023_EDITADO.csv"
*Deve conter as colunas mencionadas no código, incluindo:

*Características categóricas: TP_COR_RACA, TP_ESCOLA, TP_ENSINO, SG_UF_ESC, etc.
*Colunas de notas: NU_NOTA_CN, NU_NOTA_CH, NU_NOTA_LC, NU_NOTA_MT, NU_NOTA_REDACAO


## Como Executar

1. Instale as dependências necessárias:
   ```bash
   pip install pandas numpy scikit-learn matplotlib
   ```
2. Execute o script principal:
   ```bash
   python apv.py
   ```
OBS. Coloque o arquivo csv na mesma pasta do projeto. Para facilitar a execução do projeto, recomenda-se usar o pycharn 

## Autor

Projeto desenvolvido por Matheus Lemos.
