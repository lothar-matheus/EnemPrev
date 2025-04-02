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

## O Algoritmo K-NN (K-Nearest Neighbors)

 O K-NN é um algoritmo de aprendizado supervisionado utilizado tanto para classificação quanto para regressão.
Diferente de outros algoritmos que constroem um modelo generalizado durante o treinamento.

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

O projeto possui as seguintes funcionalidades principais:

### `Entrada de dados customisável`

Carrega e organiza os dados do ENEM 2023 para análise.

### `Ajuste do parâmetro para o usuário final`


### `Visualização()`

Visualização textual da previsão das notas.

### `Gráficos comparativos e analise de vizinhança`

Treina o modelo **K-Nearest Neighbors (KNN)** utilizando os dados processados e realiza previsões de notas e gera gráficos comparativos.

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

### `validate_results()`

Avalia o desempenho do modelo comparando as previsões com os valores reais.

### `export_results()`

Salva os resultados das previsões para um arquivo de saída (CSV, JSON, etc.).

## Tecnologias Utilizadas

* **Linguagem:** Python
* **Bibliotecas:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn

## Como Executar

1. Instale as dependências necessárias:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
   ```
2. Execute o script principal:
   ```bash
   python apv.py
   ```

## Autor

Projeto desenvolvido por [Seu Nome].
