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


1. *Carregamento e Limpeza*:
   python
   self.df = pd.read_csv(self.arquivo_csv, sep=';', encoding='latin1')
   
   # Remover linhas com valores ausentes nas colunas alvo
   for coluna in self.colunas_alvo:
       self.df = self.df[~self.df[coluna].isna()]
   
   
   Esta etapa garante que apenas registros completos sejam utilizados no treinamento, evitando viés por dados faltantes.

2. *Transformação de Dados*:
   python
   # Converter as colunas numéricas para o tipo correto
   for coluna in self.colunas_alvo:
       self.df[coluna] = pd.to_numeric(self.df[coluna], errors='coerce')
       
   # Converter colunas categóricas
   for coluna in self.colunas_categoricas:
       self.df[coluna] = self.df[coluna].astype(str)
   
   
   Esta etapa assegura que os tipos de dados estejam corretos para processamento.

3. *Codificação de Variáveis Categóricas*:
   python
   # Codificar variáveis categóricas
   for coluna in self.colunas_categoricas:
       le = LabelEncoder()
       self.df[coluna] = le.fit_transform(self.df[coluna])
       self.label_encoders[coluna] = le
   
   
   O LabelEncoder transforma as categorias textuais em valores numéricos, necessários para o cálculo de distâncias no K-NN. É importante notar que esta codificação preserva a estrutura dos dados sem introduzir relações ordinais artificiais.

4. *Normalização de Features*:
   python
   X_scaled = self.scaler.fit_transform(X)
   
   
   A normalização é crucial para o K-NN, pois evita que features com escalas maiores dominem o cálculo de distância. O StandardScaler aplica a transformação Z-score, tornando todas as features com média 0 e desvio padrão 1.

### Mecanismo de Previsão e sua Justificativa

O sistema realiza previsões seguindo um processo estatisticamente fundamentado:

#### 1. Construção do Espaço de Features

As características categóricas do estudante são transformadas em um ponto num espaço multidimensional. Por exemplo, um estudante de escola pública, ensino regular, no estado de São Paulo, seria representado como um vetor de valores numéricos após o pré-processamento:


[valor_codificado_raca, valor_codificado_escola, valor_codificado_ensino, 
 valor_codificado_uf, valor_codificado_dependencia, valor_codificado_localizacao]


Este vetor normalizado define a "posição" do estudante no espaço de características.

#### 2. Cálculo de Distâncias

A distância euclidiana é calculada entre o ponto que representa o estudante em questão e todos os pontos do conjunto de treinamento:


distância = √(∑(xi - yi)²)


Onde xi é o valor da característica i para o novo estudante e yi é o valor da característica i para um estudante do conjunto de treinamento.

#### 3. Determinação dos K Vizinhos Mais Próximos

Os K estudantes com menor distância euclidiana são selecionados como "vizinhos mais próximos". A interpretação é que estes são os K estudantes mais similares ao perfil do estudante em questão, considerando as características fornecidas.

#### 4. Cálculo da Previsão

A previsão final para cada área de conhecimento é calculada como a média aritmética das notas dos K vizinhos mais próximos:

python
previsao = model.predict(dados_normalizados)[0]


Onde model é o modelo K-NN treinado para a respectiva área de conhecimento.

#### 5. Análise Comparativa com Vizinhos

O sistema vai além da simples previsão, oferecendo uma análise comparativa:

python
vizinhos_indices = model.kneighbors(dados_normalizados, return_distance=False)[0]
notas_vizinhos = self.df.iloc[vizinhos_indices][coluna_alvo].values

### Justificativa Teórica e Prática das Previsões

A abordagem de utilizar o K-NN para previsão de notas do ENEM baseia-se em princípios estatísticos e pedagógicos sólidos:

#### 1. Fundamento Teórico:

O método parte da premissa de que estudantes com características sociodemográficas e escolares similares tendem a apresentar desempenhos acadêmicos comparáveis. Esta premissa é suportada por diversos estudos em sociologia da educação e psicometria educacional, que apontam correlações entre fatores socioeconômicos, demográficos e institucionais com o desempenho escolar.

#### 2. Relevância das Variáveis Selecionadas:

As variáveis preditoras escolhidas capturam diferentes dimensões que podem influenciar o desempenho no ENEM:

- *Dimensão Individual*: Representada pela variável de raça/cor, que pode estar associada a diferentes experiências educacionais no contexto brasileiro.
- *Dimensão Institucional*: Capturada pelas variáveis de tipo de escola, dependência administrativa e modalidade de ensino.
- *Dimensão Geográfica*: Representada pela UF e localização (urbana/rural), que refletem diferenças regionais nos sistemas educacionais.

#### 3. Valor do Parâmetro K:

A possibilidade de ajustar o valor de K introduz uma importante flexibilidade ao modelo:

- *K pequeno (ex: 3-5)*: Favorece a especificidade, capturando padrões mais localizados, porém aumenta a sensibilidade a outliers.
- *K grande (ex: 15-20)*: Favorece a generalização, reduzindo o impacto de outliers, mas pode "suavizar" excessivamente as previsões.

O sistema permite que o usuário experimente diferentes valores de K, observando como isto afeta as previsões. Esta característica é particularmente valiosa em um SAD, pois permite análises de sensibilidade e adaptação às necessidades específicas de cada aplicação.

#### 4. Interpretabilidade das Previsões:

Uma vantagem significativa da abordagem K-NN no contexto educacional é a transparência do processo preditivo:

- É possível identificar exatamente quais exemplos (estudantes do conjunto de dados) influenciaram a previsão
- A comparação com a média dos vizinhos oferece uma contextualização intuitiva da previsão
- Os gráficos gerados facilitam a visualização das notas previstas e sua comparação entre áreas de conhecimento

Esta interpretabilidade é fundamental em sistemas de apoio à decisão educacionais, onde confiança e compreensão do modelo são essenciais para sua adoção.

#### 5. Limitações Intrínsecas ao Método:

É importante destacar as limitações inerentes à abordagem:

- O modelo assume que a similaridade nas características selecionadas implica em similaridade no desempenho, o que pode não capturar toda a complexidade do processo educacional
- Fatores não mensurados como motivação, métodos de estudo, eventos pessoais e apoio familiar não são considerados, embora possam ter impacto significativo
- O conjunto de dados, mesmo sendo representativo, pode não capturar todas as nuances regionais e temporais que afetam o desempenho no ENEM


Estas limitações são explicitamente comunicadas ao usuário para garantir que as previsões sejam interpretadas como estimativas probabilísticas, não como determinações definitivas.

### Implicações Para Tomada de Decisão

As previsões geradas pelo sistema podem informar diferentes níveis de tomada de decisão:

1. *Nível Individual (Estudante)*:
   - Identificação de potenciais áreas de maior ou menor desempenho
   - Estabelecimento de metas realistas com base em dados históricos de estudantes com perfil similar
   - Oportunidade para refletir sobre como transcender limitações estruturais indicadas pelos dados

2. *Nível Institucional (Escola/Professor)*:
   - Identificação preventiva de grupos de estudantes que podem necessitar de suporte adicional
   - Personalização de estratégias pedagógicas conforme os padrões identificados
   - Alocação mais eficiente de recursos de reforço escolar com base em previsões de desempenho

3. *Nível Sistêmico (Gestores Educacionais)*:
   - Análise de disparidades regionais e socioeconômicas no desempenho esperado
   - Informação para formulação de políticas públicas direcionadas
   - Monitoramento de progresso em iniciativas de equidade educacional


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
