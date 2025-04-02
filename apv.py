import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time
import os
import threading


class EnemKNNPredictor:
    def __init__(self, arquivo_csv):
        self.arquivo_csv = arquivo_csv
        self.df = None
        self.X_train = None
        self.y_train = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.models = {}
        self.colunas_categoricas = ['TP_COR_RACA', 'TP_ESCOLA', 'TP_ENSINO',
                                    'SG_UF_ESC', 'TP_DEPENDENCIA_ADM_ESC',
                                    'TP_LOCALIZACAO_ESC']
        self.colunas_alvo = ['NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC',
                             'NU_NOTA_MT', 'NU_NOTA_REDACAO']

    def carregar_dados(self):
        """
        Carrega e faz o tratamento inicial dos dados do ENEM
        """
        print("Carregando dados...")
        try:
            self.df = pd.read_csv(self.arquivo_csv, sep=';', encoding='latin1')

            # Remover linhas com valores ausentes nas colunas alvo
            for coluna in self.colunas_alvo:
                self.df = self.df[~self.df[coluna].isna()]

            # Converter as colunas numéricas para o tipo correto
            for coluna in self.colunas_alvo:
                self.df[coluna] = pd.to_numeric(self.df[coluna], errors='coerce')

            # Converter colunas categóricas
            for coluna in self.colunas_categoricas:
                self.df[coluna] = self.df[coluna].astype(str)

            # Codificar variáveis categóricas
            for coluna in self.colunas_categoricas:
                le = LabelEncoder()
                self.df[coluna] = le.fit_transform(self.df[coluna])
                self.label_encoders[coluna] = le

            print(f"Dados carregados com sucesso. Total de registros: {len(self.df)}")
            return True
        except Exception as e:
            print(f"Erro ao carregar dados: {e}")
            return False

    def preparar_modelo(self, k=5):
        """
        Prepara o modelo K-NN para cada coluna alvo
        """
        print("Preparando modelo K-NN...")
        try:
            # Selecionar features e target
            features = self.colunas_categoricas

            # Dividir dados em treino e teste
            X = self.df[features]

            # Normalizar os dados
            X_scaled = self.scaler.fit_transform(X)
            self.X_train = X_scaled

            # Treinar um modelo para cada coluna alvo
            for coluna_alvo in self.colunas_alvo:
                y = self.df[coluna_alvo]
                self.y_train = y

                # Criar e treinar o modelo KNN
                model = KNeighborsRegressor(n_neighbors=k)
                model.fit(X_scaled, y)
                self.models[coluna_alvo] = model

            print("Modelos preparados com sucesso.")
            return True
        except Exception as e:
            print(f"Erro ao preparar modelo: {e}")
            return False

    def prever_notas(self, dados_entrada):
        """
        Fazendo a previsão de notas utilizando o modelo K-NN
        """
        try:
            # Preparar dados de entrada
            dados_processados = []

            for coluna in self.colunas_categoricas:
                if coluna in dados_entrada:
                    # Codificar valores categóricos
                    valor = dados_entrada[coluna]
                    if coluna in self.label_encoders:
                        le = self.label_encoders[coluna]
                        # Verificar se o valor existe no encoder
                        if valor in le.classes_:
                            valor_codificado = le.transform([valor])[0]
                        else:
                            # Se não existir, usar o mais comum
                            valor_codificado = le.transform([le.classes_[0]])[0]
                    else:
                        valor_codificado = 0
                    dados_processados.append(valor_codificado)
                else:
                    dados_processados.append(0)

            # Normalizar dados
            dados_normalizados = self.scaler.transform([dados_processados])

            # Fazer previsão para cada coluna alvo
            previsoes = {}
            for coluna_alvo in self.colunas_alvo:
                model = self.models[coluna_alvo]
                previsao = model.predict(dados_normalizados)[0]
                previsoes[coluna_alvo] = previsao

                # Encontrar os K vizinhos mais próximos
                vizinhos_indices = model.kneighbors(dados_normalizados, return_distance=False)[0]
                notas_vizinhos = self.df.iloc[vizinhos_indices][coluna_alvo].values
                previsoes[f"{coluna_alvo}_vizinhos"] = notas_vizinhos

            return previsoes
        except Exception as e:
            print(f"Erro ao prever notas: {e}")
            return None


class EnemKNNApp:
    def __init__(self, root, predictor):
        self.root = root
        self.predictor = predictor
        self.root.title("Sistema de Previsão de Notas do ENEM usando K-NN")
        self.root.geometry("1200x800")

        # Criar frames principais
        self.frame_entrada = ttk.LabelFrame(root, text="Dados do Aluno")
        self.frame_entrada.pack(fill="both", expand="yes", padx=10, pady=10)

        self.frame_resultados = ttk.LabelFrame(root, text="Resultados da Previsão")
        self.frame_resultados.pack(fill="both", expand="yes", padx=10, pady=10)

        self.progress_frame = ttk.Frame(root)
        self.progress_frame.pack(fill="x", padx=10, pady=10)

        self.progress = ttk.Progressbar(self.progress_frame, orient="horizontal", length=300, mode="determinate")
        self.progress.pack(side="left", padx=10, pady=10)

        self.status_label = ttk.Label(self.progress_frame, text="Status: Aguardando dados...")
        self.status_label.pack(side="left", padx=10, pady=10)

        # Criar widgets de entrada
        self.criar_campos_entrada()

        # Criar área para resultados
        self.criar_area_resultados()

    def criar_campos_entrada(self):
        # Dicionário para mapear valores numéricos para textos
        self.opcoes_cor_raca = {
            "0": "Não declarado",
            "1": "Branca",
            "2": "Preta",
            "3": "Parda",
            "4": "Amarela",
            "5": "Indígena",
            "6": "Não dispõe da informação"
        }

        self.opcoes_tipo_escola = {
            "1": "Não respondeu",
            "2": "Pública",
            "3": "Privada",
            "4": "Exterior"
        }

        self.opcoes_ensino = {
            "1": "Ensino Regular",
            "2": "Educação Especial",
            "3": "EJA"
        }

        self.opcoes_dependencia = {
            "1": "Federal",
            "2": "Estadual",
            "3": "Municipal",
            "4": "Privada"
        }

        self.opcoes_localizacao = {
            "1": "Urbana",
            "2": "Rural"
        }

        # Lista de UFs brasileiras
        self.opcoes_uf = {
            "AC": "Acre", "AL": "Alagoas", "AP": "Amapá", "AM": "Amazonas",
            "BA": "Bahia", "CE": "Ceará", "DF": "Distrito Federal", "ES": "Espírito Santo",
            "GO": "Goiás", "MA": "Maranhão", "MT": "Mato Grosso", "MS": "Mato Grosso do Sul",
            "MG": "Minas Gerais", "PA": "Pará", "PB": "Paraíba", "PR": "Paraná",
            "PE": "Pernambuco", "PI": "Piauí", "RJ": "Rio de Janeiro", "RN": "Rio Grande do Norte",
            "RS": "Rio Grande do Sul", "RO": "Rondônia", "RR": "Roraima", "SC": "Santa Catarina",
            "SP": "São Paulo", "SE": "Sergipe", "TO": "Tocantins"
        }

        # Frame para organizar os widgets em grid
        frame_grid = ttk.Frame(self.frame_entrada)
        frame_grid.pack(fill="both", padx=10, pady=10)

        # Linha 0
        ttk.Label(frame_grid, text="Raça/Cor:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.cor_raca_var = tk.StringVar()
        self.cor_raca_combo = ttk.Combobox(frame_grid, textvariable=self.cor_raca_var)
        self.cor_raca_combo['values'] = list(self.opcoes_cor_raca.values())
        self.cor_raca_combo.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        self.cor_raca_combo.current(0)

        ttk.Label(frame_grid, text="Tipo de Escola:").grid(row=0, column=2, padx=5, pady=5, sticky="w")
        self.tipo_escola_var = tk.StringVar()
        self.tipo_escola_combo = ttk.Combobox(frame_grid, textvariable=self.tipo_escola_var)
        self.tipo_escola_combo['values'] = list(self.opcoes_tipo_escola.values())
        self.tipo_escola_combo.grid(row=0, column=3, padx=5, pady=5, sticky="w")
        self.tipo_escola_combo.current(0)

        # Linha 1
        ttk.Label(frame_grid, text="Tipo de Ensino:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.tipo_ensino_var = tk.StringVar()
        self.tipo_ensino_combo = ttk.Combobox(frame_grid, textvariable=self.tipo_ensino_var)
        self.tipo_ensino_combo['values'] = list(self.opcoes_ensino.values())
        self.tipo_ensino_combo.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        self.tipo_ensino_combo.current(0)

        ttk.Label(frame_grid, text="UF da Escola:").grid(row=1, column=2, padx=5, pady=5, sticky="w")
        self.uf_var = tk.StringVar()
        self.uf_combo = ttk.Combobox(frame_grid, textvariable=self.uf_var)
        self.uf_combo['values'] = list(self.opcoes_uf.keys())
        self.uf_combo.grid(row=1, column=3, padx=5, pady=5, sticky="w")
        self.uf_combo.current(0)

        # Linha 2
        ttk.Label(frame_grid, text="Dependência Adm:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.dependencia_var = tk.StringVar()
        self.dependencia_combo = ttk.Combobox(frame_grid, textvariable=self.dependencia_var)
        self.dependencia_combo['values'] = list(self.opcoes_dependencia.values())
        self.dependencia_combo.grid(row=2, column=1, padx=5, pady=5, sticky="w")
        self.dependencia_combo.current(0)

        ttk.Label(frame_grid, text="Localização:").grid(row=2, column=2, padx=5, pady=5, sticky="w")
        self.localizacao_var = tk.StringVar()
        self.localizacao_combo = ttk.Combobox(frame_grid, textvariable=self.localizacao_var)
        self.localizacao_combo['values'] = list(self.opcoes_localizacao.values())
        self.localizacao_combo.grid(row=2, column=3, padx=5, pady=5, sticky="w")
        self.localizacao_combo.current(0)

        # Linha 3 - Parâmetro K
        ttk.Label(frame_grid, text="Número de Vizinhos (K):").grid(row=3, column=0, padx=5, pady=5, sticky="w")
        self.k_var = tk.StringVar(value="5")
        self.k_entry = ttk.Entry(frame_grid, textvariable=self.k_var, width=5)
        self.k_entry.grid(row=3, column=1, padx=5, pady=5, sticky="w")

        # Botão para fazer a previsão
        self.botao_prever = ttk.Button(frame_grid, text="Prever Notas", command=self.iniciar_previsao)
        self.botao_prever.grid(row=4, column=0, columnspan=4, padx=5, pady=20)

    def criar_area_resultados(self):
        # Frame para os resultados
        self.notebook = ttk.Notebook(self.frame_resultados)
        self.notebook.pack(fill="both", expand="yes", padx=10, pady=10)

        # Tab para resultados textuais
        self.tab_resultados = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_resultados, text="Resultados")

        # Tab para gráficos
        self.tab_graficos = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_graficos, text="Gráficos")

        # Tab para comparação
        self.tab_comparacao = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_comparacao, text="Comparação com Vizinhos")

        # Área de texto para resultados
        self.text_resultados = tk.Text(self.tab_resultados, height=15, width=80)
        self.text_resultados.pack(fill="both", expand="yes", padx=10, pady=10)

    def iniciar_previsao(self):
        """
        Inicia o processo de previsão em uma thread separada
        """
        self.botao_prever.config(state="disabled")
        self.progress["value"] = 0
        self.status_label.config(text="Status: Iniciando previsão...")

        # Iniciar em thread separada para não travar a interface
        threading.Thread(target=self.fazer_previsao, daemon=True).start()

    def fazer_previsao(self):
        """
        Realiza a previsão e atualiza a interface
        """
        try:
            # Atualizar barra de progresso
            self.atualizar_progresso(20, "Preparando dados...")

            # Obter valores selecionados
            cor_raca = str(
                list(self.opcoes_cor_raca.keys())[list(self.opcoes_cor_raca.values()).index(self.cor_raca_var.get())])
            tipo_escola = str(list(self.opcoes_tipo_escola.keys())[
                                  list(self.opcoes_tipo_escola.values()).index(self.tipo_escola_var.get())])
            tipo_ensino = str(
                list(self.opcoes_ensino.keys())[list(self.opcoes_ensino.values()).index(self.tipo_ensino_var.get())])
            uf = self.uf_var.get()
            dependencia = str(list(self.opcoes_dependencia.keys())[
                                  list(self.opcoes_dependencia.values()).index(self.dependencia_var.get())])
            localizacao = str(list(self.opcoes_localizacao.keys())[
                                  list(self.opcoes_localizacao.values()).index(self.localizacao_var.get())])

            # Valor de K
            try:
                k = int(self.k_var.get())
                if k <= 0:
                    raise ValueError("K deve ser positivo")
            except:
                self.root.after(0, lambda: messagebox.showerror("Erro", "Valor inválido para K. Usando K=5."))
                k = 5
                self.k_var.set("5")

            # Preparar dados de entrada
            dados_entrada = {
                'TP_COR_RACA': cor_raca,
                'TP_ESCOLA': tipo_escola,
                'TP_ENSINO': tipo_ensino,
                'SG_UF_ESC': uf,
                'TP_DEPENDENCIA_ADM_ESC': dependencia,
                'TP_LOCALIZACAO_ESC': localizacao
            }

            # Preparar modelo com o valor de K escolhido
            self.atualizar_progresso(40, "Preparando modelo K-NN...")
            self.predictor.preparar_modelo(k=k)

            # Fazer previsão
            self.atualizar_progresso(60, "Calculando previsões...")
            previsoes = self.predictor.prever_notas(dados_entrada)

            if previsoes:
                self.atualizar_progresso(80, "Gerando resultados...")

                # Limpar resultados anteriores
                self.text_resultados.delete(1.0, tk.END)

                # Mostrar resultados no widget Text
                self.text_resultados.insert(tk.END, "=== PREVISÃO DE NOTAS DO ENEM ===\n\n")
                self.text_resultados.insert(tk.END, f"Dados do aluno:\n")
                self.text_resultados.insert(tk.END, f"- Raça/Cor: {self.cor_raca_var.get()}\n")
                self.text_resultados.insert(tk.END, f"- Tipo de Escola: {self.tipo_escola_var.get()}\n")
                self.text_resultados.insert(tk.END, f"- Tipo de Ensino: {self.tipo_ensino_var.get()}\n")
                self.text_resultados.insert(tk.END, f"- UF da Escola: {uf}\n")
                self.text_resultados.insert(tk.END, f"- Dependência Administrativa: {self.dependencia_var.get()}\n")
                self.text_resultados.insert(tk.END, f"- Localização: {self.localizacao_var.get()}\n\n")

                self.text_resultados.insert(tk.END, "Notas previstas:\n")

                # Nomes amigáveis para as disciplinas
                nomes_disciplinas = {
                    'NU_NOTA_CN': 'Ciências da Natureza',
                    'NU_NOTA_CH': 'Ciências Humanas',
                    'NU_NOTA_LC': 'Linguagens e Códigos',
                    'NU_NOTA_MT': 'Matemática',
                    'NU_NOTA_REDACAO': 'Redação'
                }

                # Exibir as notas previstas
                for coluna_alvo in self.predictor.colunas_alvo:
                    nota_prevista = previsoes[coluna_alvo]
                    notas_vizinhos = previsoes[f"{coluna_alvo}_vizinhos"]
                    media_vizinhos = np.mean(notas_vizinhos)

                    # Determinar se a nota é maior ou menor que a média dos vizinhos
                    comparacao = "IGUAL À" if abs(
                        nota_prevista - media_vizinhos) < 0.01 else "MAIOR QUE" if nota_prevista > media_vizinhos else "MENOR QUE"

                    self.text_resultados.insert(tk.END,
                                                f"- {nomes_disciplinas[coluna_alvo]}: {nota_prevista:.1f} ({comparacao} a média dos vizinhos: {media_vizinhos:.1f})\n")

                # Calcular média geral
                notas_gerais = [previsoes[coluna] for coluna in self.predictor.colunas_alvo]
                media_geral = np.mean(notas_gerais)
                self.text_resultados.insert(tk.END, f"\nMédia Geral Prevista: {media_geral:.1f}\n")

                # Criar gráficos
                self.atualizar_progresso(90, "Gerando gráficos...")
                self.criar_graficos(previsoes, nomes_disciplinas)
                self.criar_grafico_comparacao(previsoes, nomes_disciplinas)

                self.atualizar_progresso(100, "Previsão concluída com sucesso!")
            else:
                self.root.after(0, lambda: messagebox.showerror("Erro",
                                                                "Não foi possível fazer a previsão. Verifique os dados."))
                self.atualizar_progresso(0, "Erro na previsão.")
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Erro", f"Ocorreu um erro: {str(e)}"))
            self.atualizar_progresso(0, f"Erro: {str(e)}")
        finally:
            self.botao_prever.config(state="normal")

    def atualizar_progresso(self, valor, texto):
        """
        Atualiza a barra de progresso e o texto de status
        """
        self.root.after(0, lambda: self.progress.config(value=valor))
        self.root.after(0, lambda: self.status_label.config(text=f"Status: {texto}"))
        time.sleep(0.5)  # Simular processamento

    def criar_graficos(self, previsoes, nomes_disciplinas):
        """
        Cria gráficos para visualização dos resultados
        """
        # Limpar frame de gráficos
        for widget in self.tab_graficos.winfo_children():
            widget.destroy()

        # Criar figura para o gráfico
        fig, ax = plt.subplots(figsize=(10, 6))

        # Dados para o gráfico
        disciplinas = []
        notas = []

        for coluna in self.predictor.colunas_alvo:
            disciplinas.append(nomes_disciplinas[coluna])
            notas.append(previsoes[coluna])

        # Criar gráfico de barras
        bars = ax.bar(disciplinas, notas, color='skyblue')

        # Adicionar valores nas barras
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{height:.1f}',
                    ha='center', va='bottom')

        ax.set_ylim(0, 1000)  # Limite para notas do ENEM
        ax.set_ylabel('Nota Prevista')
        ax.set_title('Previsão de Notas por Disciplina')

        # Adicionar o gráfico à interface
        canvas = FigureCanvasTkAgg(fig, master=self.tab_graficos)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill="both", expand=True)
        canvas.draw()

    def criar_grafico_comparacao(self, previsoes, nomes_disciplinas):
        """
        Cria gráfico de comparação com os vizinhos
        """
        # Limpar frame de comparação
        for widget in self.tab_comparacao.winfo_children():
            widget.destroy()

        # Criar figura para o gráfico
        fig, ax = plt.subplots(figsize=(10, 6))

        # Preparar dados para o gráfico
        disciplinas = []
        nota_prevista = []
        media_vizinhos = []

        width = 0.35  # largura das barras

        for coluna in self.predictor.colunas_alvo:
            disciplinas.append(nomes_disciplinas[coluna])
            nota_prevista.append(previsoes[coluna])
            media_vizinhos.append(np.mean(previsoes[f"{coluna}_vizinhos"]))

        # Posições das barras
        x = np.arange(len(disciplinas))

        # Criar barras
        bars1 = ax.bar(x - width / 2, nota_prevista, width, label='Nota Prevista', color='skyblue')
        bars2 = ax.bar(x + width / 2, media_vizinhos, width, label='Média dos Vizinhos', color='lightcoral')

        # Adicionar valores nas barras
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height,
                        f'{height:.1f}',
                        ha='center', va='bottom', fontsize=8)

        # Configurar eixos
        ax.set_ylabel('Nota')
        ax.set_title('Comparação: Nota Prevista vs. Média dos Vizinhos')
        ax.set_xticks(x)
        ax.set_xticklabels(disciplinas)
        ax.legend()
        ax.set_ylim(0, 1000)  # Limite para notas do ENEM

        # Adicionar o gráfico à interface
        canvas = FigureCanvasTkAgg(fig, master=self.tab_comparacao)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill="both", expand=True)
        canvas.draw()


def main():
    try:
        # Caminho para o arquivo CSV
        arquivo_csv = "MICRODADOS_ENEM_2023_EDITADO.csv"

        # Verificar se o arquivo existe
        if not os.path.exists(arquivo_csv):
            print(f"Erro: O arquivo {arquivo_csv} não foi encontrado.")
            messagebox.showerror("Erro", f"O arquivo {arquivo_csv} não foi encontrado.")
            return

        # Inicializar o preditor
        predictor = EnemKNNPredictor(arquivo_csv)

        # Carregar os dados
        if not predictor.carregar_dados():
            messagebox.showerror("Erro", "Não foi possível carregar os dados. Verifique o arquivo CSV.")
            return

        # Criar interface gráfica
        root = tk.Tk()
        app = EnemKNNApp(root, predictor)
        root.mainloop()
    except Exception as e:
        print(f"Erro inesperado: {e}")
        messagebox.showerror("Erro", f"Ocorreu um erro inesperado: {str(e)}")


if __name__ == "__main__":
    main()