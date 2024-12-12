from pypdf import PdfReader
from owlready2 import *
from langchain.tools import Tool
from langchain_google_genai import ChatGoogleGenerativeAI
from crewai import Agent, Task, Crew, Process

import streamlit as st

# IMPORTANTO MODELO DE LLM
os.environ["GOOGLE_API_KEY"] = st.secrets["GEMINI_API_KEY"]
llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature = 1)


def read_pdf(file_pdf):
    reader = PdfReader(file_pdf)
    texto_pdf = ""
    for page in reader.pages: 
        texto_pdf += page.extract_text()
    return texto_pdf
# TESTE read_pdf
#text = read_pdf("article.pdf")
#print(text)

############# TOOL1
read_pdf_tool = Tool(
    name="Leitor de arquivos PDF",
    description="Esta ferramenta possui um argumento com o valor {file_pdf} e retorna o conteúdo do arquivo em formato de texto simples.",
    func=lambda file_pdf: read_pdf(file_pdf),
)

def entrega_texto_artigo():
    return texto_artigo

adquiri_artigo_tool = Tool(
    name="Ferramenta que adquire um artigo",
    description="Esta ferramenta retorna um texto de artigo.",
    func=lambda: entrega_texto_artigo(),
)

def entrega_texto_ontologia():
    return texto_onto

adquiri_ontologia_tool = Tool(
    name="Ferramenta que adquire uma ontologia",
    description="Esta ferramenta retorna um texto de uma ontologia em owl.",
    func=lambda: entrega_texto_ontologia(),
)



############# AGENT1
agente_extrator_de_conceitos_de_artigos = Agent(
    role="Agente extrator de conceitos do arquivo {file_pdf}",
    goal="Ler o texto do artigo científico contido no arquivo {file_pdf} e extrair vários conceitos relevantes sotre {topic} para análise subsequente por outro agente.",
    backstory="Você é um especialista em {topic} em extrair vários conceitos de artigos científicos. Seu papel é garantir que todos os conceitos relevantes do arquivo {file_pdf} sobre o tema {topic} sejam extraídos com precisão, facilitando a análise subsequente por outros agentes.",
    verbose=True,
    llm=llm,
    max_iter=3,
    memory=True,
    tools=[read_pdf_tool]
)

############# TASK1
tarefa_extrair_conceitos_sobre_um_topico = Task(
    description="Extrair vários conceitos relevantes do arquivo {file_pdf} para análise subsequente.",
    expected_output="""Retorna um documento em inglês formatado em marckdonw contendo o título do artigo e uma tabela no formato markdown contendo as colunas: conceito e explicação. Onde conceito são os conceitos importantes extraídos de {file_pdf} e explicação é um parágrafo extraído de {file_pdf} onde consta o uso do conceito.
    e.g.
    # Título do artigo
    | Concept | Explanation |
    """,
    agent=agente_extrator_de_conceitos_de_artigos,
    #output_file="concepts.md"
)



# LEITOR DE ONTOLOGIA OWL
def read_owl(file_onto):
    onto = get_ontology(file_onto).load()
    response = ""
    for cls in onto.classes():
        response = response + cls.name + " "
        if cls.comment:
            for descricao in cls.comment: 
                response = response + descricao + " "
        response = response + "\n"
    return response
# TESTE read_owl
#ontology = read_owl("D:/Workspaces/Python/tentativa7/onto.owl")
#print(f"Ontologia carregada: {ontology}")

############# TOOL2
read_onto_tool = Tool(
    #name="Read Ontology Tool",
    #description="This tool receives the argument named onto_path with the value {onto_path}",
    name="Ferramenta leitora de ontologia OWL",
    description="Esta ferramenta recebe um argumento com o valor {file_onto} e retorna uma lista de classes da ontologia OWL.",
    func=lambda file_onto: read_owl(file_onto),
)
#response = read_onto_tool.run("D:/Workspaces/Python/tentativa7/onto.owl")
#print(response)

############# AGENT2
agente_extrator_de_classes_de_ontologia = Agent(
    role="Agente extrator de classes de ontologia do arquivo {file_onto}",
    goal="Ler uma ontologia no arquivo {file_onto} e extrair uma lista de classes para análise subsequente por outro agente.",
    backstory="Você é um especialista em ontologia e consegue entregar uma lista de classes para análise subsequente por outros agentes.",
    verbose=True,
    llm=llm,
    max_iter=3,
    memory=True,
    tools=[read_onto_tool]
)

############# TASK2
tarefa_extrair_classes_de_uma_ontologia = Task(
    description="Extrair todas as classes ontológicas de uma ontologia no arquivo {file_onto} para análise subsequente por outro agente.",
    expected_output="""Retorna um documento em inglês formatado em markdown com uma tabela com duas colunas: Class e Annotation.""",
    agent=agente_extrator_de_classes_de_ontologia,
    #output_file="classes.md"
)


############# TOOL SEM USO NO MOMENTO
# read_onto_txt_tool = Tool(
#     name="Leitor de arquivos de ontologia em OWL",
#     description="Esta ferramenta possui um argumento com o valor {file_onto} e retorna a ontologia do arquivo em formato de texto simples.",
#     func=lambda file_onto: read_onto_txt(file_onto),
# )
#response = read_onto_txt_tool.run("D:/Workspaces/Python/tentativa7/onto.owl")
#print(response)


############ AGENT3
agente_comparador_de_conceitos_com_ontologia = Agent(
    role="Agente comparador de conceitos extraídos de um artigo fornecido por uma tool com classes ontológicas extraídas de uma ontologia também fornecida por outra tool.",
    goal="Comparar os conceitos extraídos adiquiridos de um artigo com uma lista de classes ontológicas também adiquirida e listar os conceitos que estão presentes na ontologia e os que estão ausentes.",
    backstory="Você é um doutor especialista em {topic} consegue comparar com precisão conceitos de uma tabela estraída de um artigo com uma lista de classes ontológicas.",
    verbose=True,
    llm=llm,
    max_iter=10,
    memory=True,
    tools=[adquiri_artigo_tool, adquiri_ontologia_tool]
)

############# TASK3
tarefa_listar_conceitos_presentes_e_ausentes_da_ontologia = Task(
    description="Comparar os conceitos extraídos de um arquivo fornecido por outro agente com uma lista de classes ontológicas também fornecido por outro agente e listar os conceitos que estão presentes na ontologia e os que estão ausentes.",
    expected_output="""Compila os resultados anteriores sem resumir e retorna um documento em inglês formatado em marckdonw contendo:
    - O título do antigo análisado pelo agente extrator de conceitos,
    - Uma tabela de conceitos extraídos do artigo pelo agente extrator de conceitos,
    - Uma tabela de classes ontológicas e seus respectivos comentários, se houver, fornecida pelo agente extrator de classes de ontologia,
    - Uma tabela de conceitos do artigo melhor relacionados com as classes ontológicas fornecidas pelo agente extrator de classes de ontologia,
    - Uma tabela com os conceitos do artigo que não foram possíveis de relacionar com as classes ontológicas e uma cópia de sua respectiva explicação da tabela 'conceitos extraídos'.",
    e.g.
    # TITLE: "Title of the article"
    ## CONCEPTS
    | concept | explanation |
    ## CLASS ONTOLOGY
    | class_ontology | annotations |
    ## FOUNDED 
    | concept | class_ontology | relationship_explanation |
    ## NOT FOUNDED
    | concept | explanation |
    Formatado como markdown sem '```'
    Para compor o documento, utilize as informações fornecidas pelos outros agentes, não faça nada hipotético.
    Preencha todos os tópicos do exemplo.""",
    agent=agente_comparador_de_conceitos_com_ontologia,
    #output_file="report.md"
)



crew = Crew(
    agents=[agente_comparador_de_conceitos_com_ontologia],
    tasks=[tarefa_listar_conceitos_presentes_e_ausentes_da_ontologia],
    process=Process.sequential,
    verbose=2,
    full_output=True,
    share_crew=False,
    manager_llm=llm,
    max_iter=15,
)


#results = crew.kickoff(inputs={"topic": "learning analytics", "file_pdf": "article3.pdf", "file_onto": "onto.owl"})
#print(results)

with st.sidebar:
    st.header("Coloque o tópico da área de pesquisa, o texto do artigo e o texto da ontologia em owl")

    with st.form(key='research_form'):
        topic = st.text_input('Tópico da área de pesquisa')
        texto_artigo = st.text_area('Texto do artigo')
        texto_onto = st.text_area('Texto da ontologia em owl')
        submit_button = st.form_submit_button(label='Submit')
    
if submit_button:
    if not topic:
        st.error("Por favor, insira o tópico da área de pesquisa.")
    else:
        results = crew.kickoff(inputs={"topic": topic})
        st.subheader("Relatório da pesquisa:")
        st.write(results['final_output'])
        print(results)
