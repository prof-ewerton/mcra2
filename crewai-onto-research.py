from pypdf import PdfReader
from owlready2 import *
from langchain.tools import Tool
from langchain_google_genai import ChatGoogleGenerativeAI
from crewai import Agent, Task, Crew, Process



# IMPORTANTO MODELO DE LLM
os.environ["GOOGLE_API_KEY"] = "AIzaSyD2iQW2yaa1hxkdueSjTygd3cWCnA-sSkc"
llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature = 0)






# scientific_article_reader_agent = Agent(
#     role="Reader of Scientific Articles in {topic}",
#     goal="Read the texts of scientific articles {topic} in  and extract the relevant content.",
#     backstory="You are a specialist in reading and extracting text from scientific articles in {topic}. Your role is to ensure that all relevant content from learning analytics articles is accurately extracted, facilitating subsequent analysis by other agents.",
#     verbose=True,
#     llm=llm,
#     max_iter=3,
#     memory=True,
#     tools=[read_pdf_tool]
# )

# scientific_article_reader_task = Task(
#     description="""Read the full text of scientific articles in file_path is {file_pdf} related to learning analytics.
#     Extract the relevant content that will be used for subsequent analysis.""",
#     expected_output="Returns the title of the article and a markdown formatted table with the columns: concept and explanation. Where concept are the important concepts about {topic} extracted from the article and explanation is a paragraph from the article where its usage is found.",
#     agent=scientific_article_reader_agent,
# )

# ontology_relationship_analyst_agent = Agent(
#     role="Ontology Relationship Analyst in {topic}",
#     goal="Relate the terms table extracted from scientific articles in {topic} to the classes of a predefined ontology.",
#     backstory="You are an analyst specialized in relating concepts from scientific articles in {topic} to ontologies. With a keen eye for detail and a deep understanding of learning analytics, you identify and map the concepts extracted from the texts to the appropriate classes in the ontology, providing a clear and precise mapping.",
#     verbose=True,
#     llm=llm,
#     max_iter=3,
#     memory=True,
#     tools=[read_onto_txt_tool, read_pdf_tool]
# )

# ontology_relationship_task = Task(
#     description="""Analyze the text extracted from the Reader of Scientific Articles in {file_pdf} and relate the found terms to the classes of a predefined from ontology in {file_onto}.
#     Identify and list the terms that correspond to the classes of the ontology in {file_onto}.""",
#     expected_output="""A list with the terms found in the article and their respective classes in the ontology, along with explanations of the relationships.
#     Another separate list with the terms and your explanations that were not found in the ontology.
#     e.g.
#     # FOUNDED 
#     | term_text | class_ontology | relationship_explanation |
#     | --------- | -------------- | ------------------------ |
#     | "term" | "class" | "Paragraph extracted from the text that relates the term to the ontology class." |
#     | ... | ... | ... |
#     e.g.
#     # NOT FOUNDED
#     | term_text | relationship_explanation |
#     | --------- | -------------- | ------------------------ |
#     | "term" | "A paragraph where the word fits into the context of the article." |
#     | ... | ... |
#     Formatted as markdown without '```'.""",
#     agent=ontology_relationship_analyst_agent,
#     output_file="report.md"
# )


# crew = Crew(
#     #agents=[scientific_article_reader_agent, ontology_relationship_analyst_agent],
#     #tasks=[scientific_article_reader_task, ontology_relationship_task],
#     agents=[ontology_relationship_analyst_agent],
#     tasks=[ontology_relationship_task],
#     process=Process.hierarchical,
#     verbose=2,
#     full_output=True,
#     share_crew=False,
#     manager_llm=llm,
#     max_iter=3,
# )




def read_pdf(file_pdf):
    reader = PdfReader(file_pdf)
    texto_pdf = ""
    for page in reader.pages: 
        texto_pdf += page.extract_text()
    return texto_pdf
# TESTE read_pdf
#text = read_pdf("article.pdf")
#print(text)


# read_pdf_tool = Tool(
#     name="Read PDF Tool",
#     description="This tool receives the argument named file_path with the value {file_path} and returns the content in plain text format.",
#     func=lambda file_path: read_pdf(file_path),
# )

#response = read_pdf_tool.run("article.pdf")
#print(response)

############# TOOL1
read_pdf_tool = Tool(
    name="Leitor de arquivos PDF",
    description="Esta ferramenta possui um argumento com o valor {file_pdf} e retorna o conteúdo do arquivo em formato de texto simples.",
    func=lambda file_pdf: read_pdf(file_pdf),
)

############# AGENT1
agente_extrator_de_conceitos_de_artigos = Agent(
    role="Agente extrator de conceitos do arquivo {file_pdf}",
    goal="Ler o texto do artigo científico contido no arquivo {file_pdf} e extrair vários conceitos relevantes sotre {topic} para análise subsequente.",
    backstory="Você é um especialista em {topic} em extrair vários conceitos de artigos científicos. Seu papel é garantir que todos os conceitos relevantes do arquivo {file_pdf} sobre o tema {topic} sejam extraídos com precisão, facilitando a análise subsequente por outros agentes.",
    verbose=True,
    llm=llm,
    max_iter=5,
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
    output_file="concepts.md"
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
    goal="Ler uma ontologia no arquivo {file_onto} e extrair uma lista de classes para análise subsequente.",
    backstory="Você é um especialista em ontologia e consegue entregar uma lista de classes para análise subsequente por outros agentes.",
    verbose=True,
    llm=llm,
    max_iter=5,
    memory=True,
    tools=[read_onto_tool]
)

############# TASK2
tarefa_extrair_classes_de_uma_ontologia = Task(
    description="Extrair todas as classes ontológicas de uma ontologia no arquivo {file_onto} para análise subsequente.",
    expected_output="""Retorna um documento em inglês formatado em markdown com uma tabela com duas colunas: Class e Annotation.""",
    agent=agente_extrator_de_classes_de_ontologia,
    output_file="classes.md"
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
    role="Agente comparador de conceitos extraídos de um artigo fornecido por outro agente com classes ontológicas extraídas de uma ontologia também fornecida por outro agente",
    goal="Comparar os conceitos extraídos de um arquivo com uma lista de classes ontológicas e listar os conceitos que estão presentes na ontologia e os que estão ausentes.",
    backstory="Você é um doutor especialista em {topic} consegue comparar com precisão conceitos de uma tabela estraída de um artigo com uma lista de classes ontológicas.",
    verbose=True,
    llm=llm,
    max_iter=10,
    memory=True,
    tools=[]
)

############# TASK3
tarefa_listar_conceitos_presentes_e_ausentes_da_ontologia = Task(
    description="Comparar os conceitos extraídos de um arquivo fornecido por outro agente com uma lista de classes ontológicas também fornecido por outro agente e listar os conceitos que estão presentes na ontologia e os que estão ausentes.",
    expected_output="""Retorna um documento em inglês formatado em marckdonw contendo:
    - O título do antigo análisado pelo agente extrator de conceitos,
    - Uma tabela de conceitos extraídos do artigo pelo agente extrator de conceitos,
    - Uma tabela de classes ontológicas e seus respectivos comentários, se houver, fornecida pelo agente extrator de classes de ontologia,
    - Uma tabela de conceitos do artigo melhor relacionados com as classes ontológicas,
    - Uma tabela com os conceitos do artigo que não foram possíveis de relacionar com as classes ontollógicas,
    e.g.
    # TITLE: "Title of the article"
    ## CONCEPTS
    | concept | explanation |
    ## CLASS ONTOLOGY
    | class_ontology | annotations |
    ## FOUNDED 
    | concept | class_ontology | relationship_explanation |
    ## NOT FOUNDED
    | concept | relationship_explanation |
    Formatado como markdown sem '```'
    Para compor o documento, utilize a informação fornecida pelos outros agentes não faça nada hipotético.""",
    agent=agente_comparador_de_conceitos_com_ontologia,
    output_file="report.md"
)



crew = Crew(
    agents=[agente_comparador_de_conceitos_com_ontologia, agente_extrator_de_conceitos_de_artigos, agente_extrator_de_classes_de_ontologia],
    tasks=[tarefa_listar_conceitos_presentes_e_ausentes_da_ontologia, tarefa_extrair_conceitos_sobre_um_topico, tarefa_extrair_classes_de_uma_ontologia],
    process=Process.sequential,
    verbose=2,
    full_output=True,
    share_crew=False,
    manager_llm=llm,
    max_iter=15,
)


results = crew.kickoff(inputs={"topic": "learning analytics", "file_pdf": "article3.pdf", "file_onto": "onto.owl"})
print(results)