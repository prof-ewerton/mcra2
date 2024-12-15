from pypdf import PdfReader
from owlready2 import *
from langchain.tools import Tool
from langchain_google_genai import ChatGoogleGenerativeAI
from crewai import Agent, Task, Crew, Process


# IMPORTANTO MODELO DE LLM
os.environ["GOOGLE_API_KEY"] = "AIzaSyD2iQW2yaa1hxkdueSjTygd3cWCnA-sSkc"
llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature = 0)



def read_pdf(file_pdf):
    reader = PdfReader(file_pdf)
    texto_pdf = ""
    for page in reader.pages: 
        texto_pdf += page.extract_text()
    return texto_pdf
# TESTE
#text = read_pdf("article.pdf")
#print(text)


############# TOOL1
read_pdf_tool = Tool(
    name="PDF file reader.",
    description="This tool has an argument with the value {file_pdf} and returns the content of the file in plain text format.",
    func=lambda file_pdf: read_pdf(file_pdf),
)

############# AGENT1
agente_extrator_de_conceitos_de_artigos = Agent(
    role="Concept extractor agent from the file {file_pdf}",
    goal="Read the text of the scientific article contained in the file {file_pdf} and extract various relevant concepts about {topic} for subsequent analysis.",
    backstory="You are a specialist in {topic} extracting various concepts from scientific articles. Your role is to ensure that all relevant concepts from the file {file_pdf} on the topic {topic} are accurately extracted, facilitating subsequent analysis by other agents.",
    verbose=True,
    llm=llm,
    max_iter=3,
    memory=True,
    tools=[read_pdf_tool]
)

############# TASK1
tarefa_extrair_conceitos_sobre_um_topico = Task(
    description="Extract various relevant concepts from the file {file_pdf} for subsequent analysis.",
    expected_output="""Return a document containing the title of the article and a table with the columns: concept and context.
    Where 'Concept' includes important concepts extracted from {file_pdf} and 'Context' is a paragraph extracted from {file_pdf} where the concept is used.
    In TITLE, write the title of the studied article. Do not create a hypothetical title, article or Concepts.
    e.g.
    # LIST OF CONCEPTS AND CONTEXT OF THE PAPER
    ## TITLE
    | Concept | Context |
    Formatted as markdown without '```'
    Do not make anything hypothetical.""",
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
    name="OWL ontology reader tool",
    description="This tool takes an argument with the value {file_onto} and returns a list of classes from the OWL ontology.",
    func=lambda file_onto: read_owl(file_onto),
)
#response = read_onto_tool.run("D:/Workspaces/Python/tentativa7/onto.owl")
#print(response)

############# AGENT2
agente_extrator_de_classes_de_ontologia = Agent(
    role="Ontology class extractor agent from the file {file_onto}",
    goal="Read an ontology from the file {file_onto} and extract a list of classes for subsequent analysis.",
    backstory="You are an ontology specialist and can deliver a list of classes for subsequent analysis by other agents.",
    verbose=True,
    llm=llm,
    max_iter=3,
    memory=True,
    tools=[read_onto_tool]
)

############# TASK2
# tarefa_extrair_classes_de_uma_ontologia = Task(
#     description="Extract all ontological classes from an ontology in the file {file_onto} for subsequent analysis.",
#     expected_output="""Return a document formatted in markdown with a table with two columns: Class and Annotation.
#     e.g
#     #LIST OF ONTOLOGICAL CLASSES AND ANNOTATIONS
#     | class ontology | annotation |
#     """,
#     agent=agente_extrator_de_classes_de_ontologia,
#     #output_file="classes.md"
# )

############# TASK2
# Retirei as annotations, o agente 3 estava confundido o contexto com a annotation.
tarefa_extrair_classes_de_uma_ontologia = Task(
    description="Extract all ontological classes from an ontology in the file {file_onto} for subsequent analysis.",
    expected_output="""Return a document formatted in markdown with a table with classes of the ontology.
    e.g
    #LIST OF ONTOLOGICAL CLASSES
    | class ontology |
    """,
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
    role="Concept comparison agent between extracted concepts from an paper provided by another agent and ontological classes extracted from an ontology also provided by another agent.",
    goal="Relate the concepts provided by one agent with the ontological classes provided by another agent, justifying the relationship.",
    backstory="You are a PhD expert in {topic} capable of accurately relate concepts with ontological classes and justifying the relationship.",
    verbose=True,
    llm=llm,
    max_iter=3,
    memory=True,
    tools=[]
)

# ############# TASK3
# tarefa_listar_conceitos_presentes_e_ausentes_da_ontologia = Task(
#     description="Compare the concepts extracted from a file provided by another agent with a list of ontological classes also provided by another agent and list the concepts that are present in the ontology and those that are absent.",
#     expected_output=""""Return a document in English formatted in markdown containing:
#     - The title of the article analyzed by the concept extractor agent,
#     - A table of concepts extracted from the article by the concept extractor agent,
#     - A table of ontological classes and their respective comments, if any, provided by the ontology class extractor agent,
#     - A table of the article's concepts best related to the ontological classes,
#     - A table with the article's concepts that could not be related to the ontological classes.
#     e.g.
#     # TITLE OF THE ARTICLE
#     ## CONCEPTS
#     | concept | explanation |
#     ## CLASS ONTOLOGY
#     | class ontology | annotation |
#     ## FOUNDED 
#     | concept | class_ontology | relationship_explanation |
#     ## NOT FOUNDED
#     | concept | relationship_explanation |
#     Formatted as markdown without '```'
#     To compose the document, use the information provided by other agents, do not make anything hypothetical.""",
#     agent=agente_comparador_de_conceitos_com_ontologia,
#     output_file="report.md"
# )

# ############# TASK3
# tarefa_listar_conceitos_presentes_e_ausentes_da_ontologia = Task(
#     description="Compare the concepts extracted from a file provided by another agent with a list of ontological classes also provided by another agent and list the concepts that are present in the ontology and those that are absent.",
#     expected_output=""""Return a document in English containing:
#     - TITLE: Place the title of the paper from the concept list;
#     - RELATED CONCEPTS: A table of the paper's concepts best related to the ontological classes,
#     - In the Relationship column, explain in a paragraph what the relationship between the concept and the class is based on the context of the concept.
#     - UNRELATED CONCEPTS: A table with the paper's concepts that could not be related to the ontological classes.
#     - In the 'concept context' column, copy the context related to the concept from the concept list.
#     e.g.
#     # TITLE
#     ## RELATED CONCEPTS 
#     | concept | class ontology | Relationship |
#     ## UNRELATED CONCEPTS
#     | concept | concept context |
#     Formatted as markdown without '```'
#     To compose the document, use the information provided by other agents, do not make anything hypothetical.""",
#     agent=agente_comparador_de_conceitos_com_ontologia,
#     output_file="report.md"
# )

############# TASK3
tarefa_listar_conceitos_presentes_e_ausentes_da_ontologia = Task(
    description="Compare the concepts extracted from a file provided by another agent with a list of ontological classes also provided by another agent and list the concepts that are present in the ontology and those that are absent.",
    expected_output=""""Return a document in English, formatted as markdown without '```'.

    e.g.
    # TITLE
    ## RELATED CONCEPTS 
    | concept | class ontology | justification |
    ## UNRELATED CONCEPTS
    | concept | concept context |
    
    - In TITLE, write the title of the studied article.
    - RELATED CONCEPTS: A table containing the concepts best related to the list of ontological classes;
    - In the justification column, justify in aparagraph what the relationship between the concept and the class is based on the context of the concept;
    - UNRELATED CONCEPTS: A table with the paper's concepts that could not be related to the ontological classes;
    - In the concept context column, copy the context related to the concept.

    To compose the document, use the information provided by other agents, do not make anything hypothetical.""",
    agent=agente_comparador_de_conceitos_com_ontologia,
    output_file="report.md"
)


crew = Crew(
    agents=[agente_extrator_de_conceitos_de_artigos, agente_extrator_de_classes_de_ontologia, agente_comparador_de_conceitos_com_ontologia],
    tasks=[tarefa_extrair_conceitos_sobre_um_topico, tarefa_extrair_classes_de_uma_ontologia, tarefa_listar_conceitos_presentes_e_ausentes_da_ontologia],
    process=Process.sequential,
    verbose=2,
    full_output=True,
    share_crew=False,
    manager_llm=llm,
    max_iter=10,
    #max_rpm=15, # Limite da conta gratuita.
)


results = crew.kickoff(inputs={"topic": "learning analytics", "file_pdf": "paper5.pdf", "file_onto": "onto.owl"})

#with open("report.md", "w") as arquivo:
#    arquivo.write(results['final_output'])

#print(results['final_output'])