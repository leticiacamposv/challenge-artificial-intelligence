from langchain import Chain, Step, Input, Output, QueryContext
import spacy

nlp = spacy.load('pt_core_news_sm')

# Passo inicial para receber a pergunta do aluno
class InitialStep(Step):
    def __init__(self):
        super().__init__(
            inputs=[Input(name="question", type="str", required=True)],
            outputs=[Output(name="difficulty", type="str")]
        )

    def run(self, inputs, context: QueryContext):
        question = inputs['question']
        doc = nlp(question)
        difficulty = ""
        # Identificar dificuldade com base em alguma lógica, por exemplo:
        if len(doc.ents) == 0:
            difficulty = "Conceptual understanding required"
        else:
            difficulty = "Specific detail required"
        return {"difficulty": difficulty}

initial_step = InitialStep()

# Passo para perguntar sobre formato preferido
class PreferenceStep(Step):
    def __init__(self):
        super().__init__(
            inputs=[Input(name="difficulty", type="str", required=True)],
            outputs=[Output(name="format", type="str")]
        )

    def run(self, inputs, context: QueryContext):
        format_preference = input("Qual formato de aprendizado você prefere? (vídeo, PDF, imagem): ")
        return {"format": format_preference}

preference_step = PreferenceStep()

# Configurar conexão com Elasticsearch
es = Elasticsearch([{'host': 'localhost', 'port': 9200}])

# Definir passo para buscar no Elasticsearch
class SearchStep(Step):
    def __init__(self):
        super().__init__(
            inputs=[Input(name="question", type="str", required=True),
                    Input(name="format", type="str", required=True)],
            outputs=[Output(name="result", type="dict")]
        )

    def run(self, inputs, context: QueryContext):
        question = inputs['question']
        format_preference = inputs['format']
        # Executar busca no Elasticsearch
        response = es.search(
            index="conteudos",
            body={
                "query": {
                    "bool": {
                        "must": [
                            {"multi_match": {"query": question, "fields": ["conteudo"]}},
                            {"match": {"formato": format_preference}}
                        ]
                    }
                }
            }
        )
        results = [hit["_source"] for hit in response["hits"]["hits"]]
        # Retornar o documento mais pertinente
        return {"result": results[0] if results else "No relevant content found"}

search_step = SearchStep()

# Definir passo para retornar o conteúdo
class ReturnContentStep(Step):
    def __init__(self):
        super().__init__(
            inputs=[Input(name="result", type="dict", required=True)],
            outputs=[Output(name="content", type="str")]
        )

    def run(self, inputs, context: QueryContext):
        result = inputs['result']
        return {"content": result['conteudo']}

return_content_step = ReturnContentStep()

# Criar e executar a cadeia de passos
class AdaptiveLearningChain(Chain):
    def __init__(self):
        super().__init__(
            steps=[initial_step, preference_step, search_step, return_content_step]
        )

# Exemplo de uso
chain = AdaptiveLearningChain()
question = "Como utilizar HTML5 para criar uma página web?"

# Executar a cadeia
inputs = {"question": question}
result = chain.run(inputs)
print("Conteúdo recomendado:", result['content'])