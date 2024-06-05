# 1. Bibliotecas de terceiros utilizadas
## 1.1 Pré-processamento de textos
- nltk
- SpaCy
- sklearn(TfidfVectorizer)
- YAKE
- hugging-face transformers 
- translate

## 1.2 PDF
- PyPDF2

## 1.3 Audio/video
- moviepy
- SpeechRecognition

## 1.4 Imagens
- PIL
- pytesseract (Requer instalação na máquina do modelo e scripts para diferentes linguagens)
- sentencepiece

## 1.5 Mecanismo de busca e store
- whoosh

## 1.5 Outros
- torch (requisito para hugging-face)
- langchain

# 2. Decisão da arquitetura utilizada
Para extração de informações dos dados tentou-se primeiramente usando técnicas de NLP simples sem envolver modelos pré-treinados por questão de complexidade no caso de já atender a primeira opção, mas para imagem não obtive os resultados que gostaria, inclusive com os modelos pré-treinados hugging-face, mas performaram melhor ao menos. 
Para indexação, inicialmente se pensou no elasticsearch por ser uma ferramenta muito conhecida, versátil, poder de escalabilidade, mas tive problemas em realizar a conexão no meu pc, foi quando descobri o Whoosh que para implementação local se provou uma boa alternativa. Langchain foi utilizado para implementação de interações na cadeia do modelo, como a de captar o formato preferido.

# 3. O que melhoraria se tivesse mais tempo?
Estudaria mais sobre os motores de busca, testaria mais parâmetros, modelos e abordagens para tratamento da imagem. Além disso, caso tivesse plano premium da OpenAI, tentaria abordagens usando a api destes ou, por exemplo, modelos google

