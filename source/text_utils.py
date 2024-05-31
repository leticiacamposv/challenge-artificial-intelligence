
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
import yake
import re

def latin_text_preprocessing(text):
    """
    Encoding de textos para identificador de caracteres latinos e realização de nlp com spacy usando modulo pt. 

    Parâmetros: 
    text (str) -> Texto em português a ser tratado

    Retorno
    doc -> Texto pré-processado pela pipeline do spacy
    """
    try:
        decoded = text.encode('latin1', 'xmlcharrefreplace').decode('utf-8')
    except:
        decoded = text
        
    nlp = spacy.load('pt_core_news_sm')
    doc = nlp(decoded)
    return doc

def tfidf_keyword_extraction(text, top_n=10):
    """
    Extrai as principais palavras-chave de um texto usando o método TF-IDF (Term Frequency-Inverse Document Frequency)

    Parâmetros:
    text (str) -> Texto para extração das palavras chaves
    top_n (int) -> Quantidade de palavras-chave a serem retornadas

    Retorno: 
    list -> Lista das palavras-chave extraídas do texto, ordenadas por relevância
    """

    doc = latin_text_preprocessing(text)

    lemmatized_tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    processed_text = ' '.join(lemmatized_tokens)

    # Cálculo da matriz TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([processed_text])
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.toarray().flatten()

    # Pós-processamento
    keywords = sorted(zip(feature_names, tfidf_scores), key=lambda x: x[1], reverse=True)

    return [str(palavra) for palavra, score in keywords[:top_n]]

def text_most_relevant_prhases_extraction(text, num_sentences=30):
    """
    Extrai as frases mais relevantes de um texto com base na pontuação de relevância de cada sentença usando Rank do spacy.

    Parâmetros:
    text (str): O texto a ser processado para extração das frases mais relevantes.
    num_sentences (int, opcional): O número de frases mais relevantes a serem retornadas. O padrão é 30.

    Retorno:
    list[str]: Uma lista contendo as frases mais relevantes do texto, ordenadas por relevância decrescente.
    """

    doc = latin_text_preprocessing(text)

    # Cálculo de importância de cada sentença
    sentence_scores = {}
    for sent in doc.sents:
        score = sum(token.rank for token in sent)
        sentence_scores[sent] = score
    
    relevant_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:num_sentences]
    return [str(sentence) for sentence in relevant_sentences]

def yake_keyword_extraction(extracted_text, language='pt', max_ngram_size=3, deduplication_threshold = 0.50, deduplication_algorithm = 'seqm',  numofkeywords= 30, windowSize = 1, features = None):
    """
    Extrai as principais palavras-chave de um texto usando o método YAKE (Yet Another Keyword Extractor).

    Parâmetros:
    extracted_text (str) -> Texto extraído de um documento PDF para extração das palavras-chave.
    language (str) -> Código de linguagem do texto (padrão é 'pt' para português).
    max_ngram_size (int) -> Tamanho máximo dos n-grams considerados para extração de palavras-chave.
    deduplication_threshold (float) -> Limite para deduplicação de palavras-chave, entre 0 e 1.
    deduplication_algorithm (str) -> Algoritmo de deduplicação a ser utilizado ('seqm' ou outro).
    numofkeywords (int) -> Número de palavras-chave a serem retornadas.
    windowSize (int) -> Tamanho da janela de contexto para extração de palavras-chave.
    features (optional) -> Conjunto de features adicionais para o algoritmo de extração.

    Retorno:
    list -> Lista das palavras-chave extraídas do texto, ordenadas por relevância.
    """
    
    custom_kw_extractor = yake.KeywordExtractor(lan=language, 
                                                n=max_ngram_size, 
                                                dedupLim=deduplication_threshold, 
                                                dedupFunc=deduplication_algorithm, 
                                                top=numofkeywords, 
                                                features=features, 
                                                windowsSize=windowSize)
    
    keywords_scores = custom_kw_extractor.extract_keywords(extracted_text)
    keywords_scores = sorted(keywords_scores, key=lambda x: x[1], reverse=False) # Neste caso, quanto mais baixo o score, mais relevante é a palavra-chave

    keywords = [t[0] for t in keywords_scores]

    return keywords

def ner_author_extractor(extracted_text):
    """
    Extrai o nome do autor de um texto usando o modelo de reconhecimento de entidades nomeadas(NER).

    Parâmetros:
    extracted_text (str) -> Texto extraído de um documento PDF para extração do nome do autor.

    Retorno:
    str -> Nome do autor extraído do texto.
    """
     
    doc = latin_text_preprocessing(extracted_text)
    names = [ent.text for ent in doc.ents if ent.label_ == 'PER']

    # Capturar apenas 3 primeiros sobrenomes caso haja mais de 3
    if len(names[0].split())>4:
        name = names[0].split()[:4]
        full_name = ' '.join(name)
    else:
        full_name = names[0]
    
    return full_name

# def create_versioned_string(name):
#     return [
#         [{'LOWER': name}], 
#         [{'LOWER': {'REGEX': f'({name}\d+\.?\d*.?\d*)'}}], 
#         [{'LOWER': name}, {'TEXT': {'REGEX': '(\d+\.?\d*.?\d*)'}}],
#     ]

# def create_language_patterns():
#     versioned_languages = ['ruby', 'php', 'python', 'perl', 'java', 'haskell', 
#                            'scala', 'c', 'cpp', 'matlab', 'bash', 'delphi']
#     flatten = lambda l: [item for sublist in l for item in sublist]
#     versioned_patterns = flatten([create_versioned_string(lang) for lang in versioned_languages])

#     lang_patterns = [
#         [{'LOWER': 'objective'}, {'IS_PUNCT': True, 'OP': '?'},{'LOWER': 'c'}],
#         [{'LOWER': 'objectivec'}],
#         [{'LOWER': 'c'}, {'LOWER': '#'}],
#         [{'LOWER': 'c'}, {'LOWER': 'sharp'}],
#         [{'LOWER': 'c#'}],
#         [{'LOWER': 'f'}, {'LOWER': '#'}],
#         [{'LOWER': 'f'}, {'LOWER': 'sharp'}],
#         [{'LOWER': 'f#'}],
#         [{'LOWER': 'lisp'}],
#         [{'LOWER': 'common'}, {'LOWER': 'lisp'}],
#         [{'LOWER': 'go', 'POS': {'NOT_IN': ['VERB']}}],
#         [{'LOWER': 'golang'}],
#         [{'LOWER': 'html'}],
#         [{'LOWER': 'css'}],
#         [{'LOWER': 'sql'}],
#         [{'LOWER': {'IN': ['js', 'javascript']}}],
#         [{'LOWER': 'c++'}],
#     ]

#     return versioned_patterns + lang_patterns

def extract_programming_language(text):
    linguagens = [
    "ABAP", "ABC", "ActionScript", "Ada", "Agda", "ALGOL", "Alice", "APL", "AppleScript", "Arc", "Arduino",
    "ASP.NET", "Assembly", "Awk", "Ballerina", "Bash", "Basic", "BCPL", "Beta", "Boo", "C", "C#", "C++", "Caché ObjectScript",
    "Caml", "Ceylon", "CFML", "CHILL", "CIL", "CLIPS", "Clojure", "COBOL", "Cobra", "CoffeeScript", "ColdFusion", "Crystal",
    "Curl", "Dart", "DCL", "Delphi", "DIBOL", "Dylan", "Eiffel", "Elixir", "Elm", "Emacs Lisp", "Erlang", "Euphoria",
    "F#", "F*", "Factor", "Falcon", "Fancy", "Fantom", "Forth", "Fortran", "Fortress", "Gambas", "GAMS", "GAP", "GDScript",
    "Genie", "GML", "Go", "Google Apps Script", "Gosu", "Groovy", "Hack", "Haskell", "Haxe", "Heron", "HLSL", "HTML", "HTML5",
    "HyperTalk", "IDL", "Io", "Ioke", "J#", "JADE", "Java", "JavaScript", "JScript", "Julia", "Kotlin", "KRL", "LabVIEW",
    "Ladder Logic", "Lasso", "Lava", "LC-3", "Lisp", "LiveCode", "Logo", "Logtalk", "LotusScript", "LPC", "Lua", "MAD",
    "Magik", "Malbolge", "Maple", "Mathematica", "MATLAB", "Max", "MAXScript", "MEL", "Mercury", "Miranda", "ML", "Modula-2",
    "Modula-3", "MOO", "Mortran", "MSIL", "NATURAL", "Neko", "Nim", "NXT-G", "NXC", "Nial", "Nice", "Nickle", "NPL", "NSIS",
    "Nu", "NWScript", "Oberon", "Object Pascal", "Objective-C", "Objective-J", "OCaml", "Occam", "OpenCL", "OpenEdge ABL",
    "Oz", "Parrot", "Pascal", "Perl", "PHP", "Pico", "Pike", "PL/I", "PL/SQL", "PostScript", "PowerBuilder", "PowerShell",
    "Processing", "Prolog", "Puppet", "Pure Data", "PureBasic", "Python", "Q#", "QBasic", "QML", "R", "Racket", "Raku",
    "RAPID", "Ratfor", "REBOL", "Red", "Redcode", "Refal", "Rexx", "Ring", "RobotC", "Ruby", "Rust", "S-Lang", "SAS", "Scala",
    "Scheme", "Scratch", "sed", "Seed7", "Shell", "Simula", "Simulink", "Slate", "Smalltalk", "SNOBOL", "SPARK",
    "SPSS", "SQL", "Squirrel", "Standard ML", "Stata", "Swift", "Tcl", "Terra", "TeX", "Turing", "TypeScript", "Vala",
    "VBScript", "VHDL", "Verilog", "VimL", "Visual Basic", "Visual Basic .NET", "Wolfram", "X10", "xBase", "XC", "Xojo",
    "XQuery", "Yorick", "Z Shell"]

    text = text.lower()
    # Cria uma expressão regular que corresponde a qualquer uma das linguagens
    re_pattern = re.compile(r'\b(' + '|'.join(re.escape(linguagem) for linguagem in linguagens) + r')\b', re.IGNORECASE)
    
    # Encontra todas as correspondências no texto
    matches = re_pattern.findall(text)
    
    # Remove duplicatas e mantém a ordem original
    text_languages = list(dict.fromkeys(matches))
    
    return text_languages

