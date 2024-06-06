import speech_recognition as sr
from moviepy.editor import VideoFileClip

def extract_audio(video_file_path, audio_file_path):
    """
    Extrai o áudio de um arquivo de vídeo e o salva em um arquivo de áudio e gera lista com demais metadados do vídeo.

    Parâmetros:
    video_file_path (str): O caminho do arquivo de vídeo.
    audio_file_path (str): O caminho onde o arquivo de áudio extraído será salvo.

    Retorno:
    Audio extraido
    """
    video = VideoFileClip(video_file_path)
    audio = video.audio
    audio.write_audiofile(audio_file_path)

def extract_video_metadata(video_file_path):
    """
    Extrai metadados de um arquivo de vídeo.

    Esta função utiliza a biblioteca MoviePy para carregar o vídeo e extrair informações importantes como 
    duração, tamanho, taxa de quadros por segundo (fps), tempo de início e tempo de término do vídeo.

    Parâmetros:
    video_file_path (str): O caminho do arquivo de vídeo do qual os metadados serão extraídos.

    Retorno:
    dict: Um dicionário contendo os metadados:
        - "duration" (str): A duração do vídeo em segundos.
        - "size" (str): O tamanho do vídeo em pixels, como uma tupla (largura, altura).
        - "fps" (str): A taxa de quadros por segundo do vídeo.
        - "start_time" (str): O tempo de início do vídeo.
        - "end_time" (str): O tempo de término do vídeo.
    """
    video = VideoFileClip(video_file_path)
    metadata = {
        "video_duration": str(video.duration),
        "video_size": str(video.size),
        "video_fps": str(round(video.fps)),
        "video_start_time": str(video.start),
        "video_end_time": str(video.end)
        }
    return metadata

def transcribe_audio(audio_file_path):
    """
    Transcreve o áudio de um arquivo de áudio em texto usando a API Google Speech Recognition

    Parâmetros:
    audio_file_path (str): O caminho do arquivo de áudio a ser transcrito.

    Retorno:
    str: O texto transcrito do áudio.
    """
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file_path) as source:
        audio = recognizer.record(source)
    try:
        transcript = recognizer.recognize_google(audio, language='pt-BR')
    except sr.UnknownValueError:
        transcript = "Não foi possível entender o áudio"
    except sr.RequestError as e:
        transcript = f"Não foi possível requisitar os resultados; {e}"
    return transcript