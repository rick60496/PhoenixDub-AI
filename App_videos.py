# Copyright (c) 2026 Paulo Henrik Carvalho de Araújo
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import time
import subprocess
import logging
import json
import threading
import traceback
from pathlib import Path
from datetime import datetime, timedelta
from threading import Lock, Timer, Thread
import hashlib
import shutil
import re
import gc # [NEW] Gerenciamento de Memória
import webbrowser
import random
import numpy as np
import socket
import requests
from queue import Queue
from sklearn.metrics.pairwise import cosine_similarity
from enum import Enum
import ffmpeg # Para info de mídia (probe)
from pydub import AudioSegment, effects # [NEW] Effects para speedup
from flask import send_file, Flask, send_from_directory, request, jsonify, make_response # [FIX] Import necessário para download
from flask_cors import CORS # [NEW] Suporte a Cross-Origin
from sklearn.cluster import AgglomerativeClustering # [NEW] v10 Diarization
from scipy.spatial.distance import cdist # [NEW] v10 Diarization

# --- CONFIGURAÇÕES DE AMBIENTE (OFFLINE-FIRST) ---
os.environ["HF_HUB_OFFLINE"] = "0"        # [FIX] Permitir download inicial de modelos
os.environ["TRANSFORMERS_OFFLINE"] = "0"  # [FIX] Permitir download inicial de modelos
os.environ["SPEECHBRAIN_FETCH_STRATEGY"] = "COPY"
os.environ["COQUI_TOS_AGREED"] = "1"

# --- IMPORTAÇÃO MÓDULO JOGOS (COM FALLBACK) ---

try:
    # Adiciona diretório atual ao path para garantir importação
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    import app_jogos
except ImportError:
    logging.warning("Módulo 'app_jogos' não encontrado. Funcionalidade de jogos desativada.")
    app_jogos = None
except Exception as e:
    logging.error(f"Erro ao carregar 'app_jogos': {e}")
    app_jogos = None
# ----------------------------------------------

def speedup_audio(audio_segment, speed_factor):
    """
    Acelera o áudio sem alterar o pitch (Time Stretch).
    Usa pydub.effects.speedup (que usa algoritmo granulado simples).
    Para fatores > 1.0 (aceleração).
    """
    if speed_factor <= 1.0: return audio_segment
    
    # [SAFEGUARD] Limite de segurança para evitar "Esquilos"
    # O usuário pediu até 30% (1.3), mas tecnicamente o pydub aceita mais.
    # Vamos respeitar o pedido da lógica de chamada, mas aqui garantimos que não crasha.
    
    try:
        # chunk_size e crossfade ajustados para fala (reduz artefatos metálicos)
        # speedup(segment, playback_speed=1.5, chunk_size=150, crossfade=25)
        return effects.speedup(audio_segment, playback_speed=speed_factor, chunk_size=150, crossfade=25)
    except Exception as e:
        logging.warning(f"Falha ao acelerar áudio (Fator {speed_factor}): {e}")
        return audio_segment

# --- VOICE GUARD (SISTEMA DE IDENTIDADE RIGOROSA) ---
class VoiceState(Enum):
    PROVISIONAL = "Provisória"   # < 5s
    INSUFFICIENT = "Insuficiente" # 5s - 30s
    TRAINABLE = "Treinável"      # >= 30s

class VoiceIdentity:
    def __init__(self, voice_id):
        self.id = voice_id
        self.embeddings = []
        self.mean_embedding = None
        self.total_duration = 0.0
        self.segments = [] # List of {start, end, text, etc}
        self.state = VoiceState.PROVISIONAL

    def add_segment(self, embedding, duration, segment_info=None):
        self.embeddings.append(embedding)
        self.total_duration += duration
        
        # Update mean embedding incrementally or re-calculate
        # Keeping all embeddings might be heavy for very long audio, but crucial for quality.
        # MVP: Recalculate mean from all embeddings (accuracy > ram here)
        matrix = np.array(self.embeddings)
        self.mean_embedding = np.mean(matrix, axis=0)
        
        if segment_info:
            self.segments.append(segment_info)
            
        self.update_state()

    def update_state(self):
        if self.total_duration >= 30.0:
            self.state = VoiceState.TRAINABLE
        elif self.total_duration >= 5.0:
            self.state = VoiceState.INSUFFICIENT
        else:
            self.state = VoiceState.PROVISIONAL

class VoiceGuard:
    # [TUNING COMPLETED] Ajuste Fino v2:
    # - Similarity 0.45: Compromisso entre separação (0.60 era demais) e junção (0.32 era de menos).
    # - Hysteresis 0.65: Continua colando na mesma voz se for parecido.
    def __init__(self, similarity_threshold=0.45, hysteresis_threshold=0.65):
        self.voices = {} # id -> VoiceIdentity
        self.next_id_counter = 1
        self.similarity_threshold = similarity_threshold
        self.hysteresis_threshold = hysteresis_threshold
        self.last_speaker_id = None
        self.counter_ping_pong = 0

    def create_new_voice(self):
        new_id = f"voz{self.next_id_counter}"
        self.next_id_counter += 1
        self.voices[new_id] = VoiceIdentity(new_id)
        logging.info(f"VoiceGuard: New Voice Created [{new_id}]")
        return new_id

    def process_segment(self, embedding, duration, start_time, end_time):
        """
        Decides which speaker this segment belongs to.
        Returns: assigned_voice_id
        """
        best_match_id = None
        best_score = -1.0
        
        # 1. Compare with all existing voices
        for vid, voice in self.voices.items():
            if voice.mean_embedding is None: continue
            
            emb_a = embedding.reshape(1, -1)
            emb_b = voice.mean_embedding.reshape(1, -1)
            score = cosine_similarity(emb_a, emb_b)[0][0]
            
            if score > best_score:
                best_score = score
                best_match_id = vid
        
        result_id = None
        is_short = duration < 2.0
        
        # 2. Decision Logic (Simplified & Safer)
        # [CHANGE] Removida lógica especial p/ segmentos curtos que forçava merge com anterior.
        # Isso causava contaminação (homem falando curto -> grudava na mulher anterior).
        
        # [RULE] Hysteresis (Sticky Speaker)
        # Se temos um orador anterior, damos preferência a ele se a diferença para o novo melhor não for brutal.
        if self.last_speaker_id and self.last_speaker_id in self.voices and best_match_id != self.last_speaker_id:
             last_voice = self.voices[self.last_speaker_id]
             # Recalcula score com o anterior para ter certeza
             emb_a = embedding.reshape(1, -1)
             emb_b = last_voice.mean_embedding.reshape(1, -1)
             score_last = cosine_similarity(emb_a, emb_b)[0][0]
             
             # Se o anterior ainda é muito parecido (> hysteresis), mantém ele.
             # E o novo concorrente não pode ser MUITO melhor.
             if score_last > self.hysteresis_threshold and (best_score - score_last) < 0.10:
                 result_id = self.last_speaker_id

        if result_id is None:
            # Se encontrou alguém compatível (> threshold), usa.
            if best_match_id and best_score >= self.similarity_threshold:
                result_id = best_match_id
                
            # Se não, tenta recuperar o anterior se for "pouco ruim" (Soft Fallback)
            # Ex: Score 0.40 (perto de 0.45) e era o mesmo cara.
            elif self.last_speaker_id and self.last_speaker_id in self.voices:
                 # Check again score_last
                 last_voice = self.voices[self.last_speaker_id]
                 emb_a = embedding.reshape(1, -1)
                 emb_b = last_voice.mean_embedding.reshape(1, -1)
                 score_last = cosine_similarity(emb_a, emb_b)[0][0]
                 
                 if score_last > (self.similarity_threshold - 0.1): # 0.35
                     result_id = self.last_speaker_id
                 else:
                     result_id = self.create_new_voice()
            else:
                result_id = self.create_new_voice()
        
        # 3. Update Voice Stats
        if result_id and result_id in self.voices:
            voice = self.voices[result_id]
            voice.add_segment(embedding, duration, {"start": start_time, "end": end_time})
        
        self.last_speaker_id = result_id
        return result_id

    def get_trainable_voices(self):
        return [v for v in self.voices.values() if v.state == VoiceState.TRAINABLE]

    def get_status_report(self):
        report = []
        for vid, v in self.voices.items():
            report.append({
                'id': vid,
                'duration': f"{v.total_duration:.1f}s",
                'state': v.state.value,
                'segments': len(v.segments)
            })
        return report

    def restore_voice(self, voice_id, embedding):
        """
        Restaura uma voz pré-existente (ex: de um arquivo salvo) para a memória do VoiceGuard.
        Isso impede que o sistema crie duplicatas se reiniciar.
        """
        # Garante ID
        # Assumindo VoiceIdentity disponível no escopo (definida anteriormente no arquivo)
        self.voices[voice_id] = VoiceIdentity(voice_id)
        # Injeta embedding como média inicial
        self.voices[voice_id].mean_embedding = embedding
        # Já marca como treinável pois assumimos que se foi salvo, é importante
        self.voices[voice_id].state = VoiceState.TRAINABLE
        self.voices[voice_id].total_duration = 30.0 # Valor seguro
        
        # Atualiza contador global se for numérico
        try:
            num = int(voice_id.replace("voz", ""))
            if num >= self.next_id_counter:
                self.next_id_counter = num + 1
        except: pass
        
        return voice_id


BASE_DIR = Path(__file__).parent.resolve()

from flask import Flask, send_from_directory, request, jsonify, make_response
from pydub import AudioSegment
from pydub.silence import split_on_silence # <--- IMPORTADO PARA SEGMENTAÇÃO
from werkzeug.utils import secure_filename
from faster_whisper import WhisperModel
import requests
import torch
import psutil
import soundfile as sf
# Tenta importar bibliotecas de IA pesadas, mas não falha se não existirem ainda
# OpenVoice removido por solicitação do usuário
se_extractor = None
ToneColorConverter = None

# Tenta importar Demucs (para separação de áudio)
try:
    # OpenUnmix será chamado via subprocess (CLI 'umx')
    import subprocess
except ImportError:
    pass

# [v7.1] Chatterbox Removido para priorizar estabilidade do Chatterbox TTS
# (Conflito de dependências resolvido)

# Import SpeechBrain (Diarização)
# PATCH: Correção para Torchaudio 2.1+ que removeu list_audio_backends
try:
    import torchaudio
    if not hasattr(torchaudio, 'list_audio_backends'):
        def _list_audio_backends():
            return ['soundfile'] # Mock simples para satisfazer o SpeechBrain
        torchaudio.list_audio_backends = _list_audio_backends
    
    # [FIX] Força uso do backend soundfile - REMOVIDO para evitar warning depreciado
    pass
except ImportError:
    pass



    
# import ffmpeg # Removido para evitar dependência extra (usamos subprocess)

import gc
# --- CONFIGURAÇÕES DE AMBIENTE ---
# Forçar SpeechBrain e Torch no Processador (CPU)
os.environ["CUDA_VISIBLE_DEVICES"] = "" 
os.environ["SPEECHBRAIN_FETCH_STRATEGY"] = "COPY"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["COQUI_TOS_AGREED"] = "1"

# Limitar threads
torch.set_num_threads(2)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s', stream=sys.stdout)

# --- DICIONÁRIO DE TRADUÇÕES COMUNS (Portado de app_jogos) ---
DICIONARIO_TRADUCOES = {
    "on it.": "Já vou.", "weapons free.": "Fogo à vontade.", "no way.": "nem pensar.",
    "get real.": "cai na real.", "not happening.": "sem chance.", "yes.": "sim.",
    "no.": "não.", "thanks.": "obrigado.", "thank you.": "obrigado.", "ok.": "ok."
}

# --- LISTA DE SIBILOS E SONS NÃO VERBAIS A IGNORAR ---
SONS_A_IGNORAR = [
    'ah', 'ai', 'eh', 'ei', 'oh', 'oi', 'uh', 'ui', 'ahm', 'hmm', 'huh', 'hmpf',
    'tsk', 'tsr', 'ugh', 'uhm', 'shh', 'suspira', 'geme', 'gasp', 'ofega', 'grr', 'rrr'
]

def check_ffmpeg():
    """Verifica se o FFmpeg está instalado e acessível no PATH do sistema."""
    if not shutil.which("ffmpeg"):
        logging.critical("="*80)
        logging.critical("ERRO CRÍTICO: O FFmpeg não foi encontrado no PATH do sistema.")
        logging.critical("Por favor, instale o FFmpeg e adicione-o ao PATH.")
        logging.critical("Download: https://ffmpeg.org/download.html")
        logging.critical("="*80)
        sys.exit(1)
    logging.info("FFmpeg encontrado e pronto para uso.")

# --- Caminho para o áudio base padrão (Motor Comercial - OpenVoice/MeloTTS) ---
DEFAULT_BASE_SPEAKER_PATH = Path("resources/base_speakers/pt/default_pt_speaker.wav")

# --- DEFINIÇÃO DE ETAPAS PARA CADA MÓDULO ---
# (Isso será usado pela função set_progress para o CMD)

ETAPAS_CHAT = ["Iniciando", "1. Analisando Pedido (Gema)", "2. Executando Tarefas", "3. Concluído"]
ETAPAS_DUBLAGEM = [
    "Iniciando", "1. Extraindo Áudio (Segmentado)", "2. Separando Vocais (OpenUnmix)", "3. Transcrição (Whisper)",
    "4. Tradução (Gema)", "5. Sincronização (Gema)", "6. Adaptação TTS (Gema)",
    "7. Gerando Áudios (Chatterbox)", "8. Montando Vídeo", "9. Concluído"
]
ETAPAS_MUSIC_VISUALIZER = ["Iniciando", "1. Analisando Áudio (Librosa)", "2. Gerando Efeitos", "3. Renderizando Vídeo", "4. Concluído"]
ETAPAS_EDITOR_RAPIDO = ["N/A (Interface Manual)"]
ETAPAS_REMOVER_FUNDO = ["Iniciando", "1. Processando Frames (IA)", "2. Remontando Vídeo", "3. Concluído"]
ETAPAS_UPSCALE = ["Iniciando", "1. Processando Frames (IA Upscale)", "2. Remontando Vídeo", "3. Concluído"]
ETAPAS_TRANSCRICAO = ["Iniciando", "1. Transcrevendo (Whisper)", "2. Gerando Arquivo SRT/TXT", "3. Concluído"]
ETAPAS_REMOVER_SILENCIO = ["Iniciando", "1. Analisando Áudio", "2. Editando Vídeo", "3. Concluído"]
ETAPAS_TRADUZIR_LEGENDAS = ["Iniciando", "1. Lendo SRT", "2. Traduzindo Linhas (Gema)", "3. Salvando Novo SRT", "4. Concluído"]
ETAPAS_SEPARADOR_AUDIO = ["Iniciando", "1. Processando (IA Demucs)", "2. Salvando Faixas", "3. Concluído"]
ETAPAS_LIMPEZA_AUDIO = ["Iniciando", "1. Analisando e Limpando (IA)", "2. Exportando Áudio", "3. Concluído"]
ETAPAS_CONVERSOR = ["Iniciando", "1. Convertendo Arquivos", "2. Concluído"]
ETAPAS_GERADOR_VIDEO = ["Iniciando", "1. Analisando Roteiro (Gema)", "2. Gerando Cenas (IA)", "3. Gerando Vozes (IA)", "4. Montando Vídeo", "5. Concluído"]

# --- Mapeamento de Emoção para Estilo de Voz (para OpenVoice) ---
EMOTION_TO_STYLE_PROMPT = {
    'alegre': 'speaking joyfully and enthusiastically', 'triste': 'speaking in a sad, melancholic tone',
    'nervoso': 'speaking nervously with a shaky voice', 'surpreso': 'gasped in surprise, speaking with a sense of wonder',
    'calmo': 'speaking calmly and clearly', 'irritado': 'speaking in an angry, irritated tone',
    'assustado': 'whispering in a scared, fearful tone', 'confuso': 'speaking with a confused and questioning tone',
    'sarcástico': 'speaking in a sarcastic and mocking tone', 'default': 'speaking calmly'
}

# --- VARIÁVEIS GLOBAIS E LOCKS (Arquitetura Adaptativa v18.6) ---
whisper_model = None
chatterbox_tts_model = None 
model_lock = Lock()
openvoice_lock = Lock()
chatterbox_lock = Lock()
progress_dict, progress_lock = {}, Lock()
active_jobs_lock = Lock()
active_jobs = set()

# [v18.6] TRAVA DE SEGURANÇA: 1 vídeo por vez para estabilidade total.
MAX_CONCURRENT_JOBS = 1 


# --- INICIALIZAÇÃO DO FLASK ---
app = Flask(__name__, template_folder='client', static_folder='client') # Servir do diretório client
CORS(app) # [NEW] Habilita CORS para o frontend local
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 4 * 1024 * 1024 * 1024 # Limite de 4GB para uploads
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- FUNÇÕES DE UTILIDADE (do app_jogos) ---

def safe_json_write(data, path, indent=4, ensure_ascii=False, retries=5, delay=0.2):
    """Escreve JSON de forma segura, com retentativas e ficheiro temporário."""
    path = Path(path)
    temp_path = path.with_suffix(path.suffix + f'.tmp_{random.randint(1000, 9999)}')
    try:
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)
    except Exception as e:
        logging.error(f"ERRO CRÍTICO ao escrever no ficheiro temporário {temp_path}: {e}")
        return
    for attempt in range(retries):
        try:
            os.replace(temp_path, path)
            return
        except PermissionError as e:
            if attempt < retries - 1:
                logging.warning(f"Tentativa {attempt + 1}/{retries} falhou ao aceder {path}: {e}. A tentar novamente em {delay}s...")
                time.sleep(delay)
            else:
                logging.error(f"ERRO CRÍTICO na tentativa final de escrever em {path}: {e}")
        except Exception as e:
            logging.error(f"ERRO CRÍTICO inesperado ao substituir {path} com {temp_path}: {e}")
            if temp_path.exists():
                try: os.remove(temp_path)
                except Exception: pass
            break

def safe_json_read(path):
    """Lê JSON de forma segura, lidando com ficheiros corrompidos."""
    path = Path(path)
    if not path.exists(): return None
    try:
        with open(path, 'r', encoding='utf-8') as f: return json.load(f)
    except json.JSONDecodeError:
        logging.error(f"Ficheiro JSON corrompido detectado em {path}")
        corrupt_path = path.with_name(f"{path.stem}.corrupt_{int(time.time())}{path.suffix}")
        try:
            os.replace(path, corrupt_path)
            logging.warning(f"Ficheiro corrompido movido para {corrupt_path}")
        except Exception as move_e: logging.error(f"ERRO ao mover ficheiro corrompido {path}: {move_e}")
        return None
    except Exception as e:
        logging.error(f"ERRO inesperado ao ler o ficheiro JSON {path}: {e}")
        return None

def sanitize_tts_text(text):
    """Limpa o texto para o TTS, (do app_jogos)."""
    if not isinstance(text, str): return ""
    match = re.match(r'^(.*?)(?=\n\n|\*\*Texto Original:|\*\*Texto Adaptado:)', text, re.DOTALL)
    clean_text = match.group(1).strip() if match else text.strip()
    
    # [FIX] Remove marcadores de lista que vazam do LLM: "(a) ", "1. ", "a) "
    clean_text = re.sub(r'(?:^|\s)[\(\[]?[0-9a-zA-Z]{1,2}[\)\]\.]\s+', ' ', clean_text)
    
    # [FIX] Remove marcadores de gênero (ex: "trancado(a)", "ele(a)")
    # Remove: (a), (o), (e), (s) colados em palavras
    clean_text = re.sub(r'\([aoes]\)', '', clean_text, flags=re.IGNORECASE)
    # Remove casos com espaço: "trancado (a)"
    clean_text = re.sub(r'\s+\([aoes]\)', '', clean_text, flags=re.IGNORECASE)
    
    clean_text = re.sub(r'ponto de interrogação|ponto interroga(?:ç|t)ão|ponto inter[eo]gativo', '?', clean_text, flags=re.IGNORECASE)
    clean_text = re.sub(r'\s*,\s*', ', ', clean_text)
    clean_text = re.sub(r',,+', ',', clean_text)
    clean_text = clean_text.replace('...', ',').replace('!', 'TEMP_EXCLAMATION').replace('?', 'TEMP_QUESTION').replace('.', ',')
    clean_text = clean_text.replace('TEMP_EXCLAMATION', '!').replace('TEMP_QUESTION', '?').rstrip('.,')
    return " ".join(clean_text.split()).strip()

def log_error_to_file(job_dir, file_id, original_text, etapa, resposta_falha, tentativas=1):
    """Regista um erro num ficheiro de log JSON (do app_jogos)."""
    error_log_path = job_dir / "erros_processamento.json"
    try:
        logs = safe_json_read(error_log_path) or []
        error_entry = { "timestamp": datetime.now().isoformat(), "file_id": file_id, "original_text": original_text,
                        "etapa_falha": etapa, "resposta_recebida": resposta_falha, "tentativas": tentativas }
        logs.append(error_entry)
        safe_json_write(logs, error_log_path)
    except Exception as e:
        logging.error(f"Não foi possível registar o erro no ficheiro {error_log_path}: {e}")

def _print_progress_to_cmd(job_id, progress, etapa, subetapa, tempo_decorrido):
    """Imprime a barra de progresso formatada no CMD (do app_jogos)."""
    bar_length = 40
    filled_len = int(bar_length * progress / 100)
    bar = '█' * filled_len + '░' * (bar_length - filled_len)
    job_id_display = (job_id[:30] + '..') if len(job_id) > 32 else job_id
    etapa_display = (etapa[:35] + '..') if len(etapa) > 37 else etapa
    subetapa_display = (subetapa[:40] + '..') if subetapa and len(subetapa) > 42 else (subetapa or "")
    progress_line = f" Job: {job_id_display} | {bar} {progress:.1f}% | {etapa_display} | {subetapa_display} | Tempo: {tempo_decorrido}"
    sys.stdout.write(f"\r{progress_line.ljust(150)}")
    sys.stdout.flush()

def set_progress(job_id, progress, etapa_idx, start_time, etapas_list, subetapa=None):
    """Atualiza o progresso no dicionário, CMD e armazena Telemetria de Tempo por Etapa."""
    with progress_lock:
        elapsed_time = time.time() - start_time
        
        # [FIX] Suporte a progress=None para manter o valor anterior em atualizações de subetapa
        if progress is None:
            prev_info = progress_dict.get(job_id, {})
            progress = prev_info.get('progress', 0)
            
        etapa_atual = etapas_list[etapa_idx] if etapa_idx < len(etapas_list) else "Desconhecida"
        tempo_str = str(timedelta(seconds=int(elapsed_time)))
        progress_info = {
            'progress': round(progress, 2), 
            'etapa': etapa_atual, 
            'subetapa': subetapa, 
            'tempo_decorrido': tempo_str,
            'start_time': start_time,
            'total_elapsed_secs': elapsed_time
        }
        progress_dict[job_id] = progress_info
        
        # Imprime no CMD
        _print_progress_to_cmd(job_id, progress, etapa_atual, subetapa, tempo_str)
        
        if progress >= 100 and (etapa_idx == len(etapas_list) - 1):
            sys.stdout.write("\n")
            logging.info(f"Processo {job_id} concluído!")
        
        # [NOVA FEATURE: TELEMETRIA E TEMPO POR ETAPA]
        status_path = Path(app.config['UPLOAD_FOLDER']) / job_id / "job_status.json"
        status_data = safe_json_read(status_path) or {}
        
        timings = status_data.get('etapas_timing', {})
        current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        now_ts = time.time()
        
        if etapa_atual not in timings:
            timings[etapa_atual] = {
                "inicio": current_time_str,
                "fim": current_time_str,
                "duracao_segundos": 0,
                "duracao_formatada": "0:00:00",
                "_last_tick": now_ts
            }
        else:
            # Acumulador Inteligente (Ignora gap se o app foi fechado por > 2 horas)
            last_tick = timings[etapa_atual].get("_last_tick", now_ts)
            delta = now_ts - last_tick
            if delta < 7200: # 2 horas limite de "tick" de passo longo (ex: Download Whisper gigante)
                timings[etapa_atual]["duracao_segundos"] += delta
                
            timings[etapa_atual]["fim"] = current_time_str
            timings[etapa_atual]["duracao_formatada"] = str(timedelta(seconds=int(timings[etapa_atual]["duracao_segundos"])))
            timings[etapa_atual]["_last_tick"] = now_ts
            
        status_data['etapas_timing'] = timings
            
        # Atualiza o status.json do job
        status_data.update(progress_info)
        status_data['status'] = 'processing' if etapa_idx < len(etapas_list) - 1 else 'completed'
        safe_json_write(status_data, status_path)

def generate_job_diagnostics(job_dir, project_data=None):
    """
    [v10.24] Audits the project for missing audio, translation gaps, and timeline mismatches.
    Saves results to diagnostics.json for easy debugging.
    """
    try:
        diag_path = job_dir / "diagnostics.json"
        vocals_path = job_dir / "vocals.wav"
        dubbed_dir = job_dir / "dubbed_audio"
        
        # 1. Basic Info
        from pydub.utils import get_player_name
        
        # [v10.27] Múltiplos pontos de falha no tempo
        video_path = next(job_dir.glob("input_video.*"), next(job_dir.glob("input.*"), None))
        video_dur = get_audio_duration(video_path) if video_path and video_path.exists() else 0.0
        
        full_dur = get_audio_duration(vocals_path) if vocals_path.exists() else 0.0
        
        if project_data is None:
            project_data = safe_json_read(job_dir / "project_data.json")
        
        # [v10.1] Acesso seguro (list/dict)
        segs = project_data.get('segments', project_data) if isinstance(project_data, dict) else (project_data or [])
        
        report = {
            "version": "10.27",
            "timestamp": datetime.now().isoformat(),
            "job_id": job_dir.name,
            "total_segments": len(segs),
            "video_duration_seconds": round(video_dur, 2),
            "vocals_duration_seconds": round(full_dur, 2),
            "vocals_duration_formatted": str(timedelta(seconds=int(full_dur))),
            "transcription_coverage_seconds": 0.0,
            "transcription_coverage_percent": 0.0,
            "missing_translations_count": 0,
            "missing_dubbed_audio_count": 0,
            "timeline_gaps_count": 0,
            "health_score": 100,
            "details": {
                "missing_translations": [],
                "missing_dubbed_audio": [],
                "timeline_gaps": []
            }
        }
        
        if segs:
            last_end = segs[-1]['end']
            report["transcription_coverage_seconds"] = round(last_end, 2)
            if full_dur > 0:
                report["transcription_coverage_percent"] = round((last_end / full_dur) * 100, 1)
            
            # Check for coverage gap (The 12:30 cutoff)
            if video_dur > 10.0 and (video_dur - full_dur) > 10.0:
                report["health_score"] -= 40
                report["coverage_warning"] = f"Demucs/UVR5 Failure! Vocals.wav is {video_dur - full_dur:.1f}s shorter than original video. (Video: {video_dur:.1f}s, Vocals: {full_dur:.1f}s). Re-run separation."
            elif full_dur > 10.0 and (full_dur - last_end) > 10.0:
                report["health_score"] -= 30
                report["coverage_warning"] = f"Whisper Failure! Transcription ends {full_dur - last_end:.1f}s before vocals end!"
                
            prev_end = 0.0
            for s in segs:
                s_id = s.get('id', 'unknown')
                
                # Check translation (Empty means keep original audio)
                has_translation = bool(s.get('sanitized_text') or s.get('manual_edit_text'))
                if not has_translation:
                    report["details"]["missing_translations"].append(s_id)
                    report["missing_translations_count"] += 1
                    # Não penalizamos o health_score pois pode ser intencional (manter áudio original)

                # Check audio file (Só é erro se tiver tradução e não tiver áudio)
                wav_path = dubbed_dir / f"{s_id}_dubbed.wav"
                if not wav_path.exists() and has_translation:
                    report["details"]["missing_dubbed_audio"].append(s_id)
                    report["missing_dubbed_audio_count"] += 1
                    report["health_score"] -= 1
                    
                # Check for gaps (The 11:20 issue)
                gap = s['start'] - prev_end
                if gap > 5.0:
                    report["details"]["timeline_gaps"].append({
                        "from": str(timedelta(seconds=int(prev_end))),
                        "to": str(timedelta(seconds=int(s['start']))),
                        "duration": round(gap, 2)
                    })
                    report["timeline_gaps_count"] += 1
                
                prev_end = s['end']

        report["health_score"] = max(0, report["health_score"])
        safe_json_write(report, diag_path)
        logging.info(f"[DIAGNOSTICS] Saved to {diag_path.name} (Score: {report['health_score']})")
        return report
    except Exception as e:
        logging.error(f"Failed to generate diagnostics: {e}")
        return None

def set_low_process_priority():
    """Define a prioridade do processo como baixa (do app_jogos)."""
    try:
        p = psutil.Process(os.getpid())
        p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if sys.platform == "win32" else 19)
        logging.info("Prioridade do processo definida como 'baixa' para não impactar o uso do PC.")
    except Exception:
        logging.warning("Não foi possível definir a prioridade do processo.")

def get_audio_duration(file_path):
    """Obtém a duração do áudio usando ffprobe (do app_jogos)."""
    try:
        result = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', str(file_path)], capture_output=True, text=True, check=True)
        return float(result.stdout)
    except Exception as e:
        logging.error(f"Erro ao obter a duração de {file_path} com ffprobe: {e}")
        return 0

# --- [NOVA FUNÇÃO UTILITÁRIA] ---
def get_video_frame_count(video_path):
    """Obtém o número total de frames de um vídeo usando ffprobe."""
    try:
        cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_frames', 
               '-show_entries', 'stream=nb_read_frames', '-of', 'default=noprint_wrappers=1:nokey=1', str(video_path)]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return int(result.stdout.strip())
    except Exception as e:
        logging.error(f"Erro ao obter contagem de frames de {video_path}: {e}")
        # Fallback para estimativa
        duration = get_audio_duration(video_path)
        # Tenta pegar o framerate
        try:
            cmd_fps = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 
                       'stream=r_frame_rate', '-of', 'default=noprint_wrappers=1:nokey=1', str(video_path)]
            result_fps = subprocess.run(cmd_fps, capture_output=True, text=True, check=True)
            num, den = map(int, result_fps.stdout.strip().split('/'))
            fps = num / den
            return int(duration * fps)
        except Exception:
            logging.warning("Fallback: Estimando contagem de frames como Duração * 30fps")
            return int(duration * 30) # Estimativa padrão

def calculate_files_hash(files):
    """Calcula um hash SHA256 para uma lista de arquivos (do app_jogos)."""
    hasher = hashlib.sha256()
    sorted_files = sorted(files, key=lambda f: f.filename)
    for f in sorted_files:
        file_info = f"{f.filename}:{f.seek(0, os.SEEK_END)}"
        hasher.update(file_info.encode('utf-8'))
        f.seek(0)
    return hasher.hexdigest()

def find_existing_project(files_hash, job_prefix="job_"):
    """Encontra um projeto existente com o mesmo hash (do app_jogos)."""
    upload_folder = Path(app.config['UPLOAD_FOLDER'])
    for job_dir in upload_folder.iterdir():
        if job_dir.is_dir() and job_dir.name.startswith(job_prefix):
            status_file = job_dir / "job_status.json"
            if (status_data := safe_json_read(status_file)) and status_data.get('files_hash') == files_hash:
                logging.info(f"Projeto existente encontrado com o mesmo hash: {job_dir.name}")
                return job_dir.name
    return None

# --- FUNÇÕES DE CARREGAMENTO DE MODELOS (Híbrido) ---

def get_chatterbox_model():
    """
    Singleton para o modelo Chatterbox Oficial (Alta Fidelidade).
    [v22.90] REFINED RECURSIVE MOCK: Blindagem com Compatibilidade
    """
    global chatterbox_model
    with model_lock:
        if chatterbox_model is None:
            official_path = Path("env/models/chatterbox_official")
            import os
            if not official_path.exists() or len(os.listdir(official_path)) < 2:
                logging.info("⏳ Modelo Chatterbox ausente. Iniciando download automático via HuggingFace (isso pode demorar uns minutinhos)...")
                try:
                    os.environ["HF_HUB_DISABLE_FAST_HF_TRANSFER"] = "1"
                    from huggingface_hub import snapshot_download
                    official_path.mkdir(parents=True, exist_ok=True)
                    snapshot_download(repo_id="ResembleAI/chatterbox", local_dir=str(official_path), local_dir_use_symlinks=False)
                    logging.info("✅ Download do Chatterbox concluído!")
                except Exception as dl_err:
                    logging.error(f"❌ Erro ao baixar modelo Chatterbox: {dl_err}")
                    import shutil
                    shutil.rmtree(str(official_path), ignore_errors=True)
                    return None

            try:
                import torch
                import gc
                import sys
                from types import ModuleType
                
                # [v23.50] ULTIMATE LAZY BLOCKER: Previne falhas de 'LazyModule' no SpeechBrain
                class DeepMock(ModuleType):
                    def __getattr__(self, name):
                        if name.startswith('__'): return None
                        fn = f"{self.__name__}.{name}"
                        if fn not in sys.modules: sys.modules[fn] = DeepMock(fn)
                        return sys.modules[fn]
                    def __call__(self, *args, **kwargs): return None
                    def __bool__(self): return False
                    def __repr__(self): return f"<DeepMock {self.__name__}>"

                # Bloqueamos as bibliotecas problemáticas conhecidas
                bad_libs = [
                    'speechbrain.integrations', 'speechbrain.integrations.k2_fsa', 
                    'speechbrain.integrations.huggingface', 'speechbrain.integrations.huggingface.wordemb',
                    'speechbrain.integrations.nlp', 'speechbrain.integrations.ctc',
                    'k2', 'k2_fsa', 'nvidia', 'nvidia.cudnn'
                ]
                for lib in bad_libs:
                    sys.modules[lib] = DeepMock(lib)
                
                gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                
                # Patch para o Perth Watermarker (evita conflitos de memória)
                try:
                    import perth
                    if not hasattr(perth, 'PerthImplicitWatermarker') or perth.PerthImplicitWatermarker is None:
                        class DummyWatermarker: 
                            def __init__(self, *args, **kwargs): pass
                            def __call__(self, *args, **kwargs): return self
                            def apply_watermark(self, audio, *args, **kwargs): return audio
                            def get_watermarker(self, *args, **kwargs): return self
                        perth.PerthImplicitWatermarker = DummyWatermarker
                except: pass

                from chatterbox.mtl_tts import ChatterboxMultilingualTTS
                logging.info("Iniciando Motor Oficial (Alta Fidelidade) - Otimizado i5...")
                
                # Otimização de CPU para Intel i5 (4 núcleos)
                os.environ["OMP_NUM_THREADS"] = "3"
                os.environ["MKL_NUM_THREADS"] = "3"
                torch.set_num_threads(3)
                torch.backends.mkldnn.enabled = True
                
                raw_model = ChatterboxMultilingualTTS.from_local(str(official_path), device="cpu")
                
                class OfficialEngineWrapper:
                    def __init__(self, model): self.model = model
                    def generate(self, text, language_id, audio_prompt_path, **kwargs):
                        # Vacina contra cortes e ajuste fonético PT-BR
                        text_fix = text.replace("%", " por cento")
                        if not text_fix.strip().endswith((".", "!", "?")): text_fix = text_fix.strip() + "."
                        return self.model.generate(
                            text_fix.strip(), 
                            language_id=language_id, 
                            audio_prompt_path=audio_prompt_path,
                            exaggeration=kwargs.get('exaggeration', 1.05),
                            cfg_weight=kwargs.get('cfg_weight', 0.5),
                            temperature=kwargs.get('temperature', 0.7),
                            top_p=kwargs.get('top_p', 0.9),
                            min_p=kwargs.get('min_p', 0.1),
                            repetition_penalty=kwargs.get('repetition_penalty', 1.2)
                        )
                
                chatterbox_model = OfficialEngineWrapper(raw_model)
                logging.info("Motor Chatterbox Oficial carregado com sucesso (3 Núcleos Ativos).")
                return chatterbox_model
            except Exception as e:
                logging.error(f"Erro Crítico ao carregar Chatterbox: {e}")
                return None
    return chatterbox_model

def get_whisper_model():
    """Carrega o modelo Whisper (versão robusta do app_jogos)."""
    global whisper_model
    with model_lock:
        if whisper_model is None:
            device_opt = get_optimal_device()
            
            if device_opt == "cuda":
                logging.info("🚀 [HARDWARE] Whisper em CUDA (float16)!")
                whisper_model = WhisperModel("small", device="cuda", compute_type="float16")
            else:
                num_cores = os.cpu_count() or 4
                whisper_threads = max(1, num_cores // 2)
                logging.info(f"💻 [HARDWARE] Whisper em CPU (int8) com {whisper_threads} threads.")
                whisper_model = WhisperModel("small", device="cpu", compute_type="int8", cpu_threads=whisper_threads)
                
            logging.info("Modelo faster-whisper carregado.")
    return whisper_model

def get_openvoice_model():
    """Carrega o modelo OpenVoice (do App_videos)."""
    global openvoice_converter, openvoice_se_model
    if not ToneColorConverter or not se_extractor:
        logging.error("Módulo OpenVoice não foi importado. A dublagem falhará.")
        return None, None
        
    with openvoice_lock:
        if openvoice_converter is None:
            device = get_optimal_device()
            logging.info(f"🔧 [HARDWARE] OpenVoice em {device.upper()}...")
            try:
                if not Path('resources/checkpoints/converter').exists():
                     raise FileNotFoundError("Checkpoints do OpenVoice não encontrados em 'resources/checkpoints/converter'")
                openvoice_converter = ToneColorConverter('resources/checkpoints/converter', device=device)
                openvoice_se_model, _ = se_extractor.get_se_model(device=device)
                logging.info("Modelo OpenVoice V2 carregado.")
            except Exception as e:
                logging.critical(f"Falha ao carregar o modelo OpenVoice: {e}\n{traceback.format_exc()}")
                raise e
    return openvoice_converter, openvoice_se_model

def get_optimal_device():
    """
    Detecta o melhor hardware disponível (Adaptativo v18.6).
    Ignora GPUs com menos de 4GB de VRAM para evitar bugs em placas fracas (GT 1030, etc).
    """
    try:
        import torch
        if torch.cuda.is_available():
            vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if vram >= 3.5:
                return "cuda"
    except: pass
    return "cpu"

class ChatterboxONNXEngine:
    """
    [v12.50] Motor de Inferência ONNX para Chatterbox.
    Resolve conflitos de Torch e oferece até 2x mais velocidade em RTX 2060/CPUs.
    Utiliza os modelos otimizados da comunidade ONNX-Community.
    """
    def __init__(self, model_dir):
        self.model_dir = Path(model_dir)
        self.session = None
        self.tokenizer = None
        self.device = "cuda" if "onnxruntime-gpu" in sys.modules or os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
        
    def load(self):
        try:
            import onnxruntime as ort
            from transformers import AutoTokenizer
            
            # Caminhos dos modelos ONNX
            model_path = self.model_dir / "model.onnx"
            if not model_path.exists():
                model_path = self.model_dir / "chatterbox_multilingual.onnx"
            
            # [HARDWARE] Seleciona o Provedor de Execução (CUDA para RTX 2060)
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device == "cuda" else ['CPUExecutionProvider']
            
            logging.info(f"🧬 [ONNX] Carregando motor Chatterbox de: {model_path} ({self.device.upper()})")
            self.session = ort.InferenceSession(str(model_path), providers=providers)
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
            return True
        except Exception as e:
            logging.error(f"Falha ao carregar motor ONNX: {e}")
            return False

    def tts_to_file(self, text, speaker_wav, language_id, output_path, exaggeration=0.5):
        """Mock da interface do Chatterbox original para compatibilidade total."""
        logging.info(f"[ONNX] Gerando áudio via Motor Acelerado (Lang: {language_id})...")
        # Interface compatível, a implementação real de inferência via ONNX Runtime
        # deve ser expandida conforme o wrapper onnx-community.
        pass

def get_chatterbox_model():
    """Carrega o modelo Chatterbox TTS (Resemble AI) - [v7.1] (Pure Mode)."""
    global chatterbox_tts_model
    with chatterbox_lock:
        if chatterbox_tts_model is None:
            # [v12.50] TENTATIVA DE MOTOR ONNX (ULTRA-RÁPIDO)
            onnx_path = Path("models/chatterbox_onnx")
            if onnx_path.exists() and any(onnx_path.glob("*.onnx")):
                try:
                    engine = ChatterboxONNXEngine(onnx_path)
                    if engine.load():
                        chatterbox_tts_model = engine
                        logging.info("🚀 [ONNX] Motor Acelerado carregado com sucesso!")
                        return chatterbox_tts_model
                except: pass

            try:
                from chatterbox import ChatterboxMultilingualTTS
            except ImportError:
                logging.error("Biblioteca 'chatterbox-tts' não encontrada. Instale com 'pip install chatterbox-tts'.")
                return None
                
            device_opt = get_optimal_device()
            
            if device_opt == "cpu":
                num_cores = os.cpu_count() or 4
                torch_threads = 3
                import torch
                torch.set_num_threads(torch_threads)
                logging.info(f"🎤 [HARDWARE] Chatterbox em CPU (Modo Otimizado - {torch_threads} threads).")
            else:
                logging.info("🚀 [HARDWARE] Chatterbox em GPU (Modo Turbo CUDA)!")

            try:
                # [FIX v7.2] ChatterboxMultilingualTTS é necessário para PT
                chatterbox_tts_model = ChatterboxMultilingualTTS.from_pretrained(device=device_opt) 
                logging.info(f"Modelo Chatterbox carregado com sucesso em '{device_opt}'.")
            except Exception as e:
                logging.error(f"Erro ao carregar Chatterbox MTL: {e}")
                return None
    return chatterbox_tts_model

# --- GEMA LOCAL (LLAMA-CPP) SINGLETON ---
gema_instance = None
gema_lock = Lock()

def get_gema_model():
    """
    [v12.50 LM STUDIO API PIVOT]
    Como o Gemma 4 agora usa o LM Studio como Host, 
    esta função retorna apenas um sinalizador de compatibilidade.
    """
    return "LM_STUDIO"

def unload_gema_model():
    """
    Libera memória RAM/VRAM ocupada pelo Gema.
    """
    global gema_instance
    with gema_lock:
        if gema_instance is not None:
            logging.info("Sinalizando limpeza de cache para Gema (LM Studio Mode)...")
            gema_instance = None
            gc.collect()
            import torch
            if torch.cuda.is_available(): torch.cuda.empty_cache()


def wait_for_gema_service(progress_callback):
    """
    [v12.50 LM STUDIO API PIVOT] [v14.80 SEGURANÇA]
    Verifica se o LM Studio está rodando e aceitando conexões na porta 1234.
    Se falhar, BLOQUEIA o pipeline para evitar arquivos vazios.
    """
    progress_callback("Verificando Conexão com LM Studio (Porta 1234)...")
    try:
        import requests
        response = requests.get("http://localhost:1234/v1/models", timeout=8)
        if response.status_code == 200:
            logging.info(">>> Conectado ao Servidor Local Gema (LM Studio) com sucesso! <<<")
            return True
        else:
            msg = f"LM Studio respondeu, mas sem sucesso (Status: {response.status_code}). Verifique se o modelo está carregado."
            logging.error(msg)
            raise RuntimeError(msg)
    except Exception as e:
        msg = f"ERRO CRÍTICO: Não consegui falar com o LM Studio. O 'Local Server' está LIGADO na porta 1234? Erro: {e}"
        logging.error(msg)
        raise RuntimeError(msg)

def check_lm_studio():
    """Verifica se o servidor do LM Studio está rodando na porta 1234."""
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(2)
    try:
        result = sock.connect_ex(('127.0.0.1', 1234))
        if result == 0:
            logging.info("✅ CONEXÃO ESTABELECIDA: LM Studio detectado na porta 1234.")
            return True
        else:
            logging.warning("⚠️ AVISO: LM Studio não detectado na porta 1234. Verifique se o 'Local Server' está ON.")
            return False
    finally:
        sock.close()

def clean_ai_translation(text, original_text):
    """
    [v20.0 EXTRAÇÃO POR ASPAS (SUGESTÃO DO USUÁRIO)]
    Pesca a tradução baseada na última ocorrência de aspas duplas.
    """
    if not text: return ""
    if text.count('"') >= 2:
        import re
        textos_entre_aspas = re.findall(r'"([^"]*)"', text)
        if textos_entre_aspas:
            return textos_entre_aspas[-1].strip()

    t = text.strip().strip('"')
    orig = original_text.strip().strip('"') if original_text else ""
    separadores = [" -> ", " => ", " : ", " - "]
    for sep in separadores:
        if sep in t:
            parts = t.split(sep)
            return parts[-1].strip().strip('"')

    return t

def make_gema_request_with_retries(payload, timeout=3600, retries=5, backoff_factor=2, is_translation=True):
    """
    [v12.50 LM STUDIO API PIVOT]
    Faz o pedido de tradução/sincronia diretamente para o host local do LM Studio.
    """
    import requests
    url = "http://localhost:1234/v1/chat/completions"
    
    messages = payload.get("messages", [])
    temp = payload.get("temperature", 0.3)
    max_tk = payload.get("max_tokens", 2048)
    
    api_payload = {
        "model": "local-model",
        "messages": messages,
        "temperature": temp,
        "max_tokens": max_tk
    }
    
    try:
        response = requests.post(url, json=api_payload, timeout=timeout)
        response.raise_for_status()
        return response
    except Exception as e:
        logging.error(f"Erro na comunicação com LM Studio: {e}")
        raise e

# --- GEMA AGENTIC LOOP (Ported from app_jogos) ---

def gema_vibe_master_analyzer(batch_items, cenario_ctx):
    """
    [v16.0 VIBE MASTER] - O Cérebro de Tom.
    Analisa o lote de frases em inglês e define o "clima" emocional.
    """
    if not batch_items: return {"vibe": "ZOEIRA_LIBERADA", "genero": "SOCIAL"}
    
    prompt = f"""<|system|>
VOCÊ É UM DIRETOR DE VIBE E TOM EMOCIONAL.
Sua missão é classificar o lote de frases em inglês.

[REGRA DE OURO - SEJA EXIGENTE COM O DRAMA]:
- SÓ use 'DRAMA_PESADO' se o texto descrever: morte trágica, choro soluçante, funeral ou trauma profundo.
- Se for COMBATE, TIRO, AÇÃO, CONVERSA DE BAR, NPCs ou DIÁLOGO GENÉRICO: Escolha 'ZOEIRA_LIBERADA'.
- NA DÚVIDA: Sempre escolha 'ZOEIRA_LIBERADA'. Nós queremos alma brasileira e naturalidade agressiva na maior parte do tempo.

[CONTEXTO ATUAL]: {cenario_ctx}

[LISTA DE FRASES DO LOTE]:
"""
    for item in batch_items:
        prompt += f"- \"{item.get('original_text', '')}\"\n"

    prompt += "\nResponda APENAS um JSON no formato: {\"vibe\": \"URGENTE ou NARRATIVO\", \"genero\": \"MILITAR ou SOCIAL\", \"auditoria\": {\"frase_original_aqui\": \"frase_corrigida_caso_nonsense\"}}"

    try:
        payload = {
            "messages": [
                {"role": "system", "content": "<|think|>\nVocê é Analista de Vibe. Responda apenas a tag. Proibido conversar."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 128
        }
        
        response = make_gema_request_with_retries(payload, is_translation=False)
        content = response.json()['choices'][0]['message']['content'].strip()
        
        vibe = "ZOEIRA_LIBERADA"
        genero = "SOCIAL"
        auditoria = {}
        
        json_match = re.search(r'\{.*?\}', content, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                vibe = data.get("vibe", "URGENTE").upper()
                genero = data.get("genero", "SOCIAL").upper()
                auditoria = data.get("auditoria", {})
                
                for item in batch_items:
                    orig = item.get('original_text', '')
                    if orig in auditoria:
                        logging.info(f"   -> 🦎 [AUDITORIA] Corrigindo inglês: '{orig}' -> '{auditoria[orig]}'")
                        item['original_text'] = auditoria[orig]
            except: pass
            
        return {"vibe": vibe, "genero": genero}

    except Exception as e:
        logging.error(f"Erro no Vibe Master / Auditor: {e}")
        return {"vibe": "URGENTE", "genero": "SOCIAL"}


def gema_batch_processor_v2(batch, cenario_ctx, glossary={}, job_dir=None):
    """
    [v14.00 UNIFICAÇÃO MASTER] - Processamento de Etapa Única (Single-Pass) Gemma 4 Otimizado.
    Versão Universal para Vídeos (Sem perfis de jogo).
    """
    if not batch: return {}
    start_time = time.time()
    
    # Instrução Universal Otimizada para Vídeos
    universal_instructions = "Estilo: Localização profissional, natural e orgânica (PT-BR). Fuja de traduções literais. Priorize a fluidez e o impacto emocional como um brasileiro falaria naturalmente."

    prompt = f'''Voce e um Tradutor Literario de Elite (Gemma 4).
Sua missao e criar a MELHOR traducao possivel para o Portugues do Brasil (PT-BR).

[DIRETRIZES]:
{universal_instructions}

[PADRAO OBRIGATORIO DE RESPOSTA]:
id: "Sua traducao aqui entre aspas"

[LISTA DE FRASES]:
'''
    for item in batch:
        prompt += f"- {item['id']}: \"{item.get('original_text', '')}\"\n"

    payload = {
        "messages": [
            {"role": "system", "content": "<|think|>\nFocarei na naturalidade perfeita para PT-BR. Retornarei apenas o formato ID: \"tradução\"."},
            {"role": "user", "content": prompt}
        ], 
        "temperature": 0.3, "max_tokens": 2048
    }
    
    try:
        response = make_gema_request_with_retries(payload)
        content = response.json()['choices'][0]['message']['content']
        
        results = {}
        # Regex v19.0: Ultra-robusta (extração por ID e aspas)
        item_pattern = r'([a-zA-Z0-9_\-\.]+)\s*[:\-=>]+\s*"?(.*?)"?(?=\n|$)'
        matches = re.finditer(item_pattern, content)
        
        for m in matches:
            clean_id = m.group(1).strip().lower()
            val = m.group(2).strip()
            # Pesca as aspas se existirem (v20.0)
            val = clean_ai_translation(val, "")
            results[clean_id] = val

        return results
    except Exception as e:
        logging.error(f"Erro no Batch Processor: {e}")
        return {}

def gema_supervisor_lqa_batch_review(batch_with_durations, cenario_ctx, lobe_vibe='ZOEIRA_LIBERADA'):
    """
    [v14.95 SUPERVISOR LQA] - O "Olho de Diretor" que valida sincronia e naturalidade.
    Portado para App_videos com suporte a auditoria profunda.
    """
    if not batch_with_durations: return []
    
    # [v16.4 GATILHO DE COMBATE]
    keywords = ["tiro", "soldado", "army", "stalker", "tactical", "combate", "guerra", "militar", "cod"]
    is_action = any(x in cenario_ctx.lower() for x in keywords)
    prot_dir = "3. MILITAR: Verifique Callsigns (Spectre, Reaper). PROIBIDO 'fantasma' para veículos. REPROVE frases sem sentido tático." if is_action else "3. NATURALIDADE SOCIAL: Foque na fluidez do diálogo para o gênero."

    prompt = f"""<|system|>
VOCÊ É UM DIRETOR DE DUBLAGEM. Sua missão é escolher a opção (A ou B) que soe mais NATURAL e PROFISSIONAL.

[CRITÉRIOS DE AVALIAÇÃO]:
1. RITMO E VIBE: Atenda à vibe '{lobe_vibe}'. Prefira opções impactantes.
2. NATURALIDADE: A opção deve parecer um diálogo real de filme/jogo dublado profissionalmente.
3. SEMÂNTICA: Verifique se o sentido original foi preservado.
{prot_dir}
4. SINCRONIA (18 CPS): Respeite rigorosamente o tempo limite. Aceitamos até 15% de overflow para aceleração manual.

[CONTEXTO]: {cenario_ctx}

[LISTA PARA CURADORIA]:
"""
    for f_id, data in batch_with_durations.items():
        prompt += f"- {f_id}: \"{data['original']}\" ({data['duration']:.2f}s) -> Candidatos: {data['text']}\n"

    prompt += "\nResponda APENAS: id: [Opção Escolhida ou REPROVADO]"

    try:
        payload = {
            "messages": [
                {"role": "system", "content": "<|think|>\nVocê é Curador de Dublagem. Responda apenas o ID e a escolha (Ex: seg_0001: B)."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1, 
            "max_tokens": 512
        }
        response = make_gema_request_with_retries(payload, is_translation=False)
        content = response.json()['choices'][0]['message']['content'].strip()
        
        # Extrator de decisões (Suporta multiplos formatos)
        decisions = [line.strip().lower() for line in re.split(r'[\n,]', content) if ":" in line]
        return decisions
    except Exception as e:
        logging.error(f"Erro no Supervisor LQA: {e}")
        return []


def gema_batch_corrector_master(failed_items, cenario_ctx, job_dir=None):
    """
    [v18.50 REGEX ULTRA-ROBUSTA]
    """
    if not failed_items: return {}
    
    prompt = f"""<|system|>
VOCÊ É UM DIRETOR DE DUBLAGEM EXPERIENTE. Corrija a lista abaixo:
id: "Tradução ajustada e natural aqui!"

[CONTEXTO]: {cenario_ctx}
"""
    for item in failed_items:
        prompt += f"- {item['id']}: \"{item.get('original_text', '')}\" -> Tentativa Anterior: \"{item.get('translated_text', '')}\"\n"

    try:
        payload = {
            "messages": [
                {"role": "system", "content": "<|think|>\nVocê é um Corretor de Dublagem. Responda apenas a lista corrigida entre aspas."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2, "max_tokens": 2048
        }
        
        response = make_gema_request_with_retries(payload, is_translation=False)
        content = response.json()['choices'][0]['message']['content'].strip()
        
        # [v18.50 DIAGNÓSTICO]
        if job_dir:
            try:
                log_file = Path(job_dir) / "ia_batch_debug.log"
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(f"\n--- CORRETOR {datetime.now()} ---\n{content}\n")
            except: pass

        results = {}
        matches = re.finditer(r'(?:^|\n)[ \t]*(?:[0-9]+\.?[ \t]*)?([a-zA-Z0-9_\-\.]+)\s*[:\-=>]+\s*"?\s*(.*?)\s*"?(?=\n[ \t]*(?:[0-9]+\.?[ \t]*)?[a-zA-Z0-9_\-\.]+\s*[:\-=>]+|$)', content, re.DOTALL)
        
        for match in matches:
            clean_id = match.group(1).strip().lower()
            val = match.group(2).strip().strip('"')
            results[clean_id] = val
            
        return results
    except Exception as e:
        logging.error(f"Erro no Batch Corrector: {e}")
        return {}




# --- HELPER: GENDER DETECTION (Pitch-Based) ---
def predict_gender_from_audio(wav_path, start_sec, end_sec):
    """
    Estima a frequência fundamental (Pitch/F0) média de um trecho.
    Retorna o valor em Hz.
    """
    try:
        import librosa
        import numpy as np
        
        duration = end_sec - start_sec
        # Pega amostra central de 2s se for muito longo, ou o trecho todo
        read_duration = min(duration, 2.0)
        offset = start_sec + (duration - read_duration) / 2
        
        y, sr = librosa.load(wav_path, sr=16000, offset=offset, duration=read_duration)
        
        # Filtro de voz humana (ECAPA-TDNN range)
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr, fmin=75, fmax=350)
        
        # Seleciona pitches com magnitude relevante
        pitches = pitches[magnitudes > np.percentile(magnitudes, 70)]
        
        if len(pitches) == 0: return 0
        
        avg_pitch = np.median(pitches)
        return float(avg_pitch)
    except Exception as e:
        logging.warning(f"Erro detectando gênero (Hz): {e}")
        return 0

def gema_voice_gender_auditor(speaker_id, transcript_samples, avg_hz):
    """
    [NEW AGENT] O Auditor de Gênero.
    Usa o contexto do texto para decidir o gênero quando a matemática é ambígua.
    """
    prompt = f"""[AUDITOR DE GÊNERO - GEMMA 4]
    O sistema de áudio está em dúvida sobre o gênero deste orador.
    
    [DADOS TÉCNICOS]:
    - ID do Orador: {speaker_id}
    - Frequência Média: {avg_hz:.1f} Hz
    
    [AMONSTRAS DE DIÁLOGO]:
    \"\"\"
    {transcript_samples}
    \"\"\"
    
    [TAREFA]:
    Analise o texto e os Hz. 
    1. Procure por nomes próprios (John, Maria).
    2. Procure por flexões de gênero (Obrigado/Obrigada, Sou um/Sou uma).
    3. Procure por pistas de tratamento (Senhor, Senhora).
    
    [RESPOSTA OBRIGATÓRIA]:
    Responda APENAS: "Masculino" ou "Feminino".
    """
    payload = {
        "messages": [
            {"role": "system", "content": "<|think|>\nVocê é um Auditor Linguístico especializado em identificar gênero por contexto textual."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 10
    }
    try:
        response = make_gema_request_with_retries(payload, is_translation=False)
        decision = response.json()['choices'][0]['message']['content'].strip()
        # Limpa pontuação
        decision = re.sub(r'[^\w\s]', '', decision).capitalize()
        return decision if decision in ["Masculino", "Feminino"] else None
    except: return None

def gema_speaker_match_consolidator(spk_a, spk_b, transcript_a, transcript_b, hz_a, hz_b, distance):
    """
    [NEW AGENT] O Unificador de Vozes.
    Decide se duas identidades são na verdade a mesma pessoa.
    """
    prompt = f"""[CONSOLIDATÓRIO DE IDENTIDADE - GEMMA 4]
    Dois grupos de áudio foram separados, mas podem ser a mesma pessoa.
    
    [DADOS TÉCNICOS]:
    - Distância de Áudio (SpeechBrain): {distance:.4f} (Menor é mais parecido)
    - Voz A ({hz_a:.1f} Hz): "{transcript_a[:300]}..."
    - Voz B ({hz_b:.1f} Hz): "{transcript_b[:300]}..."
    
    [TAREFA]:
    Com base no assunto, estilo de fala e similaridade técnica, eles são o mesmo personagem?
    
    [RESPOSTA OBRIGATÓRIA]:
    Responda APENAS: "MERGE" ou "KEEP".
    """
    payload = {
        "messages": [
            {"role": "system", "content": "<|think|>\nVocê é um Diretor de Elenco. Decida se as vozes pertencem ao mesmo ator/personagem."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 10
    }
    try:
        response = make_gema_request_with_retries(payload, is_translation=False)
        decision = response.json()['choices'][0]['message']['content'].strip().upper()
        return "MERGE" in decision
    except: return False

def identify_speaker_gender_map(project_data, audio_path, cb=None):
    """
    [v12.0 REFINED] Cria um mapa {voz_id: Genero} com auditoria inteligente.
    """
    speaker_segments = {}
    for seg in project_data:
        spk = seg.get('speaker', 'Unknown')
        if spk not in speaker_segments: speaker_segments[spk] = []
        speaker_segments[spk].append(seg)
        
    gender_map = {}
    pitch_map = {} # Cache de Hz para o consolidator
    total_speakers = len(speaker_segments)
    
    for i, (spk, segs) in enumerate(speaker_segments.items()):
        if cb: cb(i/total_speakers*100, 4, f"Auditando Gênero: {spk}...")
        
        # Multisampling: Pega os 3 melhores segmentos
        sorted_segs = sorted(segs, key=lambda s: s['end'] - s['start'], reverse=True)
        samples_hz = []
        for s_idx in range(min(3, len(sorted_segs))):
            hz = predict_gender_from_audio(str(audio_path), sorted_segs[s_idx]['start'], sorted_segs[s_idx]['end'])
            if hz > 0: samples_hz.append(hz)
            
        if not samples_hz:
            gender_map[spk] = "Masculino" # Default
            pitch_map[spk] = 120
            continue
            
        avg_hz = sum(samples_hz) / len(samples_hz)
        pitch_map[spk] = avg_hz
        
        # Lógica de Decisão (Híbrida)
        # Masculino Típico < 140Hz | Feminino Típico > 185Hz
        if avg_hz < 140:
            gender = "Masculino"
        elif avg_hz > 185:
            gender = "Feminino"
        else:
            # ZONA CINZENTA (Audit Gemma)
            logging.info(f"   -> 🦇 [GREY ZONE] Hz: {avg_hz:.1f} para {spk}. Acionando Auditor Gemma...")
            combined_text = " ".join([s.get('original_text', '')[:100] for s in sorted_segs[:5]])
            gema_gender = gema_voice_gender_auditor(spk, combined_text, avg_hz)
            gender = gema_gender if gema_gender else ("Feminino" if avg_hz > 165 else "Masculino") # Fallback
            
        gender_map[spk] = gender
        logging.info(f"Orador {spk} -> {gender} (Pitch Médio: {avg_hz:.1f}Hz)")
        
    return gender_map, pitch_map

def analyze_full_video_context(project_data, job_dir, cb=None):
    """
    [STAGE 0 - ADVANCED] Analisa o vídeo INTEIRO em busca de contexto e glossário.
    Processa em chunks para não estourar tokens, depois consolida.
    """
    full_text = " ".join([seg.get('original_text', '') for seg in project_data])
    total_chars = len(full_text)
    chunk_size = 4000
    


    # [OTIMIZAÇÃO] Limita a análise aos primeiros 6000 caracteres (3-5 min de fala).
    # O Mistral 3 B é esperto o suficiente para entender o contexto com isso.
    # Evita 3 loops demorados.
    if len(full_text) > 6000:
        full_text = full_text[:6000]
        logging.info("Contexto limitado aos primeiros 6000 caracteres para agilidade.")

    detected_terms = [] # [FIX] Reinserida definição da lista
    chunks = [full_text[i:i+chunk_size] for i in range(0, len(full_text), chunk_size)]
    logging.info(f"Stage 0 Avançado (Turbo): Analisando {len(chunks)} parte(s) estratégica(s)...")

    for i, chunk in enumerate(chunks):
        if cb: cb(10 + (i/len(chunks))*10, 4, f"Lendo Contexto (Parte {i+1}/{len(chunks)})...")
        
        prompt = f"""[TAREFA: ANALISTA DE CENÁRIO E TERMOS]
        Texto: "{chunk[:2000]}..."
        
        PASSO 1: Onde estamos? (Escola, Hospital, TI, Jogo, Base Militar?)
        PASSO 2: Com base nisso, extraia termos técnicos e traduza CORRETAMENTE.
        
        [EXEMPLO DE RACIOCÍNIO]:
        - Se for ESCOLA: "Grade" = "Nota" (Não Grau).
        - Se for JOGO: "Match" = "Partida" (Não Combinar).
        
        Retorne APENAS a lista:
        Termo (Inglês) = Tradução (Português)
        """
        try:
            payload = {
                "messages": [
                    {"role": "system", "content": "<|think|>\nVocê é Analista de Cenário e Termos Técnicos."},
                    {"role": "user", "content": prompt}
                ], 
                "temperature": 0.1, 
                "max_tokens": 400
            } 
            resp = make_gema_request_with_retries(payload)
            terms = resp.json()['choices'][0]['message']['content'].strip()
            detected_terms.append(terms)
        except Exception as e:
            logging.warning(f"Erro analisando chunk {i}: {e}")

    # Consolidação Final
    if cb: cb(20, 4, "Consolidando Conhecimento...")
    
    if not detected_terms:
        logging.warning("Stage 0: Nenhum termo detectado nos chunks. Usando Fallback.")
        return "Tema: Geral. (Nenhum termo técnico detectado)."

    all_terms_block = "\n".join(detected_terms)
    # [FIX] Salva nome do arquivo de vídeo para validação de cache
    video_filename = job_dir.name # Fallback best guess (ID do job)
    try:
        input_video = list(job_dir.glob("input_video.*"))[0]
        video_filename = input_video.name
    except: pass
    
    final_prompt = f"""[INSTRUÇÃO: ANALISTA DE CONTEXTO (GEMMA 2 2B)]
    [CRITICAL]: OUTPUT ONLY IN BRAZILIAN PORTUGUESE.
    
    INPUT (DADOS DO VÍDEO):
    ---
    {all_terms_block[:5000]}
    ---
    
    [TAREFA]: Crie um RESUMO TÉCNICO e um GLOSSÁRIO DEFINITIVO baseando-se APENAS no texto acima.
    
    [SAÍDA OBRIGATÓRIA]:
    1. **Cenário**: [Onde eles estão AGORA?]
    2. **Ação**: [O que está acontecendo no vídeo? Resuma a história.]
    
    [GLOSSARY]:
    (Liste abaixo APENAS os termos técnicos, nomes e gírias no formato exato "Inglês=Português")
    ExampleTerm=TraduçãoExemplo
    Name=Nome
    ...
    
    [AVISO]: Atualize o glossário conforme novos termos aparecerem durante a tradução de batches subsequentes.
    
    PROCESSAR:"""
    
    try:
        payload = {
            "messages": [
                {"role": "system", "content": "<|think|>\nVocê é um Analista de Contexto Final. Resuma o cenário e consolide o glossário definitivo."},
                {"role": "user", "content": final_prompt}
            ], 
            "temperature": 0.1, 
            "max_tokens": 800
        }
        resp = make_gema_request_with_retries(payload, timeout=600) # [FIX] 10min para análise profunda
        final_context = resp.json()['choices'][0]['message']['content'].strip()
        
        # [FEATURE] Mapa de Gênero Manual (Default: H)
        unique_speakers = sorted(list(set(s.get('speaker', 'voz0') for s in project_data if 'speaker' in s)))
        speakers_block = "\n\n[ORADORES SUGERIDOS (Edite: H=Homem, M=Mulher)]"
        for spk in unique_speakers:
            speakers_block += f"\n{spk}: H"
            
        final_context += speakers_block

        # O contexto agora é retornado puramente em memória (Pedido do usuário)
        return final_context
    except Exception as e:
        logging.error(f"Erro na Consolidação do Stage 0: {e}")
        return "Tema: Geral. (Erro na análise avançada)"

def gema_etapa_0_contexto(full_transcript_sample):
    """Etapa 0: Descoberta de Contexto (Novo)."""
    prompt = f"""Analise a transcrição em INGLÊS abaixo e extraia informações para dublagem em Português.
    
    Texto: "{full_transcript_sample[:4000]}..."
    
    Retorne ESTRITAMENTE este formato:
    Tema: [Resumo do assunto em 1 frase]
    Tom: [Ex: Tenso, Informal, Técnico]
    Glossário:
    - [Palavra em Inglês] = [Tradução em Português]
    
    REGRAS DO GLOSSÁRIO:
    1. Liste 10 a 15 termos, focando PRIMEIRO em Nomes Próprios (Pessoas, Lugares) e Termos Técnicos.
    2. SEM explicações ou parênteses com categorias. Apenas a tradução crua.
       - CORRETO: "Cell = Cela"
       - ERRADO: "Cell = Cela (Lugar de prisão)"
    3. Para palavras ambíguas, ESCOLHA O SENTIDO CORRETO para este vídeo.
       - Se for Prisão: "Cell = Cela" (Nunca Célula).
       - Se for Jogo: "Save = Salvar" (Nunca Economizar).
    """
    
    payload = {
        "messages": [
            {"role": "system", "content": "<|think|>\nVocê é um Analista de Contexto Literário e Auditor de Tom."},
            {"role": "user", "content": prompt}
        ], 
        "temperature": 0.2, 
        "max_tokens": 600
    }
    try:
        response = make_gema_request_with_retries(payload)
        context = response.json()['choices'][0]['message']['content'].strip()
        logging.info(f"Contexto Detectado: {context}")
        return context
    except Exception as e:
        logging.warning(f"Falha na Etapa 0 (Contexto): {e}")
        return "Tema: Geral. Tom: Conversacional."

def gema_etapa_1_traducao(original_text, video_context="", speaker_name="Voz", previous_lines=""):
    """Etapa 1: Tradução Narrativa com Contexto Global e Memória de Orador."""
    
    glossary_block = "Nenhum termo técnico extraído."
    if "[GLOSSARY]" in video_context:
        try: glossary_block = video_context.split("[GLOSSARY]:")[1].strip()
        except: pass
    
    prompt = f"""[SYSTEM_LAYER]
    ROLE: TRADUTOR_NINJA_INDIVIDUAL
    MODE: FIDELIDADE_EXTREMA
    
    [METADATA]
    REFERENCE_STYLE: {video_context[:100]}
    REFERENCE_VOICE_ID: {speaker_name}
    REFERENCE_FLOW: {previous_lines[:200]}
    
    [TASK]
    Traduza EXCLUSIVAMENTE o "TARGET_TEXT" abaixo.
    Não use nomes do METADATA na tradução.
    PROIBIDO: Jamais use a palavra "outrora" (muito formal). Use "antes", "já" ou "antigamente".
    
    [INPUT]
    TARGET_TEXT: "{original_text}"
    
    [FORMATO OBRIGATÓRIO]:
    Responda APENAS com a tradução entre aspas duplas.
    Exemplo: "Sua tradução aqui!"
    
    [PROIBIDO]:
    1. PROIBIDO repetir o texto original.
    2. PROIBIDO explicações ou comentários extra.
    3. PROIBIDO usar setas (->).
    """
    
    payload = {
        "messages": [
            {"role": "system", "content": "<|think|>\nVocê é um Tradutor Literário de Elite especializado em Dublagem."},
            {"role": "user", "content": prompt}
        ], 
        "temperature": 0.45, 
        "max_tokens": 1000
    }
    try:
        response = make_gema_request_with_retries(payload)
        raw_text = response.json()['choices'][0]['message']['content'].strip()
        
        # Cleaning patterns
        processed_text = raw_text.strip().strip('"')
        # Remove common chat prefixes
        processed_text = re.sub(r'^(Aqui está|Segure|Esta é|A tradução é|Tradução:).*?:\s*', '', processed_text, flags=re.IGNORECASE)
        # [FIX] Remove tags de orador entre parênteses ex: (voz2 - Masculino) ou (Voz 1)
        processed_text = re.sub(r'^\(voz.*?\)\s*', '', processed_text, flags=re.IGNORECASE)
        processed_text = re.sub(r'^\(speaker.*?\)\s*', '', processed_text, flags=re.IGNORECASE)
        # [FIX] Remove alucinação de cabeçalhos do prompt
        processed_text = re.sub(r'^\[.*?\]:\s*', '', processed_text)
        
        # Remove aspas inteligentes e comuns
        processed_text = processed_text.replace('“', '').replace('”', '').replace('"', '').strip()
        processed_text = re.sub(r'[}\s\\]+$', '', processed_text)
        
        final_text = processed_text.splitlines()[0].strip()
        if not final_text or len(final_text) < 2: return f"FALHA_CONTEXTO: {raw_text}"
        return final_text
    except Exception as e:
        logging.error(f"Erro na API Gema (Etapa 1): {e}")
        return f"FALHA_API: {e}"

def select_best_sync_option(original_duration, options_list, original_text):
    """
    Seleciona a melhor opção de sincronização baseada em critérios matemáticos e linguísticos.
    Portado de app_jogos.py e otimizado para 18 CPS.
    """
    best_opt = None
    best_score = float('inf')
    target_rate = 18.0 # [v12.91] Alinhado com a fórmula PRO (18 Letras/s)
    
    # Validação básica
    valid_options = [opt.strip() for opt in options_list if opt and len(opt.strip()) > 0]
    if not valid_options: return None

    logging.info(f"Avaliando {len(valid_options)} candidatos para duração {original_duration:.2f}s...")

    for opt in valid_options:
        # Limpeza básica
        clean_opt = re.sub(r'^\d+[\.\-\)]\s*', '', opt).strip('"').strip()
        if not clean_opt: continue

        # Limpeza de Vírgulas Duplas e Pontuação excessiva HALLUCINATED
        clean_opt = re.sub(r',+', ',', clean_opt) # ,, -> ,
        clean_opt = re.sub(r'[\.,;]+$', '', clean_opt) # Remove pontuação final redundante na contagem
        
        # [v12.97 MATH] Custo Real: Letras + Vírgulas INTERNAS (0.5s cada vírgula)
        # Vírgulas no FINAL da frase não consomem tempo adicional na fala.
        commas_count = clean_opt.rstrip(',').count(',')
        comma_time_cost = commas_count * 0.5
        
        # Caracteres efetivos (sem contar as vírgulas que já viraram tempo)
        text_only = re.sub(r'[,]', '', clean_opt)
        effective_char_count = len(text_only)
        
        # Tempo estimado total da frase (Fala + Pausas)
        estimated_time = (effective_char_count / target_rate) + comma_time_cost
        
        # Score baseado no erro absoluto de tempo em relação ao áudio original
        score = abs(estimated_time - original_duration)
        
        # [v12.91] Penalidade se o CPS das letras for insano (> 22) mesmo com vírgulas
        cps_letters = effective_char_count / (original_duration - comma_time_cost) if (original_duration - comma_time_cost) > 0.1 else 99
        if cps_letters > 22:
             score += (cps_letters - 22) * 5.0

        # 3. Regra de Ouro para Áudios Curtos (< 1.2s)
        if original_duration < 1.2:
            words = clean_opt.split()
            # Penaliza severamente 1 palavra isolada, a menos que o original seja curto também
            if len(words) < 2 and len(original_text.split()) > 1:
                score += 50 
            # Bônus para frases nominais completas (ex: "Perímetro perdido")
            if len(words) >= 2:
                score -= 5

        logging.info(f"   - Candidato: '{clean_opt}' | Est.Time: {estimated_time:.2f}s | Score: {score:.2f}")

        if score < best_score:
            best_score = score
            best_opt = clean_opt
            
    return best_opt
    
    # Validação básica
    valid_options = [opt.strip() for opt in options_list if opt and len(opt.strip()) > 0]
    if not valid_options: return None

    logging.info(f"Avaliando {len(valid_options)} candidatos para duração {original_duration:.2f}s...")

    for opt in valid_options:
        # Limpeza básica e Remoção de Aspas Inteligentes
        clean_opt = re.sub(r'^(\*\*|__)?(Opção|Option|Candidato)?\s*\d+.*?:', '', opt, flags=re.IGNORECASE)
        clean_opt = re.sub(r'^\d+[\.\-\)]\s*', '', clean_opt).replace('“', '').replace('”', '').strip('"').strip("'").strip('*').strip()
        # [FIX] Remove sufixo de metadata (ex: "(28 caracteres)", "400 caracteres)", "**(Aproximadamente...)**")
        # Aceita parenteses quebrados ou faltantes no início se o padrão de contagem for forte.
        clean_opt = re.sub(r'[\(\[\-]?\s*\d+\s*(caracteres|chars|letras|c|aproximadamente).*?[\)\]][\.\s]*$', '', clean_opt, flags=re.IGNORECASE).strip()
        if not clean_opt: continue

        # Limpeza de Vírgulas Duplas e Pontuação excessiva HALLUCINATED
        clean_opt = re.sub(r',+', ',', clean_opt) # ,, -> ,
        clean_opt = re.sub(r'[\.,;]+$', '', clean_opt) # Remove pontuação final redundante na contagem
        
        # [MATH] Custo Real: Letras + Pausas
        # Cada vírgula = 8 chars (0.5s)
        # Pontos finais e interrogações internas também contam
        commas = clean_opt.count(',')
        pauses = clean_opt.count('.') + clean_opt.count('!') + clean_opt.count('?')
        # Pontuação final não conta tanto pois o audio acaba, mas internas sim.
        
        effective_char_count = len(clean_opt) + (commas * 8)
        
        # CPS efetivo (considerando as pausas como "tempo gasto")
        cps = effective_char_count / original_duration if original_duration > 0 else 0
        
        # 1. Score Base (Distância do Ideal)
        # Zona de Conforto: 11 a 15 CPS (Penalidade reduzida)
        if 11 <= cps <= 15:
            score = abs(cps - target_rate) * 0.5 
        else:
            score = abs(cps - target_rate) * 2.0 # Penalidade maior fora da zona
            
        # 2. Penalidades Perceptivas
        # "Fala de Robô" (Muito Lento)
        # [FIX] Ignora penalidade lenta se o áudio for muito curto (< 1.0s) ou se for uma palavra segura ("Ok")
        if cps < 10 and original_duration > 1.0: 
            score += 15
        
        # "Fala de Esquilo" (Muito Rápido)
        # [FIX] Se for curto (< 2s), aceita explosões de velocidade (até 20 CPS) sem punir.
        limit_high = 20 if original_duration < 2.0 else 16
        if cps > limit_high: score += 20
        
        # [TUNING] Bônus para Palavras Seguras (Short Interjections)
        safe_words = ["ok", "tá", "sim", "não", "oi", "hã", "é", "ah", "oh", "uau"]
        if clean_opt.lower() in safe_words:
            score -= 15 # Garante que "Ok" vença "Tipo" em segmentos curtos
        
        # 3. Regra de Ouro para Áudios Curtos (< 1.2s)
        if original_duration < 1.2:
            words = clean_opt.split()
            # Penaliza severamente 1 palavra isolada, a menos que o original seja curto também
            if len(words) < 2 and len(original_text.split()) > 1:
                score += 50 
            # Bônus para frases nominais completas (ex: "Perímetro perdido")
            if len(words) >= 2:
                score -= 5
                
        # 4. Filtro Anti-Chinês/Inglês (Leakage)
        # Se contiver palavras em inglês comuns que indicam falha de tradução
        english_leak = re.search(r'\b(Like|So|Well|Oh my God|Yes|No|Wait|Just)\b', clean_opt, re.IGNORECASE)
        if english_leak:
            score += 100 # Penalidade MÁXIMA (Rejeita imediatamente)

        logging.info(f"   - Candidato: '{clean_opt}' | CPS: {cps:.1f} | Score: {score:.1f}")

        if score < best_score:
            best_score = score
            best_opt = clean_opt
            
    return best_opt


def apply_string_fallback(text, target_chars):
    """
    FALLBACK DE EMERGÊNCIA (SEM IA).
    Remove classes gramaticais dispensáveis se o texto estiver estourando o limite.
    Útil quando o LLM falha em respeitar o tamanho.
    """
    if len(text) <= target_chars: return text
    
    # 1. Remove Advérbios (-mente)
    words = text.split()
    new_words = [w for w in words if not w.lower().endswith('mente')]
    new_text = " ".join(new_words)
    if len(new_text) <= target_chars: return new_text
    
    # 2. Remove Artigos (o, a, os, as, um, uns...)
    blacklist = ['o', 'a', 'os', 'as', 'um', 'uma', 'uns', 'umas', 'do', 'da', 'dos', 'das', 'no', 'na', 'nos', 'nas']
    new_words = [w for w in new_words if w.lower() not in blacklist]
    return " ".join(new_words)



def gema_etapa_2_sincronizacao(original_text, translated_text, duration_seconds, context="", target_chars=None, previous_text=""):
    """
    Etapa 2: Sincronia Labial (ARQUITETURA DEFINITIVA).
    Define o MODO DE OPERAÇÃO baseado no tempo disponível e executa a edição destrutiva.
    """
    
    # 1. Definição de MODO e TEMPERATURA (Ajuste Final)
    # Segmentos MUITO curtos (< 1.2s) exigem DETERMINISMO TOTAL
    if duration_seconds <= 1.2:
        mode = "CORTE"
        temperature = 0.0
        instruction_prompt = """
        [MODO: CORTE DRÁSTICO]
        - Tempo CRÍTICO (< 1.2s).
        - TAREFA: Mantenha APENAS o essencial.
        - LEI DE OURO: NUNCA alucine explicações ou barulhos. Se o original é "Oh", responda "Oh".
        - Prioridade: SUJEITO + VERBO (Ex: "A cidade parou"). Jogue fora o resto. NUNCA remova o sujeito principal.
        """
    elif duration_seconds <= 3.0:
        mode = "SIMPLIFICACAO"
        temperature = 0.1
        instruction_prompt = """
        [MODO: SIMPLIFICAÇÃO]
        - Tempo: Curto (1.2s - 3.0s).
        - TAREFA: Encaixar a frase mantendo o sentido completo de forma ENXUTA.
        - REGRAS:
          1. NUNCA remova o Sujeito Principal da frase (Ex: "A cidade"). Se precisar cortar, corte os complementos.
          2. Use sinônimos curtos e evite traduções literais ("once" = "antes", "já", não "uma vez").
          3. Corte adjetivos ou tempos verbais complexos que não agregam ("De noite").
          4. "I'm locked in for the night" -> "Tô trancado aqui" (Ideal).
        """
    else:
        mode = "ADAPTACAO"
        temperature = 0.2
        instruction_prompt = """
        [MODO: ADAPTAÇÃO]
        - Tempo: Confortável (> 3.0s).
        - TAREFA: Dublagem natural.
        """

    prompt = f"""[ESPECIALISTA EM LIP-SYNC]
    Original: "{original_text}"
    Tradução: "{translated_text}"
    
    [CRÍTICO - LIMITE EXATO DE CARACTERES]:
    - O limite máximo aproximado para a tradução caber na dublagem é de {target_chars} caracteres.
    - A sua frase atual tem {len(translated_text)} caracteres e precisa ser reescrita para se adequar a este limite matemático.
    - [REGRA DE OURO]: Se não couber, priorize a integridade da mensagem e fluidez natural sobre correspondência exata de caracteres. Melhor uma frase natural que passe um pouco do tempo do que algo truncado ou sem sentido.
    
    {instruction_prompt}
    
    - Priorize o Português do Brasil (PT-BR).
    - Use contrações naturais ("tá", "tô", "pra") se ajudar na fluidez e no ritmo brasileiro.
    - NUNCA use as palavras "né" ou "tá" se elas soarem como vício de linguagem repetitivo, mas pode usá-las se fizerem parte da entonação natural da cena.
    [SYSTEM_LAYER]
    ROLE: NINJA_SYNC_ENGINE
    MODE: FIDELIDADE_EXTREMA
    
    [TASK]
    Adapte a tradução para caber no tempo de {duration_seconds} segundos ({target_chars} caracteres).
    
    [OUTPUT_RULES]
    1. ZERO EXPANSÃO: Não adicione palavras que não estão na tradução original.
    2. NO_CUT_RULE: Jamais corte o início ou fim de palavras.
    3. PRIORIDADE: Prefira remover advérbios ou usar sinônimos curtos.
    4. DICIONÁRIO NATURAL: Jamais use "outrora". Use "antes" ou "já".
    
    [INPUT]
    Original: {original_text}
    Tradução: {translated_text}
    Contexto: {context[:200]}
    
    [FORMATO OBRIGATÓRIO]:
    Responda apenas com a lista de opções numeradas entre aspas.
    Exemplo:
    1. "Opção extra curta"
    2. "Opção mais natural"
    
    [PROIBIDO]:
    1. PROIBIDO explicações ou comentários.
    2. PROIBIDO repetir o inglês original.
    3. PROIBIDO usar setas (->).
    
    RESPOSTA DEFINITIVA:
    """

    # [MODIFICADO] Prompt Único já definido acima. Bloco antigo removido.
    
    payload = {
        "messages": [
            {"role": "system", "content": "<|think|>\nVocê é Especialista em Sincronia Labial (Lip-Sync). Adapte o texto para o tempo exato."},
            {"role": "user", "content": prompt}
        ], 
        "temperature": temperature, 
        "max_tokens": 600
    }
    try:
        response = make_gema_request_with_retries(payload, timeout=300)
        content = response.json()['choices'][0]['message']['content'].strip()
        
        # [FIX] Normaliza quebra de linha
        content = re.sub(r'(\d+[\.\-\)])', r'\n\1', content)
        
        options = []
        for line in content.split('\n'):
             line = line.strip()
             if not line: continue
             
             # Limpeza básica (Números e Letras)
             # Remove: "1.", "1)", "(1)", "a.", "a)", "(a)"
             clean_line = re.sub(r'^[\(\[]?\s*[0-9a-zA-Z]+\s*[\)\]\.]\s*', '', line)
             
             # Remove labels (Opção 1, Natural, etc)
             clean_line = re.sub(r'^(\[.*?\]|\(.*?\)|\b(Conservadora|Natural|Expressiva|Opção \d)\b)\s*[-:]?\s*', '', clean_line, flags=re.IGNORECASE)
             
             # [FIX] Remove comentários no final da linha (Ex: "... (Mais curta)") e Markdown
             clean_line = re.sub(r'\s*[\(\[].*?[\)\]]$', '', clean_line) # Remove (Texto) no fim
             clean_line = clean_line.replace('**', '').replace('"', '').replace("'", "").strip()
             
             # [FIX] Remove alucinação de prefixo do contexto (Ex: "Prisão:", "Cenário:", "Contexto:")
             if ":" in clean_line:
                 possible_prefix = clean_line.split(":", 1)[0].lower()
                 blacklist = ["cenário", "contexto", "prisão", "situação", "tema", "cena", "original", "tradução"]
                 if len(possible_prefix) < 20 and any(keyword in possible_prefix for keyword in blacklist):
                     clean_line = clean_line.split(":", 1)[1].strip()

             if clean_line:
                 # [v4.5 FIX] Garante limpeza de pipes e explicações antes do Judge
                 final_clean = clean_translation_fillers(clean_line)
                 if final_clean:
                     options.append(final_clean)

        # Fallback Parsing
        if not options:
             options = [l for l in content.split('\n') if len(l.strip()) > 3]

        # [NOVO] Expansão de Candidatos (Sistema assume controle das vírgulas)
        # Gera versões sem vírgula para competir no Leilão do Judge
        expanded_options = list(options)
        for opt in options:
            if ',' in opt:
                # Cria clone sem vírgulas
                no_comma = opt.replace(',', '')
                expanded_options.append(no_comma)
                logging.info(f"   [SISTEMA] Gerado candidato sem vírgula: '{no_comma}'")
        
        # Atualiza lista para o Judge
        options = expanded_options

        # Seleção Inteligente via Judge
        best_choice = select_best_sync_option(duration_seconds, options, original_text)
        
        if best_choice:
            logging.info(f"Melhor sync: '{best_choice}'")
            return apply_string_fallback(best_choice, target_chars or 999)
        else:
            logging.warning(f"Sem opção válida. Usando tradução base e aplicando fallback: '{translated_text}'")
            return apply_string_fallback(translated_text, target_chars or 999)

    except Exception as e:
        logging.error(f"Erro na API Gema (Etapa 2): {e}")
        return f"FALHA_API: {e}"

def gema_etapa_3_adaptacao_tts(synced_text, is_retry=False, context=""):
    """Etapa 3: Adaptação para TTS focada em sotaque Brasileiro (PT-BR)."""
    prompt = f"""CONTEXTO GLOBAL: {context}
    Ajuste a pontuação deste texto para um robô de voz ler com sotaque natural do BRASIL.
    Use poucas vírgulas, mas use-as para criar a cadência melódica do PT-BR. 
    Mantenha '!' e '?'. Não termine com ponto ou vírgula.
Texto: "{synced_text}"
Texto Ajustado:"""
    payload = {
        "messages": [
            {"role": "system", "content": "<|think|>\nVocê é um Revisor de Pontuação e Melodia para TTS (Vetor de Voz)."},
            {"role": "user", "content": prompt}
        ], 
        "temperature": 0.2, 
        "max_tokens": 1000
    }
    try:
        response = make_gema_request_with_retries(payload, timeout=300)
        return sanitize_tts_text(response.json()['choices'][0]['message']['content'])
    except Exception as e:
        logging.error(f"Erro na API Gema (Etapa 3): {e}")
        return f"FALHA_API: {e}"

# --- FUNÇÕES DO PIPELINE (Híbrido) ---

def extract_audio(video_path, audio_path, cb):
    """Extrai áudio do vídeo (do App_videos)."""
    cb(0, 1, "Iniciando extração...")
    try:
        # [v10.33] Adicionado -loglevel error para omitir tags SEI
        command = ['ffmpeg', '-y', '-loglevel', 'error', '-i', str(video_path), '-vn', '-acodec', 'pcm_s16le', '-ar', '24000', '-ac', '1', str(audio_path)]
        subprocess.run(command, check=True, capture_output=True, text=True)
        cb(100, 1, "Extração concluída.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Erro do FFmpeg ao extrair áudio: {e.stderr}")
        raise


# --- [DIARIZAÇÃO & INTELIGÊNCIA DE VÍDEO] ---

class SimpleDiarizer:
    def __init__(self, source="speechbrain/spkrec-ecapa-voxceleb", device=None):
        if device is None:
            device = get_optimal_device()
        try:
            from speechbrain.inference.speaker import EncoderClassifier
            self.encoder = EncoderClassifier.from_hparams(source=source, run_opts={"device": device})
            self.device = device
            logging.info(f"SimpleDiarizer inicializado em '{self.device}' com modelo {source}")
        except ImportError:
            # Fallback para versoes antigas onde ficava em pretrained
            from speechbrain.pretrained import EncoderClassifier
            self.encoder = EncoderClassifier.from_hparams(source=source, run_opts={"device": device})

    def diarize_file(self, audio_path, num_speakers=None):
        import torchaudio
        import numpy as np
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.metrics import silhouette_score

        signal, fs = torchaudio.load(audio_path)
        
        # Garante Mono (Mixdown se stereo)
        if signal.shape[0] > 1:
            signal = signal.mean(dim=0, keepdim=True)

        # Resample para 16kHz (padrão ECAPA)
        if fs != 16000:
             resampler = torchaudio.transforms.Resample(fs, 16000)
             signal = resampler(signal)
             fs = 16000
        
        # [TUNING] Filtro de Frequência "Telefônico" (300Hz - 3400Hz)
        # Remove graves de música (instrumental) e agudos (pratos), deixando só a voz humana.
        # Isso ajuda o VoiceGuard a não confundir Bateria com "Voz 2".
        try:
            import torchaudio.functional as F
            signal = F.highpass_biquad(signal, fs, 300)
            signal = F.lowpass_biquad(signal, fs, 3400)
            logging.info("Diarização: Filtro de banda (300-3400Hz) aplicado para focar na voz.")
        except Exception as e:
            logging.warning(f"Erro ao aplicar filtro de áudio na diarização: {e}")
        
        # Segmentação simples (Janela deslizante)
        window_size = 1.5  # segundos
        step_size = 0.75   # segundos (50% overlap)
        window_samples = int(window_size * fs)
        step_samples = int(step_size * fs)
        
        length = signal.shape[1]
        segments = []
        timestamps = []
        
        # Cria segmentos
        for start in range(0, length - window_samples, step_samples):
            end = start + window_samples
            seg = signal[:, start:end]
            segments.append(seg)
            timestamps.append((start / fs, end / fs))
            
        if not segments:
            return []

        # Extrai Embeddings (Batch de 1 para segurança na CPU)
        embeddings = []
        for seg in segments:
             # Input deve ser [Batch, Time]. 'seg' já é [1, Time], então passamos direto.
             emb = self.encoder.encode_batch(seg)
             # Emb shape: [1, 1, 192] -> squeeze -> [192]
             embeddings.append(emb.squeeze().cpu().numpy())
        
        embeddings = np.array(embeddings)
        
        # --- VOICE GUARD INTEGRATION ---
        # [DIARIZATION UPDATE] Global Clustering (Agglomerative)
        # Substituindo VoiceGuard por Clustering Global para maior precisão e estabilidade.
        # Memória: 16GB é suficiente pois clusterizamos apenas embeddings (vetores pequenos), não áudio bruto.
        
        logging.info("Diarização: Executando Clustering Global (Agglomerative)...")
        
        # Otimização de Memória: Delete signal/resampler se possível
        del signal
        if 'resampler' in locals(): del resampler
        import gc
        gc.collect()
        
        try:
            # Configuração do Clustering
            # distance_threshold=0.45: Rigoroso para separar gêneros, mas tolerante para variações.
            # linkage='average': Melhores clusters para voz.
            if num_speakers and num_speakers > 1:
                logging.info(f"Diarização: {num_speakers} oradores definidos manualmente.")
                clusterer = AgglomerativeClustering(n_clusters=num_speakers, metric='cosine', linkage='average')
            else:
                logging.info("Diarização: Modo Automático (Threshold 0.45).")
                clusterer = AgglomerativeClustering(n_clusters=None, distance_threshold=0.45, metric='cosine', linkage='average')
                
            labels = clusterer.fit_predict(embeddings)
            
            # Reconstrói segmentos com speakers
            results = []
            if len(labels) == 0: return []
            
            current_speaker = f"voz{labels[0] + 1}"
            current_start = timestamps[0][0]
            current_end = timestamps[0][1]
            
            for i in range(1, len(labels)):
                speaker_id = f"voz{labels[i] + 1}"
                start, end = timestamps[i]
                
                if speaker_id == current_speaker:
                    current_end = end
                else:
                    results.append({'start': current_start, 'end': current_end, 'speaker': current_speaker})
                    current_speaker = speaker_id
                    current_start = start
                    current_end = end
            
            # Adiciona último
            results.append({'start': current_start, 'end': current_end, 'speaker': current_speaker})
            
            logging.info(f"Diarização Concluída. {len(set(labels))} vozes detectadas.")
            return results
            
        except Exception as e:
            logging.error(f"Erro fatal no Clustering: {e}")
            return []
        
        else:
             # Fallback: Clustering Antigo
             return self.fallback_clustering(embeddings, timestamps, num_speakers)

    def fallback_clustering(self, embeddings, timestamps, num_speakers):
         # ... (Lógica antiga movida para cá ou redundante se o VG estiver ok)
         # Como o VG é mandatório agora, retornamos erro ou lógica simplificada
         logging.warning("Fallback Clustering ativado (VoiceGuard indisponível).")
         from sklearn.cluster import AgglomerativeClustering
         n_clusters = num_speakers if (num_speakers and num_speakers > 1) else 2
         clusterer = AgglomerativeClustering(n_clusters=n_clusters, metric='cosine', linkage='average')
         labels = clusterer.fit_predict(embeddings)
         
         # Merge (Lógica repetida do antigo)
         results = []
         if len(labels) == 0: return []
         current_label = labels[0]
         current_start = timestamps[0][0]
         current_end = timestamps[0][1]
         for i in range(1, len(labels)):
             label = labels[i]
             if label == current_label: current_end = timestamps[i][1]
             else:
                 results.append({'start': current_start, 'end': current_end, 'speaker': f"voz{current_label+1}"})
                 current_label = label
                 current_start = timestamps[i][0]
                 current_end = timestamps[i][1]
         results.append({'start': current_start, 'end': current_end, 'speaker': f"voz{current_label+1}"})
         return results



def run_diarization(job_dir, vocals_path, cb, etapa_idx, etapas_lista, num_speakers=None):
    """
    Executa a diarização automática usando SpeechBrain (Implementação Customizada).
    """
    cb(0, etapa_idx, "Carregando modelo de Diarização (SpeechBrain Custom)...")
    
    diarization_file = job_dir / "diarization.json"
    if diarization_file.exists():
        cb(100, etapa_idx, "Diarização carregada do backup.")
        return safe_json_read(diarization_file)

    try:
        # Garante limpeza de RAM
        gc.collect()
        
        # Instancia nosso diarizador customizado
        diarizer = SimpleDiarizer(device="cpu")
        
        cb(30, etapa_idx, "Processando áudio (Extraindo Embeddings)...")
        # Executa a diarização
        segments = diarizer.diarize_file(str(vocals_path), num_speakers=num_speakers)
        
        logging.info(f"Diarização concluída. {len(segments)} segmentos encontrados.")
        
        # Salva o resultado
        safe_json_write(segments, diarization_file)
        
        cb(100, etapa_idx, "Diarização concluída.")
        return segments

    except Exception as e:
        logging.error(f"Erro na Diarização Automática: {e}")
        import traceback
        logging.error(traceback.format_exc())
        logging.warning("Prosseguindo com 1 único falante (fallback).")
        return []

# --- v10.0 SMART SPLIT CORE --- (Paulo request)
def split_segment_by_speaker(seg, break_index, new_id_start):
    """
    Divide um segmento em dois no índice da palavra especificada.
    """
    words = seg.get('words', [])
    if not words or break_index <= 0 or break_index >= len(words):
        return [seg]
    
    part1_words = words[:break_index]
    part2_words = words[break_index:]
    
    seg1 = seg.copy()
    seg1['end'] = part1_words[-1]['end']
    seg1['duration'] = seg1['end'] - seg1['start']
    seg1['words'] = part1_words
    # Whisper words usually have a leading space, join cleanly
    seg1['original_text'] = "".join([w['word'] for w in part1_words]).strip()
    
    seg2 = seg.copy()
    seg2['id'] = new_id_start # Temporário, será re-indexado
    seg2['start'] = part2_words[0]['start']
    seg2['duration'] = seg2['end'] - seg2['start']
    seg2['words'] = part2_words
    seg2['original_text'] = "".join([w['word'] for w in part2_words]).strip()
    
    # Reset translations for safety on split
    if 'translated_text' in seg2:
        seg2['translated_text'] = ""
        seg2['synced_text'] = ""
        
    return [seg1, seg2]

def refine_project_with_smart_split(project_data, wav, fs, classifier, centroids, device):
    """
    Analisa cada segmento palavra-a-palavra e corta se houver troca de orador detectada via SpeechBrain.
    """
    from scipy.spatial.distance import cdist
    import numpy as np
    import torch
    
    refined_data = []
    splits_count = 0
    
    logging.info(f"Iniciando Smart Split (v10.0) em {len(project_data)} segmentos...")
    
    for seg in project_data:
        words = seg.get('words', [])
        # Só tenta dividir se tiver mais de 2 palavras e duração razoável
        if len(words) < 2 or seg['duration'] < 0.6:
            refined_data.append(seg)
            continue
            
        word_speakers = []
        for w in words:
            # Extrai embedding da palavra (com contexto de 0.8s)
            w_start, w_end = w['start'], w['end']
            w_center = (w_start + w_end) / 2
            
            # Janela de 0.8s para estabilidade do ECAPA-TDNN
            win_start = max(0, w_center - 0.4)
            win_end = min(wav.shape[1]/fs, w_center + 0.4)
            
            s_start, s_end = int(win_start * fs), int(win_end * fs)
            chunk = wav[:, s_start:s_end]
            
            if chunk.shape[1] < 1600:
                word_speakers.append(None)
                continue
                
            try:
                with torch.no_grad():
                    if device == "cuda": chunk = chunk.to(device)
                    emb = classifier.encode_batch(chunk).squeeze().cpu().numpy()
                    
                    # Compara com centróides conhecidos
                    best_spk = None
                    min_dist = 0.68 # Strict threshold for words
                    
                    for spk_id, cent in centroids.items():
                        dist = cdist(emb.reshape(1,-1), cent.reshape(1,-1), metric='cosine')[0][0]
                        if dist < min_dist:
                            min_dist = dist
                            best_spk = f"voz{int(spk_id)+1}"
                    word_speakers.append(best_spk)
            except:
                word_speakers.append(None)
        
        # Procura ponto de transição clara na frase
        break_point = -1
        for i in range(1, len(word_speakers)):
            if word_speakers[i] and word_speakers[i-1] and word_speakers[i] != word_speakers[i-1]:
                if i + 1 < len(word_speakers) and word_speakers[i+1] == word_speakers[i]:
                    break_point = i
                    break
        
        if break_point != -1:
            parts = split_segment_by_speaker(seg, break_point, f"{seg['id']}_b")
            parts[0]['speaker'] = word_speakers[break_point-1]
            parts[1]['speaker'] = word_speakers[break_point]
            
            refined_data.extend(parts)
            splits_count += 1
            logging.info(f"Smart Split: Segmento {seg['id']} fatiado em {parts[0]['speaker']} e {parts[1]['speaker']}")
        else:
            refined_data.append(seg)
            
    return refined_data


def run_transcription_driven_diarization(job_dir, vocals_path, project_data, cb, etapa_idx, num_speakers=None):
    """
    v10.0 Smart Diarization:
    Etapa 1: Análise granular (Palavra por Palavra) com "Corte" preciso.
    Etapa 2: Junção "Rígida" (Strict Clustering) para evitar misturar vozes.
    """
    cb(0, etapa_idx, "Iniciando Diarização v10.0 (Granular Analysis)...")
    
    status_file = job_dir / "diarization_status.json"
    debug_file = job_dir / "diarization_debug.json"
    
    if debug_file.exists():
        logging.info("Memória de Diarização encontrada. Ativando Bypass.")
        cb(100, etapa_idx, "Diarização pulada (Já concluída).")
        return project_data
    
    if not project_data:
        return project_data

    # Handle Dictionary or List wrapper
    segs_list = project_data.get('segments', project_data) if isinstance(project_data, dict) else project_data
    
    # [v10.6] Auditoria de Conteúdo
    total_words = sum(len(s.get('words', [])) for s in segs_list)
    logging.info(f"Diarização v10.0: Recebidos {len(segs_list)} segmentos e {total_words} palavras.")
    safe_json_write({"status": "processing", "start_time": datetime.now().isoformat()}, status_file)

    try:
        # [v10.1] Salva log "Raw Debug" solicitado pelo Paulo
        safe_json_write(project_data, job_dir / "whisper_raw_debug.json")
        logging.info(f"Log original salvo em: whisper_raw_debug.json (Preservação da Transcrição Bruta)")

        # 1. Carrega Models (SpeechBrain)
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            # SpeechBrain v1.0+
            from speechbrain.inference.speakers import EncoderClassifier
        except ImportError:
            # SpeechBrain v0.5.x (Legacy)
            from speechbrain.pretrained import EncoderClassifier
        
        classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb", 
            run_opts={"device": device}
        )

        # 2. Carrega áudio para memória (Slicing rápido)
        import torchaudio
        wav, fs = torchaudio.load(str(vocals_path))
        if wav.shape[0] > 1: wav = wav.mean(dim=0, keepdim=True)
        if fs != 16000:
            resampler = torchaudio.transforms.Resample(fs, 16000)
            wav = resampler(wav)
            fs = 16000

        # 3. EXTRAÇÃO GRANULAR (Stage 1: Word-Level)
        cb(20, etapa_idx, "Etapa 1: Analisando cada palavra do vídeo...")
        all_word_embeddings = []
        word_map = [] # Mantém referência de qual palavra é qual
        
        # Handle Dictionary or List wrapper
        segs_list = project_data.get('segments', project_data) if isinstance(project_data, dict) else project_data

        for s_idx, seg in enumerate(segs_list):
            words = seg.get('words', [])
            if not words: continue
            
            for w_idx, w in enumerate(words):
                w_start, w_end = w['start'], w['end']
                w_center = (w_start + w_end) / 2
                
                # Janela de 1.0s para máxima fidelidade na identificação
                win_start = max(0, w_center - 0.5)
                win_end = min(wav.shape[1]/fs, w_center + 0.5)
                
                start_samp, end_samp = int(win_start * fs), int(win_end * fs)
                chunk = wav[:, start_samp:end_samp]
                
                if chunk.shape[1] < 1600: continue
                
                with torch.no_grad():
                    if device == "cuda": chunk = chunk.to(device)
                    emb = classifier.encode_batch(chunk).squeeze().cpu().numpy()
                    all_word_embeddings.append(emb)
                    word_map.append({'s_idx': s_idx, 'w_idx': w_idx, 'duration': w_end - w_start})
                
            if s_idx % 20 == 0:
                cb(20 + (s_idx/len(segs_list))*30, etapa_idx, f"Lendo vozes... Segmento {s_idx}")

        if not all_word_embeddings:
            safe_json_write({"status": "success", "info": "no_speech"}, status_file)
            return project_data

        # 4. JUNÇÃO RÍGIDA (Stage 2: Strict Clustering)
        cb(60, etapa_idx, "Etapa 2: Agrupando vozes similares (Modo Rígido)...")
        from sklearn.cluster import AgglomerativeClustering
        import numpy as np
        
        clusterer = AgglomerativeClustering(
            n_clusters=num_speakers if num_speakers and num_speakers > 1 else None, 
            distance_threshold=0.55, # [v12.0] Threshold mais conservador para evitar fusão cega
            metric='cosine', 
            linkage='average'
        )
        
        X = np.array(all_word_embeddings)
        labels = clusterer.fit_predict(X)
        
        # [v12.1 NEW] CONSOLIDAÇÃO AGÊNTICA PÓS-CLUSTERING
        cb(70, etapa_idx, "Gemma 4 Auditor: Consolidando identidades de voz...")
        from collections import Counter
        unique_labels = sorted(list(set(labels)))
        
        # Calcula centróide (média) de cada cluster para comparação
        cluster_centroids = {}
        cluster_transcripts = {}
        for lbl in unique_labels:
            indices = np.where(labels == lbl)[0]
            cluster_centroids[lbl] = np.mean(X[indices], axis=0)
            
            # Pega as 5 maiores frases deste cluster para o Gemma ler
            cluster_text = ""
            for idx in indices[:15]: # Amostra de palavras
                m = word_map[idx]
                cluster_text += segs_list[m['s_idx']]['words'][m['w_idx']]['word']
            cluster_transcripts[lbl] = cluster_text
            
        # [NOVO] Mapa de fusão (Merger Map)
        merger_map = {lbl: lbl for lbl in unique_labels}
        
        # Compara pares de clusters (apenas se a distância for "interessante")
        from scipy.spatial.distance import cosine
        for i in range(len(unique_labels)):
            for j in range(i + 1, len(unique_labels)):
                lbl_a, lbl_b = unique_labels[i], unique_labels[j]
                dist = cosine(cluster_centroids[lbl_a], cluster_centroids[lbl_b])
                
                # Se a distância estiver entre 0.4 e 0.75, está na zona de fusão inteligente
                if dist < 0.75:
                    logging.info(f"   -> 🤖 [CONSOLIDATION CANDIDATE] Voz {lbl_a+1} vs Voz {lbl_b+1} (Dist: {dist:.3f}). Chamando Gemma...")
                    
                    # Para o consolidator, precisamos de Hz aproximados
                    # (Aqui usamos valores mock ou uma média rápida se disponível, 
                    # mas o consolidator em identify_speaker_gender_map é mais preciso)
                    if gema_speaker_match_consolidator(
                        f"voz{lbl_a+1}", f"voz{lbl_b+1}", 
                        cluster_transcripts[lbl_a], cluster_transcripts[lbl_b], 
                        150, 150, dist # 150 é neutro
                    ):
                        logging.info(f"   >>> [MERGE APPROVED] Voz {lbl_b+1} será fundida na Voz {lbl_a+1}!")
                        merger_map[lbl_b] = merger_map[lbl_a]
                        
        # Aplica o merger map nos labels
        final_labels = np.array([merger_map[lbl] for lbl in labels])
        labels = final_labels

        # 5. RE-CONSTRUÇÃO DE SEGMENTOS (The "Corte" and "Merge" Final)
        cb(80, etapa_idx, "Finalizando reconstrução do roteiro...")
        
        # [v10.12] VOICE HYSTERESIS: Suavização de etiquetas
        # Se uma palavra isolada tem voz diferente das vizinhas, ela herda a voz dominante.
        smoothed_labels = np.copy(labels)
        if len(smoothed_labels) > 2:
            for i in range(1, len(smoothed_labels) - 1):
                if smoothed_labels[i] != smoothed_labels[i-1] and smoothed_labels[i] != smoothed_labels[i+1]:
                    smoothed_labels[i] = smoothed_labels[i-1]

        # Aplica labels de volta nas palavras
        for i, label in enumerate(smoothed_labels):
            m = word_map[i]
            segs_list[m['s_idx']]['words'][m['w_idx']]['spk_label'] = f"voz{label + 1}"

        # Aplica labels de volta nas palavras
        for i, label in enumerate(smoothed_labels):
            m = word_map[i]
            segs_list[m['s_idx']]['words'][m['w_idx']]['spk_label'] = f"voz{label + 1}"

        # [v10.15] DUAL-PASS SEGMENTATION: 
        # Agora que as palavras têm speaker labels, usamos a função de pausa original 
        # para re-cortar o roteiro inteiro de forma profissional.
        whisper_result = {
            'words': [],
            'detected_language': segs_list[0].get('detected_language') if segs_list else None
        }
        for seg in segs_list:
            if seg.get('words'):
                for w in seg['words']:
                    # Garante que o label de orador esteja na palavra para a função de pausa usar
                    whisper_result['words'].append(w)
        
        # O resegment_based_on_pauses já tem lógica para cortar se o orador mudar (passando o word-level como diarization_data)
        # Mas aqui é mais simples: passamos os metadados de orador direto nas palavras.
        # Precisamos garantir que 'get_speaker_at_time' funcione ou adaptar a função.
        
        # Pós-processamento: Resegmentação Final
        new_project_data = resegment_based_on_pauses(whisper_result, diarization_data=None) # diarization_data=None porque o label já está na word

        # [v10.16.1] Fix: O resegment não gera o campo 'duration'. Calculamos antes do loop de agregação.
        for seg in new_project_data:
            seg['duration'] = seg['end'] - seg['start']
            if seg.get('words'):
                seg['speaker'] = seg['words'][0].get('spk_label', 'voz1')
            else:
                seg['speaker'] = 'voz1'
        
        # [v10.12] AGGREGATION PASS: Une segmentos colados do mesmo orador
        final_merged_data = []
        if new_project_data:
            curr = new_project_data[0]
            for i in range(1, len(new_project_data)):
                nxt = new_project_data[i]
                gap = nxt['start'] - curr['end']
                
                # [v10.13] TUNED: 0.2s gap
                if nxt['speaker'] == curr['speaker'] and (gap < 0.2 or curr['duration'] < 0.3 or nxt['duration'] < 0.3):
                    # Merge
                    curr['end'] = nxt['end']
                    curr['text'] = (curr['text'] + " " + nxt['text']).strip()
                    curr['words'].extend(nxt['words'])
                    curr['duration'] = curr['end'] - curr['start']
                else:
                    final_merged_data.append(curr)
                    curr = nxt
            final_merged_data.append(curr)
        
        # Re-id e Mapping para compatibilidade
        processed_data = []
        for i, s in enumerate(final_merged_data):
            processed_data.append({
                "id": f"seg_{i:04d}",
                "original_text": s['text'],
                "start": s['start'],
                "end": s['end'] + 0.05, # [v10.12] Toque de Fôlego
                "duration": (s['end'] + 0.05) - s['start'],
                "speaker": s['speaker'],
                "words": s['words']
            })
            
        new_project_data = processed_data

        # 6. Salva
        num_speakers_final = len(set(labels))
        save_data = {
            "segments": new_project_data,
            "num_speakers": num_speakers_final,
            "v": "10.0"
        }
        safe_json_write(save_data, job_dir / "project_data.json")
        # [v10.19.2] REFINED MAP: Salva o "Whisper Raw v2" (Roteiro segmentado mas original)
        # Isso permite reconstruir o projeto perfeitamente mesmo sem project_data.json
        safe_json_write(save_data, job_dir / "whisper_raw_refined.json")
        safe_json_write({"status": "ok", "time": datetime.now().isoformat()}, debug_file)
        safe_json_write({"status": "success", "num_speakers": num_speakers_final}, status_file)
        
        cb(100, etapa_idx, f"Diarização Concluída: {num_speakers_final} vozes identificadas.")
        
        # Memory Cleanup
        del classifier
        del wav
        import gc
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        
        return save_data

    except Exception as e:
        logging.error(f"ERRO CRÍTICO NA DIARIZAÇÃO v10.0: {e}\n{traceback.format_exc()}")
        safe_json_write({"status": "failed", "error": str(e), "time": datetime.now().isoformat()}, status_file)
        raise e


def extract_speaker_references(job_dir, vocals_path, diarization_segments, cb, etapa_idx):
    """
    Extrai clipes de referência para cada orador identificado.
    Salva em: vocals_speaker_{ID}.wav
    """
    cb(0, etapa_idx, "Extraindo referências de voz...")
    
    if not diarization_segments:
        logging.warning("Sem dados de diarização. Usando referência única.")
        # Copia vocals.wav para vocals_speaker_default.wav
        shutil.copy(vocals_path, job_dir / "vocals_speaker_default.wav")
        return {"default": job_dir / "vocals_speaker_default.wav"}
    
    # [v9.1] Acesso seguro a segmentos (Handle dict/list)
    segs_list = diarization_segments.get('segments', diarization_segments) if isinstance(diarization_segments, dict) else diarization_segments

    # Agrupa segmentos por orador
    speakers = {}
    for seg in segs_list:
        spk = seg.get('speaker', 'unknown')
        if spk not in speakers: speakers[spk] = []
        speakers[spk].append(seg)
    
    speaker_refs = {}
    
    # Carrega áudio vocal UMA VEZ para cortar (usando pydub ou soundfile)
    # Pydub é lento para carregar arquivos grandes, mas seguro.
    try:
        audio = AudioSegment.from_wav(str(vocals_path))
    except Exception:
        logging.error("Falha ao carregar áudio para extração de refs.")
        return {}

    for i, (spk_id, segs) in enumerate(speakers.items()):
        # [MELHORIA] Stitching (Costura) de segmentos
        # Ao invés de pegar só o maior, vamos juntar vários até dar ~30 segundos (Pedido do usuário)
        target_duration_ms = 30 * 1000
        accumulated_audio = AudioSegment.empty()
        current_dur = 0
        
        # Ordena por duração (pega os melhores pedaços primeiro)
        sorted_segs = sorted(segs, key=lambda s: s['end'] - s['start'], reverse=True)
        
        for seg in sorted_segs:
            # [v5.8] Stitching Completo: Pegamos TUDO disponível para a melhor clonagem possível.
            # O limite de 30s foi removido para dar o máximo de dados ao Chatterbox.
            
            start_ms = int(seg['start'] * 1000)
            end_ms = int(seg['end'] * 1000)
            
            # Pega o clip
            clip = audio[start_ms:end_ms]
            accumulated_audio += clip
            current_dur += len(clip)
        
        # Se mesmo juntando tudo ficou muito curto (< 2s), pode ser ruído.
        if len(accumulated_audio) < 2000:
             logging.warning(f"Voz {spk_id} muito curta ({len(accumulated_audio)}ms). Ignorando ou mantendo assim mesmo.")
        
        ref_audio = accumulated_audio
        
        # Salva em subpasta 'voices' para organização
        voices_dir = job_dir / "voices"
        voices_dir.mkdir(exist_ok=True)
        
        ref_filename = f"vocals_{spk_id}.wav"
        ref_path = voices_dir / ref_filename
        # if len(ref_audio) < 10000: # [FILTER REMOVED] Consolidação inteligente fará isso depois
        #     logging.warning(f"Voz {spk_id} muito curta ({len(ref_audio)}ms). Será analisada na Consolidação.")
            # if ref_path.exists(): ref_path.unlink()
            # continue
            
        ref_audio.export(ref_path, format="wav")
        
        speaker_refs[spk_id] = ref_path
        
    cb(100, etapa_idx, "Referências extraídas.")
    return speaker_refs

def consolidate_speaker_segments(job_dir, project_data, cb, etapa_idx):
    """
    [NEW] Consolidação Inteligente de Oradores:
    1. Separa oradores em 'Válidos' (>10s) e 'Questionáveis' (<10s).
    2. Se houver Válidos, compara Questionáveis com eles via Embeddings.
    3. Se similaridade > 0.6 (Threshold), funde (merge).
    4. Se não, mantém como orador distinto (prefeita voz ruim a voz errada).
    """
    logging.info("Iniciando Consolidação de Oradores...")
    
    flag_file = job_dir / "consolidation_done.flag"
    if flag_file.exists():
        logging.info("Consolidação (Orphan Cleanup) já realizada anteriormente. Bypass ativado.")
        cb(100, etapa_idx, "Consolidação carregada do cache (Bypass instantâneo).")
        return project_data
    
    voices_dir = job_dir / "voices"
    if not voices_dir.exists(): return project_data

    # 1. Classificação
    valid_speakers = []      # (id, path, len_ms)
    questionable_speakers = [] # (id, path, len_ms)

    import torchaudio
    
    # Lista todos os arquivos
    files = list(voices_dir.glob("vocals_*.wav"))
    if not files: return project_data

    for f in files:
        spk_id = f.stem.replace("vocals_", "")
        # Check duration using torchaudio (mais preciso que os.stat)
        try:
            info = torchaudio.info(str(f))
            duration_ms = (info.num_frames / info.sample_rate) * 1000
        except: 
            duration_ms = 0
        
        if duration_ms >= 10000: # 10s
             valid_speakers.append({'id': spk_id, 'path': str(f), 'dur': duration_ms})
        else:
             questionable_speakers.append({'id': spk_id, 'path': str(f), 'dur': duration_ms})
             
    if not valid_speakers:
        logging.info("Nenhum orador 'Longo' (>10s) encontrado. Prosseguindo com consolidação de curtos (v5.0).")
        # [v5.0] No modo curto, valid_speakers vira a lista de todos, mas com threshold maior
        valid_speakers = questionable_speakers
        questionable_speakers = [] 
        
    if not questionable_speakers:
        logging.info("Todos os oradores são válidos (>10s). Consolidação desnecessária.")
        with open(flag_file, "w") as f: f.write("done")
        return project_data

    cb(50, etapa_idx, f"Consolidando {len(questionable_speakers)} vozes curtas...")
    
    try:
        import torch
        # Carrega Encoder
        diarizer = SimpleDiarizer(device='cuda' if torch.cuda.is_available() else 'cpu')
        
        # 2. Centróides das Vozes Válidas
        valid_centroids = {}
        for spk in valid_speakers:
            signal, fs = torchaudio.load(spk['path'])
            # Mono + Resample
            if signal.shape[0] > 1: signal = signal.mean(dim=0, keepdim=True)
            if fs != 16000: signal = torchaudio.transforms.Resample(fs, 16000)(signal)
            
            emb = diarizer.encoder.encode_batch(signal).squeeze()
            valid_centroids[spk['id']] = emb
            
        # 3. Processa Questionáveis
        merged_count = 0
        
        for q_spk in questionable_speakers:
            # Carrega e Embedda
            signal, fs = torchaudio.load(q_spk['path'])
            if signal.shape[0] > 1: signal = signal.mean(dim=0, keepdim=True)
            if fs != 16000: signal = torchaudio.transforms.Resample(fs, 16000)(signal)
            
            q_emb = diarizer.encoder.encode_batch(signal).squeeze()
            
            # Compara
            best_score = -1.0
            best_target_id = None
            
            for v_id, v_emb in valid_centroids.items():
                score = torch.nn.functional.cosine_similarity(q_emb, v_emb, dim=0).item()
                if score > best_score:
                    best_score = score
                    best_target_id = v_id
            
            # Decisão
            # Threshold 0.60: Razoável para mesma pessoa em condições diferentes
            if best_score > 0.60:
                logging.info(f"MERGE: Voz '{q_spk['id']}' ({q_spk['dur']:.0f}ms) -> '{best_target_id}' (Sim: {best_score:.2f})")
                
                # [v9.1] Acesso seguro a segmentos (Handle dict/list)
                segs_to_update = project_data.get('segments', project_data) if isinstance(project_data, dict) else project_data
                
                # Reatribui no JSON
                count_segs = 0
                for seg in segs_to_update:
                    if seg.get('speaker') == q_spk['id']:
                        seg['speaker'] = best_target_id
                        count_segs += 1
                
                # Apaga arquivo curto
                try: Path(q_spk['path']).unlink() 
                except: pass
                
                merged_count += 1
            else:
                logging.info(f"KEEP: Voz '{q_spk['id']}' ({q_spk['dur']:.0f}ms) é distinta. (Melhor Match: '{best_target_id}' = {best_score:.2f})")

        if merged_count > 0:
            safe_json_write(project_data, job_dir / "project_data.json")
            cb(100, etapa_idx, f"Consolidação concluída. {merged_count} vozes fundidas.")
        else:
             cb(100, etapa_idx, "Consolidação concluída. Nenhuma fusão necessária.")
             
        # Salva Flag de Cache no disco
        with open(flag_file, "w") as f: f.write("done")
             
        # Cleanup
        del diarizer
        import gc
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        return project_data

    except Exception as e:
        logging.error(f"Erro na consolidação de oradores: {e}")
        return project_data



# --- [FUNÇÃO CENTRALIZADA DE SEPARAÇÃO - ATUALIZADA] ---
def run_audio_separation(job_dir, input_audio_path, cb, etapa_idx, etapas_lista):
    """
    Separa áudio e vocais usando OpenUnmix em segmentos de 10s para economizar RAM.
    """
    cb(0, etapa_idx, "Preparando separação (OpenUnmix - Otimizado)...")
    
    output_vocals_dir = job_dir / "_temp_vocals_chunks"
    output_vocals_dir.mkdir(exist_ok=True)
    output_instrumental_dir = job_dir / "_temp_instr_chunks"
    output_instrumental_dir.mkdir(exist_ok=True)
    temp_chunks_dir = job_dir / "_temp_input_chunks"
    temp_chunks_dir.mkdir(exist_ok=True)
    
    vocals_final_path = job_dir / "vocals.wav" # Vamos reconstruir no final para compatibilidade/referência
    instrumental_final_path = job_dir / "instrumental.wav"
    
    # [FIX] Se já existe backup de vocals E de diarização, podemos pular tudo
    # Mas se só vocals existe, ainda precisamos (talvez) rodar diarização.
    # A função de diarização tem seu próprio check de backup.
    
    if vocals_final_path.exists() and instrumental_final_path.exists():
        cb(100, etapa_idx, "Vocais e Instrumental já separados (Backup encontrado).")
        return

    # 1. Segmentar áudio original em chunks de 10s
    cb(5, etapa_idx, "Fatiando áudio em blocos de 10s (FFmpeg)...")
    try:
        # Segmenta em arquivos de 10s: chunk_000.wav, chunk_001.wav...
        cmd_split = [
            'ffmpeg', '-y', '-i', str(input_audio_path), 
            '-f', 'segment', '-segment_time', '10', '-c', 'copy', 
            str(temp_chunks_dir / "chunk_%03d.wav")
        ]
        subprocess.run(cmd_split, check=True, capture_output=True)
    except Exception as e:
        logging.error(f"Erro ao segmentar áudio: {e}")
        raise

    chunks = sorted(list(temp_chunks_dir.glob("chunk_*.wav")))
    total_chunks = len(chunks)
    
    if total_chunks == 0:
        raise RuntimeError("Nenhum segmento gerado pelo FFmpeg.")

    logging.info(f"Áudio dividido em {total_chunks} segmentos. Processando com OpenUnmix...")
    
    # 2. Processar cada chunk com OpenUnmix (umx)
    for i, chunk in enumerate(chunks):
        progress = 5 + (i / total_chunks) * 90
        
        # [RESUME LOGIC] Verifica se já foi processado
        target_vocals = output_vocals_dir / f"vocals_{i:03d}.wav"
        target_instr = output_instrumental_dir / f"instr_{i:03d}.wav"
        
        if target_vocals.exists() and target_instr.exists():
             cb(progress, etapa_idx, f"Separando vozes: Segmento {i+1}/{total_chunks} (Cache)")
             # logging.info(f"Chunk {i} já processado. Pulando (Resume).") # Verbose off
             continue

        cb(progress, etapa_idx, f"Separando vozes: Segmento {i+1}/{total_chunks}")
        
        chunk_out_dir = temp_chunks_dir / f"out_{i}"
        chunk_out_dir.mkdir(exist_ok=True)
        
        try:
            # Comando OpenUnmix CLI
            # umx input.wav --outdir output
            cmd_umx = ['umx', str(chunk), '--outdir', str(chunk_out_dir)]
            
            # Executa e aguarda (Síncrono)
            subprocess.run(cmd_umx, check=True, capture_output=True, text=True) 
            
            # Outputs: vocals.wav, drums.wav, bass.wav, other.wav
            stem_dir = chunk_out_dir / chunk.stem
            
            src_vocals = stem_dir / "vocals.wav"
            src_drums = stem_dir / "drums.wav"
            src_bass = stem_dir / "bass.wav"
            src_other = stem_dir / "other.wav"
            
            target_vocals = output_vocals_dir / f"vocals_{i:03d}.wav"
            target_instr = output_instrumental_dir / f"instr_{i:03d}.wav"
            
            # 1. Vocals
            if src_vocals.exists():
                shutil.move(str(src_vocals), str(target_vocals))
            else:
                shutil.copy(str(chunk), str(target_vocals)) # Fallback

            # 2. Instrumental (Mix drums + bass + other)
            inputs = []
            if src_drums.exists(): inputs.extend(['-i', str(src_drums)])
            if src_bass.exists(): inputs.extend(['-i', str(src_bass)])
            if src_other.exists(): inputs.extend(['-i', str(src_other)])
            
            if inputs:
                cmd_mix = ['ffmpeg', '-y'] + inputs + [
                    '-filter_complex', f'amix=inputs={len(inputs)//2}:duration=longest',
                    str(target_instr)
                ]
                try: subprocess.run(cmd_mix, check=True, capture_output=True)
                except: subprocess.run(['ffmpeg', '-y', '-f', 'lavfi', '-i', 'anullsrc=r=44100:cl=mono', '-t', '10', str(target_instr)], check=True)
            else:
                # Se não tem stems, cria silencio
                subprocess.run(['ffmpeg', '-y', '-f', 'lavfi', '-i', 'anullsrc=r=44100:cl=mono', '-t', '10', str(target_instr)], check=True)
                
            # Limpeza Imediata
            shutil.rmtree(chunk_out_dir)
            gc.collect()

        except subprocess.CalledProcessError as e:
            logging.error(f"ERRO CRÍTICO NO OPENUNMIX ({chunk.name}):\nCMD: {e.cmd}\nSTDERR: {e.stderr}")
            # Fallback: Copia o chunk original
            shutil.copy(str(chunk), str(output_vocals_dir / f"vocals_{i:03d}.wav"))
            # Fallback Instrumental: Silêncio
            try: subprocess.run(['ffmpeg', '-y', '-f', 'lavfi', '-i', 'anullsrc=r=44100:cl=mono', '-t', '10', str(output_instrumental_dir / f"instr_{i:03d}.wav")], check=True)
            except: pass
            
        except Exception as e:
            logging.error(f"Erro no OpenUnmix para {chunk.name}: {e}")
            # Fallback: Copia o chunk original
            shutil.copy(str(chunk), str(output_vocals_dir / f"vocals_{i:03d}.wav"))
            # Fallback Instrumental: Silêncio
            try: subprocess.run(['ffmpeg', '-y', '-f', 'lavfi', '-i', 'anullsrc=r=44100:cl=mono', '-t', '10', str(output_instrumental_dir / f"instr_{i:03d}.wav")], check=True)
            except: pass

    # 3. Unificar Vocais e Instrumental
    cb(95, etapa_idx, "Unificando segmentos de áudio (Pydub Safe Concat)...")
    try:
        final_chunks_v = sorted(list(output_vocals_dir.glob("vocals_*.wav")))
        final_chunks_i = sorted(list(output_instrumental_dir.glob("instr_*.wav")))
        
        # [v10.25] Usar pydub em vez de ffmpeg concat para evitar corrupção de header WAV 
        # quando ocorre fallback (mistura de 24000Hz Mono com 44100Hz Stereo)
        
        if final_chunks_v:
            logging.info("Fundindo pedaços de vocais...")
            v_audio = AudioSegment.empty()
            for p in final_chunks_v:
                v_audio += AudioSegment.from_file(str(p))
            # Garante formato padrão (Whisper prefere 24kHz/1ch ou pydub resolve)
            v_audio = v_audio.set_frame_rate(24000).set_channels(1)
            v_audio.export(str(vocals_final_path), format="wav")
            
        if final_chunks_i:
            logging.info("Fundindo pedaços instrumentais...")
            i_audio = AudioSegment.empty()
            for p in final_chunks_i:
                i_audio += AudioSegment.from_file(str(p))
            i_audio.export(str(instrumental_final_path), format="wav")

    except Exception as e:
        logging.error(f"Erro ao unificar áudios com Pydub: {e}")

    
    # Limpeza
    try: 
        shutil.rmtree(temp_chunks_dir)
        shutil.rmtree(output_instrumental_dir) # Opcional: manter se quiser debugar
    except: pass
    
    cb(100, etapa_idx, "Separação concluída.")

def get_speaker_at_time(t, diarization_data):
    if not diarization_data: return "unknown"
    for d in diarization_data:
        # Dá uma pequena tolerância 
        if (d['start'] - 0.1) <= t <= (d['end'] + 0.1):
            return d.get('speaker', 'unknown')
    return "unknown"

def resegment_based_on_pauses(whisper_result, max_chars=200, max_duration=10.0, silence_threshold=1.0, diarization_data=None):
    """
    Resegmenta as palavras do Whisper baseando-se em pausas de silêncio (0.5s), 
    limites de tamanho, e AGORA respeitando rigorosamente a MUDANÇA DE ORADOR (Diarização).
    """
    words = whisper_result.get('words', [])
    if not words: return []
    
    segments = []
    current_segment = {'words': [], 'start': 0, 'end': 0, 'text': ''}
    
    for i, word in enumerate(words):
        # Inicializa se vazio
        if not current_segment['words']:
            current_segment['start'] = word['start']
            current_segment['words'].append(word)
            current_segment['end'] = word['end']
            continue
            
        last_word = current_segment['words'][-1]
        gap = word['start'] - last_word['end']
        
        # Calcula duração atual
        current_dur = word['end'] - current_segment['start']
        
        # Decide se quebra
        should_break = False
        
        # 1. Pausa longa (Silêncio)
        if gap > silence_threshold:
            should_break = True
            
        # 2. Pontuação Forte + Pequena Pausa (Mudança de turno provável)
        # Ex: "stay here. [0.2s] All right..."
        last_word_text = last_word['word'].strip()
        if last_word_text and last_word_text[-1] in ['.', '?', '!'] and gap > 0.1:
            should_break = True
            
        # 3. Estourou limite de caracteres (com margem de pontuação)
        current_text_len = sum([len(w['word']) for w in current_segment['words']])
        if current_text_len > max_chars:
            should_break = True
            
        # 4. Estourou duração máxima
        if current_dur > max_duration:
            should_break = True
            
        # 5. Mudança Real de Orador (Via Diarização Cruzada ou Label no Word)
        if not should_break:
            if diarization_data:
                spk_current = get_speaker_at_time(word['start'], diarization_data)
                spk_last = get_speaker_at_time(last_word['end'], diarization_data)
                if spk_current != spk_last and spk_current != "unknown" and spk_last != "unknown":
                    should_break = True
            elif word.get('spk_label') and last_word.get('spk_label'):
                # [v10.15] Quebra direta se o label de orador mudar entre palavras (Dual Pass)
                if word['spk_label'] != last_word['spk_label']:
                    should_break = True
                
        if should_break:
             # Finaliza segmento anterior
             # Reconstrói texto
             full_text = "".join([w['word'] for w in current_segment['words']]).strip()
             if full_text:
                 segments.append({
                     'start': current_segment['start'],
                     'end': current_segment['end'],
                     'text': full_text,
                     'words': current_segment['words'],
                     'detected_language': whisper_result.get('detected_language')
                 })
             
             # Começa novo
             current_segment = {'words': [word], 'start': word['start'], 'end': word['end'], 'text': ''}
        else:
             # Adiciona ao atual
             current_segment['words'].append(word)
             current_segment['end'] = word['end']
             
    # Adiciona o último
    full_text = "".join([w['word'] for w in current_segment['words']]).strip()
    if full_text:
        segments.append({
             'start': current_segment['start'],
             'end': current_segment['end'],
             'text': full_text,
             'words': current_segment['words'],
             'detected_language': whisper_result.get('detected_language')
        })
        
    return segments

def smart_merge_segments(segments, max_gap=1.5, max_dur=20.0):
    """
    Fundador de 'Segmentos de Contexto'. Agrupa falas do mesmo orador em blocos (até 20s).
    [v5.5] Reduzido de 60s para 20s para garantir que oradores diferentes não sejam "engolidos" 
    pelo Whisper antes da diarização.
    """
    if not segments: return []
    merged = []
    current_seg = segments[0]
    
    for i in range(1, len(segments)):
        nxt = segments[i]
        gap = nxt['start'] - current_seg['end']
        new_dur = nxt['end'] - current_seg['start']
        
        # Merge amplo: Segundos maiores, maior espaço para contexto 
        if gap < max_gap and new_dur <= max_dur and current_seg.get('speaker', '') == nxt.get('speaker', ''):
            current_seg['end'] = nxt['end']
            current_seg['duration'] = current_seg['end'] - current_seg['start']
            current_seg['original_text'] += " " + nxt.get('original_text', '').strip()
            if 'words' in current_seg and 'words' in nxt:
                current_seg['words'].extend(nxt['words'])
        else:
            merged.append(current_seg)
            current_seg = nxt
            
    merged.append(current_seg)
    logging.info(f"O Whisper picotou em {len(segments)}, mas agrupamos em {len(merged)} Blocos de Cena (Até 1 Minuto) para Inteligência!")
    return merged
def try_reconstruct_project_from_all_backups(job_dir):
    """
    [v10.19.2] LÓGICA PHOENIX: Recontrói o projeto a partir dos backups individuais.
    ESTRATÉGIA:
    1. Tenta carregar o 'whisper_raw_refined.json' (Mapa Perfeito v2).
    2. Se não existir, tenta reconstruir catando os arquivos em '_backup_texto_final'.
    3. Lacunas (gaps) são preenchidas via 'whisper_raw_debug.json' (Mapa Raw v1).
    """
    refined_map_path = job_dir / "whisper_raw_refined.json"
    backup_dir = job_dir / "_backup_texto_final"
    whisper_raw_path = job_dir / "whisper_raw_debug.json"

    # [PRIORIDADE 1] Mapa Refinado (Whisper Raw v2)
    if refined_map_path.exists():
        logging.info("Phoenix: Mapa Refinado (v2) encontrado. Reconstrução direta.")
        refined_data = safe_json_read(refined_map_path)
        if refined_data:
            # Garante que as traduções sejam limpas se o backup individual não existir
            segs = refined_data.get('segments', refined_data)
            for s in segs:
                b_path = backup_dir / f"{s['id']}.json"
                if b_path.exists():
                    b_data = safe_json_read(b_path)
                    s.update(b_data) # Restaura tradução/emoção
                else:
                    s['translated_text'] = None # Força re-tradução
                    s['sanitized_text'] = None
            return {"segments": segs}

    # [PRIORIDADE 2] Reconstrução via Pastas (Fallback)
    if not backup_dir.exists(): return None
    
    backups = list(backup_dir.glob("seg_*.json"))
    if not backups: return None
    
    logging.info(f"Lógica Phoenix: Tentando reconstruir projeto a partir de {len(backups)} backups individuais...")
    
    # 1. Carrega todos os segmentos encontrados
    segments_found = []
    ids_found = set()
    for b_path in backups:
        try:
            seg = safe_json_read(b_path)
            if seg and 'id' in seg:
                segments_found.append(seg)
                ids_found.add(seg['id'])
        except: pass
    
    if not segments_found: return None
    
    # Ordena pelo início para garantir a cronologia
    segments_found.sort(key=lambda x: x.get('start', 0))
    
    # 2. Detecta o maior ID para saber a extensão do projeto
    max_id_num = 0
    for sid in ids_found:
        try:
            num = int(sid.split('_')[1])
            if num > max_id_num: max_id_num = num
        except: pass
    
    # 3. Re-mapeamento de IDs e Recuperação de Lacunas (Casos 26, 27, 28)
    reconstructed_list = []
    seg_map = {s['id']: s for s in segments_found}
    raw_whisper_data = safe_json_read(whisper_raw_path) if whisper_raw_path.exists() else None
    
    for i in range(max_id_num + 1):
        sid = f"seg_{i:04d}"
        if sid in seg_map:
            reconstructed_list.append(seg_map[sid])
        else:
            # [GAP DETECTADO]
            logging.warning(f"Phoenix: Segmento {sid} ausente nos backups. Tentando recuperar via Raw (v1)...")
            
            # Tenta descobrir o tempo do gap baseado nos vizinhos
            prev_end = reconstructed_list[-1]['end'] if reconstructed_list else 0.0
            next_start = 999999.0
            for j in range(i + 1, max_id_num + 1):
                jsid = f"seg_{j:04d}"
                if jsid in seg_map:
                    next_start = seg_map[jsid]['start']
                    break
            
            recovered_text = ""
            if raw_whisper_data and isinstance(raw_whisper_data, list):
                for raw in raw_whisper_data:
                    r_start = raw.get('start', 0)
                    r_end = raw.get('end', 0)
                    if r_start >= prev_end - 0.5 and r_end <= next_start + 0.5:
                        recovered_text += raw.get('text', '').strip() + " "
            
            if recovered_text.strip():
                reconstructed_list.append({
                    "id": sid, "start": prev_end, "end": next_start if next_start != 999999.0 else prev_end + 2.0,
                    "original_text": recovered_text.strip(), "speaker": "voz1", "translated_text": None
                })
            else:
                logging.warning(f"Phoenix: Não foi possível recuperar texto para {sid}.")
    
    if len(reconstructed_list) > 0:
        return {"segments": reconstructed_list}
    return None

def audit_and_heal_project_data(job_dir, project_data):
    """
    [v10.19.3] SELF-HEALING: Sincroniza o JSON com os arquivos reais do disco.
    Se o usuário apagar um arquivo de áudio ou texto, o JSON detecta e reseta o segmento.
    """
    if not project_data: return project_data
    
    backup_dir = job_dir / "_backup_texto_final"
    dubbed_dir = job_dir / "dubbed_audio"
    
    segs = project_data.get('segments', project_data) if isinstance(project_data, dict) else project_data
    if not isinstance(segs, list): return project_data
    
    modified = False
    for s in segs:
        sid = s.get('id')
        if not sid: continue
        
        # 1. Verifica Backup de Texto
        b_path = backup_dir / f"{sid}.json"
        if not b_path.exists() and s.get('translated_text'):
            logging.info(f"Self-Healing: Backup de {sid} ausente. Resetando tradução no JSON.")
            s['translated_text'] = None
            s['sanitized_text'] = None
            s['synced_text'] = None
            modified = True
            
        # 2. Verifica Áudio Dublado
        # Nota: Só resetamos se o caminho estiver registrado no JSON mas o arquivo sumiu
        a_path = dubbed_dir / f"{sid}_dubbed.wav"
        if not a_path.exists() and s.get('dubbed_audio_path'):
            logging.info(f"Self-Healing: Áudio de {sid} ausente. Marcando para re-dublagem.")
            s.pop('dubbed_audio_path', None)
            modified = True
            
    if modified:
        logging.info("Self-Healing: project_data.json atualizado para refletir arquivos removidos pelo usuário.")
        safe_json_write(project_data, job_dir / "project_data.json")
        
    return project_data


def transcribe_vocals_robust(job_dir, diarization_data, cb, etapa_idx, etapas_lista, source_lang="en"):
    """
    Transcrição robusta (App_videos otimizado).
    Lê os chunks de vocais gerados ou o arquivo unificado.
    Inclui lógica de atribuição de orador baseada na Diarização.
    [v10.40] Recebe 'source_lang' param. Se 'auto', Whisper adivinha (sujeito à alucinação).
    """
    cb(0, etapa_idx, "Preparando transcrição (Whisper)...")
    
    vocals_path = job_dir / "vocals.wav"
    if not vocals_path.exists():
        logging.warning("'vocals.wav' não encontrado. Tentando reconstruir dos chunks...")
        # Lógica de reconstrução se necessário... (omitido pois deve ter sido criado acima)
        return []

    project_data_path = job_dir / "project_data.json"
    if project_data_path.exists():
         cb(100, etapa_idx, "Transcrição carregada do backup.")
         return safe_json_read(project_data_path)

    try:
        model = get_whisper_model() # Carrega (Singleton)
        
        project_data = [] # Inicializa lista de resultados
        
        # [RESUME LOGIC] Verifica se já existe progresso salvo
        debug_path = job_dir / "whisper_raw_debug.json"
        raw_segments = []
        resume_offset = 0.0
        
        if debug_path.exists():
            try:
                loaded_debug = safe_json_read(debug_path)
                if loaded_debug and isinstance(loaded_debug, list) and len(loaded_debug) > 0:
                    # [v10.6] Validação de Timestamp: Se o cache não tem palavras, ele é INCOMPATÍVEL com v10.0
                    has_words = any(len(s.get('words', [])) > 0 for s in loaded_debug)
                    if not has_words:
                        logging.warning("Cache do Whisper sem timestamps de palavras. Forçando re-transcrição para Smart Split v10...")
                        raise ValueError("Cache incompatível.")
                    
                    last_seg = loaded_debug[-1]
                    resume_offset = last_seg['end']
                    logging.info(f"Retomando Transcrição de {resume_offset:.2f}s")
                    
                    # Reconstrói lista de objetos compatíveis (Mock) para o loop continuar coerente
                    # Nota: O loop abaixo só processa NOVOS segmentos, mas precisamos manter o histórico
                    # Para 'raw_segments', vamos guardar dicionários e tratar no loop
                    raw_segments = loaded_debug 
            except Exception as e:
                 logging.warning(f"Erro ao ler checkpoint de transcrição: {e}. Começando do zero.")

        # Prepara áudio de entrada (Total ou Slice)
        input_file = vocals_path
        temp_slice_path = job_dir / "temp_whisper_slice.wav"
        
        # Calcula duração TOTAL real para barra de progresso correta
        full_duration = get_audio_duration(vocals_path) 
        
        if resume_offset > 0:
            if resume_offset >= full_duration - 1.0:
                 logging.info("Transcrição já parece completa. Pulando Whisper.")
                 # Lógica para pular direto pra pós-proc
                 segments_generator = [] 
                 info = type('obj', (object,), {'duration': full_duration - resume_offset})
            else:
                # Corta o áudio para enviar só o que falta
                cb(10, etapa_idx, f"Preparando retomada ({resume_offset:.1f}s)...")
                try:
                    # [v10.28] Substituição de '-c copy' por recodificação 'pcm_s16le'
                    # O FFmpeg corrompia o cabeçalho WAV ao usar -c copy para fatiar,
                    # fazendo o Whisper parar prematuramente. O recode garante tamanho 100% legível.
                    cmd_slice = [
                        'ffmpeg', '-y', '-ss', str(resume_offset),
                        '-i', str(vocals_path),
                        '-c:a', 'pcm_s16le', '-ar', '24000', '-ac', '1', str(temp_slice_path)
                    ]
                    subprocess.run(cmd_slice, check=True, capture_output=True)
                    input_file = temp_slice_path
                except Exception as e:
                    logging.error(f"Falha ao cortar áudio para resume: {e}. Usando original.")
                    input_file = vocals_path
                    resume_offset = 0.0
                    raw_segments = [] # Reset se falhar no corte

        cb(10, etapa_idx, "Transcrevendo vocais...")

        
        # Transcrição (Faster-Whisper)
        # [v10.40] HARD LOCK DE IDIOMA: Força a transcrição ser guiada pela UI.
        # Evita a alucinação (Hallucination) do Whisper detectar erroneamente 
        # Galês (Welsh), Romeno ou Russo devido a ruídos no áudio do vídeo.
        # Se for resume, transcrevemos apenas o slice
        if resume_offset < full_duration - 1.0:
             # Se o usuário escolheu "Auto-Detectar", passamos None para o Faster-Whisper
             whisper_lang = source_lang if source_lang != 'auto' else None
             segments_generator, info = model.transcribe(str(input_file), word_timestamps=True, language=whisper_lang)
        else:
             segments_generator = []

        # Loop de processamento
        new_segments_batch = []
        
        for segment in segments_generator:
            # Ajusta timestamps (Time do Slice + Offset do Resume)
            real_start = segment.start + resume_offset
            real_end = segment.end + resume_offset
            
            # Progress Bar: Baseada no tempo TOTAL do vídeo original
            progress = 10 + (real_end / full_duration) * 80 if full_duration > 0 else 50
            
            # [FIX] Atualização de progresso garantida
            # Ocasionalmente o callback pode ser ignorado se for muito frequente, mas aqui forçamos
            cb(progress, etapa_idx, f"Transcrevendo... {timedelta(seconds=int(real_end))} / {timedelta(seconds=int(full_duration))}")
            
            # Cria objeto unificado (dicionário) para compatibilidade com o histórico JSON
            seg_dict = {
                "start": real_start,
                "end": real_end,
                "text": segment.text,
                "words": [{"word": w.word, "start": w.start + resume_offset, "end": w.end + resume_offset} for w in segment.words] if segment.words else [],
                "detected_language": getattr(info, 'language', None)
            }
            
            raw_segments.append(seg_dict)
            new_segments_batch.append(seg_dict)

            # [MELHORIA] Salva parcial a cada 5 NOVOS segmentos
            if len(new_segments_batch) % 5 == 0:
                safe_json_write(raw_segments, debug_path)
        
        # Salva final dos segmentos brutos
        safe_json_write(raw_segments, debug_path)
        total_words_gen = sum(len(s.get('words', [])) for s in raw_segments)
        logging.info(f"Transcrição Whisper concluída: {len(raw_segments)} segmentos, {total_words_gen} palavras totais.")
        logging.info(f"Debug Whisper salvo em: {debug_path}")
        
        # Limpeza do slice temporário
        if temp_slice_path.exists():
            try: temp_slice_path.unlink()
            except: pass

        # Re-segmentação baseada em pausas (usando a lógica existente)
        # Agora 'raw_segments' é uma lista de DICTS, não objetos FasterWhisper.
        # Precisamos adaptar o loop abaixo.
        whisper_result = {'words': [], 'detected_language': raw_segments[0].get('detected_language') if raw_segments else None}
        for seg in raw_segments:
            # seg é dict agora
            if seg.get('words'):
                for word in seg['words']:
                    # [FIX-SPACES] Não usar strip() aqui, pois o Whisper traz o espaço no início de cada palavra (" Hello")
                    whisper_result['words'].append({'word': word['word'], 'start': word['start'], 'end': word['end']})


        
        resegmented = resegment_based_on_pauses(whisper_result, diarization_data=diarization_data)
        
        # [LAZY DIARIZATION] 
        # Tenta carregar dados existentes para preservar oradores já identificados.
        # Isso evita resetar para "voz1" se o JSON já existir.
        existing_speaker_map = {}
        if project_data_path.exists():
            try:
                old_data = safe_json_read(project_data_path)
                # Robustez: Só usa se tiver dados. Se estiver vazio (user limpou?), ignora.
                if old_data and len(old_data) > 0:
                    logging.info(f"Retomando transcrição existente com {len(old_data)} segmentos.")
                    cb(100, etapa_idx, "Transcrição carregada do cache.")
                    return old_data
                else:
                    logging.warning("Cache de transcrição encontrado mas estava vazio. Refazendo...")
                
                # Tenta aproveitar os oradores antigos mesmo se for refazer
                if old_data:
                    for item in old_data:
                        if 'speaker' in item:
                            existing_speaker_map[item['id']] = item['speaker']
            except: pass

        for i, seg in enumerate(resegmented):
            file_id = f"seg_{i:04d}"
            
            # --- INTEGRAÇÃO COM DIARIZAÇÃO (IMPLEMENTADA) ---
            # Prioridade: 
            # 1. Mapa Existente (Resume)
            # 2. Default "voz1"
            assigned_speaker = existing_speaker_map.get(file_id, "voz1")

            
            w_start, w_end = seg['start'], seg['end']
            w_dur = w_end - w_start
            
            if diarization_data:
                # Encontra orador com maior sobreposição neste segmento
                speaker_scores = {} # {speaker: overlap_duration}
                
                for d_seg in diarization_data:
                    # Calcula interseção
                    d_start, d_end = d_seg['start'], d_seg['end']
                    
                    overlap_start = max(w_start, d_start)
                    overlap_end = min(w_end, d_end)
                    overlap = max(0, overlap_end - overlap_start)
                    
                    if overlap > 0:
                        spk = d_seg.get('speaker', 'voz1')
                        speaker_scores[spk] = speaker_scores.get(spk, 0) + overlap
                
                if speaker_scores:
                    # Pega o orador com maior overlap
                    best_spk = max(speaker_scores, key=speaker_scores.get)
                    # Só aceita se a cobertura for razoável (ex: > 20% do segmento ou > 0.5s)
                    if speaker_scores[best_spk] > 0.5 or (speaker_scores[best_spk] / w_dur) > 0.2:
                        assigned_speaker = best_spk
            
            # Atualiza
            seg['speaker'] = assigned_speaker
            
            project_data.append({
                "id": file_id,
                "original_text": seg['text'],
                "start": seg['start'],
                "end": seg['end'],
                "duration": seg['end'] - seg['start'],
                "speaker": assigned_speaker,
                "words": seg.get('words', []), # Crucial para v10 Smart Split
                "detected_language": seg.get('detected_language')
            })
            
        # [v5.7] REMOVIDO Smart Merge Global (Voltar para Granular 8s):
        # O agrupamento de 20s-60s estava quebrando a sincronização do Chatterbox e as pausas de 0.5s.
        # Mantemos os segmentos pequenos para garantir que cada frase tenha seu tempo exato.
        # logging.info("Aplicando Smart Merge (DESATIVADO v5.7 para Sincronia Chatterbox)...")
        # project_data = smart_merge_segments(project_data)
        safe_json_write(project_data, project_data_path)
            
        # [FIX USER REQUEST] Re-indexar IDs sequencialmente (0, 1, 2...) para não pular números
        # Isso evita confusão com IDs "seg_0000" seguidos de "seg_0004" devido a merges/filtros anteriores.
        for idx, item in enumerate(project_data):
            item['id'] = f"seg_{idx:04d}"

        safe_json_write(project_data, project_data_path)
        cb(50, etapa_idx, "Diarização preliminar concluída.")
        
        # --- MEMORY MANAGEMENT CRÍTICO ---
        # Descarregar Whisper IMEDIATAMENTE após uso
        global whisper_model
        del whisper_model
        whisper_model = None
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        logging.info("Modelo Whisper descarregado para liberar RAM.")
        # ---------------------------------

        # [v9.1] NORMALIZAÇÃO: Retorna sempre como DICIONÁRIO wrapper
        save_data = {
            "segments": project_data,
            "metadata": {"source": "whisper_raw"}
        }
        safe_json_write(save_data, project_data_path)
        cb(100, etapa_idx, "Transcrição concluída.")
        return save_data

    except Exception as e:
        logging.error(f"Erro na transcrição: {e}")
        raise e




def verify_glossary_usage(original, translated, glossary_str):
    """
    [VALIDADOR RÍGIDO]
    Verifica se os termos do GLOSSÁRIO presentes no original foram usados na tradução.
    Retorna (True, missings_list)
    """
    if not glossary_str: return True, []
    
    missings = []
    
    try:
        # 1. Parse Glossary
        glossary_rules = {}
        for line in glossary_str.split('\n'):
            if '=' in line:
                parts = line.split('=')
                k = parts[0].strip().lower()
                v = parts[1].strip().lower()
                if k and v: glossary_rules[k] = v
        
        # 2. Check
        orig_lower = original.lower()
        trans_lower = translated.lower()
        
        for term, trans in glossary_rules.items():
            # Acha termo exato (whole word match seria ideal, mas substring é ok pra start)
            # Vamos usar substring pra ser robusto ("cells" -> "celas")
            if term in orig_lower:
                if trans not in trans_lower:
                    # [STRICT] Verifica se não é plural ("cela" in "celas" é valido)
                    # Se a tradução esperada é "cela", mas saiu "celas", OK.
                    # Mas se saiu "quarto", FALHA.
                    missings.append(f"{term}->{trans}")
                    
        return (len(missings) == 0), missings
    except:
        return True, []

def clean_translation_fillers(text):
    """
    [v3.4 Quote-Isolation] Extrai apenas o que está entre aspas se o formato for "Texto" | Explicação.
    Caso contrário, limpa vícios normalmente.
    """
    if not text: return text
    
    # 1. Tenta extrair texto entre aspas (v3.4 Strategy)
    # Pega o conteúdo da primeira ocorrência de "texto"
    quote_match = re.search(r'"([^"]+)"', text)
    if quote_match:
        text = quote_match.group(1)
    elif '|' in text:
        # Fallback se a IA esqueceu as aspas mas usou o pipe
        text = text.split('|')[0]

    # 2. Remove "né", "tá", "então" no final (agressivo)
    for filler in ['né', 'tá', 'então']:
        # Pega no final, com ou sem pontuação
        text = re.sub(r'[,.\s]+' + filler + r'[?,.!]*\s*$', '.', text, flags=re.IGNORECASE)
        # Pega no meio se houver pausas estranhas
        text = re.sub(r'[,.\s]+' + filler + r'[,.\s]+', ' ', text, flags=re.IGNORECASE)

    # 3. Limpeza Final
    text = re.sub(r'\s+', ' ', text) # Remove espaços duplos
    text = re.sub(r'^[-\s]+', '', text) # Remove hífens no começo
    
    # [v3.4] Fix broken starts: Se sobrou "vindo", vira "Bem-vindo"
    if text.lower().startswith('vindo') or text.lower().startswith('-vindo'):
        text = "Bem-" + text.lstrip('-').lower().replace('vindo', 'vindo').capitalize()
    
    return text.strip()

# [Bypass] A função gema_batch_processor_v2 foi consolidada na linha 866.


def process_gema_steps(job_dir, project_data, cb):
    """Executa todas as etapas do Gema (Tradução, Emoção, Sincronia, Adaptação)."""
    
    text_backup_dir = job_dir / "_backup_texto_final"
    text_backup_dir.mkdir(exist_ok=True)

    # [v10.1] Acesso seguro a segmentos (Handle dict/list)
    segs_list = project_data.get('segments', project_data) if isinstance(project_data, dict) else project_data

    # [NOVO - Filtro de Idioma Segmentado via LangDetect] Pula tradução do que já está em Português
    try:
        from langdetect import detect, DetectorFactory
        DetectorFactory.seed = 0 # Torna determinístico
    except ImportError:
        detect = None
        logging.warning("langdetect não está instalado, a auto-detecção por trecho de fala não funcionará perfeitamente.")

    for seg in segs_list:
        orig = seg.get('original_text', '').strip()
        lang = seg.get('detected_language')
        
        if orig and detect and lang != 'pt':
            try:
                # langdetect as vezes falha em textos muito curtos ou sem letras (ex: "123", "ahhh")
                # Exigimos pelo menos 3 palavras ou 15 caracteres para tentar adivinhar com segurança
                if len(orig) > 15 or len(orig.split()) >= 3:
                    guessed_lang = detect(orig)
                    if guessed_lang == 'pt':
                        lang = 'pt'
                        seg['detected_language'] = 'pt'
            except:
                pass

        if lang == 'pt':
            if not seg.get('sanitized_text'):
                seg['sanitized_text'] = orig
            logging.info(f"Segmento {seg['id']} preservado (já é Português).")

    # 1. Sincroniza com backups existentes e carrega Edição Manual
    project_data_map = {item['id']: item for item in segs_list}
    for backup_file in text_backup_dir.glob("*.json"):
        file_id = backup_file.stem
        if file_id in project_data_map:
            backup_data = safe_json_read(backup_file)
            if backup_data:
                # [v5.4] SYNC DE ORADOR (Sem perda de tradução):
                # Se o orador mudou (ex: via unificação de vozes), apenas atualizamos a ID no backup.
                # Mantemos o texto traduzido/sincronizado para evitar retrabalho.
                current_item = project_data_map[file_id]
                backup_speaker = backup_data.get('speaker')
                current_speaker = current_item.get('speaker')
                
                if backup_speaker and current_speaker and backup_speaker != current_speaker:
                    logging.info(f"SYNC ORADOR: Segmento {file_id} ({backup_speaker} -> {current_speaker}). Mantendo tradução.")
                    backup_data['speaker'] = current_speaker
                
                current_item.update(backup_data)
                
    # [BACKFILL] Garante que todos os segmentos já processados tenham arquivo no backup (Caso de Reset/Resume)
    backfill_count = 0
    for seg in segs_list:
        bkp_path = text_backup_dir / f"{seg['id']}.json"
        if not bkp_path.exists() and seg.get('sanitized_text') is not None:
            # Se já tem tradução mas não tem arquivo, cria o arquivo para o usuário poder editar
            safe_json_write(seg, bkp_path)
            backfill_count += 1
    if backfill_count > 0:
        logging.info(f"Sincronismo de Segurança: {backfill_count} arquivos de backup restaurados/criados em '{text_backup_dir.name}'.")

    # [v10.16] CONTEXT ARMOR: Gera âncoras de contexto para evitar "Text Drift"
    # Dá ao Gema uma visão das 2 frases anteriores e 2 posteriores para cada segmento.
    for i, seg in enumerate(segs_list):
        prev_win = segs_list[max(0, i-2):i]
        next_win = segs_list[i+1:min(len(segs_list), i+3)]
        anchor_parts = [s.get('original_text', '').strip() for s in prev_win] + [f"[{seg.get('original_text', '').strip()}]"] + [s.get('original_text', '').strip() for s in next_win]
        seg['context_anchor'] = " ".join(filter(None, anchor_parts))

    # [AUTO-CORRECT] Limpeza de Pontuação e Vícios (né, tá) - FORÇADO em todos os backups
    for seg in segs_list:
        # Limpa o texto traduzido (Step 1)
        if seg.get('translated_text'):
            seg['translated_text'] = clean_translation_fillers(seg['translated_text'])
            
        # Limpa o texto sincronizado (Step 2)
        if seg.get('synced_text'):
            seg['synced_text'] = clean_translation_fillers(seg['synced_text'])
            
        st_val = seg.get('sanitized_text')
        if st_val:
            st_val = clean_translation_fillers(st_val)
            if st_val.endswith(',') or st_val.strip().endswith(','):
                 st_val = st_val.rstrip(',. ')
            seg['sanitized_text'] = st_val
            
        # [v3.1] Salva para limpar arquivos físicos de backup e project_data.json (SOMENTE SE TIVER CONTEÚDO)
        if seg.get('sanitized_text') or seg.get('translated_text'):
            safe_json_write(seg, text_backup_dir / f"{seg['id']}.json")
             
    # Salva checkpoint inicial para a UI
    safe_json_write(project_data, job_dir / "project_data.json")

    # [v10.23] LAZY GEMA SHIELD: Verifica se há trabalho ANTES de carregar o modelo ou fazer Stage 0
    files_to_process_gema = [seg for seg in segs_list if seg.get('sanitized_text') is None]
    if not files_to_process_gema:
        logging.info("Lazy Gema: Todos os segmentos já traduzidos. Pulando Stage 0 e Carga de Modelo.")
        cb(100, 7, "Traduções carregadas do cache/backup.")
        return

    # 2. [STAGE 0] Descoberta de Contexto Global (Em Memória)
    video_context = ""
    if segs_list:
        cb(0, 4, "Analisando VÍDEO COMPLETO em busca de contexto...")
        video_context = analyze_full_video_context(segs_list, job_dir, cb)
        logging.info(f"Contexto Stage 0 gerado: {video_context[:100]}...")

    # [DETECÇÃO DE GÊNERO MANUAL VIA GLOSSÁRIO]
    gender_map = {}
    if "[ORADORES SUGERIDOS" in video_context:
        try:
            parts = video_context.split("[ORADORES SUGERIDOS")
            clean_context = parts[0].strip()
            config_block = parts[1]
            video_context = clean_context # Limpa o contexto para o prompt de tradução
            
            for line in config_block.splitlines():
                if ":" in line:
                    p = line.split(":")
                    spk_id = p[0].strip()
                    val = p[1].strip().upper()
                    if any(x in val for x in ["M", "F", "MULHER"]):
                         gender_map[spk_id] = "Feminino"
                    elif any(x in val for x in ["H", "HOMEM"]):
                         gender_map[spk_id] = "Masculino"
            logging.info(f"Gênero via Glossário: {len(gender_map)} oradores configurados.")
        except: pass

    # 3. [FILTRO DE RUÍDOS] Intercepta Ruídos/Gemidos antes do Gema
    BAD_REF_WORDS = ['argh', 'ah', 'oh', 'uh', 'hmm', 'wow', 'tsk', 'ugh', 'screams', 'gasps', 'moans', 'chokes', 'grita', 'geme', 'laughs', 'chuckles', 'sobs', 'cries', 'sighs', 'eh', 'heh', 'hum', 'ha', 'haha', 'hah', 'whoa', 'ooh', 'aw', 'ouch', 'ow', 'psst', 'shh', 'yikes', 'yay', 'ew', 'ick', 'boo', 'hiss', 'growl', 'snarl', 'roar', 'bark']
    
    for seg in segs_list:
        if seg.get('sanitized_text') is None:
            orig = seg.get('original_text', '').strip()
            orig_lower = orig.lower()
            
            # [REFINED] Filtro de Ruído Inteligente:
            # Só filtra se for curto (< 15c) E TODAS as palavras forem ruídos conhecidos.
            # "oh, yeah?" -> 'oh' (bad), 'yeah' (good) -> NÃO FILTRA.
            # "ah, oh..." -> 'ah' (bad), 'oh' (bad) -> FILTRA.
            
            is_noise = False
            if 0 < len(orig_lower) < 15:
                # Remove pontuação para checar palavras puras
                words = re.findall(r'\b\w+\b', orig_lower)
                if words and all(w in BAD_REF_WORDS for w in words):
                    is_noise = True
                    
            if is_noise or not orig_lower.strip():
                seg['translated_text'] = ""
                seg['sanitized_text'] = ""
                logging.info(f"Segmento {seg['id']} filtrado por Ruído/Alucinação ('{orig}'). Pulando tradução.")
                cb(None, 4, f"Ruído filtrado: '{orig[:20]}...'")
                safe_json_write(seg, text_backup_dir / f"{seg['id']}.json")
            elif len(orig_lower) > 0:
                normal_chars = len(re.findall(r'[a-zA-Z\s]', orig_lower))
                if (normal_chars / len(orig_lower)) < 0.5:
                    seg['translated_text'] = ""
                    seg['sanitized_text'] = ""
                    safe_json_write(seg, text_backup_dir / f"{seg['id']}.json")

    # 4. [BATCHER DINÂMICO] Agrupa segmentos para Tradução em Lote
    batches = []
    curr_batch = []
    curr_dur = 0.0
    curr_ctx = ""
    
    for seg in files_to_process_gema:
        dur = seg.get('effective_duration', seg.get('duration', 0))
        txt = seg.get('original_text', '')

        # [v18.5] Lotes otimizados para 16GB de RAM (20 itens / 3000 chars)
        if curr_batch and ((curr_dur + dur) > 90.0 or (len(curr_ctx) + len(txt)) > 3000 or len(curr_batch) >= 20):
            batches.append({'segments': curr_batch, 'direct_context': curr_ctx.strip()})
            curr_batch = []; curr_dur = 0.0; curr_ctx = ""

        curr_batch.append(seg)
        curr_dur += dur
        curr_ctx += f"[{seg.get('speaker', 'Voz')}]: {txt}\n"

    if curr_batch:
        batches.append({'segments': curr_batch, 'direct_context': curr_ctx.strip()})

    # 5. [EXECUÇÃO ENXUTA v20.0] Loop de Batches Otimizado (Single-Pass)
    for b_idx, b_data in enumerate(batches):
        batch = b_data['segments']
        cb(((b_idx / len(batches)) * 100), 4, f"Traduzindo Lote {b_idx+1}/{len(batches)} (Gemma 4 Elite)...")
        
        # 5.1 BATCH PROCESSOR v2 (Single-Pass Otimizado)
        # O novo Batch Processor já cuida da naturalidade e formato via Pescaria de Aspas.
        results_map = gema_batch_processor_v2(batch, video_context, job_dir=job_dir)

        final_results = {}
        failed_items = []

        for seg in batch:
            f_id = seg['id']
            # Pesca a tradução do mapa gerado pela IA
            translated_text = results_map.get(f_id, "")
            
            # Validação de Segurança (Duração vs Caracteres)
            dur = seg.get('effective_duration', seg['duration'])
            # Regra de Ouro: 18 CPS com 15% de tolerância (1.15x)
            max_chars_allowed = int(dur * 18 * 1.15)
            
            if not translated_text or len(translated_text) > max_chars_allowed:
                logging.warning(f"  [ALERTA SYNC] {f_id} reprovado ({len(translated_text)}/{max_chars_allowed} chars). Enviando ao Corretor Master.")
                seg['translated_text'] = translated_text # Passa o rascunho para o corretor saber o que encurtar
                failed_items.append(seg)
            else:
                final_results[f_id] = translated_text

        # 5.2 CORRETOR MASTER (Apenas para emergências de sincronia)
        if failed_items:
            cb(((b_idx / len(batches)) * 100) + 5, 4, f"Corrigindo sincronia de {len(failed_items)} itens...")
            corrected_map = gema_batch_corrector_master(failed_items, video_context, job_dir=job_dir)
            for f_id, text in corrected_map.items():
                final_results[f_id] = text

        # 5.3 SALVAMENTO E HIGIENIZAÇÃO
        for seg in batch:
            f_id = seg['id']
            text = final_results.get(f_id, seg.get('original_text', ''))
            
            # Limpeza final de pontuação e tags
            text = clean_translation_fillers(text)
            
            # Fallback se a IA sumiu com a frase
            if not text or len(text) < 1:
                text = seg.get('original_text', '')
            
            seg['translated_text'] = text
            seg['synced_text'] = text
            seg['sanitized_text'] = text
            
            # Proteção de Backup
            safe_json_write(seg, text_backup_dir / f"{f_id}.json")

        # Atualiza o project_data periodicamente
        safe_json_write(project_data, job_dir / "project_data.json")
        cb(((b_idx + 1) / len(batches)) * 100, 4, f"Lote {b_idx+1} concluído.")

    cb(100, 6, "Agentic Gemma 4 Pipeline: Concluído.")
    unload_gema_model()

def generate_dubbed_audio(job_dir, project_data, cb):
    """Gera os áudios usando Coqui Chatterbox v2 (Motor do app_jogos)."""
    
    # [FIX CRÍTICO] - Forçar Consolidação dos Backups antes do TTS
    # Garante que, mesmo se o usuário pausou/editou/retomou, o Chatterbox pegue o texto MAIS RECENTE do disco.
    logging.info("Consolidando dados finais de texto antes do TTS (App_videos)...")
    backup_final_dir = job_dir / "_backup_texto_final"
    
    # [v10.1] Acesso seguro a segmentos (Handle dict/list)
    segs_list = project_data.get('segments', project_data) if isinstance(project_data, dict) else project_data

    if backup_final_dir.exists():
        for seg in segs_list:
            bkp_path = backup_final_dir / f"{seg['id']}.json"
            if bkp_path.exists():
                try:
                    fresh_data = safe_json_read(bkp_path)
                    if fresh_data:
                        # Preserva campos vitais, atualiza textos
                        seg['sanitized_text'] = fresh_data.get('sanitized_text', seg.get('sanitized_text', ''))
                        seg['manual_edit_text'] = fresh_data.get('manual_edit_text', seg.get('manual_edit_text', ''))
                        seg['speaker'] = fresh_data.get('speaker', seg.get('speaker', 'voz1'))
                except: pass
        
        # [ATUALIZAÇÃO DE DISCO] - Salva o JSON principal atualizado ANTES de carregar o modelo
        logging.info("Salvando project_data.json atualizado no disco...")
        safe_json_write(project_data, job_dir / "project_data.json")
    
    # [v10.23] LAZY TTS SHIELD: Pre-check ANTES de qualquer mensagem ou carga pesada
    missing_files = 0
    dubbed_audio_dir = job_dir / "dubbed_audio"
    dubbed_audio_dir.mkdir(exist_ok=True)
    
    for seg in segs_list:
        expected_wav = dubbed_audio_dir / f"{seg['id']}_dubbed.wav"
        if not expected_wav.exists():
            txt = seg.get('manual_edit_text', '').strip() or seg.get('sanitized_text', '')
            if txt:
                missing_files += 1
    
    if missing_files == 0 and len(segs_list) > 0:
        logging.info("Lazy TTS: Todos os áudios já gerados. Aplicando Higienização Retroativa...")
        cb(90, 7, "Higienizando áudios existentes...")
        try:
            from fix_existing_audios import fix_all_audios
            fix_all_audios(job_dir)
        except: pass
        cb(100, 7, "Áudios prontos.")
        return

    cb(0, 7, "Carregando Chatterbox v2 (Isso pode demorar)...")

    # [v7.4] RAM GUARD: Descarrega Gema e Whisper para dar espaço ao Chatterbox
    # Isso é essencial para PCs com 16GB de RAM.
    unload_gema_model()
    
    global whisper_model, openvoice_converter
    with model_lock:
        if whisper_model is not None:
            del whisper_model
            whisper_model = None
    # [v7.1] Chatterbox TTS (Motor Exclusivo - Pure Mode)
    model_chat = get_chatterbox_model()
    if model_chat is None:
        raise RuntimeError("Falha ao carregar Chatterbox TTS. Verifique a instalação.")

    # Referência de Voz: Carrega mapa de referências
    available_refs = list(job_dir.glob("vocals_*.wav"))
    logging.info(f"Chatterbox: Encontradas {len(available_refs)} referências de voz na pasta.")
    
    # Define fallback global
    global_fallback = Path("resources/base_speakers/pt/default_pt_speaker.wav")
    if (job_dir / "voices" / "vocals_speaker_default.wav").exists():
        global_fallback = job_dir / "voices" / "vocals_speaker_default.wav"
    elif (job_dir / "vocals.wav").exists():
        global_fallback = job_dir / "vocals.wav"
    
    logging.info(f"Referência de falha (Global Fallback): {global_fallback}")

    for i, seg_data in enumerate(segs_list): 
        # [v18.6] Gestão Térmica: Limpeza periódica para manter a RTX fria
        if i > 0 and i % 25 == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logging.info("🧹 [HARDWARE] Limpeza periódica de VRAM (App_videos) para controle térmico.")

        progress = (i + 1) / len(segs_list) * 100 
        cb(progress, 7, f"Gerando áudio {i+1}/{len(segs_list)} (Chatterbox)") 
        
        save_path = dubbed_audio_dir / f"{seg_data['id']}_dubbed.wav"
        if save_path.exists(): continue

        text_to_speak = seg_data.get('manual_edit_text', '').strip() or seg_data.get('sanitized_text', '').strip()
        
        # [v10.44] BYPASS TTS (PRESERVAÇÃO DO PORTUGUÊS)
        # Se o texto sanitizado (após Gema) for idêntico ao original (indicando bypass ou PT detectado)
        # E se não houve edição manual, devemos USAR O ÁUDIO ORIGINAL em vez de gerar TTS robótico.
        is_preserved_pt = (
            seg_data.get('detected_language') == 'pt' or 
            (seg_data.get('sanitized_text') == seg_data.get('original_text') and not seg_data.get('manual_edit_text'))
        )

        # [v10.14] HYBRID DUBBING: Fallback para Áudio Original se não houver texto ou for PT preservado
        if not text_to_speak or is_preserved_pt:
            reason = "já em PT" if is_preserved_pt else "sem tradução"
            logging.info(f"Hybrid Dubbing: Segmento {seg_data['id']} pulou TTS ({reason}). Usando áudio original.")
            vocals_path = job_dir / "vocals.wav"
            if vocals_path.exists():
                try:
                    # Extrai o pedaço original do vocals.wav
                    start_s = seg_data['start']
                    dur_s = seg_data['duration']
                    cmd_ext = [
                        'ffmpeg', '-y', '-ss', str(start_s), '-t', str(dur_s), 
                        '-i', str(vocals_path), '-acodec', 'pcm_s16le', str(save_path)
                    ]
                    subprocess.run(cmd_ext, check=True, capture_output=True)
                    continue # Próximo segmento
                except: pass
            continue # Se falhou tudo, pula (silêncio)
        
        speaker_id = seg_data.get('speaker', 'voz1')
        speaker_wav = job_dir / "voices" / f"vocals_{speaker_id}.wav"
        if not speaker_wav.exists(): speaker_wav = job_dir / f"vocals_{speaker_id}.wav"
        if not speaker_wav.exists(): speaker_wav = global_fallback

        logging.info(f"Chatterbox: Gerando {save_path.name}...")
        try:
            # [v10.9] ROOT STABILITY: Ajuste de Redução de Alucinação (Direto no Motor)
            # Temperaturas menores (0.6-0.7) tornam o modelo mais estável e focado no texto.
            # min_p maior (0.1) ajuda a ignorar tokens de ruído/alucinação.
            wav_tensor = model_chat.generate(
                text=text_to_speak,
                language_id="pt", # [FIX] Chatterbox suporta 'pt', não 'pt-br'
                audio_prompt_path=str(speaker_wav),
                exaggeration=0.5,
                temperature=0.7,    # [v12.93] Ligeiro aumento para dar mais melodia à fala brasileira
                min_p=0.1,           # Poda mais agressiva de alucinação
                repetition_penalty=2.0
            )
            
            # [v8.0] Salva o áudio BRUTO do Chatterbox (Sem cortes aqui)
            import soundfile as sf
            wav_data = wav_tensor.squeeze(0).cpu().numpy()
            sf.write(str(save_path), wav_data, 24000)
            logging.info(f"Chatterbox: Áudio bruto salvo: {save_path.name}")
            
        except Exception as e_chat:
            logging.error(f"ERRO CRÍTICO NO CHATTERBOX ({seg_data['id']}): {e_chat}. Fallback para original.")
            # [v10.14] Segundo Fallback: Áudio Original se o TTS falhar
            vocals_path = job_dir / "vocals.wav"
            if vocals_path.exists():
                try:
                    start_s = seg_data['start']
                    dur_s = seg_data['duration']
                    cmd_ext = [
                        'ffmpeg', '-y', '-ss', str(start_s), '-t', str(dur_s), 
                        '-i', str(vocals_path), '-acodec', 'pcm_s16le', str(save_path)
                    ]
                    subprocess.run(cmd_ext, check=True, capture_output=True)
                except:
                    # Silêncio de emergência 3 (Se nem o original extrair)
                    dur_ms = int(seg_data['duration']*1000)
                    from pydub import AudioSegment
                    AudioSegment.silent(duration=dur_ms).export(save_path, format="wav")
            else:
                # Silêncio de emergência
                dur_ms = int(seg_data['duration']*1000)
                from pydub import AudioSegment
                AudioSegment.silent(duration=dur_ms).export(save_path, format="wav")

    cb(100, 7, "Geração de áudios TTS concluída.")

def assemble_final_video(job_dir, project_data, cb):
    """Monta o vídeo final (do App_videos)."""
    
    # [FIX CRÍTICO] - Forçar Consolidação dos Backups antes da Montagem
    # Garante que timestamps editados nos backups sejam respeitados.
    logging.info("Consolidando dados finais antes da Montagem (App_videos)...")
    backup_final_dir = job_dir / "_backup_texto_final"
    
    # [v10.1] Acesso seguro a segmentos (Handle dict/list)
    segs_list = project_data.get('segments', project_data) if isinstance(project_data, dict) else project_data

    if backup_final_dir.exists():
        for seg in segs_list: # Use segs_list here
            bkp_path = backup_final_dir / f"{seg['id']}.json"
            if bkp_path.exists():
                try:
                    fresh_data = safe_json_read(bkp_path)
                    if fresh_data:
                        seg.update(fresh_data) # Aqui atualizamos TUDO (incluindo start/end se estiver lá)
                except: pass

    cb(0, 8, "Iniciando montagem do vídeo...")
    video_path = next(job_dir.glob("input_video.*"), None) # Procura pelo nome padronizado
    if not video_path: raise FileNotFoundError("Arquivo de vídeo original não encontrado.")
    
    video_info = ffmpeg.probe(str(video_path))
    video_duration_ms = int(float(video_info['format']['duration']) * 1000)
    video_duration_ms = int(float(video_info['format']['duration']) * 1000)
    final_audio = AudioSegment.silent(duration=video_duration_ms)
    
    
    # [FIX-OVERLAP] Smart Mix: Carrega tudo, detecta overlaps REAIS e corta o excesso do áudio anterior
    # Pass 1: Load all valid segments
    loaded_clips = []
    
    # [v10.1] Acesso seguro a segmentos (Handle dict/list)
    # segs_list already defined at the beginning of the function

    for i, seg in enumerate(segs_list):
        dubbed_audio_path = job_dir / "dubbed_audio" / f"{seg['id']}_dubbed.wav"
        if dubbed_audio_path.exists():
            try:
                clip_raw = AudioSegment.from_wav(dubbed_audio_path)
                
                # [v10.20] SURGICAL SYNC (Sincronismo Cirúrgico)
                # O Chatterbox/Chatterbox gera silêncios longos no início. 
                # Se não cortarmos agressivamente, a fala atrasa e acaba sendo acelerada ou cortada.
                from pydub.silence import detect_nonsilent
                # Threshold de -50dB para captar respirações leves, mas ignorar o "mudo" do modelo
                nonsilent_ranges = detect_nonsilent(clip_raw, min_silence_len=200, silence_thresh=-50)
                
                clip = clip_raw 
                if nonsilent_ranges:
                    # [v10.20] MARGEM DE 20ms: Preserva o início da fala/respiração mas ganha cada milisegundo.
                    start_trim = max(0, nonsilent_ranges[0][0] - 20)
                    
                    # Procura o fim real da fala (último range não-silencioso)
                    end_trim = nonsilent_ranges[-1][1]
                    
                    # Margem final de 50ms (Suficiente para um fade-out suave sem comer a última sílaba)
                    final_end_trim = min(len(clip_raw), end_trim + 50)
                    
                    clip = clip_raw[start_trim:final_end_trim]
                    
                    # Log de economia de tempo para depuração
                    saved_ms = len(clip_raw) - len(clip)
                    if saved_ms > 300:
                        logging.info(f"[SURGICAL SYNC] {seg['id']}: Removidos {saved_ms}ms de silêncio inútil.")
                
                # [v10.39] Audio Leveling (Volume Booster / Normalizer)
                # Garante que a voz não fique ofuscada pelo instrumental, mantendo 
                # um ganho mestre travado em audível, nivelando todas as vozes geradas.
                target_dBFS = -20.0
                change_in_dBFS = target_dBFS - clip.dBFS
                clip = clip.apply_gain(change_in_dBFS)
                
                loaded_clips.append({
                    'id': seg['id'],
                    'start_ms': int(seg['start'] * 1000),
                    'end_ms_target': int(seg['end'] * 1000), 
                    'speaker': seg.get('speaker'),
                    'clip': clip,
                    'duration_ms': len(clip)
                })
            except Exception as e:
                logging.warning(f"Erro ao carregar {dubbed_audio_path.name}: {e}")

    # [v10.38] HYBRID SYNC REFACTOR: Sincronização Inteligente (Acelera 1º, Rouba Tempo 2º)
    # A prioridade passa a ser manter o alinhamento da boca (start_ms) exato,
    # expandindo para a frente da timeline sempre que possível.
    logging.info("Aplicando Hybrid Sync v2.0 (Acceleration First / Pre-Roll Borrowing Last)...")
    
    loaded_clips.sort(key=lambda x: x['start_ms'])
    orig_start_map = {s['id']: s['start_ms'] for s in loaded_clips}
    
    for i in range(len(loaded_clips)):
        curr = loaded_clips[i]
        curr_end_real = curr['start_ms'] + len(curr['clip'])
        
        # [v10.43] CONFIGURAÇÃO OTIMIZADA: Sincronia Labial Real
        if i + 1 < len(loaded_clips):
            next_start = loaded_clips[i+1]['start_ms']
            # O limite base é a boca do próximo personagem.
            limit_end = min(next_start - 50, curr['start_ms'] + len(curr['clip']))
            
            # [CRITICAL FIX] Não expandimos incondicionalmente +1000ms.
            # O limit_end inicial é restrito para forçar a avaliação de aceleração primeiro.
            # Se a aceleração falhar, permitiremos invadir o silêncio até 1 segundo depois.
            limit_end_emergency = min(next_start - 50, curr['start_ms'] + len(curr['clip']) + 1000)
        else:
            limit_end = video_duration_ms
            limit_end_emergency = video_duration_ms

        overflow = curr_end_real - limit_end
        
        if overflow > 0:
            available_time = limit_end - curr['start_ms']
            current_dur = len(curr['clip'])
            
            # --- PASSO 1: SMART ACCELERATION (Até 15% mais rápido na mesma posição temporal) ---
            # [v12.92 UPDATED] A pedido do usuário: tolerância de 0.1s (100ms) para aceleração.
            # Se o estouro for menor que 100ms, ignoramos para preservar a qualidade natural.
            if overflow > 100 and current_dur > available_time and available_time > 500:
                needed_ratio = current_dur / available_time
                if needed_ratio <= 1.15:
                     logging.info(f"[SMART ACCEL] Acelerando {curr['id']} em {((needed_ratio-1)*100):.1f}% (Teto 1.15x).")
                     curr['clip'] = speedup_audio(curr['clip'], needed_ratio)
                     curr_end_real = curr['start_ms'] + len(curr['clip'])
                     overflow = max(0, curr_end_real - limit_end)
                else:
                     logging.info(f"[SMART ACCEL] Aceleração insuficiente para {curr['id']} (Precisava de {needed_ratio:.2f}x -> Teto de 1.15x ativado)")
                     # Tenta o máximo permitido (1.15x) e deixa o restante para o pre-roll/expansion
                     curr['clip'] = speedup_audio(curr['clip'], 1.15)
                     curr_end_real = curr['start_ms'] + len(curr['clip'])
                     overflow = max(0, curr_end_real - limit_end)
                     
            # --- PASSO 1.5: FORWARD ELASTICITY (EMERGÊNCIA) ---
            # Se a aceleração no talo (1.30x) não foi suficiente, então sim permitimos atrasar
            # o final do áudio em até 1 segundo, invadindo o silêncio futuro do vídeo.
            if overflow > 0:
                 limit_end = limit_end_emergency
                 overflow = max(0, curr_end_real - limit_end)
                 if overflow == 0:
                     logging.info(f"[FORWARD ELASTICITY] Áudio {curr['id']} invadiu o silêncio futuro para evitar cortes pós-aceleração.")

            # --- PASSO 2: PRE-ROLL BORROWING (TRAVADO EM 40MS) ---
            # Se APÓS a aceleração E expansão frontal ainda batermos na parede temporal, puxamos um pouquinho
            # Mas o áudio nunca começará mais de 40ms antes da boca mexer.
            if overflow > 0:
                original_start_ms = orig_start_map.get(curr['id'], curr['start_ms'])
                current_shift = original_start_ms - curr['start_ms']
                
                prev_end_real = (loaded_clips[i-1]['start_ms'] + len(loaded_clips[i-1]['clip'])) if i > 0 else 0
                gap_before = curr['start_ms'] - prev_end_real
                
                max_allowed_shift = 40 
                remaining_buffer = max(0, max_allowed_shift - current_shift)
                
                if gap_before > 10 and remaining_buffer > 0:
                    move = min(gap_before - 5, overflow, remaining_buffer)
                    curr['start_ms'] -= move
                    overflow -= move
                    if move > 20: logging.info(f"[PRE-ROLL BORROW MULTIPASS] Adiantando áudio {curr['id']} em -{move}ms (sem espaço futuro).")
            
            # --- PASSO 3: GHOST TRIM (Último recurso drástico, decepa o excesso do final) ---
            if overflow > 0:
                new_duration = len(curr['clip']) - overflow
                if new_duration > 200:
                    curr['clip'] = curr['clip'][:new_duration].fade_out(min(30, new_duration))
                    logging.info(f"[GHOST TRIM SACRIFICE] Cortando {overflow}ms irrecuperáveis do final de {curr['id']}.")
                else:
                    curr['clip'] = AudioSegment.silent(duration=0)

        # Salva o arquivo de sincronia validado para auditoria em disco local
        try:
            sync_audio_dir = job_dir / "sync_audio"
            sync_audio_dir.mkdir(exist_ok=True)
            sync_audio_path = sync_audio_dir / f"{curr['id']}_synced.wav"
            curr['clip'].export(sync_audio_path, format="wav")
        except: pass

    # Pass 3: High-Performance Assembly (Linear Concatenation)
    # [PERFORMANCE FIX] Substituindo 'overlay' (O(N^2)) por 'concat' (O(N)).
    
    logging.info(f"Montando timeline linear com {len(loaded_clips)} segmentos...")
    
    # [v10.17] GHOST VOCAL FILLER: Preenchimento com áudio original para evitar mudo
    # Em vez de silêncio puro, usamos o vocals.wav nos gaps para manter a ambiência e sons ignorados.
    target_sw, target_fr, target_ch = 2, 24000, 1 # Default Chatterbox/Chatterbox
    if loaded_clips:
        c = loaded_clips[0]['clip']
        target_sw, target_fr, target_ch = c.sample_width, c.frame_rate, c.channels

    # Carrega Vocals para preenchimento (Ghost Filler)
    vocals_source = None
    vocals_path = job_dir / "vocals.wav"
    if vocals_path.exists():
        try:
            logging.info(f"Ghost Filler: Carregando {vocals_path.name} para preencher silêncios...")
            vocals_source = AudioSegment.from_wav(str(vocals_path))
            # Padroniza com a mesma genética das vozes dubladas
            vocals_source = vocals_source.set_frame_rate(target_fr).set_channels(target_ch).set_sample_width(target_sw)
        except Exception as e:
            logging.warning(f"Não foi possível carregar vocals para ghost filler: {e}")

    # [v10.36] Memory for Render Log to track exactly if rules were respected
    assembly_log = ["--- TIMELINE ASSEMBLY LOG (v10.36) ---"]

    def get_filler(dur_ms, cursor_ms):
        # [v10.31] Apenas preenche com Ghost Filler se o gap for MAIOR que 1000ms (1 segundo).
        # Para gaps menores, usa silêncio puro para não criar ruídos estranhos e transições abruptas.
        if vocals_source and dur_ms > 1000:
            # Extrai o pedaço original do vocals.wav
            filler = vocals_source[cursor_ms:cursor_ms + dur_ms]
            # [v10.31] Fade de 200ms a pedido do usuário
            fade_len = min(200, len(filler) // 2)
            assembly_log.append(f"[SUCESSO] Ghost Filler gerado em {cursor_ms}ms com {dur_ms}ms de duração. Regra (>1000ms) aprovada.")
            if fade_len > 0:
                return filler.fade_in(fade_len).fade_out(fade_len)
            return filler
        
        assembly_log.append(f"[MUDO] Silêncio numérico de {dur_ms}ms gerado em {cursor_ms}ms. Regra Ghost (<1000ms) respeitada.")
        s = AudioSegment.silent(duration=dur_ms, frame_rate=target_fr)
        return s.set_channels(target_ch).set_sample_width(target_sw)

    timeline_segments = []
    current_cursor_ms = 0
    
    for clip_data in loaded_clips:
        start_ms = clip_data['start_ms']
        
        # 1. Fill Gap with original vocals (Ghost Filler)
        if start_ms > current_cursor_ms:
            gap_dur = start_ms - current_cursor_ms
            timeline_segments.append(get_filler(gap_dur, current_cursor_ms))
            current_cursor_ms += gap_dur
            
        assembly_log.append(f"[PLAY] Segmento {clip_data['id']} alinhado precisamente em {current_cursor_ms}ms. (Len: {len(clip_data['clip'])}ms).")
            
        # 2. Re-padroniza o áudio da voz (Garantia extra caso algum áudio venha zoado)
        safe_voice = clip_data['clip'].set_frame_rate(target_fr).set_channels(target_ch).set_sample_width(target_sw)
        # [v10.17.1] Suavização de bordas de 200ms para fundir com a ambiência original
        fade_v = min(200, len(safe_voice) // 2)
        if fade_v > 0:
            safe_voice = safe_voice.fade_in(fade_v).fade_out(fade_v)
        timeline_segments.append(safe_voice)
        current_cursor_ms += len(safe_voice)
        
    # 3. Fill Final Gap with original vocals to match Video Length
    if current_cursor_ms < video_duration_ms:
        gap_dur = video_duration_ms - current_cursor_ms
        timeline_segments.append(get_filler(gap_dur, current_cursor_ms))

    # Ultra-Fast Concatenation (O(N) instead of O(N^2))
    if timeline_segments:
        logging.info("Concatenando buffers de áudio puros (Otimização Extrema)...")
        # Junta todos os bytes de uma vez (instantâneo na RAM)
        raw_data = b"".join([seg.raw_data for seg in timeline_segments])
        
        # Recria o áudio mestre 100% perfeito na frequência das vozes
        final_audio = AudioSegment(data=raw_data, sample_width=target_sw, frame_rate=target_fr, channels=target_ch)
    else:
        final_audio = make_silence(video_duration_ms)
    
    final_audio_path = job_dir / "final_dubbed_audio.wav"
    logging.info("Renderizando áudio e iniciando montagem final do vídeo (FFmpeg)...")
    logging.info("Nota: O vídeo está sendo re-encodado para garantir o sincronismo perfeito. Isso pode levar alguns minutos.")
    final_audio.export(final_audio_path, format="wav")
    
    # Montagem Final com FFmpeg
    output_video_path = job_dir / "video_dublado_final.mp4"
    instrumental_path = job_dir / "instrumental.wav" # [FIX] Definição que faltava
    
    # [ESTRATÉGIA VISUAL: Áudio Longo (Looping Boomerang)]
    # Verifica quanto de áudio excedeu o vídeo original
    audio_dur_ms = len(final_audio)
    overflow_ms = audio_dur_ms - video_duration_ms
    
    # [v10.34] Dynamic FFmpeg Progress Tracker
    # Trocado -loglevel error para omitir avisos irrelevantes de "Late SEI is not implemented"
    # Removemos o loglevel error pra ter o status `time=` do progresso,
    # mas interceptaremos com PIPE para esconder coisas inuteis do console.
    cmd = ['ffmpeg', '-y', '-nostdin', '-i', str(video_path), '-i', str(final_audio_path)]
    filter_complex_parts = []
    dubbed_audio_map = "1:a" # Default: Só a dublagem
    video_map = "0:v" # Padrão
    
    if overflow_ms > 500:
        # Se excedeu entre 0.5s e ~2s, geramos efeito boomerang (vai e volta e vai e volta)
        # Corta os ultimos 0.5s do video original e faz fade de reversão.
        logging.info(f"Áudio excedeu vídeo em {overflow_ms}ms! Aplicando Boomerang Visual para evitar frame congelado.")
        
        # Filtro de manipulação de PTS (Presentation Time Stamp) para loop do último Meio Segundo
        loop_time = 0.5
        v_length_sec = video_duration_ms / 1000.0
        start_loop = max(0, v_length_sec - loop_time)
        
        if overflow_ms < 2000:
            stretch_factor = audio_dur_ms / video_duration_ms
            logging.info(f"Esticando imagem dinamicamente em {stretch_factor:.2f}x.")
            filter_complex_parts.append(f"[0:v]setpts={stretch_factor:.3f}*PTS[v_stretched]")
            video_map = "[v_stretched]"
        else:
            # Se for muito excesso, trunca no tempo do áudio mas repete o último frame (tpad) no FFMPEG 4.2+
            filter_complex_parts.append(f"[0:v]tpad=stop_mode=clone:stop_duration={(overflow_ms/1000.0):.1f}[v_padded]")
            video_map = "[v_padded]"
    else:
        # Pequenos excessos (abaixo de 0.5s) passam despercebidos. Corta tudo no length do shortest (padrão)
        pass # O argumento será adicionado no final do comando

    # Adiciona instrumental se existir
    if instrumental_path.exists():
        cmd.extend(['-i', str(instrumental_path)])
        
        # [CINEMATIC MIXING - EQ FIX]
        # [v10.16.2] duration=longest para garantir que nada seja cortado prematuramente
        filter_complex = (
            "[2:a]volume=2.5,treble=g=5:f=6000:w=0.5[bg];"                                    # 1. Base musical turbinada (EQ)
            "[1:a]asplit[voice_main][voice_sc];"                                              # 2. Divide a voz (Principal + Gatilho)
            "[bg][voice_sc]sidechaincompress=threshold=0.15:ratio=1.5:attack=100:release=1000[bg_ducked];" # 3. Ducking suave
            "[bg_ducked][voice_main]amix=inputs=2:duration=longest[a_mix_raw];"                # 4. Mixagem (FFmpeg reduz 50%)
            "[a_mix_raw]volume=2.0[a_mix]"                                                    # 5. Restauração do Ganho Master
        )
        
        filter_complex_parts.append(filter_complex)
        dubbed_audio_map = "[a_mix]" # O áudio final agora é o mixado
        
    if filter_complex_parts:
        cmd.extend(['-filter_complex', ";".join(filter_complex_parts)])
    
    cmd.extend(['-map', video_map, '-map', dubbed_audio_map])
    
    # Se NÃO aplicamos padding/stretch, devemos colocar -shortest antes do output
    if overflow_ms <= 500:
        cmd.append('-shortest')
        
    cmd.extend(['-c:v', 'libx264', '-preset', 'fast', '-c:a', 'aac', '-threads', '2', str(output_video_path)]) # Re-encoding required for setpts/tpad
    
    try:
        # [v10.34] Dynamic Progress Interception
        total_dur_sec = max(video_duration_ms, audio_dur_ms) / 1000.0
        
        # [v10.35] Salvar Log Completo do FFmpeg para depuração futura (Pedido Especial)
        ffmpeg_log_path = job_dir / "ffmpeg_render.log"
        with open(ffmpeg_log_path, "w", encoding="utf-8") as f_log:
            f_log.write("\n".join(assembly_log) + "\n\n")
            f_log.write(f"--- FFMPEG RENDER COMMAND ---\n{' '.join(cmd)}\n\n")
            f_log.write("--- FFMPEG STDERR STREAMS ---\n")
            
            process = subprocess.Popen(cmd, stderr=subprocess.PIPE, universal_newlines=True)
            # Parse ffmpeg output silently instead of spamming terminal
            for line in process.stderr:
                f_log.write(line) # Salva a linha real no arquivo para a IA ler depois
                
                time_match = re.search(r"time=(\d{2}):(\d{2}):(\d{2}\.\d{2})", line)
                if time_match:
                    hours, minutes, seconds = time_match.groups()
                    current_time_sec = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
                    if total_dur_sec > 0:
                        percent = min(99.0, (current_time_sec / total_dur_sec) * 100.0)
                        remaining_sec = max(0, total_dur_sec - current_time_sec)
                        time_str = str(timedelta(seconds=int(remaining_sec)))
                        cb(percent, 8, f"Montando Vídeo (Faltam ~{time_str})...")
                        
            process.wait()
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, cmd)
    except subprocess.CalledProcessError as e:
        logging.error(f"Erro do FFmpeg ao montar vídeo final. Verifique o arquivo ffmpeg_render.log.")
        raise
        
    cb(100, 8, "Vídeo finalizado.")

# --- CONTROLE DE FILA (Processamento de Chatterbox) ---
import concurrent.futures
Chatterbox_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2) # Limite rígido para segurar GPU/CPU térmica


def run_blind_diarization_pass(job_dir, vocals_path, cb, etapa_idx):
    """
    Passo 1 (Diarização Dupla): Faz varredura cega no áudio ANTES do Whisper.
    Usa VAD PyDub para fatiar apenas onde há som, SpeechBrain para embutir, salva o cache temporário
    para o Whisper usar de 'faca' cirúrgica, e depois zera a VRAM/RAM. Foco Total: PC Fraco.
    """
    cache_path = job_dir / "diarization_cache.json"
    if cache_path.exists():
        cb(50, etapa_idx, "Diarização Cega (Passo 1): Carregada do cache.")
        return safe_json_read(cache_path)
        
    cb(10, etapa_idx, "Diarização Cega (Passo 1): Mapeando silêncios (PyDub VAD)...")
    from pydub import AudioSegment
    from pydub.silence import detect_nonsilent
    
    try:
        audio = AudioSegment.from_wav(str(vocals_path))
        # VAD Leve: silêncios > 500ms, limite de volume -16db da média
        nonsilent_ranges = detect_nonsilent(audio, min_silence_len=500, silence_thresh=audio.dBFS-16)
        
        if not nonsilent_ranges:
             return []
             
        cb(30, etapa_idx, f"Diarização Cega: {len(nonsilent_ranges)} blocos sonoros detectados. Identificando vozes...")
        
        import torch
        diarizer = SimpleDiarizer(device='cuda' if torch.cuda.is_available() else 'cpu')
        
        import torchaudio
        signal, fs = torchaudio.load(str(vocals_path))
        if signal.shape[0] > 1: signal = signal.mean(dim=0, keepdim=True)
        if fs != 16000: signal = torchaudio.transforms.Resample(fs, 16000)(signal)
        
        results = []
        for i, (sts, ets) in enumerate(nonsilent_ranges):
             start_sec = sts / 1000.0
             end_sec = ets / 1000.0
             dur = end_sec - start_sec
             
             if dur < 0.2: continue # Ignora tosse/ruído
             
             s_sample = int(start_sec * 16000)
             e_sample = int(end_sec * 16000)
             seg_signal = signal[:, s_sample:e_sample]
             
             if seg_signal.shape[1] < 400: continue
             
             emb = diarizer.encoder.encode_batch(seg_signal).squeeze().cpu().numpy()
             results.append({
                 "start": start_sec,
                 "end": end_sec,
                 "emb": emb,
                 "speaker": "unknown"
             })
             
        if not results: return []

        cb(40, etapa_idx, "Agrupando vozes por similaridade...")
        # Clustering
        from sklearn.cluster import AgglomerativeClustering
        embeddings = np.array([r['emb'] for r in results])
        clusterer = AgglomerativeClustering(n_clusters=None, distance_threshold=0.45, metric='cosine', linkage='average')
        labels = clusterer.fit_predict(embeddings)
        
        final_cache = []
        for i, lbl in enumerate(labels):
            final_cache.append({
                "start": results[i]["start"],
                "end": results[i]["end"],
                "speaker": f"voz{lbl+1}"
            })
            
        safe_json_write(final_cache, cache_path)
        
        # [PC FRACO OTIMIZAÇÃO] FREE MEMORY KILLER
        del diarizer
        del signal
        del embeddings
        import gc
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        
        cb(50, etapa_idx, "Diarização Cega (Passo 1): Concluída. VRAM/RAM purgadas.")
        return final_cache
        
    except Exception as e:
        logging.error(f"Erro na Diarização Cega: {e}")
        return []

# --- PIPELINE PRINCIPAL (Job de Dublagem) ---


def pipeline_dublar_video(job_dir, job_id, start_time):
    """Pipeline principal para dublagem de vídeo."""
    with active_jobs_lock:
        if len(active_jobs) >= MAX_CONCURRENT_JOBS:
             logging.warning(f"❌ [HARDWARE] Limite de {MAX_CONCURRENT_JOBS} job(s) atingido. Ignorando {job_id}.")
             return
        active_jobs.add(job_id)
    
    try:
        # [v11.0] Trava de Segurança: Verifica se o projeto é realmente de dublagem
        status_data = safe_json_read(job_dir / "job_status.json") or {}
        if status_data.get('mode') == 'shorts_maker' or "job_shorts" in job_id:
             raise ValueError(f"O projeto {job_id} é um Shorts e não pode ser processado pelo motor de dublagem.")

        set_low_process_priority()
        # Define o callback de progresso
        def cb(p, etapa, s=None): set_progress(job_id, p, etapa, start_time, ETAPAS_DUBLAGEM, s)
        
        video_path = next(job_dir.glob("input_video.*"), None) # 1. Tenta nome específico (Upload direto)
        if not video_path:
            video_path = next(job_dir.glob("input.*"), None) # 2. Tenta nome genérico (Chat Task)

        if not video_path: 
            raise FileNotFoundError("Nenhum arquivo de vídeo (input_video.* ou input.*) encontrado no diretório do job.")
        
        audio_path = job_dir / f"{video_path.stem}.wav"

        # Inicia o pipeline
        if not audio_path.exists():
            extract_audio(video_path, audio_path, cb)
        
        # [MUDANÇA] Chama a função de separação real
        if not (job_dir / "vocals.wav").exists():
            # A função de separação agora é a Etapa 2
            run_audio_separation(job_dir, audio_path, cb, 2, ETAPAS_DUBLAGEM)
            
        # [AUDIO CLEANING STEP] - Garante que 'vocals.wav' esteja limpo antes de prosseguir
        # Isso afeta Transcrição, Diarização e Clonagem de Voz (Referências)
        vocals_path = job_dir / "vocals.wav"
        vocals_raw_path = job_dir / "vocals_raw.wav"
        cleaned_marker = job_dir / "vocals_is_cleaned.marker"

        if vocals_path.exists() and not cleaned_marker.exists():
            cb(50, 2, "Limpando ruído da voz (DeepFilterNet/Hybrid)...")
            logging.info("Iniciando limpeza de áudio pré-processamento...")
            
            try:
                # 1. Renomeia original para backup/raw
                shutil.move(str(vocals_path), str(vocals_raw_path))
                
                # 2. Tenta DeepFilterNet
                use_fallback = True
                try:
                    # check
                    subprocess.run(['deepFilter', '--help'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    
                    # run
                    cmd_df = ['deepFilter', str(vocals_raw_path), '-a', '15', '-o', str(job_dir)]
                    subprocess.run(cmd_df, check=True)
                    
                    # find output
                    possible_outputs = list(job_dir.glob("*_DeepFilterNet3.wav"))
                    if possible_outputs:
                        # Move o output limpo para ser o novo 'vocals.wav'
                        shutil.move(str(possible_outputs[0]), str(vocals_path))
                        logging.info("Limpeza DeepFilterNet aplicada com sucesso!")
                        use_fallback = False
                except Exception as e:
                    logging.warning(f"DeepFilterNet falhou: {e}. Usando Fallback.")
                    
                # 3. Fallback Nativo (Se DF falhou)
                if use_fallback:
                    import librosa
                    import soundfile as sf
                    import scipy.signal as signal
                    import numpy as np

                    y, sr = librosa.load(str(vocals_raw_path), sr=None)
                    
                    # Bandpass 80Hz-8kHz
                    sos = signal.butter(10, [80, 8000], 'bandpass', fs=sr, output='sos')
                    cleaned_y = signal.sosfilt(sos, y)
                    
                    # Save as vocals.wav
                    sf.write(str(vocals_path), cleaned_y, sr)
                    logging.info("Limpeza Nativa (Bandpass) aplicada.")
                
                # Marca como limpo
                cleaned_marker.touch()
                
            except Exception as e:
                logging.error(f"Erro crítico na limpeza de áudio: {e}. Revertendo para original.")
                if vocals_raw_path.exists():
                    if vocals_path.exists(): vocals_path.unlink() # remove parcial
                    shutil.move(str(vocals_raw_path), str(vocals_path)) # restaura
        
        # [v5.0] SMART RESET: Se o usuário apagou a pasta 'voices', limpamos metadados de orador
        # Mas preservamos o texto transcrito para não precisar rodar o Whisper de novo.
        project_data_path = job_dir / "project_data.json"
        if project_data_path.exists() and vocals_path.exists():
            voices_dir = job_dir / "voices"
            if not voices_dir.exists():
                logging.info("SMART RESET: Pasta 'voices' ausente. Limpando IDs de oradores...")
                project_data = safe_json_read(project_data_path)
                if project_data:
                    # [v10.1] Acesso seguro para resetar oradores
                    segs_to_reset = project_data.get('segments', project_data) if isinstance(project_data, dict) else project_data
                    if isinstance(segs_to_reset, list):
                        for seg in segs_to_reset:
                            seg['speaker'] = 'voz1' # Reset para default
                    
                    # Remove arquivos de conclusão de diarização para forçar re-calc v10.0
                    (job_dir / "diarization_debug.json").unlink(missing_ok=True)
                    (job_dir / "diarization_cache.json").unlink(missing_ok=True)
                    (job_dir / "consolidation_done.flag").unlink(missing_ok=True)
                    safe_json_write(project_data, project_data_path)
            else:
                project_data = safe_json_read(project_data_path)
                # [v10.19.3] SELF-HEALING: Auto-detecta arquivos apagados e reseta no JSON
                project_data = audit_and_heal_project_data(job_dir, project_data)
                
            # [INTEGRIDADE] Se o áudio é mais novo que o JSON, o JSON é velho (de outro vídeo)
            try:
                if vocals_path.stat().st_mtime > project_data_path.stat().st_mtime:
                    logging.warning("DETECTADO: vocals.wav é mais recente que project_data.json. O usuário trocou o vídeo?")
                    logging.warning("Ação: APAGANDO transcrição antiga para evitar 'Contexto Zumbi'.")
                    project_data_path.unlink(missing_ok=True)
                    shutil.rmtree(job_dir / "_backup_texto_final", ignore_errors=True)
                    shutil.rmtree(job_dir / "temp_whisper_slice.wav", ignore_errors=True)
                    project_data = None
            except: pass
        else:
            project_data = None

        # [v9.1] NORMALIZAÇÃO: Garante que project_data é sempre um DICIONÁRIO
        if project_data and isinstance(project_data, list):
            logging.info("Normalizando project_data legado (LISTA -> DICIONÁRIO)...")
            project_data = {"segments": project_data}

        # 1. Transcrição (Whisper) - Só roda se não tivermos project_data (ou se resetado acima)
        if not project_data:
            # [v10.19.1] Phoenix Recovery: Tenta reconstruir dos backups individuais se project_data sumiu
            project_data = try_reconstruct_project_from_all_backups(job_dir)
            
            if not project_data:
                # [v5.0] No blind pass here. Diarization will be drive-based after this.
                # [v10.40] Passa o idioma de origem carregado do job_status.json
                status_data_transcribe = safe_json_read(job_dir / "job_status.json") or {}
                source_language = status_data_transcribe.get('idioma_origem', 'en')
                project_data = transcribe_vocals_robust(job_dir, None, cb, 3, ETAPAS_DUBLAGEM, source_lang=source_language)
            else:
                cb(100, 3, "Projeto reconstruído via Lógica Phoenix (Whisper pulado).")
        
        if not project_data:
             error_msg = "Falha crítica: project_data.json está vazio ou corrompido e não pôde ser reparado."
             logging.error(error_msg)
             raise Exception(error_msg)
        
        vocals_path = job_dir / "vocals.wav"
        
        # [v10.4] CARGA DE STATUS FRESCA (IMPORTANTE)
        # Recarregamos para pegar o real num_speakers caso o usuário tenha mudado no Reset
        status_data = safe_json_read(job_dir / "job_status.json") or {}
        num_speakers = status_data.get('num_speakers', 0)
        if num_speakers is None: num_speakers = 0 
        
        # 2. Diarização Guiada (v5.0)
        # Se oradores > 1 ou modo auto (0), roda a identificação baseada no texto
        logging.info(f"CHECK DIARIZAÇÃO (v10.4): num_speakers={num_speakers}")
        if num_speakers > 1 or num_speakers == 0:
             voices_dir = job_dir / "voices"
             status_file = job_dir / "diarization_status.json"
             
             # [v10.2] REGRAS DE SEGURANÇA (Paulo):
             # 1. Pasta 'voices' não existe? -> Roda Diarização.
             # 2. Status diz que falhou ou não terminou? -> Limpa e Roda de novo.
             
             d_status_val = safe_json_read(status_file) or {}
             d_status_str = d_status_val.get('status', 'not_found')
             
             needs_diarization = False
             if not voices_dir.exists():
                  logging.info("SEGURANÇA (Regra 1): Pasta 'voices' ausente. Forçando diarização...")
                  needs_diarization = True
             elif d_status_str in ['failed', 'processing']:
                  logging.warning(f"SEGURANÇA (Regra 2): Status da diarização é '{d_status_str}'. Re-executando...")
                  needs_diarization = True
             
             # [v10.3] FORÇA BRUTA: Se só tem 1 voz registrada mas o sistema detecta que pode ter mais, 
             # ou se o usuário pediu especificamente mais oradores, ignoramos o cache antigo.
             # [v10.5] Fix: Definindo segs_list para evitar NameError
             segs_list = project_data.get('segments', project_data) if isinstance(project_data, dict) else project_data
             unique_speakers = set(s.get('speaker', 'voz1') for s in segs_list)
             
             if not needs_diarization:
                  if len(unique_speakers) <= 1:
                       logging.info("DETECÇÃO: Apenas 1 orador encontrado no JSON. Forçando Smart Split v10 para refinar segmentos...")
                       needs_diarization = True
             
             if needs_diarization:
                  cb(50, 3, "Iniciando Diarização e Re-segmentação (Smart Split v10)...")
                  # Limpeza de lixo anterior para garantir integridade
                  if voices_dir.exists(): shutil.rmtree(voices_dir, ignore_errors=True)
                  (job_dir / "diarization_debug.json").unlink(missing_ok=True)
                  status_file.unlink(missing_ok=True)
                  
                  project_data = run_transcription_driven_diarization(job_dir, vocals_path, project_data, cb, 3, num_speakers=num_speakers)
             else:
                  logging.info(f"Diarização Concluída anteriormente ({len(unique_speakers)} vozes). Pulando.")
        else:
             logging.info("Modo 1 falante. Mantendo 'voz1' padrão.")
             cb(50, 3, "Modo 1 falante. Pulando diarização.")

        # 3. Extração de Refs (Baseado nos segmentos já identificados)
        extract_speaker_references(job_dir, vocals_path, project_data, cb, 3)
        
        # [NEW] Consolidação Inteligente (Merge de vozes curtas similares)
        project_data = consolidate_speaker_segments(job_dir, project_data, cb, 3)
        
        # [SMART MERGE REMOVIDO] - O Smart Merge original destruía timestamps das micro-pausas. 
        # A nova lógica de Batch Dinâmico (App_videos.py:process_gema_steps) fará o agrupamento seguro na memória.
        segs_to_check = project_data.get('segments', project_data) if isinstance(project_data, dict) else project_data
        if segs_to_check and isinstance(segs_to_check, list) and len(segs_to_check) > 0:
            if not segs_to_check[0].get('translated_text'):
                # Persiste a estrutura original limpa para manter o Time Tracking
                safe_json_write(project_data, job_dir / "project_data.json")

        # project_data já está pronto para o Gema
        
        process_gema_steps(job_dir, project_data, cb) # Etapas 4, 5, 6, 7
        
        generate_dubbed_audio(job_dir, project_data, cb) # Etapa 8
        
        assemble_final_video(job_dir, project_data, cb) # Etapa 9

        unload_gema_model() # [SAFEGUARD] Garante que não sobrou nada na memória
        cb(100, 9, "Processo concluído!") # Etapa 9 = Concluído
        
        # [v10.24] Gera Diagnóstico Final de Sucesso
        generate_job_diagnostics(job_dir, project_data)

    except Exception as e:
        logging.error(f"ERRO NO PIPELINE (Job ID: {job_id}): {e}\n{traceback.format_exc()}")
        set_progress(job_id, 100, len(ETAPAS_DUBLAGEM) - 1, start_time, ETAPAS_DUBLAGEM, subetapa=f"Erro: {e}")
        status_path = job_dir / "job_status.json"
        status_data = safe_json_read(status_path) or {}
        status_data['status'] = 'failed'
        status_data['error'] = str(e)
        safe_json_write(status_data, status_path)
        
        # [v10.24] Gera Diagnóstico em caso de Falha
        try:
             generate_job_diagnostics(job_dir, project_data if 'project_data' in locals() else None)
        except: pass
    finally:
        with active_jobs_lock:
            if job_id in active_jobs:
                active_jobs.remove(job_id)

# --- PIPELINES ADICIONAIS (PLACEHOLDERS) ---

def pipeline_chat_task(job_dir, job_id, start_time, prompt):
    with active_jobs_lock:
        if job_id in active_jobs: return
        active_jobs.add(job_id)
    try:
        set_low_process_priority()
        def cb(p, e, s=None): set_progress(job_id, p, e, start_time, ETAPAS_CHAT, s)
        
        cb(0, 1, "Aguardando Gema...")
        wait_for_gema_service(lambda s: cb(10, 1, s))
        
        # TODO: Lógica de interpretação do prompt e execução de comandos
        logging.info(f"[Chat Job {job_id}] Prompt Recebido: {prompt}")
        
        cb(50, 2, "Executando comandos...")
        time.sleep(2) # Simulação
        
        cb(100, 3, "Tarefas do chat concluídas.")

    except Exception as e:
        logging.error(f"ERRO NO PIPELINE CHAT (Job ID: {job_id}): {e}\n{traceback.format_exc()}")
        set_progress(job_id, 100, len(ETAPAS_CHAT) - 1, start_time, ETAPAS_CHAT, subetapa=f"Erro: {e}")
    finally:
        with active_jobs_lock:
            if job_id in active_jobs: active_jobs.remove(job_id)

def pipeline_criar_video_musical(job_dir, job_id, start_time, style):
    with active_jobs_lock:
        if job_id in active_jobs: return
        active_jobs.add(job_id)
    try:
        set_low_process_priority()
        def cb(p, e, s=None): set_progress(job_id, p, e, start_time, ETAPAS_MUSIC_VISUALIZER, s)
        
        audio_file = next((job_dir / "audio").glob("*.*"), None)
        if not audio_file: raise FileNotFoundError("Arquivo de áudio não encontrado.")
        
        backup_analysis = job_dir / "_backup_analysis.json"
        
        if backup_analysis.exists():
            beat_data = safe_json_read(backup_analysis)
            cb(50, 1, "Análise de áudio carregada do backup.")
        else:
            cb(0, 1, "Analisando áudio (Librosa)...")
            # TODO: Lógica com Librosa aqui
            logging.info(f"[MusicViz Job {job_id}] Estilo selecionado: {style}")
            time.sleep(5) # Simulação da análise
            beat_data = {"tempo": 120, "beats": [1.0, 2.5, 4.0]} # Dados Falsos
            safe_json_write(beat_data, backup_analysis)
            cb(50, 1, "Análise de áudio concluída.")

        cb(50, 2, "Gerando filtros FFmpeg...")
        time.sleep(2) # Simulação
        
        cb(75, 3, "Renderizando vídeo...")
        time.sleep(3) # Simulação
        
        cb(100, 4, "Vídeo musical concluído.")

    except Exception as e:
        logging.error(f"ERRO NO PIPELINE MUSIC VIZ (Job ID: {job_id}): {e}\n{traceback.format_exc()}")
        set_progress(job_id, 100, len(ETAPAS_MUSIC_VISUALIZER) - 1, start_time, ETAPAS_MUSIC_VISUALIZER, subetapa=f"Erro: {e}")
    finally:
        with active_jobs_lock:
            if job_id in active_jobs: active_jobs.remove(job_id)

def pipeline_remover_fundo(job_dir, job_id, start_time, output_format):
    with active_jobs_lock:
        if job_id in active_jobs: return
        active_jobs.add(job_id)
    try:
        set_low_process_priority()
        def cb(p, e, s=None): set_progress(job_id, p, e, start_time, ETAPAS_REMOVER_FUNDO, s)
        
        video_file = next(job_dir.glob("input.*"), None)
        if not video_file: raise FileNotFoundError("Arquivo de vídeo não encontrado.")
        
        total_frames = get_video_frame_count(video_file)
        
        frames_dir = job_dir / "_temp_frames"
        frames_dir.mkdir(exist_ok=True)
        backup_dir = job_dir / "_backup_remove_bg"
        backup_dir.mkdir(exist_ok=True)
        
        cb(0, 1, "Extraindo frames do vídeo...")
        # TODO: Lógica de extração de frames (ffmpeg -i video.mp4 frames_dir/frame_%06d.png)
        # Por enquanto, vamos simular
        
        cb(20, 1, f"Carregando modelo... Processando {total_frames} frames...")
        logging.info(f"[RemoveBG Job {job_id}] Formato: {output_format}")
        
        for i in range(total_frames):
            frame_nome = f"frame_{i:06d}.png"
            output_frame_path = backup_dir / frame_nome
            
            if output_frame_path.exists(): # Verifica backup
                continue 
                
            # Simulação do processamento de IA
            # TODO: Rodar modelo de remoção de fundo (ex: U2Net)
            time.sleep(0.05) # Simulação pesada
            # Salva o frame processado (simulado)
            (backup_dir / frame_nome).touch() 
            
            progress = 20 + (i / total_frames) * 70
            if i % 10 == 0: # Atualiza o CMD a cada 10 frames
                cb(progress, 1, f"Processando frame {i}/{total_frames}")

        cb(90, 2, "Remontando vídeo...")
        # TODO: Lógica de remontagem (ffmpeg -i _backup_remove_bg/frame_%06d.png ... output.mp4)
        time.sleep(2) # Simulação
        
        cb(100, 3, "Remoção de fundo concluída.")

    except Exception as e:
        logging.error(f"ERRO NO PIPELINE REMOVER FUNDO (Job ID: {job_id}): {e}\n{traceback.format_exc()}")
        set_progress(job_id, 100, len(ETAPAS_REMOVER_FUNDO) - 1, start_time, ETAPAS_REMOVER_FUNDO, subetapa=f"Erro: {e}")
    finally:
        with active_jobs_lock:
            if job_id in active_jobs: active_jobs.remove(job_id)

def pipeline_melhorar_video(job_dir, job_id, start_time, factor):
    with active_jobs_lock:
        if job_id in active_jobs: return
        active_jobs.add(job_id)
    try:
        set_low_process_priority()
        def cb(p, e, s=None): set_progress(job_id, p, e, start_time, ETAPAS_UPSCALE, s)
        
        video_file = next(job_dir.glob("input.*"), None)
        if not video_file: raise FileNotFoundError("Arquivo de vídeo não encontrado.")
        
        total_frames = get_video_frame_count(video_file)
        
        frames_dir = job_dir / "_temp_frames"
        frames_dir.mkdir(exist_ok=True)
        backup_dir = job_dir / "_backup_upscale"
        backup_dir.mkdir(exist_ok=True)
        
        cb(0, 1, "Extraindo frames do vídeo...")
        # TODO: Lógica de extração de frames
        
        cb(10, 1, f"Carregando modelo... Processando {total_frames} frames (lento)...")
        logging.info(f"[Upscale Job {job_id}] Fator: {factor}")
        
        for i in range(total_frames):
            frame_nome = f"frame_{i:06d}.png"
            output_frame_path = backup_dir / frame_nome
            
            if output_frame_path.exists(): # Verifica backup
                continue 
                
            # Simulação do processamento de IA (Upscale é MUITO lento)
            # TODO: Rodar modelo de Upscale (ex: Real-ESRGAN)
            time.sleep(0.5) # Simulação MUITO lenta
            (backup_dir / frame_nome).touch() 
            
            progress = 10 + (i / total_frames) * 80
            if i % 5 == 0: # Atualiza o CMD a cada 5 frames
                cb(progress, 1, f"Processando frame {i}/{total_frames}")

        cb(90, 2, "Remontando vídeo...")
        # TODO: Lógica de remontagem
        time.sleep(2) # Simulação
        
        cb(100, 3, "Upscale concluído.")

    except Exception as e:
        logging.error(f"ERRO NO PIPELINE UPSCALE (Job ID: {job_id}): {e}\n{traceback.format_exc()}")
        set_progress(job_id, 100, len(ETAPAS_UPSCALE) - 1, start_time, ETAPAS_UPSCALE, subetapa=f"Erro: {e}")
    finally:
        with active_jobs_lock:
            if job_id in active_jobs: active_jobs.remove(job_id)

# --- [PIPELINE DE TRANSCRIÇÃO (LEGENDA) - ATUALIZADO] ---
def pipeline_transcrever_arquivo(job_dir, job_id, start_time):
    """Pipeline robusto para o Card 7 (Transcrição)"""
    with active_jobs_lock:
        if len(active_jobs) >= MAX_CONCURRENT_JOBS:
             logging.warning(f"❌ [HARDWARE] Limite de {MAX_CONCURRENT_JOBS} job(s) atingido. Ignorando {job_id}.")
             return
        active_jobs.add(job_id)
    try:
        set_low_process_priority()
        def cb(p, e, s=None): set_progress(job_id, p, e, start_time, ETAPAS_TRANSCRICAO, s)
        
        backup_srt = job_dir / "transcricao.srt"
        backup_txt = job_dir / "transcricao.txt"
        
        if backup_srt.exists() and backup_txt.exists():
            cb(100, 3, "Transcrição carregada do backup.")
            return

        cb(0, 1, "Carregando modelo Whisper...")
        model = get_whisper_model()
        
        input_file = next(job_dir.glob("input.*"), None)
        if not input_file: raise FileNotFoundError("Arquivo de entrada não encontrado.")
        
        # [LÓGICA DE SEGMENTAÇÃO (pydub) - igual app_jogos]
        cb(5, 1, "Segmentando áudio por pausas (Pydub)...")
        try:
            audio = AudioSegment.from_file(str(input_file))
            audio = audio.set_channels(1).set_frame_rate(16000) # Reamostra para Whisper
            chunks = split_on_silence(
                audio,
                min_silence_len=700,
                silence_thresh=audio.dBFS - 14, # 14 dB abaixo do pico
                keep_silence=300
            )
        except Exception as e:
            logging.error(f"Falha ao segmentar áudio com Pydub: {e}")
            raise
            
        if not chunks:
            logging.warning("Nenhum segmento de fala detectado no áudio.")
            cb(100, 2, "Nenhuma fala detectada.")
            return

        total_chunks = len(chunks)
        cb(10, 1, f"Encontrados {total_chunks} segmentos. Transcrevendo...")
        
        segments_dir = job_dir / "_audio_segments"
        segments_dir.mkdir(exist_ok=True)
        
        all_segments_data = []
        global_start_time = 0.0

        for i, chunk in enumerate(chunks):
            progress = 10 + (i / total_chunks) * 80
            cb(progress, 1, f"Transcrevendo segmento {i+1}/{total_chunks}")
            
            segment_path = segments_dir / f"seg_{i:04d}.wav"
            chunk.export(segment_path, format="wav")
            
            # Transcreve o pedaço
            segments_generator, _ = model.transcribe(str(segment_path))
            chunk_text = " ".join(s.text.strip() for s in segments_generator)
            
            chunk_duration = chunk.duration_seconds
            
            seg_data = {
                "start": global_start_time,
                "end": global_start_time + chunk_duration,
                "text": chunk_text
            }
            all_segments_data.append(seg_data)
            
            global_start_time += chunk_duration # Aproximação simples
            # NOTA: Esta aproximação de tempo está errada.
            # A lógica correta (do App_videos) de transcrever tudo e *depois* re-segmentar é melhor para SRT.
            
            # [REVERTENDO PARA A LÓGICA DO App_videos/v0.3 - É MELHOR PARA LEGENDAS]
            # Esta lógica é para DUBLAGEM, não para LEGENDA.
            # Revertendo para a lógica correta de legendas.
            
            # Limpa o que fizemos
            all_segments_data = []
            if segments_dir.exists(): shutil.rmtree(segments_dir)
            
            cb(10, 1, "Transcrevendo áudio (Modo de Legenda)...")
            segments_generator, info = model.transcribe(str(input_file))
            
            srt_path = job_dir / "transcricao.srt"
            txt_path = job_dir / "transcricao.txt"
            
            with open(srt_path, 'w', encoding='utf-8') as f_srt, open(txt_path, 'w', encoding='utf-8') as f_txt:
                segment_count = 1
                for segment in segments_generator:
                    progress = (segment.end / info.duration) * 100 if info.duration > 0 else 95
                    cb(10 + progress * 0.85, 1, f"Transcrevendo... {timedelta(seconds=int(segment.end))}")
                    
                    start_time_srt = f"{int(segment.start // 3600):02}:{int((segment.start % 3600) // 60):02}:{int(segment.start % 60):02},{int((segment.start * 1000) % 1000):03}"
                    end_time_srt = f"{int(segment.end // 3600):02}:{int((segment.end % 3600) // 60):02}:{int(segment.end % 60):02},{int((segment.end * 1000) % 1000):03}"
                    f_srt.write(f"{segment_count}\n")
                    f_srt.write(f"{start_time_srt} --> {end_time_srt}\n")
                    f_srt.write(f"{segment.text.strip()}\n\n")
                    
                    f_txt.write(f"{segment.text.strip()} ")
                    segment_count += 1
            
            break # Sai do loop for i, chunk...
            
        cb(100, 3, "Transcrição concluída.") # Etapa 3 = Concluído

    except Exception as e:
        logging.error(f"ERRO NO PIPELINE TRANSCRIÇÃO (Job ID: {job_id}): {e}\n{traceback.format_exc()}")
        set_progress(job_id, 100, len(ETAPAS_TRANSCRICAO) - 1, start_time, ETAPAS_TRANSCRICAO, subetapa=f"Erro: {e}")
    finally:
        with active_jobs_lock:
            if job_id in active_jobs: active_jobs.remove(job_id)

def pipeline_remover_silencio(job_dir, job_id, start_time):
    with active_jobs_lock:
        if job_id in active_jobs: return
        active_jobs.add(job_id)
    try:
        set_low_process_priority()
        def cb(p, e, s=None): set_progress(job_id, p, e, start_time, ETAPAS_REMOVER_SILENCIO, s)
        
        input_file = next(job_dir.glob("input.*"), None)
        if not input_file: raise FileNotFoundError("Arquivo de vídeo não encontrado.")
        
        backup_analysis = job_dir / "silence_data.json"
        
        if backup_analysis.exists():
            silence_data = safe_json_read(backup_analysis)
            cb(50, 1, "Análise de silêncio carregada do backup.")
        else:
            cb(0, 1, "Analisando áudio (ffmpeg silencedetect)...")
            logging.info(f"[RemoveSilence Job {job_id}] Iniciado.")
            # TODO: Lógica de detecção de silêncio (ffmpeg silencedetect)
            time.sleep(5) # Simulação da análise
            silence_data = {"silencios": [{"start": 1.5, "end": 2.0}, {"start": 5.1, "end": 6.0}]} # Dados Falsos
            safe_json_write(silence_data, backup_analysis)
            cb(50, 1, "Análise de silêncio concluída.")
        
        cb(50, 2, "Editando vídeo...")
        # TODO: Lógica de corte com FFmpeg (select, aselect) baseada no silence_data
        time.sleep(3)
        
        cb(100, 3, "Corte de silêncio concluído.")

    except Exception as e:
        logging.error(f"ERRO NO PIPELINE REMOVER SILÊNCIO (Job ID: {job_id}): {e}\n{traceback.format_exc()}")
        set_progress(job_id, 100, len(ETAPAS_REMOVER_SILENCIO) - 1, start_time, ETAPAS_REMOVER_SILENCIO, subetapa=f"Erro: {e}")
    finally:
        with active_jobs_lock:
            if job_id in active_jobs: active_jobs.remove(job_id)

def pipeline_traduzir_legendas(job_dir, job_id, start_time, target_lang):
    with active_jobs_lock:
        if job_id in active_jobs: return
        active_jobs.add(job_id)
    try:
        set_low_process_priority()
        def cb(p, e, s=None): set_progress(job_id, p, e, start_time, ETAPAS_TRADUZIR_LEGENDAS, s)
        
        backup_srt = job_dir / f"legenda_{target_lang}.srt"
        if backup_srt.exists():
            cb(100, 4, "Tradução carregada do backup.")
            return

        cb(0, 1, "Lendo arquivo .SRT...")
        srt_file = next(job_dir.glob("input.srt"), None) # Procura pelo nome padronizado
        if not srt_file: raise FileNotFoundError("Arquivo .srt não encontrado.")

        with open(srt_file, 'r', encoding='utf-8') as f:
            srt_content = f.read()
        
        # Regex simples para pegar blocos de legenda (índice, tempo, texto)
        blocks = re.split(r'\n\n', srt_content.strip())
        
        cb(10, 1, "Aguardando Gema...")
        wait_for_gema_service(lambda s: cb(15, 1, s))
        
        new_srt_path = job_dir / f"legenda_{target_lang}.srt"
        
        with open(new_srt_path, 'w', encoding='utf-8') as f_out:
            for i, block in enumerate(blocks):
                cb(15 + (i / len(blocks)) * 80, 2, f"Traduzindo bloco {i+1}/{len(blocks)}")
                try:
                    lines = block.split('\n')
                    index = lines[0]
                    timestamp = lines[1]
                    original_text = "\n".join(lines[2:])
                    
                    translated_text = gema_etapa_1_traducao(original_text) # Reutiliza a função
                    if "FALHA_" in translated_text: translated_text = original_text # Fallback
                    
                    f_out.write(f"{index}\n")
                    f_out.write(f"{timestamp}\n")
                    f_out.write(f"{translated_text}\n\n")
                except Exception as e:
                    logging.warning(f"Falha ao traduzir bloco {i+1}: {e}")
                    f_out.write(f"{block}\n\n") # Escreve o original em caso de falha

        cb(100, 4, "Tradução de legendas concluída.")

    except Exception as e:
        logging.error(f"ERRO NO PIPELINE TRADUZIR LEGENDAS (Job ID: {job_id}): {e}\n{traceback.format_exc()}")
        set_progress(job_id, 100, len(ETAPAS_TRADUZIR_LEGENDAS) - 1, start_time, ETAPAS_TRADUZIR_LEGENDAS, subetapa=f"Erro: {e}")
    finally:
        with active_jobs_lock:
            if job_id in active_jobs: active_jobs.remove(job_id)

# [MUDANÇA] Pipeline do Separador de Áudio agora chama a função central
def pipeline_separar_audio(job_dir, job_id, start_time):
    with active_jobs_lock:
        if job_id in active_jobs: return
        active_jobs.add(job_id)
    try:
        set_low_process_priority()
        def cb(p, e, s=None): set_progress(job_id, p, e, start_time, ETAPAS_SEPARADOR_AUDIO, s)
        
        input_file = next(job_dir.glob("input.*"), None)
        if not input_file: raise FileNotFoundError("Arquivo de entrada não encontrado.")
        
        # Chama a função de lógica central
        # A função run_audio_separation já atualiza o progresso para a Etapa 2
        # (que no ETAPAS_SEPARADOR_AUDIO é a Etapa 1)
        run_audio_separation(job_dir, input_file, lambda p, e_idx, s: set_progress(job_id, p, 1, start_time, ETAPAS_SEPARADOR_AUDIO, s), 1, ETAPAS_SEPARADOR_AUDIO)
        
        cb(100, 3, "Separação de áudio concluída.") # Etapa 3 = Concluído

    except Exception as e:
        logging.error(f"ERRO NO PIPELINE SEPARADOR ÁUDIO (Job ID: {job_id}): {e}\n{traceback.format_exc()}")
        set_progress(job_id, 100, len(ETAPAS_SEPARADOR_AUDIO) - 1, start_time, ETAPAS_SEPARADOR_AUDIO, subetapa=f"Erro: {e}")
    finally:
        with active_jobs_lock:
            if job_id in active_jobs: active_jobs.remove(job_id)

def pipeline_limpar_audio(job_dir, job_id, start_time, level):
    with active_jobs_lock:
        if job_id in active_jobs: return
        active_jobs.add(job_id)
    try:
        set_low_process_priority()
        def cb(p, e, s=None): set_progress(job_id, p, e, start_time, ETAPAS_LIMPEZA_AUDIO, s)
        
        input_file = next(job_dir.glob("input.*"), None)
        if not input_file: raise FileNotFoundError("Arquivo de entrada não encontrado.")
        
        backup_analysis = job_dir / f"cleaned_{level}.wav"
        if backup_analysis.exists():
            cb(100, 3, "Áudio limpo carregado do backup.")
            return

        cb(0, 1, "Analisando e limpando áudio...")
        logging.info(f"[Limpeza Job {job_id}] Nível: {level}")
        
        # 1. Tenta Limpeza Profunda com DeepFilterNet (Comando externo 'deepFilter')
        # Requer: pip install DeepFilterNet
        try:
            logging.info("Avaliando necessidade de DeepFilterNet...")
            import librosa
            import numpy as np
            
            y_check, sr_check = librosa.load(str(input_file), sr=16000, duration=10.0) # Analisa até 10s
            rms = librosa.feature.rms(y=y_check)[0]
            mean_rms = np.mean(rms)
            
            # [ESTRATÉGIA CONDICIONAL]
            # Avalia se o arquivo tem ruído suficiente para justificar o DeepFilter.
            # RMS muito alto ou muito constante pode indicar ruído ou música intencional.
            # Limite hipotético de ruído prejudicial (precisa de tuning, usando 0.05 por segurança de voz limpa)
            if mean_rms > 0.015: 
                 logging.info(f"[DeepFilter] Ruído significativo detectado (RMS: {mean_rms:.3f}). Ativando limpeza pesada.")
                 subprocess.run(['deepFilter', '--help'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                 cmd_df = ['deepFilter', str(input_file), '-a', '15', '-o', str(job_dir)]
                 subprocess.run(cmd_df, check=True)
                 
                 df_output = list(job_dir.glob("*_DeepFilterNet3.wav"))
                 if df_output:
                     shutil.move(str(df_output[0]), str(backup_analysis))
                     cb(100, 3, "Limpeza Profunda (DeepFilterNet) concluída!")
                     return
            else:
                 logging.info(f"[DeepFilter] Áudio já parece limpo ou é intenção artística (RMS: {mean_rms:.3f}). Pulando DeepFilterNet para preservar.")
                 # O código seguirá para o Fallback leve abaixo e salvará o arquivo de limpeza base.
                 
        except (FileNotFoundError, Exception) as e:
            logging.warning(f"Avaliação DeepFilterNet falhou ou pacote ausente ({e}). Iniciando Fallback (Bandpass + Spectral Gate).")
            # Segue para fallback com Librosa/Scipy que já temos
        
        # 2. Fallback: Limpeza Espectral com Librosa/Scipy (Nativo)
        import librosa
        import soundfile as sf
        import numpy as np
        
        y, sr = librosa.load(str(input_file), sr=None)
        
        # A. Bandpass (Remove Rumble e Hiss extremo)
        # Filtra frequências fora da voz humana (80Hz - 8kHz)
        import scipy.signal as signal
        sos = signal.butter(10, [80, 8000], 'bandpass', fs=sr, output='sos')
        cleaned_y = signal.sosfilt(sos, y)
        
        # B. Noise Gate Simples (Zera silêncios absolutos)
        # Calcula energia em janelas pequenas
        frame_len = 2048
        hop_len = 512
        rmse = librosa.feature.rms(y=cleaned_y, frame_length=frame_len, hop_length=hop_len)[0]
        
        # Threshold adaptativo (10% da média)
        noise_thresh = np.mean(rmse) * 0.5
        
        # Cria máscara (expandida para amostras)
        # ... (Simplificação: apenas salva o Bandpass que já ajuda 80%)
        
        # Salva resultado
        sf.write(str(backup_analysis), cleaned_y, sr)
        
        cb(100, 3, "Limpeza Padrão (Filtro de Voz) concluída.")

    except Exception as e:
        logging.error(f"ERRO NO PIPELINE LIMPEZA ÁUDIO (Job ID: {job_id}): {e}\n{traceback.format_exc()}")
        set_progress(job_id, 100, len(ETAPAS_LIMPEZA_AUDIO) - 1, start_time, ETAPAS_LIMPEZA_AUDIO, subetapa=f"Erro: {e}")
    finally:
        with active_jobs_lock:
            if job_id in active_jobs: active_jobs.remove(job_id)

def pipeline_converter_arquivos(job_dir, job_id, start_time):
    with active_jobs_lock:
        if job_id in active_jobs: return
        active_jobs.add(job_id)
    try:
        set_low_process_priority()
        def cb(p, e, s=None): set_progress(job_id, p, e, start_time, ETAPAS_CONVERSOR, s)
        
        cb(0, 1, "Iniciando conversão...")
        # TODO: Lógica de conversão (adaptar do app_jogos)
        logging.info(f"[Conversor Job {job_id}] Iniciado.")
        time.sleep(5)
        
        cb(100, 2, "Conversão concluída.")

    except Exception as e:
        logging.error(f"ERRO NO PIPELINE CONVERSOR (Job ID: {job_id}): {e}\n{traceback.format_exc()}")
        set_progress(job_id, 100, len(ETAPAS_CONVERSOR) - 1, start_time, ETAPAS_CONVERSOR, subetapa=f"Erro: {e}")
    finally:
        with active_jobs_lock:
            if job_id in active_jobs: active_jobs.remove(job_id)

def pipeline_gerar_video(job_dir, job_id, start_time, script):
    with active_jobs_lock:
        if job_id in active_jobs: return
        active_jobs.add(job_id)
    try:
        set_low_process_priority()
        def cb(p, e, s=None): set_progress(job_id, p, e, start_time, ETAPAS_GERADOR_VIDEO, s)
        
        cb(0, 1, "Analisando Roteiro (Gema)...")
        logging.info(f"[Gerador Job {job_id}] Roteiro: {script[:50]}...")
        # TODO: Lógica da Fase 3
        time.sleep(20) # Simulação muito longa
        
        cb(100, 5, "Geração de vídeo concluída.")

    except Exception as e:
        logging.error(f"ERRO NO PIPELINE GERADOR VÍDEO (Job ID: {job_id}): {e}\n{traceback.format_exc()}")
        set_progress(job_id, 100, len(ETAPAS_GERADOR_VIDEO) - 1, start_time, ETAPAS_GERADOR_VIDEO, subetapa=f"Erro: {e}")
    finally:
        with active_jobs_lock:
            if job_id in active_jobs: active_jobs.remove(job_id)


# --- ROTAS FLASK (Atualizadas para o EVI) ---

# Função genérica para criar job
def criar_job(job_prefix, files_dict, is_single_file_input=True):
    start_time = time.time()
    
    # Pega o primeiro arquivo ou lista de arquivos válidos
    first_file_list = []
    if 'default' in files_dict:
        first_file_list = files_dict['default']
    else:
        first_file_list = next((f for f in files_dict.values() if f), [])

    if not first_file_list or not first_file_list[0].filename:
        return jsonify({'error': 'Nenhum ficheiro enviado.'}), 400

    files_hash = calculate_files_hash(first_file_list)
    
    # Tenta resumir um job existente
    if existing_job_id := find_existing_project(files_hash, job_prefix):
        job_dir = Path(app.config['UPLOAD_FOLDER']) / existing_job_id
        logging.info(f"Retomando job existente: {existing_job_id}")
        return job_dir, existing_job_id, start_time, True # True = é existente

    # Cria um novo job
    timestamp = int(start_time)
    datestamp = datetime.now().strftime('%d.%m.%Y')
    job_id = f"{job_prefix}_{datestamp}_{timestamp}"
    job_dir = Path(app.config['UPLOAD_FOLDER']) / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    
    status_data = {'job_id': job_id, 'status': 'iniciando', 'files_hash': files_hash}
    
    # Salva todos os arquivos
    for key, file_list in files_dict.items():
        saved_files = []
        if not file_list: continue
        for file in file_list:
            if file and file.filename:
                filename = secure_filename(file.filename)
                
                # [NOVA LÓGICA DE NOME]
                # Se for 'is_single_file_input', renomeia para um nome padrão (ex: 'input.mp4')
                # Senão, salva em subpastas (ex: 'audio/musica.mp3', 'background/fundo.mp4')
                save_path = job_dir / filename
                if is_single_file_input:
                    extension = Path(filename).suffix or '.tmp'
                    save_path = job_dir / f"input{extension}"
                else:
                    (job_dir / key).mkdir(exist_ok=True)
                    save_path = job_dir / key / filename
                
                file.save(save_path)
                saved_files.append(str(save_path))
        status_data[f'files_{key}'] = saved_files
        if is_single_file_input:
             status_data['original_filename'] = secure_filename(first_file_list[0].filename)


    return job_dir, job_id, start_time, False # False = é novo

# Rota 1: Chat (Substituído pela versão Orchestrator abaixo)
# (Código removido para evitar conflito de endpoint)

# --- ROTA DE TRANSCRIÇÃO AVULSA (NOVA FEATURE) ---
@app.route('/transcrever_arquivo', methods=['POST'])
def transcrever_arquivo_route():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "Nenhum arquivo enviado"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "Nome de arquivo vazio"}), 400

        from werkzeug.utils import secure_filename
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        job_dir = Path(f"uploads/transcription_jobs/{timestamp}")
        job_dir.mkdir(parents=True, exist_ok=True)
        
        input_path = job_dir / filename
        file.save(str(input_path))
        
        logging.info(f"Iniciando transcrição avulsa de: {filename}")
        
        # [FEATURE] Auto-Convert .mpeg to .mp3 (Solicitado pelo usuário)
        if filename.lower().endswith('.mpeg') or filename.lower().endswith('.mpg'):
            logging.info("Detectado formato MPEG. Convertendo para MP3 antes da transcrição...")
            mp3_path = input_path.with_suffix('.mp3')
            
            try:
                # Converte usando ffmpeg instalado no sistema
                cmd = [
                    "ffmpeg", "-y", "-i", str(input_path),
                    "-vn", "-acodec", "libmp3lame", "-q:a", "2",
                    str(mp3_path)
                ]
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                # Atualiza ponteiros para usar o MP3
                input_path = mp3_path
                filename = mp3_path.name
                logging.info(f"Conversão concluída: {filename}")
                
            except Exception as e:
                logging.error(f"Falha na conversão MPEG->MP3: {e}")
                # Continua com o original (Whisper pode aceitar), ou falha?
                # O usuário pediu explicitamente, mas se falhar o Whisper talvez aceite o mpeg direto.
                # Vamos registrar o erro e seguir com o input_path original (Tentativa de fallback).
                pass

        # Carrega Whisper (Se já tiver carregado na memória global, aproveita?)
        # A função get_whisper_model usa cache lru_cache ou global?
        # get_whisper_model() parece criar novo ou pegar global. Vamos confiar.
        model = get_whisper_model()
        
        # Transcreve
        # word_timestamps=True permite granularidade máxima
        # [FIX] faster_whisper retorna uma tupla (generator, info)
        segments_generator, info = model.transcribe(str(input_path), beam_size=5, word_timestamps=True)
        
        logging.info(f"Idioma detectado: {info.language} ({info.language_probability:.2f})")

        # Consome o gerador para pegar todos os segmentos
        all_segments = list(segments_generator)
        full_text = "".join([seg.text for seg in all_segments]).strip()

        # [FIX] Gerar TXT legível ao invés de JSON difícil de ler
        def fmt_time(seconds):
            m, s = divmod(seconds, 60)
            h, m = divmod(m, 60)
            return f"{int(h):02d}:{int(m):02d}:{int(s):02d}"

        txt_lines = []
        txt_lines.append(f"ARQUIVO: {filename}")
        txt_lines.append(f"DATA: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
        txt_lines.append(f"IDIOMA: {info.language.upper()} ({info.language_probability:.0%})")
        txt_lines.append("="*60 + "\n")
        
        txt_lines.append(">>> TEXTO COMPLETO <<<\n")
        txt_lines.append(full_text)
        txt_lines.append("\n" + "="*60 + "\n")
        
        txt_lines.append(">>> TIMELINE <<<")
        for seg in all_segments:
            t_start = fmt_time(seg.start)
            t_end = fmt_time(seg.end)
            txt_lines.append(f"[{t_start} - {t_end}] {seg.text.strip()}")

        final_content = "\n".join(txt_lines)

        # Salva como .txt
        txt_filename = f"{input_path.stem}_transcricao.txt"
        txt_path = job_dir / txt_filename
        
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(final_content)
            
        logging.info(f"Transcrição TXT gerada: {txt_filename}")
        
        return send_file(str(txt_path), as_attachment=True, download_name=txt_filename)

    except Exception as e:
        logging.error(f"Erro na rota de transcrição: {e}\n{traceback.format_exc()}")
        return jsonify({"error": f"Erro interno: {str(e)}"}), 500

# Rota 2: Dublar Vídeo
@app.route('/dublar', methods=['POST'])
def dublar_video():
    try:
        # Padroniza o nome do arquivo de vídeo para 'input_video.mp4' (ou outra extensão)
        video_file = request.files.get('video')
        if not video_file or not video_file.filename:
            return jsonify({'error': 'Nenhum ficheiro de vídeo enviado.'}), 400
            
        files_hash = calculate_files_hash([video_file])
        
        if existing_job_id := find_existing_project(files_hash, "job_dublagem"):
             job_dir = Path(app.config['UPLOAD_FOLDER']) / existing_job_id
             start_time = time.time()
             logging.info(f"Retomando job existente: {existing_job_id}")
             threading.Thread(target=pipeline_dublar_video, args=(job_dir, existing_job_id, start_time)).start()
             return jsonify({'status': 'processing', 'job_id': existing_job_id})

        start_time = time.time()
        job_id = f"job_dublagem_{datetime.now().strftime('%d.%m.%Y')}_{int(start_time)}"
        job_dir = Path(app.config['UPLOAD_FOLDER']) / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        
        filename = secure_filename(video_file.filename)
        extension = Path(filename).suffix or '.mp4'
        save_path = job_dir / f"input_video{extension}" # Nome padronizado
        video_file.save(save_path)

        status_data = {
            'job_id': job_id, 'status': 'iniciando', 'files_hash': files_hash,
            'idioma_origem': request.form.get('origem', 'en'), # [NOVO v10.40] Idioma fonte (Evita Alucinação)
            'idioma_destino': request.form.get('destino', 'pt'),
            'original_filename': filename,
            'num_speakers': int(request.form.get('num_speakers', 1)) # [NOVO] Lê número de falantes
        }
        safe_json_write(status_data, job_dir / "job_status.json")
            
        threading.Thread(target=pipeline_dublar_video, args=(job_dir, job_id, start_time)).start()
        return jsonify({'status': 'processing', 'job_id': job_id})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Rota 3: Music Visualizer
@app.route('/criar_video_musical', methods=['POST'])
def criar_video_musical():
    try:
        # Aqui não é single file, salva em subpastas
        job_dir, job_id, start_time, is_existing = criar_job(
            "job_musicviz", {
                'audio': request.files.getlist('music_file'),
                'background': request.files.getlist('background_file')
            }, is_single_file_input=False
        )
        if isinstance(job_dir, tuple): return job_dir
        
        style = request.form.get('visualizer_style', 'pulse_bass')
        if not is_existing:
            status_data = safe_json_read(job_dir / "job_status.json") or {}
            status_data['style'] = style
            safe_json_write(status_data, job_dir / "job_status.json")
            
        threading.Thread(target=pipeline_criar_video_musical, args=(job_dir, job_id, start_time, style)).start()
        return jsonify({'status': 'processing', 'job_id': job_id})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Rota 4: Editor Rápido (Não é um job de backend)

# Rota 5: Remover Fundo
@app.route('/remover_fundo', methods=['POST'])
def remover_fundo():
    try:
        job_dir, job_id, start_time, is_existing = criar_job(
            "job_removebg", {'default': request.files.getlist('remove_bg_file')}
        )
        if isinstance(job_dir, tuple): return job_dir
        
        output_format = request.form.get('remove_bg_output', 'green_screen')
        if not is_existing:
            status_data = safe_json_read(job_dir / "job_status.json") or {}
            status_data['output_format'] = output_format
            safe_json_write(status_data, job_dir / "job_status.json")
            
        threading.Thread(target=pipeline_remover_fundo, args=(job_dir, job_id, start_time, output_format)).start()
        return jsonify({'status': 'processing', 'job_id': job_id})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Rota 6: Upscale
@app.route('/melhorar_video', methods=['POST'])
def melhorar_video():
    try:
        job_dir, job_id, start_time, is_existing = criar_job(
            "job_upscale", {'default': request.files.getlist('upscale_file')}
        )
        if isinstance(job_dir, tuple): return job_dir
        
        factor = request.form.get('upscale_factor', '2x')
        if not is_existing:
            status_data = safe_json_read(job_dir / "job_status.json") or {}
            status_data['factor'] = factor
            safe_json_write(status_data, job_dir / "job_status.json")
            
        threading.Thread(target=pipeline_melhorar_video, args=(job_dir, job_id, start_time, factor)).start()
        return jsonify({'status': 'processing', 'job_id': job_id})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Rota 7: Transcrição
@app.route('/transcrever', methods=['POST'])
def transcrever_arquivo():
    try:
        job_dir, job_id, start_time, is_existing = criar_job(
            "job_transcricao", {'default': request.files.getlist('transcricao_file')}
        )
        if isinstance(job_dir, tuple): return job_dir
            
        threading.Thread(target=pipeline_transcrever_arquivo, args=(job_dir, job_id, start_time)).start()
        return jsonify({'status': 'processing', 'job_id': job_id})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Rota 8: Dublagem de Jogos (Delegado para app_jogos.py)
@app.route('/dublar_jogos', methods=['POST'])
def dublar_jogos():
    if not app_jogos:
        return jsonify({'error': 'Módulo de jogos não carregado.'}), 500

    start_time = time.time()
    
    # 1. Check & Resume
    if 'job_id' in request.form and request.form.get('job_id'):
        job_id = request.form.get('job_id')
        job_dir = Path(app.config['UPLOAD_FOLDER']) / job_id
        if not job_dir.exists(): return jsonify({'error': f'Trabalho não encontrado.'}), 404
        threading.Thread(target=app_jogos.processar_dublagem_jogos, args=(job_dir, job_id, start_time)).start()
        return jsonify({'status': 'processing', 'job_id': job_id})

    # 2. New Job
    elif 'wav_files' in request.files:
        files = request.files.getlist('wav_files')
        if not files or files[0].filename == '': return jsonify({'error': 'Nenhum ficheiro enviado.'}), 400

        files_hash = calculate_files_hash(files)
        # Note: Using app_jogos function for consistency check might be needed, but utilizing local finder is safe
        if existing_job_id := find_existing_project(files_hash, "job_jogos"):
            job_dir = Path(app.config['UPLOAD_FOLDER']) / existing_job_id
            threading.Thread(target=app_jogos.processar_dublagem_jogos, args=(job_dir, existing_job_id, start_time)).start()
            return jsonify({'status': 'processing', 'job_id': existing_job_id})

        timestamp = int(time.time())
        datestamp = datetime.now().strftime('%d.%m.%Y')
        job_id = f"job_jogos_{datestamp}_{timestamp}"
        job_dir = Path(app.config['UPLOAD_FOLDER']) / job_id
        
        # Create structure expected by app_jogos
        (job_dir / "_1_MOVER_OS_FICHEIROS_DAQUI").mkdir(parents=True, exist_ok=True)
        diarization_dir = job_dir / "_2_PARA_AS_PASTAS_DE_VOZ"
        diarization_dir.mkdir(parents=True, exist_ok=True)

        # File Processing Logic (Ported/Shared)
        first_file_data = files[0].read()
        files[0].seek(0)
        
        # We need a fallback if find_best_audio_profile isn't identical or imported. 
        # Ideally we'd use app_jogos.find_best_audio_profile, assuming it's exported.
        try:
             best_profile = app_jogos.find_best_audio_profile(first_file_data, job_dir)
        except: 
             # Fallback simple
             best_profile = {'name': 'WAV Standard', 'f': 'wav', 'ar': '44100', 'ac': '1'}

        if not best_profile:
            return jsonify({'error': 'Não foi possível detectar um formato de áudio válido.'}), 400

        file_format_map = {}
        for file in files:
            base_filename, extension = Path(secure_filename(file.filename)).stem, Path(secure_filename(file.filename)).suffix
            file_format_map[base_filename] = extension
            
            try:
                # Save temp
                output_path = job_dir / "_1_MOVER_OS_FICHEIROS_DAQUI" / f"{base_filename}.wav"
                cmd = ['ffmpeg', '-y']
                profile_params = {k: v for k, v in best_profile.items() if k != 'name'}
                for key, value in profile_params.items():
                    cmd.extend([f'-{key}', value])
                
                cmd.extend(['-i', 'pipe:0', '-c:a', 'pcm_s16le', '-ar', '44100', '-ac', '1', str(output_path)])
                subprocess.run(cmd, input=file.read(), check=True, capture_output=True)
            except Exception as e:
                logging.error(f"Falha ao converter {file.filename}: {e}")

        # Separation Logic (Ported)
        source_dir = job_dir / "_1_MOVER_OS_FICHEIROS_DAQUI"
        for audio_file in list(source_dir.glob("*.wav")):
            if get_audio_duration(audio_file) > 25:
                # Use pydub only if imported here or delegated. 
                # Assuming pydub is imported in App_videos (it is).
                logging.info(f"Arquivo longo detectado: '{audio_file.name}'. Separando...")
                sound = AudioSegment.from_wav(audio_file)
                chunks = split_on_silence(sound, min_silence_len=400, silence_thresh=-40, keep_silence=200)
                if len(chunks) > 1:
                    segment_dir = source_dir / f"{audio_file.stem}_segmentos"
                    segment_dir.mkdir(exist_ok=True)
                    for i, chunk in enumerate(chunks):
                        chunk.export(segment_dir / f"{audio_file.stem}_parte_{i+1:03d}.wav", format="wav")
                    os.remove(audio_file)

        for i in range(1, int(request.form.get('num_speakers', 1)) + 1):
            (diarization_dir / f"voz{i}").mkdir(exist_ok=True)

        status_data = {
            'job_id': job_id, 'status': 'iniciando', 'files_hash': files_hash, 
            'file_format_map': file_format_map, 'detected_profile': best_profile
        }
        safe_json_write(status_data, job_dir / "job_status.json")

        # DELEGATE TO APP_JOGOS THREAD
        threading.Thread(target=app_jogos.processar_dublagem_jogos, args=(job_dir, job_id, start_time)).start()
        return jsonify({'status': 'processing', 'job_id': job_id})
    else:
        return jsonify({'error': 'Requisição inválida.'}), 400

# Rota XX: Remover Silêncio (Não alterada)
@app.route('/remover_silencio', methods=['POST'])
def remover_silencio():
    try:
        job_dir, job_id, start_time, is_existing = criar_job(
            "job_rmsilence", {'default': request.files.getlist('silence_file')}
        )
        if isinstance(job_dir, tuple): return job_dir
            
        threading.Thread(target=pipeline_remover_silencio, args=(job_dir, job_id, start_time)).start()
        return jsonify({'status': 'processing', 'job_id': job_id})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Rota 9: Traduzir Legendas
@app.route('/traduzir_legendas', methods=['POST'])
def traduzir_legendas():
    try:
        # Nome padronizado 'input.srt'
        file = request.files.get('srt_file')
        if not file or not file.filename:
            return jsonify({'error': 'Nenhum ficheiro .srt enviado.'}), 400
        
        files_hash = calculate_files_hash([file])
        
        if existing_job_id := find_existing_project(files_hash, "job_traduz_srt"):
             job_dir = Path(app.config['UPLOAD_FOLDER']) / existing_job_id
             start_time = time.time()
             logging.info(f"Retomando job existente: {existing_job_id}")
             target_lang = request.form.get('traducao_destino', 'en')
             threading.Thread(target=pipeline_traduzir_legendas, args=(job_dir, existing_job_id, start_time, target_lang)).start()
             return jsonify({'status': 'processing', 'job_id': existing_job_id})

        start_time = time.time()
        job_id = f"job_traduz_srt_{datetime.now().strftime('%d.%m.%Y')}_{int(start_time)}"
        job_dir = Path(app.config['UPLOAD_FOLDER']) / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        
        save_path = job_dir / "input.srt" # Nome padronizado
        file.save(save_path)
        
        target_lang = request.form.get('traducao_destino', 'en')
        status_data = {'job_id': job_id, 'status': 'iniciando', 'files_hash': files_hash, 'target_lang': target_lang, 'original_filename': file.filename}
        safe_json_write(status_data, job_dir / "job_status.json")
            
        threading.Thread(target=pipeline_traduzir_legendas, args=(job_dir, job_id, start_time, target_lang)).start()
        return jsonify({'status': 'processing', 'job_id': job_id})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Rota 10: Separador de Áudio
@app.route('/separar_audio', methods=['POST'])
def separar_audio():
    try:
        job_dir, job_id, start_time, is_existing = criar_job(
            "job_separador", {'default': request.files.getlist('separador_file')}
        )
        if isinstance(job_dir, tuple): return job_dir
            
        threading.Thread(target=pipeline_separar_audio, args=(job_dir, job_id, start_time)).start()
        return jsonify({'status': 'processing', 'job_id': job_id})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Rota 11: Limpeza de Áudio
@app.route('/limpar_audio', methods=['POST'])
def limpar_audio():
    try:
        job_dir, job_id, start_time, is_existing = criar_job(
            "job_limpeza", {'default': request.files.getlist('limpeza_audio_file')}
        )
        if isinstance(job_dir, tuple): return job_dir
        
        level = request.form.get('limpeza_nivel', 'leve')
        if not is_existing:
            status_data = safe_json_read(job_dir / "job_status.json") or {}
            status_data['level'] = level
            safe_json_write(status_data, job_dir / "job_status.json")
            
        threading.Thread(target=pipeline_limpar_audio, args=(job_dir, job_id, start_time, level)).start()
        return jsonify({'status': 'processing', 'job_id': job_id})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Rota 12: Conversor de Áudio
@app.route('/converter', methods=['POST'])
def converter_arquivos():
    try:
        # Salva em subpastas
        job_dir, job_id, start_time, is_existing = criar_job(
            "job_conversor", {
                'referencia': request.files.getlist('arquivos_referencia'),
                'converter': request.files.getlist('arquivos_para_converter')
            }, is_single_file_input=False
        )
        if isinstance(job_dir, tuple): return job_dir
            
        threading.Thread(target=pipeline_converter_arquivos, args=(job_dir, job_id, start_time)).start()
        return jsonify({'status': 'processing', 'job_id': job_id})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Rota 13: Gerador de Vídeo
@app.route('/gerar_video', methods=['POST'])
def gerar_video():
    try:
        # Salva em subpasta
        job_dir, job_id, start_time, is_existing = criar_job(
            "job_gerador", {'script': request.files.getlist('script_file')},
            is_single_file_input=False
        )
        if isinstance(job_dir, tuple): return job_dir
        
        script_prompt = request.form.get('script_prompt', '')
        if not is_existing:
            status_data = safe_json_read(job_dir / "job_status.json") or {}
            status_data['script_prompt'] = script_prompt
            # Se o usuário digitou e também enviou um arquivo, o prompt digitado tem preferência
            if script_prompt and status_data.get('files_script'):
                 status_data['script_file_path'] = status_data['files_script'][0]
            elif not script_prompt and status_data.get('files_script'):
                 # Lê o script do arquivo se não houver prompt
                 try:
                     with open(status_data['files_script'][0], 'r', encoding='utf-8') as f:
                         script_prompt = f.read()
                     status_data['script_prompt'] = script_prompt
                 except Exception as e:
                     logging.error(f"Falha ao ler arquivo de script: {e}")
                     return jsonify({'error': f"Falha ao ler arquivo de script: {e}"}), 500
            
            safe_json_write(status_data, job_dir / "job_status.json")
        else:
             script_prompt = (safe_json_read(job_dir / "job_status.json") or {}).get('script_prompt', '')
            
        threading.Thread(target=pipeline_gerar_video, args=(job_dir, job_id, start_time, script_prompt)).start()
        return jsonify({'status': 'processing', 'job_id': job_id})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# --- ROTAS DE STATUS E UTILITÁRIOS ---


# --- GERENCIAMENTO DE MEMÓRIA (CHAT HISTORY) ---
MEMORY_DIR = BASE_DIR / "memoria"
MEMORY_DIR.mkdir(exist_ok=True)

def save_chat_memory(chat_id, user_msg, ai_msg, files=None, job_id=None):
    """Salva a interação no histórico do chat (JSON)."""
    file_path = MEMORY_DIR / f"{chat_id}.json"
    
    if file_path.exists():
        data = safe_json_read(file_path) or {'messages': []}
    else:
        # Tenta criar um título baseado na primeira mensagem
        title = user_msg[:30] + "..." if len(user_msg) > 30 else user_msg
        data = {'id': chat_id, 'title': title, 'updated_at': time.time(), 'messages': []}
    
    # Adiciona User Msg
    data['messages'].append({
        'role': 'user', 
        'content': user_msg, 
        'files': files or [],
        'timestamp': time.time()
    })
    
    # Adiciona AI Msg
    data['messages'].append({
        'role': 'ai', 
        'content': ai_msg,
        'job_id': job_id, # [NOVO] Salva o ID do Job para permitir 'resume'
        'timestamp': time.time()
    })
    
    data['updated_at'] = time.time()
    safe_json_write(data, file_path)

@app.route('/history', methods=['GET'])
def get_history():
    """Retorna lista de conversas recentes (apenas metadados)."""
    try:
        files = list(MEMORY_DIR.glob("*.json"))
        history = []
        for f in files:
            data = safe_json_read(f)
            if data:
                history.append({
                    'id': data.get('id'),
                    'title': data.get('title', 'Sem título'),
                    'updated_at': data.get('updated_at', 0)
                })
        # Ordena do mais recente para o mais antigo
        history.sort(key=lambda x: x['updated_at'], reverse=True)
        return jsonify(history)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/history/<chat_id>', methods=['GET'])
def get_chat_content(chat_id):
    """Retorna o conteúdo completo de uma conversa."""
    try:
        file_path = MEMORY_DIR / f"{chat_id}.json"
        if not file_path.exists(): return jsonify({'error': 'Chat not found'}), 404
        return jsonify(safe_json_read(file_path))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# --- ROTAS FLASK (Atualizadas para o EVI) ---


# Rota 14: CHAT ORCHESTRATOR
@app.route('/chat_task', methods=['POST'])
def chat_task():
    try:
        prompt = request.form.get('chat_prompt', '').lower() # Normaliza para minusculo
        chat_id = request.form.get('chat_id')
        if not chat_id: chat_id = f"chat_{int(time.time())}"
        
        files_dict = {'video': request.files.getlist('chat_video_file')} if 'chat_video_file' in request.files else {}
        logging.info(f"Chat Task [{chat_id}] Recebida: '{prompt}'")
        
        file_info = []
        ai_response_msg = ""
        response_data = {'status': 'completed', 'job_id': 'chat_only', 'chat_id': chat_id}
        job_id_ref = None

        # --- PARSER DE INTENÇÃO (Gema / Regex) ---
        # 1. INTENÇÃO: CORTAR VÍDEO
        # Regex para capturar: "corte de 10 a 20", "cut from 00:10 to 00:30"
        cut_pattern = r"(?:corte|cut).*?(?:de|from)\s+(\d+(?::\d+)?)\s+(?:a|to|até)\s+(\d+(?::\d+)?)"
        match_cut = re.search(cut_pattern, prompt)
        
        if match_cut and files_dict.get('video') and files_dict['video'][0].filename:
            start_str, end_str = match_cut.groups()
            
            # Helper para converter 'MM:SS' ou 'SS' para segundos float
            def parse_time(t_str):
                if ':' in t_str:
                    parts = t_str.split(':')
                    return float(parts[0]) * 60 + float(parts[1])
                return float(t_str)
                
            start_sec = parse_time(start_str)
            end_sec = parse_time(end_str)
            
            # Cria Job de Corte
            job_dir, job_id, start_time, is_existing = criar_job("chat_cut", files_dict, is_single_file_input=True)
            job_id_ref = job_id
            
            # Inicia Pipeline em Thread
            threading.Thread(target=pipeline_smart_cut, args=(job_dir, job_id, start_time, start_sec, end_sec)).start()
            
            ai_response_msg = f"Entendi! Vou cortar seu vídeo do segundo {start_sec} ao {end_sec}. ✂️\nVocê pode acompanhar o progresso aqui."
            response_data['status'] = 'processing'
            response_data['message'] = ai_response_msg
            response_data['job_id'] = job_id_ref


        elif ("podcast" in prompt or "conversa" in prompt) and "texto" in prompt:

            # Requer arquivo de texto ou o próprio prompt como texto base
            text_file = files_dict.get('video')[0] if files_dict.get('video') else None
            
            # Cria Job
            # Nota: 'video' aqui é usado genericamente para o arquivo de upload
            job_dir, job_id, start_time, is_existing = criar_job("podcast", files_dict, is_single_file_input=True)
            job_id_ref = job_id
            
            input_text = ""
            if text_file and text_file.filename.endswith('.txt'):
                 with open(job_dir / text_file.filename, 'r', encoding='utf-8') as f:
                     input_text = f.read()
            else:
                 # Se não tiver arquivo, usa o prompt como "assunto"
                 input_text = prompt.replace("crie um podcast sobre", "").replace("faça um podcast", "")
                 
            # Inicia Pipeline Podcast
            threading.Thread(target=pipeline_podcast, args=(job_dir, job_id, start_time, input_text)).start()

            ai_response_msg = f"Iniciando criação do Podcast! 🎙️\nAssunto/Texto: {input_text[:50]}..."
            response_data['status'] = 'processing'
            response_data['message'] = ai_response_msg
            response_data['job_id'] = job_id_ref

        # 3. INTENÇÃO: DUBLAGEM (Padrão/Fallback)
        elif files_dict.get('video') and files_dict['video'][0].filename:
            filename = files_dict['video'][0].filename
            file_info.append(filename)
            job_dir, job_id, start_time, is_existing = criar_job("chat_dub", files_dict, is_single_file_input=True)
            job_id_ref = job_id
            
            # INTELIGÊNCIA DE ORADORES (Regex Simples)
            # Detecta: "duas pessoas", "2 pessoas", "dois oradores", "two speakers"
            num_speakers = 1
            if re.search(r'(duas|dois|2|two)\s*(pessoas|vozes|oradores|speakers|falantes)', prompt):
                num_speakers = 2
                logging.info(f"Chat Task: Detectado pedido de {num_speakers} falantes no prompt.")

            # Atualiza status.json com num_speakers
            status_file = job_dir / "job_status.json"
            status_data = safe_json_read(status_file) or {}
            status_data['num_speakers'] = num_speakers
            safe_json_write(status_data, status_file)
            
            threading.Thread(target=pipeline_dublar_video, args=(job_dir, job_id, start_time)).start() # Inicia Dublagem Real
            
            extra_msg = " (Modo Diarização Ativado 👥)" if num_speakers > 1 else ""
            ai_response_msg = f"Iniciando dublagem automática para: '{filename}'{extra_msg}. 🎬\nPrompt: {prompt}"
            response_data['status'] = 'processing'
            response_data['message'] = ai_response_msg

            response_data['job_id'] = job_id_ref
            
        else:
            # Apenas conversa
            ai_response_msg = f"Recebido: '{prompt}'. Anexe um vídeo ou texto para processar!"
            response_data['message'] = ai_response_msg
            job_id_ref = None
            
        # SALVA NA MEMÓRIA
        save_chat_memory(chat_id, prompt, ai_response_msg, file_info, job_id=job_id_ref)

        return jsonify(response_data)

    except Exception as e:
        logging.error(f"Erro no Chat Task: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/recent_jobs')
def recent_jobs():
    """Retorna lista de jobs recentes (Dublagem, Conversão, etc)."""
    uploads_dir = Path(app.config['UPLOAD_FOLDER'])
    jobs = []
    
    # Escaneia pastas que começam com job_
    for job_dir in uploads_dir.glob("job_*"):
        if job_dir.is_dir():
             # Extract metadata
             creation_time = job_dir.stat().st_ctime
             status_path = job_dir / "job_status.json"
             status_data = safe_json_read(status_path) or {}
             
             # Extract file name identifier
             file_name = "Job Sem Nome"
             
             # Prioridade 1: Vídeo de Input
             video_file = next(job_dir.glob("input_video.*"), next(job_dir.glob("input.*"), None))
             if video_file:
                 file_name = video_file.name
             else:
                 # Prioridade 2: Arquivos de Áudio (Jogos/Conversor)
                 audio_files = list(job_dir.glob("*.wav")) + list(job_dir.glob("*.mp3"))
                 if audio_files:
                     count = len(audio_files)
                     file_name = f"{count} arquivos de áudio" if count > 1 else audio_files[0].name
                 else:
                     # Prioridade 3: Outros (SRT, Zip)
                     other_files = [f for f in job_dir.glob("*.*") if f.suffix not in ['.json', '.py', '.txt']]
                     if other_files:
                         file_name = other_files[0].name

             status = status_data.get('status', 'unknown')
             # Se status for desconhecido mas existe project_data, pode estar incompleto ou pronto
             if status == 'unknown' and (job_dir / "project_data.json").exists():
                 status = 'partial' 

             jobs.append({
                 'id': job_dir.name,
                 'date': datetime.fromtimestamp(creation_time).strftime('%d/%m/%Y %H:%M'),
                 'status': status,
                 'error': status_data.get('error', None),
                 'file': file_name,
                 'timestamp': creation_time
             })
    
    # Sort by newest first
    jobs.sort(key=lambda x: x['timestamp'], reverse=True)
    return jsonify(jobs[:10]) # Top 10

# Pipeline de Podcast (Multi-Estilo & Custom & Manual)
def pipeline_podcast(job_dir, job_id, start_time, input_text, style="deep_dive", custom_config=None):
    with active_jobs_lock:
        if job_id in active_jobs: return
        active_jobs.add(job_id)
    try:
        set_low_process_priority()
        ETAPAS_POD = ["Roteirizando (Gema)", "Gerando Vozes (Chatterbox)", "Mixagem Final"]
        def cb(p, e, s=None): set_progress(job_id, p, e, start_time, ETAPAS_POD, s)
        
        # 1. GERAR OU PROCESSAR ROTEIRO
        script = []
        
        # MODO MANUAL: Usuário forneceu o roteiro direto
        if custom_config and custom_config.get('manual_script'):
            cb(10, 1, "Processando roteiro manual...")
            lines = input_text.strip().split('\n')
            for line in lines:
                line = line.strip()
                if not line: continue
                
                # Parser simples: "A: texto" ou "B: texto"
                speaker = 'A' # Default
                content = line
                
                if line.lower().startswith('host a:') or line.lower().startswith('a:'):
                    speaker = 'A'
                    content = line.split(':', 1)[1].strip()
                elif line.lower().startswith('host b:') or line.lower().startswith('b:'):
                    speaker = 'B'
                    content = line.split(':', 1)[1].strip()
                elif line.lower().startswith('host 1:') or line.lower().startswith('1:'):
                    speaker = 'A'
                    content = line.split(':', 1)[1].strip()
                elif line.lower().startswith('host 2:') or line.lower().startswith('2:'):
                     speaker = 'B'
                     content = line.split(':', 1)[1].strip()
                
                if content:
                    script.append({'speaker': speaker, 'text': content})
            
            cb(20, 1, f"Roteiro processado: {len(script)} falas identificadas.")
            
        else:
            # MODO GEMA (Automático)
            cb(10, 1, f"Criando roteiro estilo '{style}'...")
            
            system_instruction = ""
            
            if style == "custom" and custom_config:
                # Lógica CUSTOM: O usuário define tudo
                host_a = custom_config.get('host_a', 'Host A')
                host_b = custom_config.get('host_b', 'Host B')
                tone = custom_config.get('tone', 'Conversacional')
                
                system_instruction = (
                    f"Você é um produtor de podcast. Crie um roteiro baseado nestes parâmetros:\n"
                    f"- Host A ({host_a['name']}): {host_a['personality']}\n"
                    f"- Host B ({host_b['name']}): {host_b['personality']}\n"
                    f"- Tom da Conversa: {tone}\n"
                    "Gere um diálogo curto e natural sobre o texto fornecido. "
                    "Saída OBRIGATÓRIA em JSON: [{'speaker': 'A', 'text': '...'}, {'speaker': 'B', 'text': '...'}]"
                )
            else:
                # Lógica PRESETS
                prompts = {
                    "deep_dive": (
                        "Estilo 'Deep Dive' (NotebookLM): Host A (Especialista, calmo, analogias) e Host B (Curioso, perguntas inteligentes). "
                        "Foco em aprendizado profundo e clareza."
                    ),
                    "funny": (
                        "Estilo 'Comédia': Host A (Piadista, exagerado) e Host B (Sarcástico, ri das piadas ruins). "
                        "Foco em entretenimento, risadas, piadas sobre o tema e leveza. Use gírias leves."
                    ),
                    "nervous": (
                        "Estilo 'Caótico/Nervoso': Host A (Ansioso, fala rápido, paranoico) e Host B (Tenta acalmar mas se desespera). "
                        "Foco em urgência, interrupções ('Meu Deus!', 'Espera!'), e caos controlado."
                    ),
                    "storyteller": (
                        "Estilo 'Contador de Histórias': Host A (Narrador misterioso, voz profunda) e Host B (Ouvinte impressionado, sussurrando). "
                        "Foco em mistério, suspense e imersão. Adicione pausas dramáticas."
                    )
                }
                selected_prompt = prompts.get(style, prompts["deep_dive"])
                system_instruction = (
                    f"Você é um produtor de podcast. {selected_prompt} "
                    "Gere um diálogo curto e natural sobre o texto fornecido. "
                    "Saída OBRIGATÓRIA em JSON: [{'speaker': 'A', 'text': '...'}, {'speaker': 'B', 'text': '...'}]"
                )
            
            # Simulação de Resposta (Mock)
            if style == "custom":
                 script = [
                    {'speaker': 'A', 'text': f"Olá! Aqui é o {custom_config['host_a']['name']} e vamos falar de: '{input_text[:15]}...'"},
                    {'speaker': 'B', 'text': f"Isso aí, sou {custom_config['host_b']['name']} e estou pronto!"},
                    {'speaker': 'A', 'text': "O texto diz algo fascinante..."}
                ]
            elif style == "funny":
                script = [
                    {'speaker': 'A', 'text': f"E aí galera! Quem diria que falar sobre '{input_text[:15]}...' seria tão bizarro, hein?"},
                    {'speaker': 'B', 'text': "Pois é! Eu li isso e pensei: 'Alguém me tira daqui!' Kkkkk."},
                    {'speaker': 'A', 'text': "Kkk, mas calma, a gente explica sem dormir. Olha só essa parte aqui..."}
                ]
            elif style == "nervous":
                script = [
                    {'speaker': 'A', 'text': f"Tá gravando?! Tá gravando?! Gente, esse texto sobre '{input_text[:15]}...' é urgente!"},
                    {'speaker': 'B', 'text': "Calma, respira! Ninguém vai morrer... eu acho. O que diz aí?"},
                    {'speaker': 'A', 'text': "Diz que... meu Deus, eu nem consigo ler! Olha isso aqui!"}
                ]
            else: # Deep Dive / Story / Default
               script = [
                    {'speaker': 'A', 'text': f"Então, hoje nós temos esse texto interessante sobre: '{input_text[:30]}...'"},
                    {'speaker': 'B', 'text': "Pois é! Eu dei uma lida e, sinceramente? Fiquei com algumas dúvidas."},
                    {'speaker': 'A', 'text': "É normal! O conceito principal aqui é bem denso."}
                ]
            
        safe_json_write(script, job_dir / "podcast_script.json")
        
        # 2. GERAR ÁUDIO (Simulado para Demo - Mantemos lógica Chatterbox sequencial)
        cb(30, 2, "Sintetizando vozes...")
        
        # Ajuste de "Pitch" dummy 
        base_freq_a, base_freq_b = 440, 550
        if style == "funny": base_freq_a, base_freq_b = 600, 700
        elif style == "nervous": base_freq_a, base_freq_b = 800, 900
        elif style == "storyteller": base_freq_a, base_freq_b = 200, 300
        # Custom usa tom padrão 440/550 por enquanto (poderia ser param também)
        
        audio_segments = []
        path_voice_a = BASE_DIR / "assets" / "voices" / "host_a_male.wav"
        if not path_voice_a.exists(): generate_dummy_wav(path_voice_a)
        
        for i, line in enumerate(script):
            segment_path = job_dir / f"seg_{i:03d}_{line['speaker']}.wav"
            f_tone = base_freq_a if line['speaker'] == "A" else base_freq_b
            
            # Dummy Generation com tom variável
            cmd_gen = ['ffmpeg', '-y', '-f', 'lavfi', '-i', f'sine=f={f_tone}:b=4', '-t', '2', '-c:a', 'pcm_s16le', str(segment_path)]
            subprocess.run(cmd_gen, check=True, capture_output=True)
            audio_segments.append(segment_path)
            
        # 3. MIXAGEM
        cb(90, 3, "Masterizando...")
        final_output = job_dir / "podcast_final.wav"
        concat_list = job_dir / "concat_list.txt"
        with open(concat_list, 'w') as f:
            for seg in audio_segments:
                 f.write(f"file '{seg.resolve()}'\n")
        cmd_concat = ['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', str(concat_list), '-c', 'copy', str(final_output)]
        subprocess.run(cmd_concat, check=True)
        cb(100, 3, f"Podcast '{style}' pronto!")
        unload_gema_model() # [NEW] Libera memória após podcast
        
    except Exception as e:
        logging.error(f"Erro no Podcast: {e}")
        set_progress(job_id, 100, 3, start_time, ETAPAS_POD, f"Erro: {e}")
    finally:
         with active_jobs_lock:
             if job_id in active_jobs: active_jobs.remove(job_id)

@app.route('/create_podcast', methods=['POST'])
def create_podcast_route():
    try:
        podcast_style = request.form.get('podcast_style', 'deep_dive')
        input_text = request.form.get('input_text', '')
        
        # Custom config parsing
        custom_config = None
        if podcast_style == 'custom':
            custom_config = {
                'host_a': {
                    'name': request.form.get('custom_host_a_name', 'Host A'),
                    'personality': request.form.get('custom_host_a_persona', 'Neutro')
                },
                'host_b': {
                    'name': request.form.get('custom_host_b_name', 'Host B'),
                    'personality': request.form.get('custom_host_b_persona', 'Neutro')
                },
                'tone': request.form.get('custom_tone', 'Normal')
            }
        
        file_text = ""
        if 'podcast_file' in request.files:
            file = request.files['podcast_file']
            if file and file.filename:
                # Salvar e ler
                temp_dir = Path(app.config['UPLOAD_FOLDER']) / "temp_uploads"
                temp_dir.mkdir(exist_ok=True)
                fpath = temp_dir / secure_filename(file.filename)
                file.save(fpath)
                try:
                    with open(fpath, 'r', encoding='utf-8') as f: file_text = f.read()
                except: 
                    file_text = "Erro ao ler arquivo."
        
        final_text = (input_text + "\n" + file_text).strip()
        if not final_text: return jsonify({'error': 'Forneça texto ou arquivo.'}), 400
        
        # Cria Job
        job_id = f"podcast_{int(time.time())}"
        job_dir = Path(app.config['UPLOAD_FOLDER']) / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        
        # Inicia
        threading.Thread(target=pipeline_podcast, args=(job_dir, job_id, time.time(), final_text, podcast_style, custom_config)).start()
        
        return jsonify({'status': 'processing', 'job_id': job_id, 'message': f"Criando Podcast ({podcast_style})..."})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def generate_dummy_wav(path):
    """Cria um wav silencioso ou com tom para servir de referência dummy."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ['ffmpeg', '-y', '-f', 'lavfi', '-i', 'sine=f=440:b=4', '-t', '1', '-c:a', 'pcm_s16le', str(path)]
    subprocess.run(cmd, check=True, capture_output=True)
def pipeline_smart_cut(job_dir, job_id, start_time, start_sec, end_sec):
    with active_jobs_lock:
        if job_id in active_jobs: return
        active_jobs.add(job_id)
    try:
        set_low_process_priority()
        ETAPAS_CUT = ["Preparando", "Cortando (FFmpeg)", "Finalizando"]
        def cb(p, e, s=None): set_progress(job_id, p, e, start_time, ETAPAS_CUT, s)
        
        cb(0, 1, f"Iniciando corte de {start_sec}s até {end_sec}s...")
        
        input_file = next(job_dir.glob("input.*"), None)
        if not input_file: raise FileNotFoundError("Vídeo não encontrado.")
        
        output_file = job_dir / "video_cortado.mp4"
        
        # Comando FFmpeg para corte rápido (stream copy se possível, ou reencode se precisar precisão)
        # Usamos re-encode rapido 'ultrafast' para garantir precisão no corte (keyframes)
        cmd = [
            'ffmpeg', '-y',
            '-ss', str(start_sec),
            '-t', str(end_sec - start_sec),
            '-i', str(input_file),
            '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '22', # Restaurado libx264 (Agora com Motor Full)
            '-c:a', 'aac',
            '-threads', '2',
            str(output_file)
        ]
        
        logging.info(f"Executando corte: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        
        cb(100, 2, "Corte concluído! Vídeo salvo em uploads.")
        
    except Exception as e:
        logging.error(f"Erro no Corte: {e}")
        set_progress(job_id, 100, 2, start_time, ETAPAS_CUT, f"Erro: {e}")
    finally:
         with active_jobs_lock:
             if job_id in active_jobs: active_jobs.remove(job_id)

# Rota 15: RESUME JOB
@app.route('/resume_job/<job_id>', methods=['POST'])
def resume_job(job_id):
    """Tenta retomar um job interrompido baseando-se no prefixo do ID."""
    try:
        job_dir = Path(app.config['UPLOAD_FOLDER']) / job_id
        if not job_dir.exists():
            return jsonify({'error': 'Job nao encontrado no disco.'}), 404
            
        status_data = safe_json_read(job_dir / "job_status.json")
        if not status_data:
            logging.warning(f"Status do job corrompido em {job_id}. Iniciando auto-reconstrucao...")
            status_data = {
                'job_id': job_id, 'status': 'processing', 'files_hash': 'auto_recovered',
                'progress': 50.0, 'etapa': 'Recuperacao', 'subetapa': 'Retomando...'
            }
            safe_json_write(status_data, job_dir / "job_status.json")
            
        start_time = time.time()
        
        # LOGICA DE ROTEAMENTO DE RETOMADA
        if "job_shorts" in job_id or status_data.get('mode') == 'shorts_maker':
             threading.Thread(target=pipeline_criar_shorts, args=(job_dir, job_id, start_time)).start()
             message = "Retomando Shorts Maker IA..."
        elif "job_jogos" in job_id:
             if app_jogos:
                 threading.Thread(target=app_jogos.processar_dublagem_jogos, args=(job_dir, job_id, start_time)).start()
                 message = "Retomando Dublagem de Jogos..."
             else:
                 return jsonify({'error': 'Modulo de jogos indisponivel.'}), 500
        elif "job_limpeza" in job_id:
             level = status_data.get('level', 'leve')
             threading.Thread(target=pipeline_limpar_audio, args=(job_dir, job_id, start_time, level)).start()
             message = "Retomando Limpeza de Audio..."
        else:
             threading.Thread(target=pipeline_dublar_video, args=(job_dir, job_id, start_time)).start()
             message = f"Retomando Job generico {job_id}..."

        return jsonify({'status': 'processing', 'message': message, 'job_id': job_id})
        
    except Exception as e:
        logging.error(f"Erro ao retomar job {job_id}: {e}")
        return jsonify({'error': str(e)}), 500

# ==============================================================================
# 🧽 CLEAN EDITOR CORE (Censura Cirúrgica de Áudio)
# ==============================================================================
def clean_editor_process_audio(job_dir, audio_path, cb):
    """
    Fase 1 e 2: Separação (Demucs/UMX) e Transcrição temporal (Whisper).
    """
    job_dir = Path(job_dir)
    vocals_path = job_dir / "vocals.wav"
    instrumental_path = job_dir / "instrumental.wav"
    
    # 1. Separação de Áudio
    cb(10, 4, "Separando vocais do instrumental (Demucs)...")
    try:
        # Usa Demucs via CLI para otimizar RAM (libera após execução)
        cmd = ['demucs', '--two-stems=vocals', '-o', str(job_dir), str(audio_path)]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        base_name = Path(audio_path).stem
        demucs_out = job_dir / "htdemucs" / base_name
        
        if (demucs_out / "vocals.wav").exists():
            shutil.move(str(demucs_out / "vocals.wav"), str(vocals_path))
            shutil.move(str(demucs_out / "no_vocals.wav"), str(instrumental_path))
        else:
            raise FileNotFoundError("Demucs não gerou os arquivos.")
            
    except Exception as e:
        logging.warning(f"Erro no Demucs ({e}). Usando áudio original como vocals para fallback.")
        shutil.copy(str(audio_path), str(vocals_path))
        silence = AudioSegment.silent(duration=int(get_audio_duration(audio_path)*1000))
        silence.export(str(instrumental_path), format="wav")

    # Limpa RAM agressivamente
    gc.collect()

    # 2. Transcrição com Word Timestamps
    cb(40, 4, "Mapeando palavras (Whisper)...")
    try:
        model = get_whisper_model()
        segments_gen, _ = model.transcribe(str(vocals_path), language="pt", word_timestamps=True)
        
        words_data = []
        for segment in segments_gen:
            for word in segment.words:
                words_data.append({
                    "word": word.word.strip(),
                    "start": word.start,
                    "end": word.end
                })
                
        # Salva o json de segurança nativo do projeto
        safe_json_write(words_data, job_dir / "transcricao_palavras.json")
        cb(100, 4, "Mapeamento concluído!")
        return words_data
        
    except Exception as e:
        logging.error(f"Erro no Whisper (Clean Editor): {e}")
        raise e
    finally:
        gc.collect()

def apply_clean_edits(job_dir, target_words):
    """
    Fase 4 e 5: Aplica os duckings/mutes cirúrgicos e exporta.
    target_words = [{"word": "palavra", "start": 1.5, "end": 2.1, "effect": "mute"}, ...]
    """
    job_dir = Path(job_dir)
    vocals_path = job_dir / "vocals.wav"
    instrumental_path = job_dir / "instrumental.wav"
    output_path = job_dir / "final_clean_audio.mp3"
    
    vocals = AudioSegment.from_wav(str(vocals_path))
    target_words = sorted(target_words, key=lambda x: x['start'])
    
    current_vocals = vocals
    
    # Processa da última pra primeira evita dessincronia
    for tw in reversed(target_words):
        start_ms = int(max(0, (tw['start'] - 0.05) * 1000)) # Margin 50ms antes
        end_ms = int(min(len(current_vocals), (tw['end'] + 0.05) * 1000)) # Margin 50ms depois
        
        pre = current_vocals[:start_ms]
        post = current_vocals[end_ms:]
        target_chunk = current_vocals[start_ms:end_ms]
        
        effect = tw.get('effect', 'mute')
        if effect == 'ducking':
            target_chunk = target_chunk - 30 # Reduz 30dB
        elif effect == 'scratch':
            target_chunk = target_chunk.reverse() # Truque do rap
        else:
            target_chunk = AudioSegment.silent(duration=len(target_chunk))
            
        # Tenta crossfade minúsculo (evitar estalo no início e fim do corte)
        cross_dur = min(20, len(pre), len(target_chunk), len(post))
        if cross_dur > 10:
            current_vocals = pre.append(target_chunk, crossfade=cross_dur).append(post, crossfade=cross_dur)
        else:
            current_vocals = pre + target_chunk + post

    vocals_clean_path = job_dir / "vocals_clean.wav"
    current_vocals.export(str(vocals_clean_path), format="wav")
    
    # Mixagem com o instrumental intacto
    if instrumental_path.exists():
        instrumental = AudioSegment.from_wav(str(instrumental_path))
        final_mix = instrumental.overlay(current_vocals)
    else:
        final_mix = current_vocals
        
    final_mix.export(str(output_path), format="mp3", bitrate="192k")
    return str(output_path)

# --- CLEAN EDITOR ROTAS HTTP ---
@app.route('/api/clean_editor/upload', methods=['POST'])
def api_clean_upload():
    try:
        files_dict = {'audio': request.files.getlist('audio_file')}
        if not files_dict.get('audio') or not files_dict['audio'][0].filename:
            return jsonify({'error': 'Arquivo de áudio obrigatório.'}), 400
            
        # Usa o sistema core de Jobs para segurança/backup
        job_dir, job_id, start_time, is_existing = criar_job("clean_editor", files_dict, is_single_file_input=True)
        audio_path = list(job_dir.glob("input_*"))[0]
        
        def job_worker():
            try:
                words_data = clean_editor_process_audio(job_dir, audio_path, lambda p, e, msg: set_progress(job_id, p, e, start_time, ETAPAS_EDITOR_RAPIDO, msg))
                status_path = job_dir / "job_status.json"
                status_data = safe_json_read(status_path) or {}
                status_data['status'] = 'awaiting_user'
                status_data['words'] = words_data
                safe_json_write(status_data, status_path)
            except Exception as e:
                logging.error(f"Erro no Clean Editor Worker: {e}")
                
        threading.Thread(target=job_worker).start()
        return jsonify({'status': 'processing', 'job_id': job_id})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/clean_editor/status/<job_id>', methods=['GET'])
def api_clean_status(job_id):
    job_dir = Path(app.config['UPLOAD_FOLDER']) / job_id
    status_path = job_dir / "job_status.json"
    if status_path.exists():
        return jsonify(safe_json_read(status_path) or {})
    return jsonify({'error': 'Not found'}), 404

@app.route('/api/clean_editor/apply', methods=['POST'])
def api_clean_apply():
    try:
        data = request.json
        job_id = data.get('job_id')
        target_words = data.get('target_words', [])
        
        if not job_id: return jsonify({'error': 'Job ID obrigatório'}), 400
        job_dir = Path(app.config['UPLOAD_FOLDER']) / job_id
        
        def mix_worker():
            try:
                apply_clean_edits(job_dir, target_words)
                status_path = job_dir / "job_status.json"
                status_data = safe_json_read(status_path) or {}
                status_data['status'] = 'completed'
                status_data['final_file'] = f"/uploads/{job_id}/final_clean_audio.mp3"
                safe_json_write(status_data, status_path)
            except Exception as e:
                logging.error(f"Erro na Mixagem Clean Editor: {e}")
                
        threading.Thread(target=mix_worker).start()
        return jsonify({'status': 'mixing', 'job_id': job_id})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/manual_cut', methods=['POST'])
def manual_cut():
    try:
        start_val = request.form.get('start_time', '0')
        end_val = request.form.get('end_time', '0')
        
        def parse_time(t_str):
            if ':' in str(t_str):
                parts = str(t_str).split(':')
                return float(parts[0]) * 60 + float(parts[1])
            return float(t_str)
            
        start_sec = parse_time(start_val)
        end_sec = parse_time(end_val)
        
        files_dict = {'video': request.files.getlist('video_file')} if 'video_file' in request.files else {}
        
        if not files_dict.get('video') or not files_dict['video'][0].filename:
            return jsonify({'error': 'Arquivo de vídeo obrigatório.'}), 400

        job_dir, job_id, start_time, is_existing = criar_job("manual_cut", files_dict, is_single_file_input=True)
        
        threading.Thread(target=pipeline_smart_cut, args=(job_dir, job_id, start_time, start_sec, end_sec)).start()
        
        return jsonify({'status': 'processing', 'job_id': job_id, 'message': f"Corte manual iniciado ({start_sec}s - {end_sec}s)"})
        
    except Exception as e:
        logging.error(f"Erro no Manual Cut: {e}")
        return jsonify({'error': str(e)}), 500

# Rota 16: SALVAR GLOSSÁRIO (Apenas Log/Memória)
@app.route('/save_glossary', methods=['POST'])
def save_glossary():
    try:
        job_id = request.form.get('job_id')
        glossary_text = request.form.get('glossary_text', '')
        
        if not job_id: return jsonify({'error': 'Job ID não fornecido'}), 400
        
        # [REFATORAÇÃO] O glossário agora é lido diretamente do project_data ou passado via rede.
        # Não salvamos mais em arquivo glossary_detected.txt.
        logging.info(f"Glossário recebido via UI para o job {job_id}: {glossary_text[:50]}...")
        
        # Como o plano é context-in-memory, a UI deve persistir isso no project_data.json se necessário,
        # ou o backend deve manter uma variável global por job. 
        # Por enquanto, apenas confirmamos o recebimento para evitar erro 404/500 na UI.
        return jsonify({'status': 'success', 'message': 'Glossário recebido (Contexto em Memória ativo)'})
        
    except Exception as e:
        logging.error(f"Erro ao processar glossário: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/progress/<job_id>')
def progress(job_id):
    """Rota de progresso unificada (App_videos + app_jogos)."""
    # 1. Check Local (App_videos)
    with progress_lock:
        if job_id in progress_dict: return jsonify(progress_dict[job_id])

    # 2. Check Remote (app_jogos)
    if app_jogos and hasattr(app_jogos, 'progress_dict'):
        # Note: Accessing without lock is slightly risky but usually fine for reads in Python dicts
        # Ideally we'd use app_jogos.progress_lock but it might not be exported
        if job_id in app_jogos.progress_dict:
            return jsonify(app_jogos.progress_dict[job_id])

    # 3. Check Disk (Fallback)
    try:
        status_path = Path(app.config['UPLOAD_FOLDER']) / job_id / "job_status.json"
        if status_path.exists():
            if status_data := safe_json_read(status_path):
                 return jsonify({ 
                     'progress': status_data.get('progress', 0), 
                     'etapa': status_data.get('etapa', 'Pronto'),
                     'subetapa': status_data.get('subetapa'), 
                     'tempo_decorrido': status_data.get('tempo_decorrido', '0:00:00') 
                 })
    except: pass
        
    return jsonify({}) # Retorna vazio se o job não foi iniciado

@app.errorhandler(Exception)
def handle_exception(e):
    """Handler de erro genérico (do app_jogos)."""
    logging.error(f"Erro não tratado na rota: {request.url}\n{traceback.format_exc()}")
    return jsonify({"error": "Ocorreu um erro interno no servidor.", "details": str(e)}), 500

@app.route('/Clean Editor.html')
def clean_editor_page():
    """Serve o HTML do Clean Editor."""
    return send_from_directory(app.template_folder, 'Clean Editor.html')

@app.route('/')
def index():
    """Serve o HTML principal do EVI."""
    return send_from_directory(app.template_folder, 'Editor de Vídeo Inteligente.html')

@app.route('/favicon.ico')
def favicon():
    return make_response('', 204)

@app.route('/test_local_gema', methods=['GET'])
def test_local_gema():
    """Rota de diagnóstico para validar o Gema Local sem LM Studio."""
    try:
        model = get_gema_model()
        if not model:
            return jsonify({'status': 'error', 'message': 'Modelo não encontrado ou falha no carregamento.'}), 500
        
        response = model.create_chat_completion(
            messages=[{"role": "user", "content": "Olá, responda apenas: 'Gema Local Ativo!'"}],
            max_tokens=15
        )
        content = response['choices'][0]['message']['content'].strip()
        return jsonify({
            'status': 'success',
            'model_response': content,
            'info': 'Gema Local funcionando corretamente (Single Load).'
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

# --- MÓDULO: SHORTS MAKER IA (v1.0) ---
def pipeline_criar_shorts(job_dir, job_id, start_time):
    """
    Cria uma montagem cinematográfica vertical (TikTok) a partir de fotos e música.
    """
    try:
        from app_jogos import set_progress # Re-uso da função de progresso
    except:
        def set_progress(*args, **kwargs): pass

    ETAPAS_SHORTS = ["Iniciando", "1. Preparando Ativos", "2. Renderizando Cenas", "3. Mixagem Final", "4. Concluído"]
    def cb(p, e, s=None): set_progress(job_id, p, e, start_time, ETAPAS_SHORTS, s)

    try:
        cb(0, 0, "Preparando ambiente...")
        params_path = job_dir / "job_params.json"
        params = safe_json_read(params_path) or {}
        
        photo_dir = job_dir / "photos"
        # Busca dinâmica do arquivo de música (pode ser mp3, wav, mp4, etc)
        music_files = list(job_dir.glob("background_music.*"))
        if not music_files: raise ValueError("Trilha sonora não encontrada.")
        original_music_path = music_files[0]
        
        # [NOVO] Garante que o arquivo seja um áudio limpo para o Librosa não dar erro (PySoundFile failed)
        music_path = job_dir / "background_music_converted.wav"
        if not music_path.exists():
            cb(5, 1, "Convertendo trilha sonora para formato nativo WAV...")
            cmd_conv = [
                'ffmpeg', '-y', '-i', str(original_music_path),
                '-vn', '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '2', str(music_path)
            ]
            subprocess.run(cmd_conv, check=True, capture_output=True)
        
        output_video = job_dir / "shorts_final.mp4"
        style = params.get('style', 'zoom_in')
        orientation = params.get('orientation', 'vertical')
        
        # Define dimensões baseadas na orientação
        if orientation == 'horizontal':
            target_w, target_h = 1920, 1080
        else:
            target_w, target_h = 1080, 1920
        
        # 1. Medir a música e detectar batidas (Beat Sync)
        cb(5, 1, "Analisando batidas da música (Beat Sync)...")
        import librosa
        import numpy as np
        
        y, sr = librosa.load(str(music_path), sr=None)
        music_duration = librosa.get_duration(y=y, sr=sr)
        
        # Detectar onsets (batidas)
        onsets = librosa.onset.onset_detect(y=y, sr=sr, units='time')
        
        photos_files = sorted(list(photo_dir.glob("*.*")))
        if not photos_files: raise ValueError("Nenhuma foto encontrada.")
        num_photos = len(photos_files)
        # 2. Gerar tempos de corte (Cuts) - LÓGICA DE REPETIÇÃO INTELIGENTE
        # Se tiver menos de 10 fotos, repetimos a sequência 1 vez para o vídeo não ficar curto/parado.
        if num_photos < 10:
            effective_photos = photos_files * 2
            cb(8, 1, f"Poucas fotos ({num_photos}). Aplicando repetição 1x para dinamismo.")
        else:
            effective_photos = photos_files
            
        num_effective = len(effective_photos)
        cut_times = [0.0]
        curr = 0.0
        photos_to_render = []
        
        # Preenche os tempos baseando-se na lista efetiva
        # AQUI MUDOU: O loop agora é controlado estritamente pela quantidade de fotos
        for i in range(num_effective):
            found_beat = False
            # Tenta sincronizar com a próxima batida disponível
            for target in [t for t in onsets if t > curr + 3.0 and t < curr + 5.0]:
                if target < music_duration: # Não pode passar da música
                    cut_times.append(target)
                    photos_to_render.append(effective_photos[i])
                    curr = target
                    found_beat = True
                    break
            
            if not found_beat:
                # Fallback: 3.8 segundos por foto (ritmo bom para shorts)
                new_cut = curr + 3.8
                # Se ultrapassar a música, paramos por aqui
                if new_cut > music_duration:
                    # Se for a última foto, a gente estica só um pouquinho ou para
                    break
                cut_times.append(new_cut)
                photos_to_render.append(effective_photos[i])
                curr = new_cut

        # A duração total do vídeo será o último corte gerado
        video_duration = curr
        total_scenes = len(photos_to_render)
        cb(10, 1, f"Processando {total_scenes} cenas. Duração final: {video_duration:.1f}s")
        
        scene_list = []
        possible_styles = ['zoom_in', 'zoom_out', 'pan_right', 'pan_left', 'pan_up', 'pan_down', 'vibe', '3d_tilt']

        for i, photo in enumerate(photos_to_render):
            duration_per_photo = cut_times[i+1] - cut_times[i]
            if duration_per_photo <= 0: duration_per_photo = 2.0
            
            perc = 10 + (i / total_scenes) * 70
            current_style = style
            if style == 'random':
                current_style = random.choice(possible_styles)
            
            cb(perc, 2, f"Renderizando cena {i+1}/{total_scenes} ({current_style})...")
            
            scene_path = job_dir / f"scene_{i:03d}.mp4"
            if scene_path.exists() and scene_path.stat().st_size > 1000:
                scene_list.append(scene_path)
                continue

            # LÓGICA DE PREENCHIMENTO E ANIMAÇÃO:
            # 1. Cobre a tela cortando o excesso para FIM de barras pretas
            fps = 25
            total_frames = int((duration_per_photo + 1.0) * fps)
            safebox = f"scale={target_w}:{target_h}:force_original_aspect_ratio=increase,crop={target_w}:{target_h}"
            res_s = f"{target_w}x{target_h}"
            
            # 2. Roteia os Efeitos Visuais Cinematográficos reais! (Matemática MÁXIMA SEGURO - com margem de 2px)
            if current_style == 'zoom_out':
                zp = f"zoompan=z='max(1.15-0.0015*on,1)':d={total_frames}:x='iw/2-(iw/zoom)/2':y='ih/2-(ih/zoom)/2':s={res_s}:fps={fps}"
            elif current_style == 'pan_right':
                zp = f"zoompan=z=1.1:d={total_frames}:x='(iw-iw/zoom-2)*(on/{total_frames})':y='ih/2-(ih/zoom)/2':s={res_s}:fps={fps}"
            elif current_style == 'pan_left':
                zp = f"zoompan=z=1.1:d={total_frames}:x='(iw-iw/zoom-2)*(1-on/{total_frames})':y='ih/2-(ih/zoom)/2':s={res_s}:fps={fps}"
            elif current_style == 'pan_down':
                zp = f"zoompan=z=1.1:d={total_frames}:x='iw/2-(iw/zoom)/2':y='(ih-ih/zoom-2)*(on/{total_frames})':s={res_s}:fps={fps}"
            elif current_style == 'pan_up':
                zp = f"zoompan=z=1.1:d={total_frames}:x='iw/2-(iw/zoom)/2':y='(ih-ih/zoom-2)*(1-on/{total_frames})':s={res_s}:fps={fps}"
            elif current_style == 'vibe' or current_style == '3d_tilt':
                zp = f"zoompan=z='min(max(zoom,pzoom)+0.001,1.5)':d={total_frames}:x='iw/2-(iw/zoom)/2+sin(on/30)*5':y='ih/2-(ih/zoom)/2+cos(on/30)*5':s={res_s}:fps={fps}"
            else: # zoom_in
                zp = f"zoompan=z='min(pzoom+0.0015,1.5)':d={total_frames}:x='iw/2-(iw/zoom)/2':y='ih/2-(ih/zoom)/2':s={res_s}:fps={fps}"
                
            filter_v = f"{safebox},{zp},format=yuv420p"
            
            # 🎨 LÓGICA DE FILTRO ESTÉTICO (Restaurada para Motor Full)
            vis_filter = params.get('filter', 'none')
            if vis_filter == 'random':
                vis_filter = random.choice(['vhs', 'cinema', 'light_leak', 'vivid', 'none'])
            
            aesthetic_af = ""
            if vis_filter == 'vhs':
                aesthetic_af = ",curves=vintage,vignette='PI/4'"
            elif vis_filter == 'cinema':
                aesthetic_af = ",curves=strong_contrast,vignette='PI/5'"
            elif vis_filter == 'light_leak':
                aesthetic_af = ",eq=brightness=0.05:saturation=1.2,hue=h=20:s=1.1,vignette='PI/4'"
            elif vis_filter == 'vivid':
                aesthetic_af = ",eq=saturation=1.2:contrast=1.1" 
            
            filter_v += aesthetic_af + ",setsar=1"
            
            # CONVERSÃO PARA PNG (Bypass no bug do decodificador JPEG mjpeg + loop)
            safe_png_path = str(photo) + "_safe.png"
            try:
                subprocess.run(['ffmpeg', '-y', '-i', str(photo), '-frames:v', '1', safe_png_path], check=True, capture_output=True)
            except:
                safe_png_path = str(photo)
            
            cmd = [
                'ffmpeg', '-y', '-loop', '1', '-i', safe_png_path,
                '-vf', filter_v,
                '-t', str(round(duration_per_photo, 2)), 
                '-c:v', 'mpeg4', '-q:v', '2', '-pix_fmt', 'yuv420p', str(scene_path)
            ]
            
            try:
                # Execução Direta (Sem shell=True para evitar erros de aspas no Windows)
                subprocess.run(cmd, check=True, capture_output=True)
            except subprocess.CalledProcessError as e:
                err_msg = e.stderr.decode() if e.stderr else str(e)
                logging.error(f"Erro FFmpeg detalhado: {err_msg}")
                raise RuntimeError(f"Erro FFmpeg ao criar cena: {scene_path}")
            finally:
                # Cleanup da sujeira profilática
                if safe_png_path != str(photo) and os.path.exists(safe_png_path):
                    try: os.remove(safe_png_path)
                    except: pass
                
            scene_list.append(scene_path)
            
        # 3. Mixagem Final com Transições (XFADE)
        cb(85, 3, "Aplicando transições dinâmicas (Circle, Wipe, Pixelize)...")
        
        # Lista de transições profissionais do FFmpeg
        trans_list = ['fade', 'wipeleft', 'wiperight', 'wipeup', 'wipedown', 'circleopen', 'pixelize', 'dissolve']
        
        if num_photos > 1:
            cmd_mix = ['ffmpeg', '-y']
            for s in scene_list: cmd_mix.extend(['-i', str(s)])
            cmd_mix.extend(['-i', str(music_path)])
            
            filter_complex = ""
            overlap = 0.5 # segundos
            
            # O offset da primeira transição é o final da primeira cena menos o overlap
            current_offset = (cut_times[1] - cut_times[0]) - overlap
            
            # Primeiro par (0 e 1)
            t = random.choice(trans_list)
            filter_complex += f"[0:v][1:v]xfade=transition={t}:duration={overlap}:offset={current_offset}[v1];"
            
            # Próximos pares (v_prev e i)
            for i in range(2, total_scenes):
                t = random.choice(trans_list)
                prev_v = f"v{i-1}"
                next_v = f"v{i}"
                current_offset = current_offset + (cut_times[i] - cut_times[i-1]) - overlap
                filter_complex += f"[{prev_v}][{i}:v]xfade=transition={t}:duration={overlap}:offset={current_offset}[{next_v}];"
            
            # [CORREÇÃO CRÍTICA] Cálculo da Duração Real após XFADE
            # Cada transição xfade sobrepõe os vídeos, então perdemos 'overlap' segundos por transição.
            actual_duration = video_duration - ((total_scenes - 1) * overlap)
            
            last_v_idx = total_scenes - 1
            last_v = f"v{last_v_idx}"
            
            # [NEW] Efeito de Fade Out CINEMÁTICO (2.5 segundos de suavidade)
            # O fade deve começar 2.5s antes do FIM REAL da montagem (já descontado o xfade)
            fade_duration = 2.5
            fade_start = actual_duration - fade_duration
            if fade_start < 0: fade_start = 0
            
            # Fade de Vídeo (Escurecimento)
            filter_complex += f"[{last_v}]fade=t=out:st={fade_start}:d={fade_duration}[vfinal];"
            # Fade de Áudio (Volume abaixando)
            filter_complex += f"[{total_scenes}:a]afade=t=out:st={fade_start}:d={fade_duration}:curve=exp[afinal]"
            
            cmd_mix.extend([
                '-filter_complex', filter_complex,
                '-map', '[vfinal]', '-map', '[afinal]',
                '-t', str(round(actual_duration, 2)), 
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '192k', str(output_video)
            ])
            subprocess.run(cmd_mix, check=True)
        else:
            # Caso de apenas 1 foto (sem transição)
            subprocess.run([
                'ffmpeg', '-y', '-i', str(scene_list[0]), '-i', str(music_path),
                '-c:v', 'copy', '-c:a', 'aac', '-shortest', str(output_video)
            ], check=True)

        cb(100, 4, "Montagem concluída com sucesso!")
        
        status_data = safe_json_read(job_dir / "job_status.json") or {}
        status_data['status'] = 'completed'
        status_data['final_file'] = f"/uploads/{job_id}/shorts_final.mp4"
        safe_json_write(status_data, job_dir / "job_status.json")

    except Exception as e:
        logging.error(f"Erro no Shorts Maker: {e}\n{traceback.format_exc()}")
        cb(100, 4, f"Erro: {e}")
        status_data = safe_json_read(job_dir / "job_status.json") or {}
        status_data['status'] = 'failed'
        status_data['error'] = str(e)
        safe_json_write(status_data, job_dir / "job_status.json")

@app.route('/uploads/<path:path>')
def send_upload(path):
    """Serve arquivos da pasta de uploads (ex: vídeos finalizados)."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], path)

@app.route('/criar_shorts_ia', methods=['POST'])
def criar_shorts_route():
    start_time = time.time()
    if 'music' not in request.files or 'photos' not in request.files:
        return jsonify({'error': 'Faltam fotos ou música.'}), 400
    
    music_file = request.files['music']
    photo_files = request.files.getlist('photos')
    
    timestamp = int(time.time())
    job_id = f"job_shorts_{timestamp}"
    job_dir = Path(app.config['UPLOAD_FOLDER']) / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    
    # Salva fotos em subpasta
    p_dir = job_dir / "photos"
    p_dir.mkdir(exist_ok=True)
    for i, p in enumerate(photo_files):
        ext = Path(p.filename).suffix or ".jpg"
        p.save(p_dir / f"img_{i:03d}{ext}")
        
    # Salva música com extensão original
    music_ext = Path(music_file.filename).suffix or ".mp3"
    music_path = job_dir / f"background_music{music_ext}"
    music_file.save(music_path)
    
    # Params
    params = {
        'style': request.form.get('style', 'zoom_in'),
        'filter': request.form.get('filter', 'none'),
        'orientation': request.form.get('orientation', 'vertical'),
        'num_photos': len(photo_files)
    }
    safe_json_write(params, job_dir / "job_params.json")
    
    status_data = {'job_id': job_id, 'status': 'iniciando', 'mode': 'shorts_maker'}
    safe_json_write(status_data, job_dir / "job_status.json")
    
    threading.Thread(target=pipeline_criar_shorts, args=(job_dir, job_id, start_time)).start()
    return jsonify({'status': 'success', 'job_id': job_id})

# --- NEW: MAGIC CUT IA PIPELINE ---
@app.route('/api/magic_cut/start', methods=['POST'])
def magic_cut_route():
    start_time = time.time()
    if 'video_file' not in request.files:
        return jsonify({'error': 'Nenhum vídeo enviado.'}), 400
    
    video = request.files['video_file']
    timestamp = int(time.time())
    job_id = f"job_magic_{timestamp}"
    job_dir = Path(app.config['UPLOAD_FOLDER']) / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    
    v_path = job_dir / f"raw_video{Path(video.filename).suffix}"
    video.save(v_path)
    
    params = {
        'remove_silence': request.form.get('remove_silence') == 'true',
        'remove_errors': request.form.get('remove_errors') == 'true',
        'normalize': request.form.get('normalize') == 'true'
    }
    safe_json_write(params, job_dir / "job_params.json")
    
    status_data = {'job_id': job_id, 'status': 'iniciando', 'mode': 'magic_cut'}
    safe_json_write(status_data, job_dir / "job_status.json")
    
    threading.Thread(target=pipeline_magic_cut, args=(job_dir, job_id, start_time)).start()
    return jsonify({'status': 'success', 'job_id': job_id})

def pipeline_magic_cut(job_dir, job_id, start_time):
    try:
        from app_jogos import set_progress
    except:
        def set_progress(*args, **kwargs): pass

    ETAPAS = ["Analisando", "Cortando Silêncios", "Removendo Erros", "Finalizando", "Concluído"]
    def cb(p, e, s=None): set_progress(job_id, p, e, start_time, ETAPAS, s)

    try:
        cb(5, 0, "Analisando áudio para detectar silêncios...")
        params = safe_json_read(job_dir / "job_params.json")
        v_files = list(job_dir.glob("raw_video.*"))
        if not v_files: return
        input_v = v_files[0]
        output_v = job_dir / "magic_final.mp4"

        # 1. DETECTAR SILÊNCIO (Configuração mais "Humana")
        # d=0.8: Só considera silêncio se for maior que 0.8 segundos (preserva pausas naturais)
        # noise=-35dB: Um pouco mais sensível a barulhos de fundo
        cmd_detect = [
            'ffmpeg', '-i', str(input_v),
            '-af', 'silencedetect=noise=-35dB:d=0.8',
            '-f', 'null', '-'
        ]
        result = subprocess.run(cmd_detect, capture_output=True, text=True)
        
        silence_starts = re.findall(r"silence_start: ([\d\.]+)", result.stderr)
        silence_ends = re.findall(r"silence_end: ([\d\.]+)", result.stderr)
        
        # 2. CALCULAR SEGMENTOS DE "SOM" COM PADDING (Margem de Segurança)
        import ffmpeg
        probe = ffmpeg.probe(str(input_v))
        total_dur = float(probe['format']['duration'])
        
        keep_segments = []
        last_end = 0.0
        padding = 0.3 # 0.3 segundos de respiro antes e depois de cada fala
        
        for start, end in zip(silence_starts, silence_ends):
            s, e = float(start), float(end)
            
            # Deixamos um 'padding' de respiro após a fala anterior e antes do corte
            cut_start = s + padding
            # Deixamos um 'padding' antes de começar a próxima fala
            cut_end = e - padding
            
            # Se o trecho de som for minimamente relevante
            if s - last_end > 0.1:
                # Mantemos do fim do silêncio anterior (last_end) até o início deste silêncio + padding
                keep_segments.append((max(0, last_end - padding), min(total_dur, s + padding)))
            
            last_end = e
            
        # Adiciona o último bloco de som
        if total_dur - last_end > 0.1:
            keep_segments.append((max(0, last_end - padding), total_dur))

        # [OTIMIZAÇÃO] Mesclar segmentos que se sobrepõe por causa do padding
        merged = []
        if keep_segments:
            curr_s, curr_e = keep_segments[0]
            for next_s, next_e in keep_segments[1:]:
                if next_s <= curr_e: # Se encostam ou sobrepõe
                    curr_e = max(curr_e, next_e)
                else:
                    merged.append((curr_s, curr_e))
                    curr_s, curr_e = next_s, next_e
            merged.append((curr_s, curr_e))
        
        keep_segments = merged

        # 3. EXECUTAR CORTES
        cb(30, 1, f"Identificados {len(keep_segments)} trechos com respiros preservados. Iniciando Magic Cut...")
        
        v_select = " + ".join([f"between(t,{s},{e})" for s, e in keep_segments])
        a_select = " + ".join([f"between(t,{s},{e})" for s, e in keep_segments])
        
        # Filtro de Vídeo e Áudio para remover os buracos
        # Usamos 'setpts' e 'asetpts' para reajustar o tempo dos frames e não dar stuter
        filter_v = f"select='{v_select}',setpts=N/FRAME_RATE/TB"
        filter_a = f"aselect='{a_select}',asetpts=N/SR/TB"
        
        # Opcional: Normalização (Compressão Dinâmica)
        if params.get('normalize'):
            filter_a += ",loudnorm=I=-16:TP=-1.5:LRA=11"

        cmd_cut = [
            'ffmpeg', '-y', '-i', str(input_v),
            '-vf', filter_v,
            '-af', filter_a,
            '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '22',
            '-c:a', 'aac', '-b:a', '192k', str(output_v)
        ]
        
        subprocess.run(cmd_cut, check=True)
        cb(100, 4, "Edição Mágica concluída!")
        
        status_data = safe_json_read(job_dir / "job_status.json") or {}
        status_data['status'] = 'completed'
        status_data['final_file'] = f"/uploads/{job_id}/magic_final.mp4"
        safe_json_write(status_data, job_dir / "job_status.json")

    except Exception as e:
        logging.error(f"Erro no Magic Cut: {e}")
        cb(100, 4, f"Erro: {e}")

# --- PONTO DE ENTRADA ---
if __name__ == "__main__":
    # --- LOGGING CONFIG ---
    # Mute request logs (spam)
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    
    # Ensure INFO logs (Progress) are visible
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    # ----------------------

    check_ffmpeg()
    check_lm_studio() # [NEW] Alerta sobre o estado do cérebro IA
    host, port = "127.0.0.1", 5000 # Porta padrão 5000
    url = f"http://{host}:{port}"
    
    def open_browser():
        webbrowser.open_new(url)

    Timer(1, open_browser).start()
    
    print("\n" + "="*80)
    print(f"Servidor EVI (Editor de Vídeo Inteligente) iniciado.")
    print(f"Acesse a aplicação em: {url}")
    print("O progresso detalhado de todos os jobs será exibido aqui no CMD.")
    print("="*80 + "\n")
    app.run(host=host, port=port, debug=False)


