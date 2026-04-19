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

import requests
import json
import logging
import warnings
# [v2026.10 FIX] Esconde avisos chatos do SpeechBrain que parecem erros mas não são
warnings.filterwarnings("ignore", category=UserWarning, module="speechbrain")
warnings.filterwarnings("ignore", message="Module 'speechbrain.*' was deprecated")
import random
import re
import os
import sys
import time
import subprocess
import threading
import torch
import gc
import traceback
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock, Timer, Thread
import hashlib
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
import zipfile # [NOVO] Para upload de lotes grandes
import re
import webbrowser
import stat
import random
import numpy as np
from enum import Enum
from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.cluster import AgglomerativeClustering # [NOVO] Para clustering fixo

from flask import Flask, send_from_directory, request, jsonify, make_response
try:
    from flask_cors import CORS # [NEW] Suporte a Cross-Origin
    HAS_CORS = True
except ImportError:
    HAS_CORS = False
    print("[AVISO] flask_cors não instalado. O painel web pode ter problemas de acesso se rodar em portas diferentes.")
    print("Para corrigir, use: pip install flask-cors")
from pydub.silence import split_on_silence

try:
    from llama_cpp import Llama
    HAS_LLAMA_CPP = True
except ImportError:
    HAS_LLAMA_CPP = False
    logging.warning("llama-cpp-python não instalado. O sistema usará LM Studio para IA.")

# --- GLOSSÁRIO FONÉTICO GLOBAL ---
PHONETIC_CORRECTIONS = {}

def corrigir_sotaque_pt_br(texto):
    """
    Normaliza o texto para o motor TTS.
    Converte números para extenso em PT-BR para evitar leitura em inglês.
    O motor Multilíngue cuida de termos em inglês (ex: upload) nativamente.
    """
    if not texto: return ""
    
    import re
    try:
        from num2words import num2words
        # Encontra números no texto
        padrao_nums = re.compile(r'\b\d+([.,]\d+)?\b')
        
        def substituir_num(match):
            num_str = match.group(0).replace(',', '.')
            try:
                val = float(num_str)
                return num2words(val, lang='pt_BR')
            except:
                return num_str
        
        texto_final = padrao_nums.sub(substituir_num, texto)
        return texto_final
    except:
        # Se falhar/não tiver num2words, retorna o original sem travar
        return texto

# --- CONFIGURAÇÕES DE AMBIENTE (OFFLINE-FIRST) ---
os.environ["HF_HUB_OFFLINE"] = "0"        # [FIX] Permitir download inicial de modelos
os.environ["TRANSFORMERS_OFFLINE"] = "0"  # [FIX] Permitir download inicial de modelos
os.environ["SPEECHBRAIN_FETCH_STRATEGY"] = "COPY"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["COQUI_TOS_AGREED"] = "1"

# --- CONFIGURAÇÃO DE LOGGING (DEBUG MODE) ---
import sys
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def log_uncaught(exctype, value, tb):
    logging.critical("ERRO NÃO TRATADO (CRASH):", exc_info=(exctype, value, tb))

sys.excepthook = log_uncaught

# --- SILENCE NOISY LOGGERS ---
logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('torchaudio').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)

# PATCH: Correção para Torchaudio movida para uso local
# try:
#     import torchaudio
#     if not hasattr(torchaudio, 'list_audio_backends'):
#         def _list_audio_backends():
#             return ['soundfile'] 
#         torchaudio.list_audio_backends = _list_audio_backends
# except ImportError:
#     pass

try:
    from pydub import AudioSegment, effects # [NEW] Effects para speedup
    from werkzeug.utils import secure_filename
    from werkzeug.utils import secure_filename
    # from faster_whisper import WhisperModel
    import requests
    # import torch
    import psutil
    # from TTS.api import TTS
    # from TTS.tts.configs.Chatterbox_config import ChatterboxConfig
    # from TTS.tts.models.Chatterbox import ChatterboxAudioConfig, ChatterboxArgs
    # from TTS.config.shared_configs import BaseDatasetConfig
    # from TTS.tts.layers.Chatterbox.tokenizer import VoiceBpeTokenizer
    from collections import OrderedDict
    # import torch.serialization
except ImportError as e:
    logging.critical(f"Erro: Dependência essencial não encontrada - {e}.")
    logging.critical("Certifique-se de que todas as dependências estão instaladas corretamente.")
    sys.exit(1)

# --- [v12.75] VERIFICAÇÃO DE DEPENDÊNCIAS CRÍTICAS ---
def check_ffmpeg():
    """Tenta localizar o FFmpeg Full (com suporte a MP3/Lame)."""
    # [v2026 FIX] Prioridade total para a pasta local onde o usuário deve colocar o Full
    local_full_bin = os.path.join(os.getcwd(), 'env', 'Library', 'bin', 'ffmpeg.exe')
    
    if os.path.exists(local_full_bin):
        logging.info(f"FFmpeg FULL detectado na pasta local: {local_full_bin}")
        os.environ["PATH"] = os.path.dirname(local_full_bin) + os.pathsep + os.environ["PATH"]
        return True

    try:
        # Verifica se o ffmpeg do PATH tem suporte a libmp3lame
        output = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, shell=True)
        if 'libmp3lame' in output.stdout:
            logging.info("FFmpeg (Full c/ MP3) encontrado no PATH.")
            return True
        else:
            logging.warning("⚠️ FFmpeg do PATH não tem suporte a MP3 (libmp3lame).")
    except:
        pass

    logging.error("❌ ERRO CRÍTICO: FFmpeg FULL não encontrado!")
    print("\n" + "!"*60)
    print("ERRO: Sua versão do FFmpeg é limitada e não suporta MP3.")
    print("Para resolver:")
    print("1. Baixe o 'FFmpeg Full' no Gyan.dev")
    print("2. Extraia o ffmpeg.exe para: env\\Library\\bin\\")
    print("Siga o manual: MANUAL_FFMPEG_FULL.md")
    print("!"*60 + "\n")
    return False

def check_lm_studio():
    """Verifica se o modelo GGUF local existe, já que não usamos mais o LM Studio externo."""
    model_path = Path("./Models/gemma-4-E4B-it-Q4_K_M.gguf")
    if model_path.exists():
        logging.info(f"Cérebro IA (Gemma 4) detectado localmente: {model_path}")
        return True
    else:
        logging.warning(f"⚠️ AVISO: Modelo GGUF não encontrado em {model_path}")
        logging.warning("O 'Cérebro IA' (Gema Local) não funcionará. Coloque o arquivo .gguf na pasta Models.")
        return False

# --- [v12.70] DICIONÁRIO DE PERFIS DE JOGO (IA + ÁUDIO) ---
# Adicione novos jogos aqui para otimizar Tradução, Glossário e Som.
GAME_PROFILES = {
    "padrao": {
        "name": "Estilo Padrão",
        "ai_instructions": "Estilo: Localização profissional e orgânica (PT-BR). Fuja de traduções literais robóticas. Priorize como um brasileiro falaria naturalmente naquela situação (use gírias de games/combate se necessário). A intenção da fala e o impacto emocional são mais importantes que as palavras exatas.",
        "glossary": {},
        "audio_settings": {
            "loudnorm": "I=-16:TP=-1.5:LRA=11",
            "volume_boost_default": 0
        }
    },
    "cod": {
        "name": "Call of Duty (MW3 Style)",
        "ai_instructions": "Estilo: Militar e Adrenalina. Foco no desespero de combate. Mantenha APENAS nomes próprios como Frost, Soap, Price, Ghost, Task Force 141 e Delta Force em Inglês. Callsigns como 'Metal 04' devem ser mantidos. TRADUZA TODO O RESTO para o Português (ex: 'upload' vira 'envio' ou 'carregamento', 'checkpoint' vira 'ponto de controle', 'copy that' vira 'entendido', 'roger' vira 'na escuta'). NUNCA suavize fatalidades: 'KIA' deve ser 'Abatidos'. O tom deve ser seco e profissional.",
        "glossary": {"Frost": "Frost", "Soap": "Soap", "Price": "Price", "Ghost": "Ghost"},
        "audio_settings": {
            "loudnorm": "I=-10:TP=-0.5:LRA=11",
            "acompressor": "threshold=-18dB:ratio=4:attack=5:release=50:makeup=2",
            "bass": "g=3:f=100[bassout];[bassout]treble=g=2:f=3500",
            "volume_boost_default": 10.0
        }
    },
    "bioshock": {
        "name": "BioShock (Dystopian 50s)",
        "ai_instructions": "Estilo: Retro-Futurista e Sombrio. Narrativa teatral e densa. Mantenha nomes como Andrew Ryan, Fontaine e Little Sisters.",
        "glossary": {"Andrew Ryan": "Andrew Ryan", "Fontaine": "Fontaine"},
        "audio_settings": {
            "loudnorm": "I=-14:TP=-1.0:LRA=11",
            "volume_boost_default": 12.0
        }
    },
    "rpg": {
        "name": "RPG (Natural/Medieval)",
        "ai_instructions": "Estilo: Imersivo e Épico. Diálogos naturais para Fantasia/Aventura (Ex: Witcher).",
        "glossary": {"Geralt": "Geralt", "Yennefer": "Yennefer"},
        "audio_settings": {
            "loudnorm": "I=-20:TP=-1.5:LRA=7",
            "volume_boost_default": 4.0
        }
    },
    "xcom": {
        "name": "The Bureau: XCOM Declassified",
        "ai_instructions": "Estilo: Anos 1960, Invasão Alienígena e Investigação de Agentes Especiais. O tom deve ser tático, mais formal e com suspense de Guerra Fria. Mantenha gírias de época e formalidade militar onde adequado.",
        "glossary": {
            "The Bureau": "The Bureau",
            "Carter": "Carter",
            "Outsider": "Forasteiro",
            "Sleepwalker": "Sonâmbulo",
            "Sectoid": "Sectoid",
            "Muton": "Muton"
        },
        "audio_settings": {
            "loudnorm": "I=-16:TP=-1.5:LRA=11",
            "volume_boost_default": 0.0
        }
    },
    "state_of_decay": {
        "name": "State of Decay (Survival Style)",
        "ai_instructions": "Estilo: Apocalipse Zumbi e Sobrevivência. O tom deve ser de cansaço, tensão constante e urgência. Use gírias de sobreviventes. Mantenha termos como 'Zeds', 'Ferals', 'Screamers' e 'Juggernauts' se o contexto pedir, ou use traduções consagradas (ex: Zumbis, Selvagens, Gritadores).",
        "glossary": {
            "Zeds": "Zeds",
            "Feral": "Selvagem",
            "Screamer": "Gritador",
            "Bloater": "Inchado",
            "Juggernaut": "Juggernaut",
            "Infestation": "Infestação"
        },
        "audio_settings": {
            "loudnorm": "I=-12:TP=-1.0:LRA=11",
            "acompressor": "threshold=-20dB:ratio=3:attack=5:release=50",
            "volume_boost_default": 8.0
        }
    }
}

def load_game_profile(profile_id):
    """
    [v12.70] Carrega as configurações de IA e Som de um perfil específico.
    """
    return GAME_PROFILES.get(profile_id, GAME_PROFILES.get('padrao'))

# --- FUNÇÃO DE PRÉ-PROCESSAMENTO DE ÁUDIO (NOVO) ---
def preprocess_audio_for_diarization(input_path, output_path):
    """
    Aplica tratamento de áudio.
    1. Tenta DeepFilterNet (OBRIGATÓRIO para redução de ruído).
    2. Se não tiver, APENAS normaliza (dynaudnorm) sem filtros destrutivos.
    """
    try:
        # [REQ] Tenta importar DeepFilterNet (Melhor qualidade)
        import librosa
        
        y_check, sr_check = librosa.load(str(input_path), sr=16000, duration=5.0)
        rms = librosa.feature.rms(y=y_check)[0]
        mean_rms = np.mean(rms)
        
        # [ESTRATÉGIA DE ATIVAÇÃO CONDICIONAL - JOGOS (Nexus)]
        if mean_rms > 0.025: # Jogo tolera mais efeitos de rádio e sujeira. Threshold mais agressivo.
             logging.info(f"RMS: {mean_rms:.3f}. Ruído pesado detectado. Aplicando DeepFilterNet.")
             from df.enhance import enhance, init_df, load_audio, save_audio
             model, df_state, _ = init_df()
             audio, _ = load_audio(input_path, sr=df_state.sr())
             enhanced = enhance(model, df_state, audio)
             save_audio(output_path, enhanced, df_state.sr())
             return True
        else:
             logging.info(f"RMS: {mean_rms:.3f}. Som limpo/intencional (Rádio/Eco). Bypass DeepFilterNet.")
             raise ImportError("Bypass condicional DeepFilterNet via librosa") # Pula direto pro Fallback (Dynaudnorm conservador)
             
    except ImportError as e:
        if "Bypass condicional" not in str(e):
             logging.warning("="*60)
             logging.warning("[AVISO] DeepFilterNet não encontrado!")
             logging.warning("Instale com: pip install deepfilternet")
             logging.warning("="*60)
        logging.warning("O áudio será apenas normalizado, SEM redução de ruído.")
        logging.warning("Instale com: pip install deepfilternet")
        logging.warning("="*60)
        
        # Fallback: APENAS Normalização (Sem filtros destrutivos do FFmpeg)
        # Respeitando pedido do usuário para não usar 'tapa-buraco'
        try:
            threads = str(max(1, (os.cpu_count() or 4) // 2))
            
            # Apenas converte para 16k mono e normaliza volume
            # SEM highpass/lowpass/afftdn
            af_filter = "dynaudnorm=f=150:g=15" 
            
            cmd = [
                'ffmpeg', '-threads', threads, '-y', 
                '-i', str(input_path),
                '-af', af_filter,
                '-ar', '16000', 
                '-ac', '1',     
                str(output_path)
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            return True
        except:
             # Se até o FFmpeg falhar, copia
             pass

    except Exception as e:
        logging.error(f"Erro no pré-processamento (DeepFilterNet): {e}")

    # Fallback final: Copia o original
    try:
        shutil.copy(str(input_path), str(output_path))
    except: pass
    return False

    return False

# --- DIARIZAÇÃO INTELIGENTE (v12.0 OFFLINE) ---
class SimpleDiarizer:
    def __init__(self, source="speechbrain/spkrec-ecapa-voxceleb", device="cpu"):
        save_dir = os.path.join(os.path.expanduser("~"), ".cache", "speechbrain_models")
        try:
            from speechbrain.inference.speaker import EncoderClassifier
            # Tenta carregar do cache local PRIMEIRO (Offline First)
            self.encoder = EncoderClassifier.from_hparams(
                source=source, 
                run_opts={"device": device},
                savedir=save_dir,
                local_files_only=True
            )
            self.device = device
            logging.info("✅ Diarizador: Carregamento Offline concluído.")
        except Exception:
            logging.info(f"📡 Diarizador {source} não encontrado localmente. Tentando conectar...")
            try:
                from speechbrain.inference.speaker import EncoderClassifier
                self.encoder = EncoderClassifier.from_hparams(
                    source=source, 
                    run_opts={"device": device},
                    savedir=save_dir,
                    local_files_only=False
                )
                self.device = device
            except:
                raise RuntimeError("\n[ERRO CRÍTICO] Modelo de Diarização não encontrado!\n"
                                 "Você precisa conectar à internet UMA VEZ para baixar os modelos base.")

    def detect_splits_surgical(self, audio_path):
        """
        [v10.60 SURGICAL VAD SPLIT]
        Detecta silêncios primeiro (Pydub) e analisa se houve troca de voz entre os blocos de fala.
        Garante que nunca corte no meio de uma palavra.
        """
        import torchaudio
        from scipy.spatial.distance import cosine
        from pydub import AudioSegment
        from pydub.silence import detect_nonsilent
        
        sound = AudioSegment.from_wav(str(audio_path))
        # Detecta trechos de fala (VAD simples mas eficaz)
        # min_silence_len=300ms para garantir que é um espaço entre frases ou palavras longas
        nonsilent_ranges = detect_nonsilent(sound, min_silence_len=300, silence_thresh=-40)
        
        if len(nonsilent_ranges) < 2: return []
        
        signal, fs = torchaudio.load(str(audio_path))
        if signal.shape[0] > 1: signal = signal.mean(dim=0, keepdim=True)
        if fs != 16000:
             resampler = torchaudio.transforms.Resample(fs, 16000)
             signal = resampler(signal)
             fs = 16000
        
        embeddings = []
        valid_ranges = []
        
        for start_ms, end_ms in nonsilent_ranges:
            # Converte ms para samples (16kHz)
            s_start = int((start_ms / 1000.0) * fs)
            s_end = int((end_ms / 1000.0) * fs)
            
            # Se o trecho for muito curto (<meio segundo), ignoramos para estabilidade do embedding
            if (end_ms - start_ms) < 500: continue
            
            try:
                # Extrai embedding do bloco de fala inteiro
                chunk = signal[:, s_start:s_end]
                emb = self.encoder.encode_batch(chunk)
                embeddings.append(emb.squeeze().cpu().numpy())
                valid_ranges.append((start_ms, end_ms))
            except: continue
            
        if len(embeddings) < 2: return []
        
        # [v21.20] SENSIBILIDADE RECALIBRADA
        # Aumentado de 0.5 para 0.8 para evitar "falsos positivos" em frases curtas.
        # Só corta se a diferença de voz for gritante.
        splits_ms = []
        for i in range(len(embeddings) - 1):
            dist = cosine(embeddings[i], embeddings[i+1])
            if dist > 0.8: # Mudança de voz entre blocos detectada (Exige Certeza Absoluta)
                # O ponto de corte é no meio do silêncio entre os dois blocos
                silence_start = valid_ranges[i][1]
                silence_end = valid_ranges[i+1][0]
                split_point = (silence_start + silence_end) / 2
                splits_ms.append(split_point / 1000.0) # Retorna em segundos
        
        return splits_ms

def split_audio_by_speaker(audio_path, job_dir):
    """
    Analisa se houve troca de voz e divide o arquivo usando VAD Cirúrgico.
    """
    try:
        from pydub import AudioSegment
        # [v21.20] TRAVA DE SEGURANÇA: Áudios curtos (<6s) não devem ser picotados.
        # Geralmente são interações simples onde o Whisper acertou o grupo.
        duration = get_audio_duration(audio_path)
        if duration < 6.0:
            return False
            
        # Inicializa o diarizador (CPU para evitar conflito com generation se ocorrer em paralelo)
        diarizer = SimpleDiarizer(device="cpu")
        splits = diarizer.detect_splits_surgical(audio_path)
        
        if not splits: return False
        
        logging.info(f"Diarização Cirúrgica v10.60: detectadas {len(splits)} trocas de voz em '{audio_path.name}'.")
        sound = AudioSegment.from_wav(str(audio_path))
        
        # Corte real (Pontos em MS)
        points = [0] + [int(s * 1000) for s in splits] + [len(sound)]
        for i in range(len(points) - 1):
            start, end = points[i], points[i+1]
            if end - start < 300: continue # Ignora micro-cortes
            chunk = sound[start:end]
            # Exporta DIRETAMENTE na raiz da pasta (sem criar subpastas de _segmentos)
            chunk.export(audio_path.parent / f"{audio_path.stem}_p{i+1:02d}.wav", format="wav")
            
        # Move original para backup
        backup_dir = job_dir / "_0_ORIGINAIS_BACKUP"
        backup_dir.mkdir(exist_ok=True)
        shutil.move(str(audio_path), str(backup_dir / audio_path.name))
        return True
    except Exception as e:
        logging.error(f"Falha na Diarização Cirúrgica v10.60: {e}")
        return False

# --- FUNÇÕES DE ÁUDIO AUXILIARES ---

def speedup_audio(audio_segment, speed_factor):
    """
    Acelera o áudio sem alterar o pitch (Time Stretch).
    Usa pydub.effects.speedup (que usa algoritmo granulado simples).
    Para fatores > 1.0 (aceleração).
    """
    if speed_factor <= 1.0: return audio_segment
    
    # [SAFEGUARD] Limite de segurança para evitar "Esquilos"
    try:
        # chunk_size e crossfade ajustados para fala (reduz artefatos metálicos)
        return effects.speedup(audio_segment, playback_speed=speed_factor, chunk_size=150, crossfade=25)
    except Exception as e:
        logging.warning(f"Falha ao acelerar áudio (Fator {speed_factor}): {e}")
        return audio_segment

def consolidate_speaker_segments(job_dir, project_data, cb, etapa_idx):
    """
    [NEW] Consolidação Inteligente de Oradores (Portado do App_videos):
    1. Separa oradores em 'Válidos' (>10s) e 'Questionáveis' (<10s).
    2. Se houver Válidos, compara Questionáveis com eles via Embeddings.
    3. Se similaridade > 0.6 (Threshold), funde (merge).
    """
    logging.info("Iniciando Consolidação de Oradores...")
    
    voices_dir = job_dir / "voices" # Em jogos, as vozes ficam em subpastas? Não, aqui no jogos o fluxo é diferente.
    # Em jogos, as vozes já estão separadas em pastas no _2_PARA_AS_PASTAS_DE_VOZ ou similar.
    # Mas esta função é para limpar o JSON (project_data) baseada em referências.
    # NO APP_JOGOS: A estrutura é diferente. As vozes são pastas.
    # A função `unify_speaker_files` já faz algo parecido (merge de pastas).
    # ENTÃO: Talvez não precisemos de `consolidate_speaker_segments` exatamente como no App_videos,
    # mas sim melhorar a `unify_speaker_files`!
    
    # VOU ABORTAR A INSERÇÃO DESTA FUNÇÃO AQUI E MELHORAR A UNIFY_SPEAKER_FILES NA PRÓXIMA ETAPA.
    return project_data

# --- SEPARAÇÃO DE VOCAL E FUNDO (OPENUNMIX) ---
def run_openunmix_batch(source_dir, job_dir, cb):
    logging.info("Iniciando separação de áudio com OpenUnmix...")
    
    stem_vocal_dir = job_dir / "_0a_SEPARACAO_VOCAL"
    stem_bg_dir = job_dir / "_0b_SEPARACAO_FUNDO"
    stem_vocal_dir.mkdir(parents=True, exist_ok=True)
    stem_bg_dir.mkdir(parents=True, exist_ok=True)
    
    files = list(source_dir.rglob("*.wav"))
    if not files: return
    
    # Verifica se já processou tudo
    all_processed = True
    for file_path in files:
        rel_path = file_path.relative_to(source_dir)
        vocal_out = stem_vocal_dir / rel_path
        bg_out = stem_bg_dir / rel_path
        if not (vocal_out.exists() and bg_out.exists()):
            all_processed = False
            break
            
    if all_processed:
        logging.info("Separação OpenUnmix já realizada (cache encontrado).")
        cb(100, 1, "Separação já concluída (Cache).")
        return

    try:
        import torch
        import torchaudio
        from openunmix import predict
    except ImportError:
        logging.warning("OpenUnmix não encontrado. Tentando instalar...")
        try:
             cb(0, 1, "Instalando OpenUnmix (pode demorar)...")
             subprocess.check_call([sys.executable, "-m", "pip", "install", "openunmix", "torchaudio", "scikit-learn"])
             import torch
             import torchaudio
             from openunmix import predict
        except Exception as e:
             logging.error(f"Falha ao instalar OpenUnmix: {e}")
             cb(100, 1, "Erro: OpenUnmix não instalado.")
             return

    # Check CUDA
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"OpenUnmix usando dispositivo: {device}")
    
    for i, file_path in enumerate(files):
        cb((i / len(files)) * 100, 1, f"Separando Fundo/Voz: {file_path.name}")
        
        rel_path = file_path.relative_to(source_dir)
        vocal_out = stem_vocal_dir / rel_path
        bg_out = stem_bg_dir / rel_path
        
        vocal_out.parent.mkdir(parents=True, exist_ok=True)
        bg_out.parent.mkdir(parents=True, exist_ok=True)
        
        if vocal_out.exists() and bg_out.exists():
            continue

        try:
            # Load Audio (resample to 44.1k required by UMX usually, but let's try direct)
            # UMX models are trained on 44.1kHz.
            audio, rate = torchaudio.load(file_path)
            if rate != 44100:
                resampler = torchaudio.transforms.Resample(rate, 44100)
                audio = resampler(audio)
                rate = 44100

            # Predict
            # estimates = predict.separate(audio, rate=rate, device=device)
            # Retorna numpy arrays ou tensores dependendo da versao.
            # A versao mais recente retorna um dicionario de tensores.
            estimates = predict.separate(
                audio[None], rate=rate, targets=['vocals', 'drums', 'bass', 'other'], residual=True, device=device
            )
            
            # Extract
            # estimates['vocals'] shape: (posteriores, channels, time) -> [0]
            vocal_audio = estimates['vocals'][0].cpu()
            
            # Background (Sum rest)
            # drums, bass, other might be None if not requested? But separate returns all targets by default or if requested.
            bg_audio = estimates['drums'][0] + estimates['bass'][0] + estimates['other'][0]
            bg_audio = bg_audio.cpu()
            
            # Save
            torchaudio.save(str(vocal_out), vocal_audio, rate)
            torchaudio.save(str(bg_out), bg_audio, rate)
            
        except Exception as e:
            logging.error(f"Erro OpenUnmix em {file_path.name}: {e}")
            # Fallback: Copy original to vocal (treat as if no separation happened)
            shutil.copy(file_path, vocal_out)
            # Create silence for BG (or copy original? No, silence is safer to avoid double stacking)
            # Mas se a pessoa quer preservar o fundo, talvez copiar original pro BG?
            # Se deu erro no unmix, melhor copiar original pro vocal e deixar BG vazio.
            # Assim o DeepFilter limpa o vocal (que é o original sujo) e o BG fica mudo (sem duplicação de ruído).

    cb(100, 1, "Separação de áudio concluída.")

def run_batch_cleaning(source_dir, dest_dir, cb):
    """
    Executa limpeza de áudio em LOTE usando DeepFilterNet.
    Lê de source_dir, salva em dest_dir.
    PULA arquivos que já existem em dest_dir (Cache).
    """
    if not source_dir.exists(): return
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    files = list(source_dir.rglob("*.wav"))
    if not files: return
    
    # Filtra apenas o que precisa ser processado
    to_process = []
    for f in files:
        dest_path = dest_dir / f.name
        if not dest_path.exists():
            to_process.append(f)
            
    if not to_process:
        cb(0, 1, "Todos os arquivos já estão limpos (Cache). Pulando DeepFilterNet.")
        return True

    cb(0, 1, f"Inicializando DeepFilterNet para limpar {len(to_process)} novos arquivos...")
    
    try:
        from df.enhance import enhance, init_df, load_audio, save_audio
        
        # Carrega modelo UMA VEZ
        model, df_state, _ = init_df()
        logging.info("Modelo DeepFilterNet carregado na memória.")
        
        total = len(to_process)
        success_count = 0
        
        for i, f in enumerate(to_process):
            dest_path = dest_dir / f.name
            try:
                # Carrega áudio
                audio, _ = load_audio(str(f), sr=df_state.sr())
                
                # [VAD/GATING FIX] Limita a atenuação a 18dB para evitar que a voz "suma" 
                # subitamente quando o fundo de rádio for muito forte.
                try:
                    enhanced = enhance(model, df_state, audio, atten_lim_db=18.0)
                except TypeError:
                    enhanced = enhance(model, df_state, audio)
                
                # [v10.65 CLEAN CLONE UPDATE]
                # Anteriormente usávamos 50/50 mix para manter ambiência, mas isso causa alucinações (eeee) no TTS.
                # Agora usamos 100% Clean para garantir que a referência de clonagem seja pura.
                # enhanced = (enhanced * 0.50) + (audio * 0.50) # [DEPRECATED v10.65]
                save_audio(str(dest_path), enhanced, df_state.sr()) 
                
                success_count += 1
                if i % 5 == 0: # Atualiza UI a cada 5
                    cb((i / total) * 100, 1, f"Limpando: {f.name} ({i+1}/{total})")
            except Exception as e_file:
                logging.error(f"Erro ao limpar {f.name}: {e_file}")
                # Fallback: Copia o original se falhar a limpeza
                try: shutil.copy(str(f), str(dest_path))
                except: pass
        
        cb(100, 1, f"Limpeza Concluída: {success_count}/{total} arquivos processados.")
        return True

    except ImportError:
        logging.warning("[BATCH] DeepFilterNet não instalado. Apenas copiando arquivos.")
        # Fallback: Copia tudo
        for f in to_process:
            try: shutil.copy(str(f), str(dest_dir / f.name))
            except: pass
        return False
    except Exception as e:
        logging.error(f"[BATCH] Erro fatal no DeepFilterNet: {e}")
        return False

def check_ffmpeg():
    """Verifica se o FFmpeg está instalado e acessível no PATH do sistema."""
    if not shutil.which("ffmpeg"):
        logging.critical("="*80)
        logging.critical("ERRO CRÍTICO: O FFmpeg não foi encontrado no PATH do sistema.")
        logging.critical("O FFmpeg é essencial para o funcionamento deste programa.")
        logging.critical("Por favor, instale o FFmpeg e certifique-se de que ele está no PATH.")
        logging.critical("Download: https://ffmpeg.org/download.html")
        logging.critical("="*80)
        sys.exit(1)
    logging.info("FFmpeg encontrado e pronto para uso.")

# --- VOICE GUARD (SISTEMA DE IDENTIDADE RIGOROSA) ---
class VoiceState(Enum):
    PROVISIONAL = "Provisória" 
    INSUFFICIENT = "Insuficiente"
    TRAINABLE = "Treinável"

class VoiceIdentity:
    def __init__(self, voice_id):
        self.id = voice_id
        self.embeddings = []
        self.mean_embedding = None
        self.total_duration = 0.0
        self.segments = []
        self.state = VoiceState.PROVISIONAL

    def add_segment(self, embedding, duration, segment_info=None):
        self.embeddings.append(embedding)
        self.total_duration += duration
        matrix = np.array(self.embeddings)
        self.mean_embedding = np.mean(matrix, axis=0)
        self.update_state()

    def update_state(self):
        if self.total_duration >= 10.0: # Jogos: arquivos curtos, limiar menor
            self.state = VoiceState.TRAINABLE
        elif self.total_duration >= 2.0:
            self.state = VoiceState.INSUFFICIENT
        else:
            self.state = VoiceState.PROVISIONAL

class VoiceGuard:
    def __init__(self, similarity_threshold=0.42, hysteresis_threshold=0.35):
        self.voices = {} 
        self.next_id_counter = 1
        self.similarity_threshold = similarity_threshold
        self.hysteresis_threshold = hysteresis_threshold 
        self.last_speaker_id = None

    def create_new_voice(self):
        new_id = f"voz{self.next_id_counter}"
        self.next_id_counter += 1
        self.voices[new_id] = VoiceIdentity(new_id)
        return new_id

    def process_segment(self, embedding, duration, start_time, end_time):
        from sklearn.metrics.pairwise import cosine_similarity
        best_match_id = None
        best_score = -1.0
        
        # [OPTIMIZATION] Verifica primeiro o último orador (Hysteresis Check)
        # Se a similaridade com o último for "ok" (> hysteresis_threshold), mantém!
        # Isso evita que pequenos ruídos ou pausas quebrem a continuidade.
        if self.last_speaker_id and self.last_speaker_id in self.voices:
            last_voice = self.voices[self.last_speaker_id]
            if last_voice.mean_embedding is not None:
                emb_a = embedding.reshape(1, -1)
                emb_b = last_voice.mean_embedding.reshape(1, -1)
                last_score = cosine_similarity(emb_a, emb_b)[0][0]
                
                if last_score >= self.hysteresis_threshold:
                    # [STICKY] Mantém o orador mesmo que não seja o "melhor de todos"
                    # desde que seja aceitável.
                    # logging.info(f"Hysteresis Active: Kept {self.last_speaker_id} (Score: {round(last_score, 2)})")
                    self.voices[self.last_speaker_id].add_segment(embedding, duration, {"start": start_time, "end": end_time})
                    return self.last_speaker_id

        # Se não caiu no hysteresis, procura o melhor match global
        for vid, voice in self.voices.items():
            if voice.mean_embedding is None: continue
            emb_a = embedding.reshape(1, -1)
            emb_b = voice.mean_embedding.reshape(1, -1)
            score = cosine_similarity(emb_a, emb_b)[0][0]
            if score > best_score:
                best_score = score
                best_match_id = vid
        
        result_id = None
        
        if best_match_id and best_score >= self.similarity_threshold:
            result_id = best_match_id
        else:
            result_id = self.create_new_voice()
        
        if result_id and result_id in self.voices:
            voice = self.voices[result_id]
            voice.add_segment(embedding, duration, {"start": start_time, "end": end_time})
        
        self.last_speaker_id = result_id
        return result_id

    def get_trainable_voices(self):
        return [v for v in self.voices.values() if v.state == VoiceState.TRAINABLE]

    # [NEW] Post-Processing Merge Logic
    # Verifica vozes muito parecidas que acabaram separadas e as une.
    def merge_similar_voices(self, threshold=0.65):
        """
        Une vozes duplicadas.
        Threshold: 0.65 (Mais alto que o de entrada, para garantir fusão segura).
        """
        import shutil
        from sklearn.metrics.pairwise import cosine_similarity
        
        merged_count = 0
        sorted_ids = sorted(self.voices.keys())
        
        # Compara todos contra todos
        # (Naive O(N^2), mas N é pequeno em jogos, < 20 speakers)
        for i in range(len(sorted_ids)):
            id_a = sorted_ids[i]
            if id_a not in self.voices: continue # Já foi mergeado
            
            voice_a = self.voices[id_a]
            if voice_a.mean_embedding is None: continue

            for j in range(i + 1, len(sorted_ids)):
                id_b = sorted_ids[j]
                if id_b not in self.voices: continue
                
                voice_b = self.voices[id_b]
                if voice_b.mean_embedding is None: continue
                
                # Calcula similaridade
                emb_a = voice_a.mean_embedding.reshape(1, -1)
                emb_b = voice_b.mean_embedding.reshape(1, -1)
                score = cosine_similarity(emb_a, emb_b)[0][0]
                
                if score > threshold:
                    # MERGE! (B -> A)
                    # logging.info(f"Merging {id_b} into {id_a} (Similarity: {round(score, 2)})")
                    
                    # 1. Transfere segmentos
                    voice_a.segments.extend(voice_b.segments)
                    voice_a.total_duration += voice_b.total_duration
                    voice_a.embeddings.extend(voice_b.embeddings)
                    
                    # 2. Recalcula média
                    matrix = np.array(voice_a.embeddings)
                    voice_a.mean_embedding = np.mean(matrix, axis=0)
                    
                    # 3. Importante: Atualiza o mapeamento para que o resto do código saiba
                    # (Isso requer que o caller use essa função e atualize seus dicionários)
                    # Aqui só atualizamos o estado interno do VoiceGuard
                    del self.voices[id_b]
                    merged_count += 1
                    
        return merged_count

class SimpleDiarizer:
    def __init__(self, source="speechbrain/spkrec-ecapa-voxceleb", device=None):
        try:
            import torchaudio
            if not hasattr(torchaudio, 'list_audio_backends'):
                def _list_audio_backends(): return ['soundfile']
                torchaudio.list_audio_backends = _list_audio_backends
            
            # [v12.5] HARDWARE ADAPTIVE DIARIZATION
            if device is None:
                device = get_optimal_device()
            
            from speechbrain.inference.speaker import EncoderClassifier
            self.encoder = EncoderClassifier.from_hparams(source=source, run_opts={"device": device})
            self.device = device
            logging.info(f"Diarizador iniciado em '{self.device}'.")
        except ImportError:
             logging.error("SpeechBrain não encontrado. Diarização falhará.")
             self.encoder = None

    def get_file_embedding(self, audio_path):
        """Gera um embedding único para o arquivo inteiro. Com Filtro de Banda."""
        import torchaudio
        
        load_path = str(audio_path)

        try:
            signal, fs = torchaudio.load(load_path)
            
            # Resample se necessário (ECAPA espera 16kHz)
            if fs != 16000:
                import torchaudio.transforms as T
                resampler = T.Resample(fs, 16000)
                signal = resampler(signal)
                fs = 16000
            
            # Garante mono
            if signal.shape[0] > 1:
                signal = signal.mean(dim=0, keepdim=True)
                
            # [TUNING] Filtro de Frequência "Telefônico" (300Hz - 3400Hz)
            # CRÍTICO PARA JOGOS: Remove música de fundo e efeitos que confundem a IA.
            try:
                import torchaudio.functional as F
                signal = F.highpass_biquad(signal, fs, 300)
                signal = F.lowpass_biquad(signal, fs, 3400)
            except Exception as e_filter:
                logging.warning(f"Erro no filtro de banda: {e_filter}")
                
            # Embed
            embeddings = self.encoder.encode_batch(signal)
            
            # [FIX] Garante que o dado saia da GPU (.cpu()) antes de converter para Numpy
            return embeddings[0, 0].cpu().numpy()
            
        except Exception as e:
            logging.error(f"Erro ao gerar embedding: {e}")
            return None

    def detect_splits_surgical(self, audio_path):
        """
        [v10.60 SURGICAL VAD SPLIT]
        Detecta silêncios primeiro (Pydub) e analisa se houve troca de voz entre os blocos de fala.
        Garante que nunca corte no meio de uma palavra.
        """
        import torchaudio
        from scipy.spatial.distance import cosine
        from pydub import AudioSegment
        from pydub.silence import detect_nonsilent
        
        sound = AudioSegment.from_wav(str(audio_path))
        # Detecta trechos de fala (VAD simples mas eficaz)
        nonsilent_ranges = detect_nonsilent(sound, min_silence_len=300, silence_thresh=-40)
        
        if len(nonsilent_ranges) < 2: return []
        
        signal, fs = torchaudio.load(str(audio_path))
        if signal.shape[0] > 1: signal = signal.mean(dim=0, keepdim=True)
        if fs != 16000:
             import torchaudio.transforms as T
             resampler = T.Resample(fs, 16000)
             signal = resampler(signal)
             fs = 16000
        
        embeddings = []
        valid_ranges = []
        
        for start_ms, end_ms in nonsilent_ranges:
            s_start = int((start_ms / 1000.0) * fs)
            s_end = int((end_ms / 1000.0) * fs)
            
            if (end_ms - start_ms) < 500: continue
            
            try:
                chunk = signal[:, s_start:s_end]
                emb = self.encoder.encode_batch(chunk)
                # [FIX] Garante .cpu() aqui também
                embeddings.append(emb.squeeze().cpu().numpy())
                valid_ranges.append((start_ms, end_ms))
            except: continue
            
        if len(embeddings) < 2: return []
        
        splits_ms = []
        for i in range(len(embeddings) - 1):
            dist = cosine(embeddings[i], embeddings[i+1])
            if dist > 0.5: # Mudança de voz entre blocos detectada
                silence_start = valid_ranges[i][1]
                silence_end = valid_ranges[i+1][0]
                split_point = (silence_start + silence_end) / 2
                splits_ms.append(split_point / 1000.0)
        
        return splits_ms

    def cluster_batch_embeddings(self, embeddings_dict, num_speakers=None):
        from sklearn.cluster import AgglomerativeClustering
        """
        Agrupa embeddings.
        Se num_speakers > 1: Clustering Fixo.
        Se num_speakers == 0 ou None: Clustering Automático (Threshold 0.45).
        """
        if not embeddings_dict: return {}
        
        filenames = list(embeddings_dict.keys())
        matrix = np.array([embeddings_dict[f] for f in filenames])
        
        # [AUTO MODE]
        if not num_speakers or num_speakers < 2:
            logging.info("Clustering Automático (Agglomerative, Threshold=0.45)...")
            # [TUNING] Aumentado para 0.60 (era 0.45) para forçar mais agrupamento (menos vozes)
            clusterer = AgglomerativeClustering(n_clusters=None, distance_threshold=0.60, metric='cosine', linkage='average')
        else:
            # [FIXED MODE]
            if len(filenames) < num_speakers:
                logging.warning("Menos arquivos que falantes. Clusterização cancelada.")
                return {f: 'voz1' for f in filenames}
            clusterer = AgglomerativeClustering(n_clusters=num_speakers, metric='cosine', linkage='average')
            
        labels = clusterer.fit_predict(matrix)
        
        results = {}
        for i, label in enumerate(labels):
            results[filenames[i]] = f"voz{label+1}"
            
        return results

# --- ETAPAS DO PROCESSO NA INTERFACE ---
ETAPAS_JOGOS = [
    "Iniciando",                      # 0
    "1. Diarização Manual",           # 1
    "2. Transcrição",                 # 2
    "3. Tradução (Gema)",             # 3
    "4. Sincronização (Gema)",        # 4
    "5. Adaptação para TTS",          # 5
    "6. Gerando Áudios (Chatterbox)", # [MODIFIED] Chatterbox to Chatterbox
    "7. Finalização e Masterização",  # 7
    "8. Concluído"                    # 8
]
ETAPAS_CONVERSAO = [
    "Iniciando", "1. Convertendo Arquivos", "2. Concluído"
]
ETAPAS_TRANSCRICAO = [
    "Iniciando", "1. Transcrevendo com Whisper", "2. Gerando Arquivos Finais", "3. Concluído"
]
# Atualizado para refletir a mudança de ferramenta
ETAPAS_SEPARACAO = [
    "Iniciando", "1. Removendo Efeito de Rádio (FFmpeg)", "2. Finalizando", "3. Concluído"
]

# --- DICIONÁRIO DE TRADUÇÕES COMUNS ---
DICIONARIO_TRADUCOES = {
    "on it.": "Já vou.", "weapons free.": "Fogo à vontade.", "no way.": "nem pensar.",
    "get real.": "cai na real.", "not happening.": "sem chance.", "yes.": "sim.",
    "no.": "não.", "thanks.": "obrigado.", "thank you.": "obrigado.", "ok.": "ok."
}

# --- LISTA DE SIBILOS E SONS NÃO VERBAIS A IGNORAR ---
SONS_A_IGNORAR = [
    'ah', 'ai', 'eh', 'ei', 'oh', 'oi', 'uh', 'ui', 'ahm', 'hmm', 'huh', 'hmpf',
    'tsk', 'tsr', 'ugh', 'uhm', 'shh', 'suspira', 'geme', 'gasp', 'ofega', 'grr', 'rrr',
    'click', 'breath', 'respira', 'chora', 'risos', 'haha', 'hahaha', 'hihi', 'hehe'
]

def is_junk_text(text):
    if not text: return True
    t = text.lower().strip()
    
    # 1. Repetição de padrão curto (ex: "DA DA DA", "E E E")
    words = t.split()
    if len(words) > 5:
        # Se as primeiras 5 palavras forem idênticas, é lixo/alucinação
        if all(w == words[0] for w in words[:5]): 
            return True
            
    # 2. Heurística de Variedade de Caracteres (Anti-Hallucination)
    clean_chars = t.replace(" ", "")
    if len(clean_chars) > 15:
        from collections import Counter
        counts = Counter(clean_chars)
        # Se 1 ou 2 letras dominam 85% de um texto longo, é junk
        if counts:
            most_common_sum = sum(v for k, v in counts.most_common(2))
            if most_common_sum / len(clean_chars) > 0.85:
                return True
            
    # 3. Padrões de Alucinação Frequentes (Whisper Hallucination)
    junk_patterns = [
        "thanks for watching", "subtitles by", "amara.org", "please subscribe",
        "da da da", "la la la", "ha ha ha", "pa pa pa", "huh huh", "um um um"
    ]
    for p in junk_patterns:
        if p in t: return True
        
    return False

# --- VARIÁVEIS GLOBAIS E LOCKS (Adaptativo v18.6) ---
whisper_model = None
chatterbox_model = None
model_lock = Lock()
progress_dict, progress_lock = {}, Lock()
active_jobs_lock = Lock()
active_jobs = set()

# [v18.6] TRAVA DE SEGURANÇA: 1 vídeo por vez para estabilidade total.
MAX_CONCURRENT_JOBS = 1 

# --- INICIALIZAÇÃO DO FLASK ---
app = Flask(__name__, template_folder='client', static_folder='client')
if HAS_CORS:
    CORS(app) # [NEW] Habilita CORS para o frontend local
else:
    print("[AVISO] Rodando sem CORS. Instale com: pip install flask-cors")
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 * 1024 
app.config['MAX_FORM_PARTS'] = 10000
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- FUNÇÕES DE SEGURANÇA ---
def safe_json_write(data, path, indent=4, ensure_ascii=False, retries=5, delay=0.2):
    path = Path(path)
    # [PRUDENCE FIX] Garante que a pasta existe antes de tentar escrever
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + '.tmp')
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
            break
    logging.error(f"NÃO FOI POSSÍvel escrever em {path} após {retries} tentativas.")

def safe_json_read(path):
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
    if not isinstance(text, str): return ""
    match = re.match(r'^(.*?)(?=\n\n|\*\*Texto Original:|\*\*Texto Adaptado:)', text, re.DOTALL)
    clean_text = match.group(1).strip() if match else text.strip()
    # [FIX] Remove marcadores de lista que vazam do LLM: "(a) ", "1. ", "a) "
    clean_text = re.sub(r'(?:^|\s)[\(\[]?[0-9a-zA-Z]{1,2}[\)\]\.]\s+', ' ', clean_text)
    
    # [FIX] Remove marcadores de gênero (ex: "trancado(a)", "ele(a)")
    clean_text = re.sub(r'\([aoes]\)', '', clean_text, flags=re.IGNORECASE)
    clean_text = re.sub(r'\s+\([aoes]\)', '', clean_text, flags=re.IGNORECASE)
    
    clean_text = re.sub(r'ponto de interrogação|ponto interroga(?:ç|t)ão|ponto inter[eo]gativo', '?', clean_text, flags=re.IGNORECASE)
    
    # [TTS OPTIMIZATION] Remove TODAS as vírgulas para evitar pausas indesejadas na fala.
    # O texto visual pode ter vírgula, mas o robô vai ler direto ("correndo").
    clean_text = clean_text.replace(',', ' ').replace('...', ' ').replace('.', ' ')
    
    clean_text = clean_text.replace('!', 'TEMP_EXCLAMATION').replace('?', 'TEMP_QUESTION')
    clean_text = clean_text.replace('TEMP_EXCLAMATION', '!').replace('TEMP_QUESTION', '?')
    
    # Remove espaços duplos criados pela remoção de pontuação
    return " ".join(clean_text.split()).strip()

def log_error_to_file(job_dir, file_id, original_text, etapa, resposta_falha, tentativas=1):
    error_log_path = job_dir / "erros.json"
    try:
        logs = safe_json_read(error_log_path) or []
        error_entry = { "timestamp": datetime.now().isoformat(), "file_id": file_id, "original_text": original_text,
                        "etapa_falha": etapa, "resposta_recebida": resposta_falha, "tentativas": tentativas }
        logs.append(error_entry)
        safe_json_write(logs, error_log_path)
    except Exception as e:
        logging.error(f"Não foi possível registar o erro no ficheiro {error_log_path}: {e}")

# --- FUNÇÕES DE LÓGICA DO PIPELINE ---
def _print_progress_to_cmd(job_id, progress, etapa, subetapa, tempo_decorrido):
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
    with progress_lock:
        elapsed_time = time.time() - start_time
        etapa_atual = etapas_list[etapa_idx] if etapa_idx < len(etapas_list) else "Desconhecida"
        tempo_str = str(timedelta(seconds=int(elapsed_time)))
        progress_info = {
            'progress': round(progress, 2), 
            'etapa': etapa_atual, 
            'subetapa': subetapa, 
            'tempo_decorrido': tempo_str,
            'start_time': start_time, # [NATIVO] Permite Animação Real-Time no Frontend
            'total_elapsed_secs': elapsed_time # [PERSISTÊNCIA] Salva segundos totais para retomar sem zerar
        }
        progress_dict[job_id] = progress_info
        _print_progress_to_cmd(job_id, progress, etapa_atual, subetapa, tempo_str)
        if progress >= 100 and (etapa_idx == len(etapas_list) - 1):
            sys.stdout.write("\n")
            logging.info(f"Processo {job_id} concluído!")
        status_path = Path(app.config['UPLOAD_FOLDER']) / job_id / "job_status.json"
        status_data = safe_json_read(status_path) or {}
        status_data.update(progress_info)
        status_data['status'] = 'processing' if etapa_idx < len(etapas_list) - 1 else 'completed'
        safe_json_write(status_data, status_path)

def get_optimal_device():
    """
    Detecta o melhor hardware disponível (Adaptativo v12.7).
    [v12.7] Adicionada verificação rigorosa de device_count preventivamente para evitar
    o erro 'Attempting to deserialize object on CUDA device 0 but torch.cuda.device_count() is 0'.
    """
    import torch
    
    # Se PyTorch achar que tem CUDA, mas a contagem for 0, os drivers estão quebrados/incompatíveis.
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        try:
            device_idx = torch.cuda.current_device()
            vram = torch.cuda.get_device_properties(device_idx).total_memory / (1024**3)
            if vram >= 3.5:
                # [v12.75] REQUISITO SPEECHBRAIN: Retornar dispositivo indexado (ex: cuda:0)
                logging.info(f"🚀 [HARDWARE] Placa NVIDIA detectada ({vram:.1f}GB VRAM). Ativando GPU!")
                return f"cuda:{device_idx}"
            else:
                logging.info(f"🐢 [HARDWARE] GPU detectada, mas é fraca ({vram:.1f}GB VRAM). Usando CPU.")
        except Exception as e:
            logging.warning(f"⚠️ Erro ao detectar VRAM ({e}). Drivers possivelmente corrompidos. Forçando CPU.")
    else:
        logging.info("💻 [HARDWARE] Nenhuma GPU válida detectada ou driver corrompido (count=0). Usando CPU.")
    return "cpu"

def get_whisper_model():
    global whisper_model
    with model_lock:
        if whisper_model is None:
            import torch
            from faster_whisper import WhisperModel
            
            device = get_optimal_device()
            if device == "cuda":
                logging.info("🚀 [HARDWARE] Whisper em CUDA (float16)!")
                whisper_model = WhisperModel("small", device="cuda", compute_type="float16")
            else:
                num_cores = os.cpu_count() or 4
                whisper_threads = max(1, num_cores // 2)
                logging.info(f"💻 [HARDWARE] Whisper em CPU (int8) com {whisper_threads} threads.")
                whisper_model = WhisperModel("small", device="cpu", compute_type="int8", cpu_threads=whisper_threads)
                
            logging.info("Modelo faster-whisper carregado.")
    return whisper_model

def unload_whisper_model():
    global whisper_model
    with model_lock:
        if whisper_model is not None:
            logging.info("Liberando memória do Whisper...")
            del whisper_model
            whisper_model = None
            import gc
            import torch # [FIX] Import torch inside function
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logging.info("Whisper descarregado da memória.")

# O motor ONNX foi removido para priorizar a qualidade de estúdio do modelo oficial.


def get_optimal_device():
    import logging
    import torch
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        try:
            device_idx = torch.cuda.current_device()
            # Pega memória livre e total em bytes
            free_m, total_m = torch.cuda.mem_get_info()
            free_vram_gb = free_m / (1024**3)
            total_vram_gb = total_m / (1024**3)

            # TRAVA INTELIGENTE: Baixei para 800MB para ser mais permissivo em placas de 6GB
            if free_vram_gb < 0.8:
                logging.warning(f"⚠️ [ALERTA DE HARDWARE] A placa de vídeo está quase cheia ({free_vram_gb:.1f}GB livres).")
                logging.warning("🔥 PARA VELOCIDADE MÁXIMA: Feche o LM Studio ou outros jogos antes de dublar.")
                logging.warning("Forçando modo CPU (Lento) para evitar crash por falta de memória...")
                return "cpu"
            
            if total_vram_gb >= 2.0: # Baixei de 3.5 para 2.0 para aceitar placas de entrada
                logging.info(f"🚀 [RTX DETECTADA] Usando GPU NVIDIA ({free_vram_gb:.1f}GB livres).")
                return f"cuda:{device_idx}"
        except Exception as e:
            logging.warning(f"⚠️ Erro ao checar VRAM: {e}. Usando CPU.")
    return "cpu"

def get_chatterbox_model():
    """
    Singleton para o modelo Chatterbox Oficial (Alta Fidelidade).
    O motor ONNX foi desativado para priorizar a qualidade de estúdio.
    """
    global chatterbox_model
    with model_lock:
        if chatterbox_model is None:
            official_path = Path("env/models/chatterbox_official")
            
            import os
            # Checa se não existe ou se está vazia
            if not official_path.exists() or len(os.listdir(official_path)) < 2:
                logging.info("⏳ Modelo Chatterbox ausente. Iniciando download automático via HuggingFace (isso pode demorar uns minutinhos)...")
                try:
                    os.environ["HF_HUB_DISABLE_FAST_HF_TRANSFER"] = "1"
                    from huggingface_hub import snapshot_download
                    official_path.mkdir(parents=True, exist_ok=True)
                    snapshot_download(repo_id="ResembleAI/chatterbox", local_dir=str(official_path), local_dir_use_symlinks=False)
                    logging.info("✅ Download do Chatterbox concluído!")
                except Exception as dl_err:
                    logging.error(f"❌ Erro ao baixar modelo Chatterbox (Link Invalido?): {dl_err}")
                    import shutil
                    shutil.rmtree(str(official_path), ignore_errors=True)
                    return None

            try:
                import torch
                import gc
                import sys
                from types import ModuleType
                
                # [v22.80] ULTIMATE RECURSIVE MOCK: Blindagem Total contra Lazy Imports
                # Criamos um protetor que gera submódulos infinitos para o SpeechBrain não reclamar.
                # [v22.90] REFINED RECURSIVE MOCK: Blindagem com Compatibilidade
                # Criamos um protetor inteligente que só finge existir o que for necessário.
                # [v23.50] ULTIMATE LAZY BLOCKER: Previne falhas de 'LazyModule' no SpeechBrain
                # Algumas bibliotecas tentam carregar módulos pesadores (k2, k2_fsa) no meio do processo.
                # Esse Mock avançado intercepta essas tentativas e devolve 'nada' com sucesso.
                class DeepMock(ModuleType):
                    def __getattr__(self, name):
                        if name.startswith('__'): return None
                        fn = f"{self.__name__}.{name}"
                        if fn not in sys.modules: sys.modules[fn] = DeepMock(fn)
                        return sys.modules[fn]
                    def __call__(self, *args, **kwargs): return None
                    def __bool__(self): return False
                    def __repr__(self): return f"<DeepMock {self.__name__}>"

                # 1. Detecta se vamos usar CPU ou GPU (já com a trava do LM Studio)
                device = get_optimal_device()

                # 2. Só bloqueia a NVIDIA se o dispositivo escolhido for CPU
                # [v23.50] ULTIMATE LAZY BLOCKER: Previne falhas de 'LazyModule' no SpeechBrain
                bad_libs = [
                    'speechbrain.integrations', 'speechbrain.integrations.k2_fsa', 
                    'speechbrain.integrations.huggingface', 'speechbrain.integrations.huggingface.wordemb',
                    'speechbrain.integrations.nlp', 'speechbrain.integrations.ctc',
                    'k2', 'k2_fsa'
                ]
                if device == "cpu":
                    bad_libs.extend(['nvidia', 'nvidia.cudnn'])
                
                for lib in bad_libs:
                    sys.modules[lib] = DeepMock(lib)
                
                # Forçamos a limpeza de memória das etapas anteriores
                gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                
                # [v22.51] MONKEY-PATCH PERTH: Resolve o erro 'NoneType' de conflito em memória
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
                logging.info("Iniciando Motor Oficial (Alta Fidelidade)...")
                
                # Otimização de CPU para Intel i5 (4 núcleos)
                # Usamos 3 núcleos para IA e deixamos 1 livre para o sistema
                num_cores = 3
                os.environ["OMP_NUM_THREADS"] = "3"
                os.environ["MKL_NUM_THREADS"] = "3"
                torch.set_num_threads(3)
                torch.backends.mkldnn.enabled = True
                
                logging.info(f"Usando {num_cores} núcleos para geração (1 núcleo livre para o sistema).")
                
                logging.info(f"Iniciando Motor Chatterbox em: {device}")
                raw_model = ChatterboxMultilingualTTS.from_local(str(official_path), device=device)
                
                class OfficialEngineWrapper:
                    def __init__(self, model): 
                        self.model = model
                        import torch
                        torch.set_grad_enabled(False) # [CPU OPTIMIZATION] Desativa gradiente globalmente na inferência
                    
                    @torch.no_grad() # [CPU OPTIMIZATION] Bloqueador de Autograd para Pytorch
                    def generate(self, text, language_id, audio_prompt_path, **kwargs):
                        # Ajuste fonético apenas para palavras críticas que o oficial erra
                        text_fix = re.sub(r"\btorre\b", "tôr-re", text, flags=re.IGNORECASE)
                        
                        # Vacina contra cortes: Expande símbolos matemáticos
                        text_fix = text_fix.replace("%", " por cento")
                        if not text_fix.strip().endswith((".", "!", "?")):
                            text_fix = text_fix.strip() + "."
                        
                        # [v22.40] REMOVIDO: Espaço extra (ele causava os suspiros/alucinações no final)
                        text_fix = text_fix.strip()
                        
                        # Parâmetros Dinâmicos (Vem lá da Etapa 6)
                        import torch
                        with torch.no_grad(): # Garantia dupla
                            return self.model.generate(
                                text_fix, 
                                language_id=language_id, 
                                audio_prompt_path=audio_prompt_path,
                                exaggeration=kwargs.get('exaggeration', 1.05),
                                cfg_weight=kwargs.get('cfg_weight', 0.5),
                                temperature=kwargs.get('temperature', 0.7),
                                top_p=kwargs.get('top_p', 0.9),
                                min_p=kwargs.get('min_p', 0.1),
                                repetition_penalty=kwargs.get('repetition_penalty', 1.2)
                            )
                    def to(self, device): return self 
                
                chatterbox_model = OfficialEngineWrapper(raw_model)
                logging.info("Motor Chatterbox Oficial carregado com sucesso.")
                return chatterbox_model
            except Exception as e:
                import traceback
                logging.error(f"Erro Crítico ao carregar Chatterbox: {e}")
                logging.error(traceback.format_exc())
                return None
                return None
    return chatterbox_model


def unload_chatterbox_model():
    """Libera a memória RAM/VRAM ocupada pelo Chatterbox."""
    global chatterbox_model
    with model_lock:
        if chatterbox_model is not None:
            logging.info("Liberando memória do Chatterbox...")
            del chatterbox_model
            chatterbox_model = None
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logging.info("Chatterbox descarregado da memória.")

def transcribe_audio(model, audio_path, source_lang="en"):
    # [CPU OPTIMIZATION] beam_size=1 (Greedy Search) é 3x mais rápido que o padrão (5).
    # condition_on_previous_text=False previne alucinações e loops.
    # [v10.41] HARD LOCK DE IDIOMA
    whisper_lang = source_lang if source_lang != 'auto' else None
    segments_generator, info = model.transcribe(audio_path, beam_size=1, condition_on_previous_text=False, language=whisper_lang)
    return {
        "text": "".join(s.text for s in segments_generator).strip(),
        "detected_language": getattr(info, 'language', None)
    }

def get_audio_duration(file_path):
    try:
        result = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', str(file_path)], capture_output=True, text=True, check=True)
        return float(result.stdout)
    except Exception as e:
        logging.error(f"Erro ao obter a duração de {file_path} com ffprobe: {e}")
        return 0

def get_audio_metadata(file_path):
    try:
        result = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'a:0', '-show_entries', 'stream=sample_rate,channels,bit_rate', '-of', 'json', str(file_path)], capture_output=True, text=True, check=True)
        stream_data = json.loads(result.stdout).get('streams', [{}])[0]
        bit_rate = stream_data.get('bit_rate')
        if not bit_rate or bit_rate == 'N/A':
            result_format = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=bit_rate', '-of', 'json', str(file_path)], capture_output=True, text=True, check=True)
            bit_rate = json.loads(result_format.stdout).get('format', {}).get('bit_rate')
        return stream_data.get('sample_rate', '44100'), stream_data.get('channels', 1), bit_rate
    except Exception as e:
        logging.error(f"Erro ao obter metadados de {file_path}: {e}")
        return '44100', 1, None

def get_audio_peak_dbfs(file_path):
    try:
        # Executa o filtro volumedetect para encontrar o pico máximo
        # [CPU THOTTLE] Limita threads
        threads = str(max(1, (os.cpu_count() or 4) // 2))
        cmd = ['ffmpeg', '-threads', threads, '-i', str(file_path), '-af', 'volumedetect', '-vn', '-sn', '-dn', '-f', 'null', 'NUL']
        result = subprocess.run(cmd, capture_output=True, text=True)
        # A saída do volumedetect vai para stderr
        output = result.stderr
        
        # Procura por "max_volume: -XX.X dB"
        match = re.search(r"max_volume:\s*(-?[\d\.]+)\s*dB", output)
        if match:
            return float(match.group(1))
        return None
    except Exception as e:
        logging.error(f"Erro ao detectar pico de áudio em {file_path}: {e}")
        return None

def find_existing_project(files_hash):
    upload_folder = Path(app.config['UPLOAD_FOLDER'])
    for job_dir in upload_folder.iterdir():
        if job_dir.is_dir() and job_dir.name.startswith("job_jogos_"):
            status_file = job_dir / "job_status.json"
            if (status_data := safe_json_read(status_file)) and status_data.get('files_hash') == files_hash:
                logging.info(f"Projeto existente encontrado com o mesmo hash: {job_dir.name}")
                return job_dir.name
    return None

def find_best_audio_profile(audio_data, job_dir):
    temp_dir = job_dir / "_temp_detection"
    temp_dir.mkdir(exist_ok=True)
    output_path = temp_dir / "test.wav"
    
    profiles = [
        {'name': 'native_wav'}, # [v10.84] FIX: Tenta detectar como WAV normal PRIMEIRO, sem forçar taxa de amostragem
        {'f': 'mp3', 'name': 'MP3_em_WAV'},
        {'f': 's16le', 'ar': '44100', 'ac': '2', 'name': 's16le_44100Hz_Estereo'},
        {'f': 's16le', 'ar': '22050', 'ac': '2', 'name': 's16le_22050Hz_Estereo'},
        {'f': 's16le', 'ar': '44100', 'ac': '1', 'name': 's16le_44100Hz_Mono'},
        {'f': 's16le', 'ar': '22050', 'ac': '1', 'name': 's16le_22050Hz_Mono'},
        {'c:a': 'adpcm_ms', 'ar': '44100', 'ac': '2', 'name': 'adpcm_ms_44100Hz_Estereo'},
        {'c:a': 'adpcm_ms', 'ar': '22050', 'ac': '2', 'name': 'adpcm_ms_22050Hz_Estereo'},
    ]

    for profile in profiles:
        logging.info(f"Tentando perfil de áudio: {profile['name']}")
        threads = str(max(1, (os.cpu_count() or 4) // 2))
        cmd = ['ffmpeg', '-threads', threads, '-y']
        
        profile_params = {k: v for k, v in profile.items() if k != 'name'}
        for key, value in profile_params.items():
            cmd.extend([f'-{key}', value])

        cmd.extend(['-i', 'pipe:0', '-c:a', 'pcm_s16le', str(output_path)])
        
        try:
            subprocess.run(cmd, input=audio_data, check=True, capture_output=True)
            if output_path.exists() and output_path.stat().st_size > 0 and get_audio_duration(output_path) > 0.01:
                logging.info(f"SUCESSO! Melhor perfil de áudio detectado: {profile['name']}")
                shutil.rmtree(temp_dir)
                return profile
        except subprocess.CalledProcessError as e:
            error_message = e.stderr.decode('utf-8', errors='ignore') if e.stderr else 'Nenhum erro reportado.'
            logging.warning(f"Perfil {profile['name']} falhou: {error_message}")
            continue
            
    shutil.rmtree(temp_dir)
    return None

def wait_for_diarization_manual(job_id, cb):
    source_dir = Path(app.config['UPLOAD_FOLDER']) / job_id / "_1_MOVER_OS_FICHEIROS_DAQUI"
    if not any(source_dir.iterdir()):
        logging.info("Nenhum arquivo para diarização manual.")
        return
        
    total_files_in_subdirs = len(list(source_dir.rglob("*.wav")))
    if total_files_in_subdirs == 0:
        logging.info("Nenhum arquivo para diarização manual.")
        return

    while True:
        num_files_remaining = len(list(source_dir.rglob("*.wav")))
        if num_files_remaining == 0:
            logging.info(f">>> Diarização manual para o job '{job_id}' concluída. Retomando pipeline. <<<")
            cb(100, 1, "Diarização manual concluída.")
            break
        else:
            msg = f"Arquivos longos foram separados. Organize todos os {num_files_remaining} segmentos/arquivos."
            sys.stdout.write(f"\r{''.ljust(150)}\r") 
            logging.warning(f">>> PAUSA: {msg} <<<")
            progress = ((total_files_in_subdirs - num_files_remaining) / total_files_in_subdirs) * 100 if total_files_in_subdirs > 0 else 0
            cb(progress, 1, msg)
            time.sleep(5)


def run_auto_diarization_batch(job_dir, job_id, cb):
    """
    Diarização Automática para Lotes de Arquivos (Jogos).
    Analisa cada arquivo como um segmento único e agrupa por voz.
    Suporta:
    1. Num Speakers Fixo (Clustering)
    2. Num Speakers == 1 (Bypass)
    3. Auto (VoiceGuard)
    """
    # [CACHE] Define pastas
    # [SAFEGUARD] A pasta abaixo é READ-ONLY. O sistema NUNCA deve apagar arquivos dela.
    # O usuário exigiu preservação total dos originais em "_1_MOVER_OS_FICHEIROS_DAQUI".
    source_dir = job_dir / "_1_MOVER_OS_FICHEIROS_DAQUI"
    clean_audio_dir = job_dir / "_1b_AUDIO_LIMPO"
    segmented_dir = job_dir / "_1c_AUDIO_SEGMENTADO" # [FIX] Definido explicitamente
    backup_dir = job_dir / "_backup_transcricao"     # [FIX] Definido explicitamente
    target_dir = job_dir / "_2_PARA_AS_PASTAS_DE_VOZ"
    
    # [FIX] Pular diarização inteira se já foi feita (Resumo do Projeto)
    marker_path = target_dir / "unification_done.marker"
    project_data_path = job_dir / "project_data.json"
    if marker_path.exists() and project_data_path.exists():
        logging.info("Diarização e unificação já concluídas neste projeto. Pulando Fase 1 inteira para acelerar o reinício.")
        cb(100, 1, "Diarização restaurada do cache.")
        return

    # Cria diretórios de trabalho
    clean_audio_dir.mkdir(parents=True, exist_ok=True)
    segmented_dir.mkdir(parents=True, exist_ok=True)
    backup_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)

    # [FEATURE] OpenUnmix: Verifica se deve separar fundo
    status_path = job_dir / "job_status.json"
    status_data = safe_json_read(status_path) or {}
    use_openunmix = str(status_data.get('preserve_background', 'false')).lower() == 'true'

    if use_openunmix:
        logging.info("[OPENUNMIX] Separação de Fundo ATIVADA. Iniciando...")
        run_openunmix_batch(source_dir, job_dir, cb)
        
        # Atualiza a fonte para a pasta de vocais isolados (assim o DeepFilter limpa só a voz)
        source_dir = job_dir / "_0a_SEPARACAO_VOCAL"
        logging.info(f"[OPENUNMIX] Fonte de áudio alterada para: {source_dir}")

    # 1. LIMPEZA DE ÁUDIO
    source_files = list(source_dir.rglob("*.wav"))
    if not source_files:
        logging.info("Nenhum arquivo para processar.")
        return

    run_batch_cleaning(source_dir, clean_audio_dir, cb)
    
    clean_files = sorted(list(clean_audio_dir.rglob("*.wav")))
    if not clean_files:
         clean_files = source_files # Fallback

    status_path = job_dir / "job_status.json"
    status_data = safe_json_read(status_path) or {}
    try:
        num_speakers = int(status_data.get('num_speakers', '0'))
    except: num_speakers = 0
    source_lang = status_data.get('source_language', 'auto') # [v10.99] Recomendado 'auto' para jogos multi-idioma (COD, etc)

    # 2. TRANSCRIÇÃO & SEGMENTAÇÃO (Phase 1)
    # Gera arquivos em _1c_AUDIO_SEGMENTADO e JSONs em _backup_transcricao
    cb(0, 1, "Fase 1: Transcrição e Segmentação...")
    
    whisper_model = None
    
    for i, audio_file in enumerate(clean_files):
        cb((i / len(clean_files)) * 40, 1, f"Transcrevendo: {audio_file.name}")
        
        duration = get_audio_duration(audio_file)
        
        # [v10.86] DIARIZATION GRANULARITY
        # Subido de 12s para 25s nativos por recomendação do usuário, garantindo fôlego máximo
        # para falas, parando cirurgicamente no último limite de palavra antes de estourar 25s.
        should_split = duration > 25.0 and num_speakers != 1
        
        if not should_split:
            # Caso simples: Copia inteiro (Preserva arquivo original)
            dest_wav = segmented_dir / audio_file.name
            dest_json = backup_dir / f"{audio_file.stem}.json"
            
            if dest_wav.exists() and dest_json.exists():
                continue # Skip
                
            shutil.copy(str(audio_file), str(dest_wav))
            
            # Transcreve (se não tiver JSON)
            if not whisper_model: whisper_model = get_whisper_model()
            whisper_lang = source_lang if source_lang != 'auto' else None
            segments, info = whisper_model.transcribe(str(dest_wav), beam_size=1, word_timestamps=False, language=whisper_lang)
            text = "".join([s.text for s in list(segments)]).strip()
            
            safe_json_write({
                "id": audio_file.stem,
                "original_text": text,
                "duration": duration,
                "source_file": str(dest_wav),
                "detected_language": getattr(info, 'language', None)
            }, dest_json)
            
        else:
            # Caso complexo: Whisper Split
            # Verifica se já foi feito
            # Procura JSONs que começam com esse stem
            related_jsons = list(backup_dir.glob(f"{audio_file.stem}_seg*.json"))
            if related_jsons:
                continue # Já foi fatiado e transcrito
            
            if not whisper_model: whisper_model = get_whisper_model()
            
            whisper_lang = source_lang if source_lang != 'auto' else None
            segments, info = whisper_model.transcribe(str(audio_file), beam_size=1, word_timestamps=False, language=whisper_lang)
            segments = list(segments)
            detected_lang = getattr(info, 'language', None)
            
            if not segments:
                # Falha no silence/speech detection? Copia inteiro como fallback
                dest_wav = segmented_dir / audio_file.name
                shutil.copy(str(audio_file), str(dest_wav))
                continue

            audio_seg = AudioSegment.from_wav(str(audio_file))
            
            # [SMART CHUNKING] Agrupa segmentos até ~15 segundos.
            # Evita que o Whisper crie 14 arquivos de 2s, agrupando-os em blocos lógicos.
            grouped_chunks = []
            current_group_start = -1
            current_group_end = -1
            
            for seg in segments:
                start_ms = max(0, int(seg.start * 1000) - 50)
                end_ms = min(len(audio_seg), int(seg.end * 1000) + 50)
                
                if (end_ms - start_ms) < 200: continue
                
                if current_group_start == -1:
                    current_group_start = start_ms
                    current_group_end = end_ms
                else:
                    proposed_duration = (end_ms - current_group_start) / 1000.0
                    # [v10.86] MICRO-CHUNKING: Permite agrupar blocos até 25s.
                    # Se adicionar a próxima palavra ultrapassar 25s, o bloco atual é fechado
                    # e um novo bloco começa com a palavra excedente.
                    if proposed_duration <= 25.0:
                        current_group_end = end_ms
                    else:
                        grouped_chunks.append((current_group_start, current_group_end))
                        current_group_start = start_ms
                        current_group_end = end_ms
                        
            # Adiciona o último bloco que ficou sobrando
            if current_group_start != -1:
                grouped_chunks.append((current_group_start, current_group_end))
                
            for idx, (g_start, g_end) in enumerate(grouped_chunks):
                chunk = audio_seg[g_start:g_end]
                chunk_name = f"{audio_file.stem}_seg{idx:03d}_{int(g_start/1000)}s.wav"
                chunk_path = segmented_dir / chunk_name
                chunk.export(chunk_path, format="wav")
                
                # Transcreve o bloco recém criado
                json_path = backup_dir / f"{chunk_path.stem}.json"
                try:
                    c_segments, _ = whisper_model.transcribe(str(chunk_path), beam_size=1, word_timestamps=False, language=whisper_lang)
                    chunk_text = "".join([s.text for s in list(c_segments)]).strip()
                except:
                    chunk_text = ""
                    
                safe_json_write({
                    "id": chunk_path.stem,
                    "original_text": chunk_text,
                    "duration": len(chunk) / 1000.0,
                    "source_file": str(chunk_path),
                    "detected_language": detected_lang
                }, json_path)

    # Libera Whisper
    if whisper_model:
        del whisper_model
        import gc
        gc.collect()

    # 3. DIARIZAÇÃO & CLUSTERING (Phase 2)
    cb(40, 1, "Fase 2: Diarização Global...")
    
    diarizer = SimpleDiarizer()
    all_segments = sorted(list(segmented_dir.glob("*.wav")))
    
    if not all_segments:
        logging.warning("Fase 2 abortada: Nenhum segmento para diarizar.")
        return

    embeddings_map = {}
    
    # Gera Embeddings
    for i, seg_path in enumerate(all_segments):
        cb(40 + (i / len(all_segments)) * 30, 1, f"Analisando voz: {seg_path.name}")
        emb = diarizer.get_file_embedding(str(seg_path))
        if emb is not None:
             embeddings_map[seg_path.name] = emb
             
    # Clusteriza
    cb(70, 1, "Agrupando Falantes...")
    if num_speakers == 1:
        file_to_voice = {f.name: 'voz1' for f in all_segments}
    else:
        n_clusters = num_speakers if num_speakers > 1 else None
        file_to_voice = diarizer.cluster_batch_embeddings(embeddings_map, n_clusters)
        
    # Move para Pastas Finais
    cb(80, 1, "Organizando pastas...")
    
    for seg_path in all_segments:
        fname = seg_path.name
        voice_id = file_to_voice.get(fname, "voz_desconhecida")
        
        # Destino
        voice_folder = target_dir / voice_id
        voice_folder.mkdir(parents=True, exist_ok=True)
        final_path = voice_folder / fname
        
        if not final_path.exists():
            shutil.copy(str(seg_path), str(final_path))
            
            
    cb(90, 1, "Finalizando: Gerando metadados do projeto...")
    
    # 4. RECONSTRUÇÃO DO PROJECT_DATA.JSON
    # Cruza os arquivos organizados nas pastas de voz com os backups de texto
    final_project_data = []
    
    # Garante que target_dir existe
    if target_dir.exists():
        for voice_folder in target_dir.iterdir():
            if not voice_folder.is_dir(): continue
            speaker_id = voice_folder.name
            
            for wav_path in voice_folder.glob("*.wav"):
                if wav_path.name.startswith("_REF_"): continue
                # Busca JSON de backup pelo nome do arquivo (stem igual)
                json_backup_path = backup_dir / f"{wav_path.stem}.json"
                
                original_text = ""
                duration = 0.0
                
                if json_backup_path.exists():
                    try:
                        meta = safe_json_read(json_backup_path)
                        original_text = meta.get('original_text', '')
                        duration = meta.get('duration', 0.0)
                    except: pass
                else:
                    # Fallback
                    try: duration = get_audio_duration(wav_path)
                    except: pass
                
                final_project_data.append({
                    "id": wav_path.stem,
                    "file_name": wav_path.name,
                    "original_text": original_text,
                    "translated_text": "",
                    "speaker": speaker_id,
                    "start_time": 0,
                    "end_time": duration,
                    "duration": duration,
                    "file_path": str(wav_path),
                    "status": "pending_translation" 
                })
    
    # Salva
    safe_json_write(final_project_data, job_dir / "project_data.json")
    
    # [v10.68] Calcula duração total para estimativa do usuário ("Quanto tempo de áudio tem?")
    total_seconds = sum(item.get('duration', 0) for item in final_project_data)
    duracao_total_formatada = str(timedelta(seconds=int(total_seconds)))
    
    status_path = job_dir / "job_status.json"
    status_data = safe_json_read(status_path) or {}
    status_data['duracao_total_formatada'] = duracao_total_formatada
    status_data['total_wav_seconds'] = total_seconds
    safe_json_write(status_data, status_path)

    logging.info(f"Project Data gerado com {len(final_project_data)} segmentos. Duração Total de Áudio: {duracao_total_formatada}")

    cb(100, 1, "Diarização Concluída.")

def unify_speaker_files(job_dir, cb):
    diarization_dir = job_dir / "_2_PARA_AS_PASTAS_DE_VOZ"
    
    # [OPTIMIZATION] Verifica se a unificação já foi concluída em execução anterior
    marker_path = diarization_dir / "unification_done.marker"
    if marker_path.exists():
        logging.info("Unificação de vozes já concluída anteriormente. Pulando.")
        return

    voice_folders = [d for d in diarization_dir.iterdir() if d.is_dir() and d.name.startswith('voz')]
    
    if not voice_folders: return

    cb(0, 1, "Iniciando unificação inteligente de vozes...")
    diarizer = SimpleDiarizer()
    
    # 1. COLETA DE CENTROIDES (Média dos Embeddings por Pasta)
    folder_centroids = []
    
    for i, folder in enumerate(voice_folders):
        wavs = list(folder.glob("*.wav"))
        if not wavs: continue
        
        # Analisa até 5 amostras aleatórias para formar o perfil da voz
        samples = random.sample(wavs, min(len(wavs), 5)) 
        embeddings = []
        for wav in samples:
            try:
                emb = diarizer.get_file_embedding(str(wav))
                if emb is not None: embeddings.append(emb)
            except: pass
            
        if embeddings:
            centroid = np.mean(embeddings, axis=0)
            folder_centroids.append({'folder': folder, 'centroid': centroid, 'count': len(wavs)})
    
    # 2. MERGE INTERATIVO (Agrupa pastas similares)
    # [TUNING FINAL] Ajustado para 0.65 (era 0.45). Como o usuário usa DeepFilter,
    # o áudio já está limpo, então podemos ser mais exigentes ("strict") para não misturar homem e mulher.
    MERGE_THRESHOLD = 0.65
    merged_map = {} # {original_folder_name: target_folder_name}
    
    # Ordena por tamanho (pastas maiores tendem a ser as 'principais')
    folder_centroids.sort(key=lambda x: x['count'], reverse=True)
    
    final_folders = []
    
    for item in folder_centroids:
        current_folder = item['folder']
        current_emb = item['centroid']
        
        merged = False
        for target in final_folders:
            dist = cosine_similarity([current_emb], [target['centroid']])[0][0]
            if dist > MERGE_THRESHOLD:
                # Merge!
                logging.info(f"Mesclando {current_folder.name} -> {target['folder'].name} (Sim: {round(dist, 2)})")
                
                # Move arquivos
                for f in current_folder.glob("*.wav"):
                    try:
                        shutil.move(str(f), str(target['folder'] / f.name))
                    except: pass # Nome duplicado?
                
                # Remove pasta vazia
                try: current_folder.rmdir() 
                except: pass
                
                merged = True
                break
        
        if not merged:
            final_folders.append(item)

    # [CLEANUP STEP] "Smart Cleanup"
    # Remove pastas com menos de 2 arquivos SE e SOMENTE SE o conteúdo for irrelevante.
    # Se for uma frase válida ("Abra a porta!"), mantém mesmo sendo 1 arquivo.
    # Se for ruído ("Argh!"), move para a principal.

    # Carrega dados do projeto para checar o texto
    project_data_path = job_dir / "project_data.json"
    project_text_map = {}
    try:
        if project_data_path.exists():
            pdata = safe_json_read(project_data_path) or []
            for item in pdata:
                # Normaliza texto para comparação
                txt = item.get('original_text', '').lower().strip()
                # Remove pontuação básica
                txt = re.sub(r'[^\w\s]', '', txt)
                project_text_map[item['id']] = txt
    except: pass

    # [NEW] 4. CONSOLIDAÇÃO INTELIGENTE (Smart Consolidation)
    # Substitui a antiga "Limpeza Final" que forçava merge.
    # Agora:
    # - Se for parecido (> 0.60): Funde (corrige fragmentação).
    # - Se for diferente (< 0.60): Mantém (respeita personagens secundários).
    
    cb(90, 1, "Realizando Consolidação Inteligente de vozes...")
    
    # Recarrega estado atual (pois pastas podem ter mudado no Merge Interativo)
    voice_folders = [d for d in diarization_dir.iterdir() if d.is_dir() and d.name.startswith('voz')]
    
    valid_voices = []      # > 10s ou > 5 arquivos (Vozes Principais)
    questionable_voices = [] # < 10s e < 5 arquivos (Vozes Curta/Duvidosas)
    
    # Recalcula centroides rapidinho
    import torchaudio
    
    def get_folder_stats(folder):
        wavs = list(folder.glob("*.wav"))
        wavs = [w for w in wavs if "_REF_" not in w.name]
        if not wavs: return None
        
        # [OPTIMIZATION] Fast Pass - Critério de Duração
        # O usuário pediu: se < 10s, tenta juntar. Se >= 10s, é válido.
        # Se tiver MUITOS arquivos (> 15), assumimos que é válido para não rodar ffprobe em tudo.
        if len(wavs) > 15:
             total_duration = 999.0
        else:
             total_duration = 0.0
             for w in wavs:
                 try: total_duration += get_audio_duration(w)
                 except: pass
                 if total_duration >= 10.0: break # Já bateu a meta, para de gastar CPU

        # Centroid Calculation
        embeddings = []
        samples = random.sample(wavs, min(len(wavs), 5))
        for w in samples:
             try:
                 emb = diarizer.get_file_embedding(str(w))
                 if emb is not None: embeddings.append(emb)
             except: pass
        
        if not embeddings: return None
        centroid = np.mean(embeddings, axis=0)
        
        return {
            'folder': folder,
            'centroid': centroid,
            'duration': total_duration,
            'file_count': len(wavs)
        }

    stats_list = []
    for vf in voice_folders:
        s = get_folder_stats(vf)
        if s: stats_list.append(s)
        
    # Classifica
    for s in stats_list:
        # [CRITÉRIO DE OURO] Duração > 10s define 'Voz Válida'
        if s['duration'] >= 10.0:
            valid_voices.append(s)
        else:
            questionable_voices.append(s)
            
    # Processa Questionáveis
    count_merged = 0
    count_kept = 0
    
    for q in questionable_voices:
        best_match = None
        best_score = -1.0
        
        # Compara com Válidos
        for v in valid_voices:
            dist = cosine_similarity([q['centroid']], [v['centroid']])[0][0]
            if dist > best_score:
                best_score = dist
                best_match = v
                
        # threshold = 0.60 (Smart Merge)
        if best_match and best_score > 0.60:
            logging.info(f"[SMART MERGE] Fundindo {q['folder'].name} -> {best_match['folder'].name} (Sim: {round(best_score, 2)})")
            for f in q['folder'].glob("*.wav"):
                try: shutil.move(str(f), str(best_match['folder'] / f.name))
                except: pass
            try: q['folder'].rmdir()
            except: pass
            count_merged += 1
        else:
            logging.info(f"[SMART KEEP] Mantendo {q['folder'].name} (Sim Máx: {round(best_score, 2)} < 0.60)")
            count_kept += 1
            
    cb(100, 1, f"Consolidação: {count_merged} fundidos, {count_kept} mantidos.")
    # Fim da Consolidação

    # [FIX CLEANUP] Código residual removido.
    # A consolidação já foi feita no loop anterior (valid_voices vs questionable_voices).
    # As vozes que sobraram em 'questionable_voices' permanecem como vozes independentes (curtas).
    if not valid_voices and questionable_voices:
         logging.info("Aviso: Todas as vozes detectadas são curtas (<10s). Mantendo originais.")

    # [NEW] 5. GERAÇÃO DO ARQUIVO DE REFERÊNCIA GATO_NET (Agora sempre NO FINAL DE TUDO)
    cb(95, 1, "Gerando áudios de referência unificados para as vozes...")
    
    # Recarrega as pastas agora que a fusão final terminou
    final_voice_folders = [d for d in diarization_dir.iterdir() if d.is_dir() and d.name.startswith('voz')]
    
    BAD_REF_WORDS = [
        'argh', 'ah', 'oh', 'uh', 'hmm', 'wow', 'tsk', 'ugh', 'screams', 'gasps', 'moans', 'chokes', 'grita', 'geme', 
        'laughs', 'chuckles', 'sobs', 'cries', 'sighs', 'eh', 'heh', 'hum', 'ha', 'haha', 'hah', 'whoa', 'ooh', 'aw', 
        'ouch', 'ow', 'psst', 'shh', 'yikes', 'yay', 'ew', 'ick', 'boo', 'hiss', 'growl', 'snarl', 'roar', 'bark', 
        'meow', 'purr', 'chirp', 'squeak', 'whimper', 'pant', 'gasp', 'cough', 'sneeze', 'burp', 'hiccup', 'yawn',
        'sniff', 'spit', 'swallow', 'gulp', 'choke', 'rasp', 'groan', 'grunt', 'mumble', 'mutter', 'shout', 'yell',
        'scream', 'shriek', 'wail', 'cry', 'sob', 'laugh', 'giggle', 'chuckle', 'snicker', 'snort', 'wheeze', 'breath',
        'breathing', 'inhale', 'exhale', 'noise', 'sound', 'static', 'interference', 'radio', 'beep', 'boop', 'click',
        'clack', 'bang', 'boom', 'crash', 'thud', 'thump', 'smash', 'crack', 'snap', 'pop', 'fizz', 'buzz', 'whir', 
        'clank', 'clatter', 'rattle', 'rustle', 'scratch', 'scrape', 'scuff'
    ]

    for i, voice_folder in enumerate(final_voice_folders):
        output_ref_path = voice_folder / "_REF_VOZ_UNIFICADA.wav"
        
        # [v10.79] FORCE REFRESH: Se houver mais de um WAV e o unificado for antigo, apaga e refaz.
        wav_files = sorted(list(voice_folder.glob("*.wav")))
        actual_wavs = [w for w in wav_files if not w.name.startswith("_REF_")]
        
        if output_ref_path.exists():
            # [v21.26] Verificação Dupla: MTime + Contagem de Arquivos
            folder_mtime = voice_folder.stat().st_mtime
            ref_mtime = output_ref_path.stat().st_mtime
            
            # Se a pasta é mais nova ou se o usuário mudou os arquivos manualmente
            if ref_mtime < folder_mtime:
                logging.info(f"[REF UPDATE] Pasta '{voice_folder.name}' atualizada. Regenerando referência...")
                try: output_ref_path.unlink()
                except: pass
            else:
                continue
        if not wav_files: continue
        
        valid_wavs = []
        for w in wav_files:
            if w.name.startswith("_REF_"): continue
            if w.stat().st_size < 8000: continue 

            fid = w.stem
            if fid in project_text_map:
                text = project_text_map[fid]
                if len(text) < 15 and any(bad in text.lower() for bad in BAD_REF_WORDS): continue
                if len(text) < 2: continue
            
            valid_wavs.append(w)

        if not valid_wavs: 
            valid_wavs = [w for w in wav_files if not w.name.startswith("_REF_") and w.stat().st_size > 8000]

        valid_wavs.sort(key=lambda x: x.stat().st_size, reverse=True)
        top_files = valid_wavs[:50] # Pega mais arquivos para garantir duração
        
        combined_audio = AudioSegment.empty()
        total_dur = 0
        
        for wav_file in top_files:
            try: 
                # O áudio original já passou pelo DeepFilter na Fase 1.
                # Não devemos passar de novo para não abafar/apagar a voz.
                seg = AudioSegment.from_wav(wav_file)
                
                if len(seg) < 800: continue
                
                # [v10.79] VAD Trimming na Referência: Remove silêncios e ruidos nas pontas
                from pydub.silence import detect_nonsilent
                nonsilent = detect_nonsilent(seg, min_silence_len=100, silence_thresh=-40)
                if nonsilent:
                    seg = seg[nonsilent[0][0]:nonsilent[-1][1]]
                
                # Adiciona com crossfade para um som mais contínuo
                if len(combined_audio) > 0:
                    combined_audio = combined_audio.append(seg, crossfade=50)
                else:
                    combined_audio = seg

                total_dur += len(seg)
                if total_dur > 15000: break # [v10.66] 15s é o ideal para o Chatterbox
            except Exception as e: logging.error(f"Erro ref unificada {wav_file}: {e}")
        
        if len(combined_audio) > 0:
            temp_combined_path = voice_folder / "_temp_combined.wav"
            combined_audio.export(temp_combined_path, format="wav")
            try:
                # [v10.66] INTELLIGENT BYPASS & STUDIO-SAFE ANALYTICS
                # Mede se o áudio realmente precisa de limpeza pesada (Radio/Ruído)
                threads = str(max(1, (os.cpu_count() or 4) // 2))
                samples_health = np.array(combined_audio.get_array_of_samples())
                ref_rms = np.sqrt(np.mean(samples_health.astype(np.float32)**2)) / 32768.0
                
                is_noisy = ref_rms > 0.008 # Threshold cirúrgico para detectar rádio/chuveiro
                
                if is_noisy:
                    logging.info(f"Voz {voice_folder.name}: Ruído detectado (RMS {ref_rms:.4f}). Removendo apenas graves inúteis (rumble).")
                    # O "Brilho" agressivo estava deixando a voz fina.
                    # Como o DeepFilter já atuou na Fase 1, aplicamos apenas um filtro passa-alta leve para rumble.
                    af_filters = "highpass=f=80,aresample=22050"
                else:
                    logging.info(f"Voz {voice_folder.name}: Qualidade de Estúdio detectada (RMS {ref_rms:.4f}). Bypass Cleaner ativado.")
                    af_filters = "aresample=22050" # Apenas resample padrão para o Chatterbox

                # [v10.67] Persiste o perfil acústico para o "Meio Termo" posterior
                speaker_profile = voice_folder / "acoustic_profile.json"
                safe_json_write({"is_noisy": bool(is_noisy), "rms": float(ref_rms)}, speaker_profile)

                cmd = ['ffmpeg', '-threads', threads, '-y', '-i', str(temp_combined_path), 
                       '-af', af_filters, '-ac', '1', '-ar', '22050', str(output_ref_path)]
                subprocess.run(cmd, check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as e: combined_audio.export(output_ref_path, format="wav")
            finally:
                if temp_combined_path.exists(): os.remove(temp_combined_path)

    # [OPTIMIZATION] Marca a unificação como concluída
    with open(marker_path, 'w') as f:
        f.write("done")
    
    cb(100, 1, "Vozes unificadas e limpas.")

def detect_game_genre(segments):
    """
    [v12.18 CACHALEÃO] Identifica o gênero do jogo baseado nos diálogos iniciais.
    Permite que o programa mude de "personalidade" conforme o jogo.
    """
    if not segments: return "Ação (Geral)"
    
    # Pega os primeiros 15 textos para dar uma boa amostragem
    sample_text = " / ".join([s['original_text'] for s in segments[:15]])
    
    prompt = f'''
Diga APENAS qual e o Genero deste jogo (Ex: 'Acao e Guerra', 'Corrida', 'RPG', 'Terror') baseado nas seguintes falas da cena:
"{sample_text}"
'''
    
    payload = {
        "messages": [{"role": "user", "content": prompt}], 
        "temperature": 0.1, 
        "max_tokens": 50
    }
    
    try:
        # [v12.28 FIX] Usa is_translation=False(pt->pt) para modo analista (sem Parrot)
        response = make_gema_request_with_retries(payload, is_translation=False)
        genre = response.json()['choices'][0]['message']['content'].strip().replace('"', "")
        
        # Filtra parroting extremo
        if "Ação e Guerra" in genre or "Gênero deste jogo" in genre:
             genre = "Ação e Tiro"
             
        logging.info(f"🦎 [CAMALEÃO] Gênero detectado pela IA: {genre}")
        return genre

    except Exception as e:
        logging.warning(f"Aviso: Falha na deteção automática de gênero ({e}). Usando fallback 'Ação'.")
        return "Ação (Geral)"

# --- GEMA LOCAL (LLAMA-CPP) SINGLETON ---
gema_instance = None
gema_lock = Lock()

def get_gema_model():
    """
    [v22.55 DIAGNÓSTICO]
    Inicializa e carrega o GGUF com verificações extras de hardware.
    """
    global gema_instance
    with gema_lock:
        if gema_instance is None:
            # Verificação de segurança para o motor local
            if not HAS_LLAMA_CPP:
                logging.warning("Motor local indisponível. Continuando via API...")
                return None
            import torch
            import os
            
            # Detecta Hardware
            gpu_layers = -1 if torch.cuda.is_available() else 0
            model_path_abs = str(Path(__file__).parent / "Models" / "gemma-4-E4B-it-Q4_K_M.gguf")
            
            if not os.path.exists(model_path_abs):
                 logging.error(f"❌ MODELO NÃO ENCONTRADO: {model_path_abs}")
                 return None

            # Diagnóstico de arquivo
            file_size_gb = os.path.getsize(model_path_abs) / (1024**3)
            logging.info(f"📂 Verificando arquivo GGUF: {file_size_gb:.2f} GB")
            
            if file_size_gb < 1.0:
                logging.error("❌ ERRO: O arquivo do modelo parece estar corrompido (muito pequeno).")
                return None

            logging.info(f"🚀 Iniciando carregamento (Modo Seguro)...")
            try:
                if not HAS_LLAMA_CPP: return None
                gema_instance = Llama(
                    model_path=model_path_abs,
                    n_ctx=4096, 
                    n_threads=2,
                    n_gpu_layers=gpu_layers,
                    verbose=False
                )
                logging.info("✅ Gema Local carregada com sucesso!")
            except Exception as e:
                logging.warning(f"⚠️ O motor local não suporta este modelo específico (Erro: {e})")
                logging.info("💡 Sugestão: Abra este modelo no LM Studio e o programa usará ele como ponte automaticamente!")
                gema_instance = None
    return gema_instance

def gema_inference(prompt, system_prompt="Você é um tradutor profissional.", model_type="gema"):
    """
    [v22.60 INDESTRUTÍVEL] 
    Tenta Local GGUF -> Se falhar ou não existir, tenta LM Studio (Porta 5000 ou 1234)
    """
    # 1. Tenta Local (Llama-cpp)
    local_gema = get_gema_model()
    if local_gema:
        try:
            response = local_gema.create_chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            return response['choices'][0]['message']['content']
        except Exception as e:
            logging.error(f"Erro na geração nativa (llama.cpp): {e}")

    # 2. Se falhar o local, tenta o LM Studio (Bypass / Fallback)
    urls = ["http://127.0.0.1:1234/v1/chat/completions", "http://localhost:1234/v1/chat/completions"]
    for url in urls:
        try:
            import requests
            payload = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3,
                "model": "local-model"
            }
            res = requests.post(url, json=payload, timeout=60)
            if res.status_code == 200:
                return res.json()['choices'][0]['message']['content']
        except:
            continue

    return "ERRO: IA não disponível (Nem local, nem LM Studio)."

# Aliases de compatibilidade para não quebrar o código existente
get_llama_instance = get_gema_model

def unload_gema_model():
    """
    Libera memória RAM ocupada pelo LLM.
    """
    global gema_instance
    with gema_lock:
        if gema_instance is not None:
            logging.info("Descarregando Gema Local da RAM...")
            del gema_instance
            gema_instance = None
            gc.collect()

def wait_for_gema_service(progress_callback):
    """
    [v22.70 ESTRATÉGIA PRIORIDADE LM STUDIO]
    Tenta se conectar primeiro ao LM Studio (que suporta Gemma 4 perfeitamente).
    Se não encontrar nada ligado lá, tenta o motor local.
    """
    progress_callback("Verificando Conexão com LM Studio (Gemma 4)...")
    
    # 1. Tenta ver se o LM Studio está aberto (Porta 1234)
    import requests
    try:
        res = requests.get("http://127.0.0.1:1234/v1/models", timeout=3)
        if res.status_code == 200:
            logging.info("✅ LM Studio Detectado! Usando cérebro externo (Alta Performance).")
            return True
    except:
        logging.info("ℹ️ LM Studio não detectado na porta 1234. Tentando motor local...")

    # 2. Se falhar o LM Studio, tenta carregar o modelo local
    progress_callback("LM Studio offline. Carregando motor nativo (GGUF)...")
    try:
        llm = get_llama_instance()
        if llm:
            logging.info("✅ Motor Local (llama.cpp) ativo como plano B.")
            return True
        else:
            raise RuntimeError("Nenhum motor de IA disponível (Abra o LM Studio!)")
    except Exception as e:
        msg = f"ERRO: IA não encontrada. Certifique-se de que o LM Studio está aberto na porta 1234."
        logging.error(msg)
        raise RuntimeError(msg)

# Bypassed. Redefinido no topo.
def check_lm_studio_placeholder():
    return True

def clean_ai_translation(text, original_text):
    """
    [v20.0 EXTRAÇÃO POR ASPAS (SUGESTÃO DO USUÁRIO)]
    Pesca a tradução baseada na última ocorrência de aspas duplas.
    Isso ignora completamente formatos como "Inglês" -> "Português".
    """
    if not text: return ""
    
    # 1. PESCARIA DE ASPAS (A SOLUÇÃO DEFINITIVA)
    # Se existirem aspas no texto (ex: "Inglês" -> "Português")
    if text.count('"') >= 2:
        import re
        # Pega tudo que está entre aspas
        textos_entre_aspas = re.findall(r'"([^"]*)"', text)
        if textos_entre_aspas:
            # Pega sempre a última aspas (que obrigatoriamente será o português)
            candidato = textos_entre_aspas[-1].strip()
            
            # [v20.5 ANTI-HALUCINAÇÃO]: Limpa a tag (Limite: Xs) se a IA for preguiçosa e copiar.
            candidato = re.sub(r'\(Limite:.*?\)', '', candidato).strip()
            
            # Validação anti-falha: Se por algum motivo bizarro ele pegar o inglês
            orig = original_text.strip().strip('"').lower() if original_text else ""
            
            # Remove pontuação básica para comparar se é só um eco do inglês
            candidato_limpo = re.sub(r'[^\w\s]', '', candidato.lower())
            orig_limpo = re.sub(r'[^\w\s]', '', orig)
            
            if orig_limpo and candidato_limpo == orig_limpo:
                # Retorna o texto sujo para o fallback limpar depois, 
                # em vez de retornar o idioma errado.
                pass 
            else:
                return candidato

    # --- FALLBACK DE LIMPEZA CLÁSSICA CASO ELE ESQUEÇA AS ASPAS ---
    t = text.strip().strip('"')
    
    # [v20.5 ANTI-HALUCINAÇÃO]
    t = re.sub(r'\(Limite:.*?\)', '', t).strip()
    
    orig = original_text.strip().strip('"') if original_text else ""
    
    separadores = [" -> ", " => ", " : ", " - "]
    
    for sep in separadores:
        if sep in t:
            parts = t.split(sep)
            primeira_parte = parts[0].strip().strip('"').lower()
            if orig and (orig.lower() in primeira_parte or primeira_parte in orig.lower()):
                return parts[-1].strip().strip('"')
            if len(parts) > 1:
                return parts[-1].strip().strip('"')

    if orig and t.lower().startswith(orig.lower()):
        rest = t[len(orig):].strip()
        rest = re.sub(r'^[:\-= \t>]+', '', rest).strip().strip('"')
        if rest: return rest

    return t

def make_gema_request_with_retries(payload, timeout=3600, retries=5, backoff_factor=2, is_translation=True):
    """
    [v22.70 MULTI-MODO]
    Prioriza o LM Studio via HTTP. Se falhar, usa o motor local llama-cpp.
    """
    import requests
    
    # --- TENTATIVA 1: LM STUDIO (RECOMENDADO PARA GEMMA 4) ---
    url = "http://127.0.0.1:1234/v1/chat/completions"
    try:
        res = requests.post(url, json=payload, timeout=60)
        if res.status_code == 200:
            return res # Retorna o objeto original do requests
    except:
        pass # Segue pro plano B

    # --- TENTATIVA 2: MOTOR LOCAL (LLAMA-CPP) ---
    llm = get_llama_instance()
    if llm:
        try:
            messages = payload.get("messages", [])
            temp = payload.get("temperature", 0.3)
            max_tk = payload.get("max_tokens", 4096)
            
            response_data = llm.create_chat_completion(
                messages=messages,
                temperature=temp,
                max_tokens=max_tk
            )
            
            # Simula objeto do requests para não quebrar o resto do app
            class MockResponse:
                def __init__(self, json_data):
                    self._json_data = json_data
                    self.status_code = 200
                def json(self): return self._json_data
                def raise_for_status(self): pass
            
            return MockResponse(response_data)
        except Exception as e:
            logging.error(f"Erro no motor local (llama-cpp): {e}")

    raise RuntimeError("❌ FALHA GERAL: LM Studio está fechado e o motor local deu erro.")

def gema_batch_processor_v2(batch, cenario_ctx, glossary={}, profile_id='padrao', job_dir=None):
    """
    [v14.00 UNIFICAÇÃO MASTER] - Processamento de Etapa Única (Single-Pass)
    [v16.50 TRAVA DE SEGURANÇA]: Detecta respostas vazias e ultra-rápidas.
    [v18.50 REGEX ULTRA-ROBUSTA]: Suporta numerações (1., 2.) e logs de diagnóstico.
    """
    if not batch: return {}
    
    start_time = time.time()
    profile = load_game_profile(profile_id)
    lore_text = profile.get("lore", "Gênero: Jogo de Aventura/Ação (Autodetecção Ativada)")
    
    keywords = ["cod", "combate", "militar", "guerra", "tiro", "soldado", "army", "stalker", "tactical"]
    is_action = any(x in lore_text.lower() or x in cenario_ctx.lower() or x in profile_id.lower() for x in keywords)
    
    protocolo_extra = (
        "3. PROTOCOLO DE RÁDIO: 'Roger' = Copiado (OBRIGATÓRIO). 'Over' = Câmbio.\n"
        "4. CALLSIGNS E VEÍCULOS (VITAL):... \n"
    ) if is_action else (
        "3. FIDELIDADE EMOCIONAL: Foque na naturalidade...\n"
    )

    prompt = f'''
Voce e um Tradutor Literario de Elite.
Sua missao e criar a MELHOR traducao possivel para o Portugues do Brasil (PT-BR).
Foque 100% na naturalidade do dialogo, na forca narrativa e na emocao da cena, para que soe como uma dublagem oficial de estudio.
Mantenha a COERENCIA e o FLUXO da conversa entre as frases da lista, garantindo que as respostas facam sentido com as perguntas anteriores.

[PERFIL DO JOGO E LORE]:
{lore_text}

[GLOSSARIO DE TERMOS]:
{glossary if glossary else "Nenhum termo especifico definido."}

[CONTEXTO DA CENA]:
{cenario_ctx if cenario_ctx else "Cena de jogo padrao."}
{protocolo_extra}

[DIRETRIZES TECNICAS E ARTISTICAS]:
1. CONCISAO: Priorize frases concisas para caber no tempo de dublagem (Max 18 CPS).
2. NATURALIDADE: Use linguagem coloquial brasileira (ex: 'pra' em vez de 'para', 'voce' ou o jeito que o personagem falaria).
3. VERBOS DE ACAO: Nao seja literal. 'Hit the gate' pode ser 'Va ao portao'. 'Grab' pode ser 'Pegue'.
4. FLUIDEZ: Se a traducao soar como um robo, reescreva de forma que um dublador real diria.

[PADRAO OBRIGATORIO DE RESPOSTA]:
meu_id_01: "Sua traducao brilhante OBRIGATORIAMENTE PT-BR aqui"

[LISTA DE FRASES]:
'''
    for item in batch:
        speaker = item.get('speaker', 'Voz Desconhecida')
        prompt += f"- {item['id']} ({speaker}): \"{item.get('original_text', '')}\"\n"

    prompt += "\nResponda APENAS com o formato acima. Use aspas APENAS para o texto em português."

    payload = {
        "messages": [
            {"role": "system", "content": "<|think|>\nFocarei na emoção e na naturalidade perfeita para PT-BR. Retornarei apenas o formato ID: \"tradução livre e fluida\"."},
            {"role": "user", "content": prompt}
        ], 
        "temperature": 0.3, "max_tokens": 4096
    }
    
    try:
        response = make_gema_request_with_retries(payload, is_translation=True)
        content = response.json()['choices'][0]['message']['content']
        
        # [v18.50 LOG DE DIAGNÓSTICO ATÔMICO]
        if job_dir:
            try:
                log_file = Path(job_dir) / "ia_batch_debug.log"
                with open(log_file, "a", encoding="utf-8") as f_log:
                    f_log.write(f"\n--- BATCH {datetime.now()} ---\n{content}\n")
            except: pass

        results_map = {}
        # Mapeia IDs para textos originais para uso na limpeza de eco
        orig_map = {str(item['id']).lower(): item.get('original_text', '') for item in batch}
        
        # Regex v19.0: Ultra-robusta
        item_pattern = r'(?:^|\n)[ \t]*(?:[0-9]+\.?[ \t]*)?(?:id\s*[:\-]\s*)?([a-zA-Z0-9_\-\.]+)\s*[:\-=>]+\s*"?\s*(.*?)\s*"?(?=\n[ \t]*(?:[0-9]+\.?[ \t]*)?(?:id\s*[:\-]\s*)?[a-zA-Z0-9_\-\.]+\s*[:\-=>]+|$)'
        matches_found = re.finditer(item_pattern, content, re.DOTALL)
        
        for m_obj in matches_found:
            clean_id = m_obj.group(1).strip().lower()
            val = m_obj.group(2).strip()
            
            original_of_id = orig_map.get(clean_id, "")
            val = clean_ai_translation(val, original_of_id)
            
            if val.startswith('"') and val.endswith('"'): val = val[1:-1]
            results_map[clean_id] = val

        if not results_map and content.strip():
            logging.info("   -> 🦎 [PARSER] Regex principal falhou. Tentando captura de emergência por IDs diretos...")
            for item in batch:
                f_id = item['id']
                escaped_id = re.escape(f_id)
                emergency_pattern = rf'{escaped_id}\s*[:\-=>]+\s*(.*?)(?=\n|$)'
                m_emergency = re.search(emergency_pattern, content, re.IGNORECASE)
                if m_emergency:
                    results_map[f_id.lower()] = m_emergency.group(1).strip().strip('"')
            
        elapsed_time = time.time() - start_time
        if not results_map and elapsed_time < 5.0:
            raise RuntimeError(f"FALHA NA TRADUÇÃO: Resposta vazia em {round(elapsed_time, 2)}s.")

        if not results_map and len(batch) == 1:
            quoted_fallback = re.search(r'"(.*?)"', content, re.DOTALL)
            if quoted_fallback:
                results_map[batch[0]['id'].lower()] = quoted_fallback.group(1).strip()

        return results_map
    except Exception as exc:
        if "FALHA NA TRADUÇÃO" in str(exc): raise exc
        logging.error(f"Erro no Master Sync (Lote): {exc}")
        return {}

def gema_atomic_processor_v3(item, context_window_str, glossary={}, profile_id='padrao', job_dir=None):
    """
    [v21.10 AGENTE ATÔMICO REFINADO]
    Usa o perfil do jogo para injetar personalidade e contexto tático na tradução.
    """
    profile = load_game_profile(profile_id)
    ai_style = profile.get("ai_instructions", "Estilo: Tradução natural e orgânica (PT-BR).")
    
    # [v21.11] Injeção de Glossário
    glossary_str = ""
    if glossary:
        glossary_items = [f"- {k}: {v}" for k, v in glossary.items()]
        glossary_str = "\n[GLOSSÁRIO OBRIGATÓRIO]:\n" + "\n".join(glossary_items)
    
    prompt = f"""Você é um Diretor de Localização especializado em Dublagem de Games.
{ai_style}
{glossary_str}

[OBJETIVO]: Traduza para um Português (Brasil) fluído, que soe como uma conversa real entre pessoas naquela situação. Evite traduções literais.

[REGRAS DE OURO]:
1. **CONCISÃO TÁTICA**: Em cenas de ação, use frases curtas e diretas.
2. **CONTEXTO INTELIGENTE**: Use as frases ao redor para entender o que está acontecendo (Ex: se falam de explosivos, "Reach" provavelmente é "Breach/Invasão").
3. **PONTUAÇÃO EMOCIONAL**: Use vírgulas para pausas naturais e exclamações para urgência.
4. **LIMITE SEGURO**: Tente não ultrapassar {int(item.get("duration", 0) * 18.2)} caracteres para a frase caber no tempo.

[CONTEXTO RECENTE]:
{context_window_str}

[FRASE PARA ADAPTAR]:
ID: {item['id']} | EN: "{item.get('original_text', '')}"

[RESPOSTA]: Retorne APENAS a tradução final entre aspas duplas.
"""

    payload = {
        "messages": [
            {"role": "system", "content": "Serei ultra-preciso. Retornarei apenas a tradução da frase alvo entre aspas, respeitando o contexto das vizinhas."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3, "max_tokens": 4096
    }

    try:
        response = make_gema_request_with_retries(payload, is_translation=True)
        content = response.json()['choices'][0]['message']['content'].strip()
        
        # v19.5 Trava de Pureza 2.0
        final_text = clean_ai_translation(content, item.get('original_text', ''))
        
        # Limpeza caso a IA teime em colocar o ID
        if ":" in final_text and item['id'].lower() in final_text.lower()[:20]:
            final_text = final_text.split(":", 1)[-1].strip()
            
        return final_text.strip().strip('"')
    except Exception as e:
        logging.error(f"Erro no Processador Atômico [{item['id']}]: {e}")
        return item.get('original_text', '') # Fallback seguro

def same_word_count_check(original, translated):
    """
    Heurística simples para detectar tradução literal (Google Tradutor).
    Se o número de palavras for idêntico e a frase for longa, há risco de literalidade excessiva.
    """
    words_orig = len(original.split())
    words_trans = len(translated.split())
    
    # Se a contagem de palavras é idêntica e a frase tem mais de 5 palavras
    if words_orig == words_trans and words_orig > 5:
        return True
    return False

def gema_etapa_correcao_master(original_text, current_translation, duration, reason="sincronia", profile_id='padrao'):
    """
    [v14.50 CORRETOR MASTER] - O Agente que resolve tudo.
    Recebe um diagnóstico (ex: muito longo, robótico) e corrige a frase.
    """
    profile = load_game_profile(profile_id)
    lore_text = profile.get("lore", "Gênero: Jogo de Aventura/Ação")
    
    target_chars = int(duration * 18)
    
    instrucao_especifica = ""
    if reason == "sincronia":
        instrucao_especifica = f"A frase está MUITO LONGA ({len(current_translation)} letras para {round(duration, 2)}s). ADAPTE E ENCURTE para no máximo {target_chars} letras, mantendo a alma do diálogo."
    elif reason == "qualidade":
        instrucao_especifica = "A tradução está 'robótica' ou parece Google Tradutor. RE-ESCREVA de forma mais natural, usando expressões que um dublador brasileiro usaria."
    else:
        instrucao_especifica = "A tradução falhou ou está inconsistente. Corrija para um Português fluído e natural."

    dur_format = round(duration, 2)
    prompt = f'''
VOCE E UM DIRETOR DE DUBLAGEM EXPERIENTE. Sua missao e CORRIGIR uma traducao que falhou nos criterios de qualidade.

[DIAGNOSTICO DO ERRO]:
{instrucao_especifica}

[DADOS]:
Original (EN): "{original_text}"
Traducao Atual (RUIM): "{current_translation}"
Tempo Limite: {dur_format}s (Maximo {target_chars} letras para 18 CPS)
Lore: {lore_text}

[MISSAO]:
Responda APENAS com a versao corrigida, natural e dentro do tempo. Use aspas duplas: "Sua versao corrigida aqui!"
'''

    try:
        payload = {
            "messages": [
                {"role": "system", "content": "<|think|>\nVocê é um Diretor de Localização. Responda apenas o texto corrigido entre aspas. Proibido conversar."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 4096
        }
        
        response = make_gema_request_with_retries(payload, is_translation=False)
        content = response.json()['choices'][0]['message']['content'].strip()
        
        quoted_match = re.search(r'"(.*?)"', content, re.DOTALL)
        if quoted_match:
             return quoted_match.group(1).strip()
        
        return current_translation # Fallback se a IA falhar na correção
        
    except Exception as e:
        logging.error(f"Erro no Agente de Correção Master: {e}")
        return current_translation


def gema_batch_corrector_master(failed_items, cenario_ctx, profile_id='padrao', job_dir=None):
    """
    [v14.60 SUPER TURBO BATCH CORRECTOR]
    Corrige múltiplas traduções ruins de uma só vez para máxima performance.
    [v18.50 REGEX ULTRA-ROBUSTA]
    """
    if not failed_items: return {}
    
    profile = load_game_profile(profile_id)
    lore_text = profile.get("lore", "Gênero: Jogo de Aventura/Ação")
    
    prompt = f'''
VOCE E UM CORRETOR DE DUBLAGEM SENIOR. Sua missao e ajustar traducoes que falharam na qualidade ou sincronia.

[REGRAS CRITICAS]:
1. Responda APENAS a lista no formato id: "Traducao Corrigida"
2. PROIBIDO repetir o ingles original.
3. PROIBIDO explicacoes ou comentarios.
4. Use APENAS aspas duplas: "..."
5. NUNCA use setas (->).

[EXEMPLO]:
id_01: "Essa e a versao limpa e corrigida!"

[LORE]: {lore_text} | Contexto: {cenario_ctx}
'''
    for item in failed_items:
        prompt += f"- {item['id']}: \"{item.get('original_text', '')}\" -> Atualmente: \"{item.get('translated_text', '')}\"\n"

    try:
        payload = {
            "messages": [
                {"role": "system", "content": "<|think|>\nVocê é um Corretor de Dublagem. Responda apenas o ID e o texto entre aspas. Proibido conversar."},
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

        # Extrator Robusto Master (Mesmo do Batch v19.0)
        results = {}
        item_pattern = r'(?:^|\n)[ \t]*(?:[0-9]+\.?[ \t]*)?(?:id\s*[:\-]\s*)?([a-zA-Z0-9_\-\.]+)\s*[:\-=>]+\s*"?\s*(.*?)\s*"?(?=\n[ \t]*(?:[0-9]+\.?[ \t]*)?(?:id\s*[:\-]\s*)?[a-zA-Z0-9_\-\.]+\s*[:\-=>]+|$)'
        matches = re.finditer(item_pattern, content, re.DOTALL)
        
        for match in matches:
            clean_id = match.group(1).strip().lower()
            val = match.group(2).strip().strip('"')
            results[clean_id] = val
            
        return results

    except Exception as e:
        logging.error(f"Erro no Batch Corrector Master: {e}")
        return {}

def gema_vibe_master_analyzer(batch_items, cenario_ctx):
    """
    [v16.0 VIBE MASTER] - O Cérebro de Tom.
    Analisa o lote de frases em inglês e define o "clima" emocional.
    """
    if not batch_items: return "ZOEIRA_LIBERADA"
    
    prompt = f'''
VOCE E UM DIRETOR DE VIBE E TOM EMOCIONAL.
Sua missao e classificar o lote de frases em ingles.

[REGRA DE OURO - SEJA EXIGENTE COM O DRAMA]:
- SO use 'DRAMA_PESADO' se o texto descrever: morte tragica, choro solucante, funeral ou trauma profundo.
- Se for COMBATE, TIRO, ACAO, CONVERSA DE BAR, NPCs ou DIALOGO GENERICO: Escolha 'ZOEIRA_LIBERADA'.
- NA DUVIDA: Sempre escolha 'ZOEIRA_LIBERADA'. Nos queremos alma brasileira e naturalidade agressiva na maior parte do tempo.

[CONTEXTO ATUAL]: {cenario_ctx}

[LISTA DE FRASES DO LOTE]:
'''
    for item in batch_items:
        prompt += f"- \"{item.get('original_text', '')}\"\n"

    prompt += "\nResponda APENAS um JSON no formato: {\"vibe\": \"URGENTE ou NARRATIVO\", \"genero\": \"MILITAR ou SOCIAL\", \"auditoria\": {\"frase_original_aqui\": \"frase_corrigida_caso_nonsense\"}}"

    try:
        payload = {
            "messages": [
                {"role": "system", "content": "Você é Analista de Vibe. Responda apenas a tag. Proibido conversar."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 64
        }
        
        response = make_gema_request_with_retries(payload, is_translation=False)
        content = response.json()['choices'][0]['message']['content'].strip()
        
        # Tenta extrair o JSON da IA
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
            except: pass
            
        # [v18.0 HEURÍSTICA E AUDITORIA]
        # Aplica a auditoria nas strings do lote
        for item in batch_items:
            orig = item.get('original_text', '')
            if orig in auditoria:
                logging.info(f"   -> 🦎 [AUDITORIA] Corrigindo inglês: '{orig}' -> '{auditoria[orig]}'")
                item['original_text'] = auditoria[orig]

        # Se IA falhar no JSON, mas palavras de combate estiverem lá, força o gênero MILITAR
        all_text = " ".join([item.get('original_text', '').lower() for item in batch_items])
        combat_keywords = ["clear", "contact", "roger", "frag", "hostile", "target", "enemy", "fire", "secure", "sector", "area", "balcony"]
        if any(w in all_text for w in combat_keywords):
            genero = "MILITAR"

        return {"vibe": vibe, "genero": genero}

    except Exception as e:
        logging.error(f"Erro no Vibe Master / Auditor: {e}")
        return {"vibe": "URGENTE", "genero": "SOCIAL"}

def agente_2_matematico_python(texto_pt, duration):
    """
    [O Fiscal Matemático Frio - Agente 2]
    Calcula se a tradução PT-BR caberá mecanicamente no limitador TTS.
    """
    if not texto_pt or duration <= 0:
        return {"aprovado": False, "dossie": "Dados insuficientes ou texto vazio."}
        
    # [ESTRUTURA DE TEMPO DO CHATTERBOX 2026]
    # O Chatterbox consegue falar de forma compreensível até 18.5 caracteres por segundo
    MAX_CPS = 18.5
    limite_max_caracteres = int(duration * MAX_CPS)
    
    # O tamanho visual do texto afeta menos o motor de voz do que as vírgulas (que causam pausas forçadas)
    commas = texto_pt.count(',')
    pontos = texto_pt.count('.') + texto_pt.count('!') + texto_pt.count('?')
    
    # Cada pausa artificial equivale a mais ou menos meio segundo perdidos.
    # Em "letras virtuais" que ocupam espaço:
    peso_pausas_em_caracteres = (commas * 8) + (pontos * 10)
    
    tamanho_efetivo = len(texto_pt) + peso_pausas_em_caracteres
    
    # Aprovação direta!
    if tamanho_efetivo <= limite_max_caracteres:
        return {"aprovado": True, "dossie": ""}
        
    estouro = tamanho_efetivo - limite_max_caracteres
    
    # Elabora o dossiê perfeitamente mastigado para a Inteligência do Agente 3
    dossie = (
        f"ALERTA DE SINCRONIA DE TEMPO! "
        f"Nós temos apenas {round(duration, 2)} segundos, o que permite um tamanho MÁXIMO de {limite_max_caracteres} letras. "
        f"A sua tradução bateu {tamanho_efetivo} letras (estimadas com pausas). "
        f"Você ESTOUROU o tempo. É estritamente OBRIGATÓRIO que você corte, no mínimo, {estouro + 5} letras dessa tradução "
        f"reescrevendo-a de forma natural e resumida."
    )
    return {"aprovado": False, "dossie": dossie}

def agente_3_adaptador_final_lqa(original_text, translated_text, dossie, timeout=3600):
    """
    [O Editor Chefe - Agente 3]
    Só é chamado quando a sirene toca no Fiscal. Ele encurta orações com extremo senso crítico.
    """
    prompt = f'''
VOCE E UM EDITOR DE DUBLAGEM GENIO. A traducao chegou, mas ELA E GRANDE DEMAIS PARA O TEMPO DO AUDIO.

[DIAGNOSTICO DO FISCAL DE TEMPO]:
{dossie}

[INGLES ORIGINAL A TITULO DE CONTEXTO]:
"{original_text}"

[TRADUCAO ORIGINAL - VOCE DEVE ENCURTAR ISSO]:
"{translated_text}"

[SUA TAREFA]:
Reescreva a [TRADUCAO ORIGINAL]. Seja agressivo nos cortes de palavras inuteis. Use contracoes ("Nos estamos" vira "Estamos", "De o" vira "Do", "Para" vira "Pra"). Mantenha a emocao natural do Brasil.

[FORMATO EXIGIDO]:
"Sua adaptacao curtinha final vai aqui dentro das aspas, e MAIS NADA."

Responda APENAS com a nova traducao resumida e perfeita. Nenhuma palavra de explicacao.
'''

    payload = {
        "messages": [
            {"role": "system", "content": "<|think|>\nO texto não cabe! Adaptando, resumindo e retornando só a versão PT-BR reescrita e ultra-condensada dentro de aspas."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2, # Um pouco mais de criatividade para resumir
        "max_tokens": 1024
    }
    
    try:
        response = make_gema_request_with_retries(payload, timeout=timeout, is_translation=False)
        content = response.json()['choices'][0]['message']['content'].strip()
        
        # O modelo já tem a mesma trava de aspas, então herdamos esse parseamento
        return clean_ai_translation(content, original_text)
    except Exception as e:
        logging.error(f"Erro Crítico no Agente 3 Adaptador: {e}")
        return translated_text # Fallback para o excedente, pois é melhor fala estourada do que erro vazio

def select_best_sync_option(original_duration, options_list, original_text):
    """
    Seleciona a melhor opção de sincronização baseada em critérios matemáticos e linguísticos.
    """
    best_opt = None
    best_score = float('inf')
    target_rate = 18.0 # [v12.95 UPDATED] Sincronia de 18 Letras/s (Fórmula do Usuário)
    
    # Validação básica
    valid_options = [opt.strip() for opt in options_list if opt and len(opt.strip()) > 0]
    if not valid_options: return None

    logging.info(f"Avaliando {len(valid_options)} candidatos para duração {round(original_duration, 2)}s...")

    for opt in valid_options:
        # Limpeza básica
        clean_opt = re.sub(r'^\d+[\.\-\)]\s*', '', opt).strip('"').strip()
        if not clean_opt: continue

        if not clean_opt: continue
        
        # Limpeza de Vírgulas Duplas e Pontuação excessiva HALLUCINATED
        clean_opt = re.sub(r',+', ',', clean_opt) # ,, -> ,
        clean_opt = re.sub(r'[\.,;]+$', '', clean_opt) # Remove pontuação final redundante na contagem
        
        # [v12.97 MATH] Custo Real: Letras + Vírgulas INTERNAS (meio segundo cada vírgula)
        # O usuário explicou que vírgulas no FINAL da frase não devem consumir tempo.
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

        e_t_f = round(estimated_time, 2)
        sc_f = round(score, 2)
        logging.info(f"   - Candidato: '{clean_opt}' | Est.Time: {e_t_f}s | Score: {sc_f}")
        if original_duration < 1.2:
            words = clean_opt.split()
            # Penaliza severamente 1 palavra isolada, a menos que o original seja curto também
            if len(words) < 2 and len(original_text.split()) > 1:
                score += 50 
            # Bônus para frases nominais completas (ex: "Perímetro perdido")
            if len(words) >= 2:
                score -= 5

        logging.info(f"   - Candidato: '{clean_opt}' | CPS: {cps_letters:.1f} | Score: {score:.1f}")

        if score < best_score:
            best_score = score
            best_opt = clean_opt
            
    return best_opt

def apply_string_fallback(text, max_chars):
    """
    Fallback de emergência: Remove artigos e advérbios se o texto estourar o limite.
    Útil quando o LLM falha em respeitar o tamanho.
    """
    if len(text) <= max_chars: return text
    
    # 1. Remove Advérbios (-mente)
    words = text.split()
    new_words = [w for w in words if not w.lower().endswith('mente')]
    new_text = " ".join(new_words)
    if len(new_text) <= max_chars: return new_text
    
    # 2. Remove Artigos (o, a, os, as, um, uns...)
    blacklist = ['o', 'a', 'os', 'as', 'um', 'uma', 'uns', 'umas', 'do', 'da', 'dos', 'das', 'no', 'na', 'nos', 'nas']
    new_words = [w for w in new_words if w.lower() not in blacklist]
    return " ".join(new_words)

def gema_lqa_reviewer_pro(original_en, candidate_pt, duration):
    """
    [PASSO 2 - LQA] O Gemma atua como um Editor Sênior de Dublagem para revisar a naturalidade.
    """
    dur_f = round(duration, 2)
    prompt = f'''
Voce e o Revisor-Chefe de Dublagem. Seu trabalho e GARANTIR que a traducao NAO pareca "traducao", mas sim uma fala natural de um filme brasileiro.

CENA:
Original (EN): "{original_en}"
Opcao Candidata (PT-BR): "{candidate_pt}"
Tempo disponivel: {dur_f}s

SUA MISSAO:
1. Analise se a frase em PT-BR soa natural, "cool" e narrativa.
2. GIRIA MILITAR: So aceite "Copiado" se for Roger/Copy. Se for "Gotcha", "Incoming" ou "Target", use termos de acao (Te peguei, Acertei, Alvo).
3. Se a frase estiver robotica ou muito literal, CORRIJA-A agora.
4. CONTAGEM DE TEMPO: (Letras / 18) + (Virgulas INTERNAS * 0.5) deve ser proximo de {dur_f}s.
5. REGRA DA VIRGULA: Somente virgulas no MEIO da frase consomem meio segundo de tempo. Virgulas no FINAL da frase sao gratuitas (0s).
6. NUNCA USE PONTOS FINAIS. Use apenas virgulas ou exclamacoes.

Responda APENAS com a versao final refinada entre aspas duplas.
Se a opcao candidata ja for perfeita, apenas repita-a entre aspas duplas. Irrelevante se precisar mudar pouca coisa: foque na naturalidade e no tempo {dur_f}s.
'''
    try:
        payload = {
            "messages": [
                {"role": "system", "content": "Você é o Juiz Sênior de Localização. Responda apenas o texto final entre aspas duplas."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2,
            "max_tokens": 512
        }
        response = make_gema_request_with_retries(payload, is_translation=False)
        content = response.json()['choices'][0]['message']['content'].strip()
        
        quoted_match = re.search(r'"(.*?)"', content, re.DOTALL)
        if quoted_match:
             return quoted_match.group(1).strip()
        return candidate_pt # Se falhar o parse, mantemos o candidato original
    except:
        return candidate_pt

def gema_etapa_2_sincronizacao(original_text, duration, previous_context=None, profile_id='padrao'):
    """
    [v14.10 FALLBACK MASTER] - Sincronia Individual Autônoma.
    Usada como redundância caso o lote falhe.
    """
    profile = load_game_profile(profile_id)
    lore_text = profile.get("lore", "Gênero: Jogo de Aventura/Ação (Autodetecção Ativada)")
    target_chars = int(duration * 18)
    temperature = 0.2
    
    prompt = f'''
# DIRETRIZES DE DUBLAGEM INDIVIDUAL (MASTER SYNC)

[TAREFA]: Traduza e SINCRONIZE a frase abaixo mantendo a "vibe" do jogo.

[CONTRATO DE SINCRONIA]:
- LIMITE DE TEMPO: {round(duration, 2)} segundos.
- CALCULO: (Letras / 18) + (Virgulas Internas * 0.5) <= {round(duration, 2)}s.
- PONTUACAO: PROIBIDO PONTOS (.). Use virgulas ou !/?.
- TABELA DE GIRIAS: "Roger" -> "Copiado!", "Gotcha" -> "Na mira!", "Cover me" -> "Me cobre!".

[LORE]: {lore_text} 
[HISTORICO]: {previous_context if previous_context else "Inicio"}

[FRASE ORIGINAL (EN)]: 
"{original_text}"

Responda APENAS com a traducao final entre aspas duplas: "Sua traducao aqui!"
'''

    try:
        payload = {
            "messages": [
                {"role": "system", "content": "Você é um Diretor de Sincronia. Sua resposta deve conter APENAS o texto traduzido final entre aspas duplas. Proibido conversar."},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": 512
        }
        
        response = make_gema_request_with_retries(payload, is_translation=False)
        content = response.json()['choices'][0]['message']['content'].strip()
        
        # Extrator Inteligente de Aspas
        quoted_match = re.search(r'"(.*?)"', content, re.DOTALL)
        if quoted_match:
             final_text = quoted_match.group(1).strip()
             return apply_string_fallback(final_text, target_chars or 999), 1
        
        # Fallback se não usar aspas
        return apply_string_fallback(original_text, target_chars or 999), 2
        
    except Exception as e:
        logging.error(f"Erro no Fallback Sync Gema: {e}")
        return original_text, 2

def sanitize_tts_text(text):
    """
    Remove pontuação e resíduos de prompt da IA (Contexto Anterior, RESPOSTA DEFINITIVA, etc).
    """
    if not text: return ""
    
    # [v12.21 PENTE-FINO] Remove labels de alucinação (ex: "Contexto Anterior: ...")
    prompt_labels = r"(?im)^(Contexto|Resposta|Original|Tradução|Style|Timing|Scenario|Note|Tradução Adaptada|Texto).*?:.*?\n?"
    text = re.sub(prompt_labels, "", text)

    # 1. Normalização Básica (Trocamos reticências por nada, e não por vírgulas mais)
    text = text.replace("...", " ").replace("..", " ").replace("—", " ")
    
    # 2. [NOVO] Expansor de Números Automático para TTS (Anti-Engasgo)
    nums_map = {
        "01": "zero um", "02": "zero dois", "03": "zero três", "04": "zero quatro",
        "05": "zero cinco", "06": "zero seis", "07": "zero sete", "08": "zero oito", "09": "zero nove",
        "1": "um", "2": "dois", "3": "três", "4": "quatro", "5": "cinco", "6": "seis", "7": "sete", "8": "oito", "9": "nove", "0": "zero",
        "10": "dez", "11": "onze", "12": "doze", "13": "treze", "14": "quatorze", "15": "quinze",
        "16": "dezesseis", "17": "dezessete", "18": "dezoito", "19": "dezenove", "20": "vinte",
        "30": "trinta", "40": "quarenta", "50": "cinquenta", "60": "sessenta", "70": "setenta", "80": "oitenta", "90": "noventa"
    }
    # Substitui números de dois dígitos primeiro para evitar conflito (ex: 10 virar um-zero)
    for n_str, n_ext in sorted(nums_map.items(), key=lambda x: len(x[0]), reverse=True):
        # Usa regex com \b para garantir que só troque números soltos e não no meio de IDs
        pattern = r'\b' + n_str + r'\b'
        text = re.sub(pattern, n_ext, text)


    # 4. Whitelist de Símbolos e Caracteres (Permite %, $, +, @, vídeo-games vibes)
    # [v12.97 UPDATED] Agora inclui símbolos vitais para narrativa de jogos.
    text = re.sub(r'[^\w\s\,\!\?\-\%\$\+\@áéíóúâêîôûãõàèìòùçÁÉÍÓÚÂÊÎÔÛÃÕÀÈÌÒÙÇ]', '', text)
    
    # [v12.98 CLEANUP] Remove conflitos de pontuação no final (ex: ",!" ou "!,")
    # Isso evita problemas no Chatterbox que "trava" com pontuação dupla exótica.
    text = text.replace(",!", "!").replace("!,", "!").replace("?,", "?").replace(",?", "?")
    
    # 3. [v12.96 PROIBIÇÃO DO PONTO]
    # O usuário reportou que pontos (.) bugam o Chatterbox e causam alucinações.
    # Exterminamos todos os pontos remanescentes.
    text = text.replace(".", "")
    
    # [INJEÇÃO DE ENERGIA] Frases curtas de ação ganham exclamação (!) no Chatterbox.
    if 0 < len(text) < 15 and not text.endswith(('!', '?')):
        text = text.strip() + "!"
    
    # 4. Limpa espaços duplos e evita vírgulas duplas/triplas
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r',+', ',', text) # Garante que não tenha ",,"
    
    # [v12.96 GPT-SOVITS EOS TOKEN FIX]
    # Usamos Exclamação (!) ou Interrogação (?) como fallback EOS, NUNCA ponto.
    if text and not text.endswith(('!', '?')):
        text += "!"
        
    return text
        
    return text

# Alias para compatibilidade
def gema_etapa_3_sanitizacao(text):
    return sanitize_tts_text(text)

def gema_etapa_3_adaptacao_tts(synced_text, is_retry=False):
    prompt_normal = f"""Você é um editor de roteiros para o motor de voz Chatterbox (TTS). Adapte o texto a seguir para uma leitura 100% natural.
**REGRAS CRÍTICAS:**
1.  **PAUSAS NATURAIS:** Use vírgulas para indicar pausas curtas onde o orador deve respirar (Cada vírgula = meio segundo).
2.  **PROIBIÇÃO TOTAL DE PONTOS:** NUNCA use o caractere de ponto (.). O Chatterbox entra em colapso e alucina se ler um ponto final. Use vírgulas (,) ou exclamações (!).
3.  **NÚMEROS POR EXTENSO:** OBRIGATORIAMENTE escreva números por extenso para o robô ler certo (ex: transforme "04" em "zero quatro", "25%" em "vinte e cinco por cento").
4.  **HÍFENS PERMITIDOS:** Use hífens normalmente em palavras compostas.
5.  **FORMATO:** Responda APENAS com o texto adaptado entre aspas duplas.
**Texto Original:** "{synced_text}"
**Texto Adaptado:**"""
    prompt_retry = f"""Ajuste a pontuação deste texto para um robô de voz ler. Responda entre aspas duplas.
Texto: "{synced_text}"
Texto Ajustado:"""
    prompt = prompt_retry if is_retry else prompt_normal
    payload = {"messages": [{"role": "user", "content": prompt}], "temperature": 0.2, "max_tokens": 1000}
    try:
        response = make_gema_request_with_retries(payload)
        return sanitize_tts_text(response.json()['choices'][0]['message']['content'])
    except Exception as e:
        logging.error(f"Erro na API Gema (Etapa 3): {e}")
        return f"FALHA_API: {e}"

def set_low_process_priority():
    try:
        p = psutil.Process(os.getpid())
        p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if sys.platform == "win32" else 19)
        logging.info("Prioridade do processo definida como 'baixa' para não impactar o uso do PC.")
    except Exception:
        logging.warning("Não foi possível definir a prioridade do processo.")

# ... (O resto das funções permanece o mesmo, para economizar espaço)
def gerar_relatorio_final(job_dir, job_id, project_data, file_format_map):
    relatorio_path = job_dir / "relatorio_processamento.txt"
    durations_cache_path = job_dir / "durations_cache.json"
    durations_cache = safe_json_read(durations_cache_path) or {}
    mastering_cache_path = job_dir / "mastering_cache.json"
    mastering_cache = safe_json_read(mastering_cache_path) or {}

    logging.info(f"Gerando relatório final em: {relatorio_path}")
    total_arquivos = len(project_data)
    sucesso_traducao, sucesso_audio, sucesso_finalizacao = 0, 0, 0
    
    with open(relatorio_path, 'w', encoding='utf-8') as f:
        f.write(f"--- Relatório de Processamento do Job: {job_id} ---\n")
        
        # [v10.68] Info de Duração Total no Topo do Relatório
        total_seconds = sum(item.get('duration', 0) for item in project_data)
        duracao_total_formatada = str(timedelta(seconds=int(total_seconds)))
        f.write(f"Duração Total de Áudio (Original): {duracao_total_formatada}\n")
        f.write(f"Total de Segmentos: {len(project_data)}\n")
        f.write("-" * 40 + "\n")
        # [FEATURE] Manual Volume Boost - Info no Relatório
        try:
            boost_file = job_dir / "volume_boost.txt"
            if boost_file.exists():
                val_raw = boost_file.read_text().strip()
                val_clean = val_raw.split('\n')[0].split('#')[0].strip()
                logging.info(f"Lendo volume_boost.txt para relatório: '{val_clean}'")
                if val_clean.isdigit():
                    val = int(val_clean)
                    if val > 0:
                        f.write(f"--- CONFIGURAÇÃO ESPECIAL ---\n")
                        f.write(f"Volume Boost Manual: ATIVADO (+{val}dB)\n")
                        f.write(f"---------------------------\n")
                else:
                    logging.warning(f"Valor numérico não encontrado em volume_boost.txt: '{val_clean}'")
        except Exception as e:
            logging.error(f"Erro ao escrever boost no relatório: {e}")
        
        f.write("\n")
        
        for seg_data in project_data:
            file_id = seg_data['id']
            # [FIX] Fallback seguro para file_name
            file_name = seg_data.get('file_name', f"{file_id}.wav")
            f.write(f"--- Arquivo: {file_name} ---\n")

            if seg_data.get('processing_status') == 'Copiado Diretamente (Som Não-Verbal)':
                f.write("  - Status:             Copiado Diretamente (Som Não-Verbal)\n")
                f.write(f"  - Texto Original:     '{seg_data.get('original_text', '')}'\n")
                sucesso_finalizacao += 1
            else:
                f.write("  - Status:             Processado para Dublagem\n")
                trans_text = seg_data.get('translated_text', '')
                if trans_text and "FALHA_" not in trans_text:
                    if seg_data.get('translation_fallback', False): f.write("  - Tradução:           FALLBACK (Usado texto original em inglês)\n")
                    else: f.write("  - Tradução:           OK\n"); sucesso_traducao += 1
                else: f.write(f"  - Tradução:           FALHOU (Motivo: {trans_text or 'N/A'})\n")

                dubbed_audio_exists = (job_dir / "_dubbed_audio" / f"{file_id}_dubbed.wav").exists()
                if dubbed_audio_exists: f.write("  - Geração de Áudio:   OK\n"); sucesso_audio += 1
                else: f.write("  - Geração de Áudio:   FALHOU\n")

            final_file_exists = (job_dir / "_saida_final" / f"{file_id}{file_format_map.get(file_id, '.wav')}").exists()
            if final_file_exists:
                f.write("  - Finalização:        OK\n")
                if seg_data.get('processing_status') != 'Copiado Diretamente (Som Não-Verbal)': sucesso_finalizacao += 1
            else: f.write("  - Finalização:        FALHOU\n")

            original_duration = seg_data.get('duration', 0)
            duration_info = durations_cache.get(file_id, {})
            Chatterbox_duration = duration_info.get('Chatterbox_duration', 0)
            final_duration = duration_info.get('duration', 0)
            speed_factor = duration_info.get('speed_factor')

            orig_dur_f = round(original_duration, 2)
            f.write(f"  - Duração Original:   {orig_dur_f}s\n")
            
            if seg_data.get('processing_status') != 'Copiado Diretamente (Som Não-Verbal)':
                cb_dur_f = round(Chatterbox_duration, 2)
                f_dur_f = round(final_duration, 2)
                if Chatterbox_duration > 0: f.write(f"  - Duração Chatterbox:       {cb_dur_f}s\n")
                if final_duration > 0: f.write(f"  - Duração Final:      {f_dur_f}s\n")
                else: f.write("  - Duração Final:      N/A (Erro ou não processado)\n")

                if speed_factor: 
                    sf_f = round(speed_factor, 2)
                    f.write(f"  - Ajuste de Velocidade: Sim (fator {sf_f}x)\n")
                elif Chatterbox_duration > 0: f.write("  - Ajuste de Velocidade: Não necessário\n")
                
                mastering_info = mastering_cache.get(file_id, {})
                if mastering_info.get('status') == 'mastered':
                    f.write("  - Masterização:       Aplicada (dynaudnorm)\n")
                    original_peak = mastering_info.get('original_peak_dbfs')
                    dubbed_peak = mastering_info.get('dubbed_peak_before_mastering_dbfs')
                    if original_peak is not None: 
                        op_f = round(original_peak, 2)
                        f.write(f"    - Pico Original:    {op_f} dBFS\n")
                    if dubbed_peak is not None: 
                        dp_f = round(dubbed_peak, 2)
                        f.write(f"    - Pico Dublado:     {dp_f} dBFS (antes da masterização)\n")
                elif mastering_info.get('status') == 'fallback_copied':
                    f.write("  - Masterização:       Falhou. Cópia do original utilizada.\n")

            else: 
                f.write(f"  - Duração Final:      {orig_dur_f}s (Cópia do original)\n")
            f.write("\n")

        total_para_dublar = len([s for s in project_data if s.get('processing_status') != 'Copiado Diretamente (Som Não-Verbal)'])
        total_copiados = total_arquivos - total_para_dublar

        f.write(f"--- RESUMO GERAL ---\n")
        f.write(f"Total de Arquivos: {total_arquivos}\n")
        f.write(f"  - Arquivos para Dublagem:        {total_para_dublar}\n")
        f.write(f"  - Arquivos Copiados (Não-Verbal): {total_copiados}\n\n")
        if total_para_dublar > 0:
            f.write(f"  - Traduções Bem-sucedidas:      {sucesso_traducao}/{total_para_dublar}\n")
            f.write(f"  - Áudios Gerados:               {sucesso_audio}/{total_para_dublar}\n")
        f.write(f"  - Finalizados com Sucesso:      {sucesso_finalizacao}/{total_arquivos}\n")


# --- FUNÇÕES PRINCIPAIS DOS PIPELINES ---
def processar_transcricao(job_dir, job_id, start_time):
    with active_jobs_lock:
        if len(active_jobs) >= MAX_CONCURRENT_JOBS:
            logging.warning(f"❌ [HARDWARE] Limite de {MAX_CONCURRENT_JOBS} job(s) atingido. Ignorando {job_id}.")
            return
        active_jobs.add(job_id)

    try:
        set_low_process_priority()
        def cb(p, etapa, s=None): set_progress(job_id, p, etapa, start_time, ETAPAS_TRANSCRICAO, s)
        
        input_file = next(job_dir.glob('input.*'), None)
        if not input_file:
            raise FileNotFoundError("Nenhum arquivo de entrada encontrado no diretório do job.")

        backup_dir = job_dir / "_backup_transcricao_whisper"
        backup_dir.mkdir(exist_ok=True)
        
        cb(5, 1, "Carregando modelo Whisper...")
        model = get_whisper_model()
        
        cb(10, 1, "Iniciando transcrição...")
        
        segments, info = model.transcribe(str(input_file))
        total_duration = info.duration
        
        all_segments_data = []
        
        for segment in segments:
            progress = (segment.end / total_duration) * 100 if total_duration > 0 else 100
            tempo_atual = str(timedelta(seconds=int(segment.end)))
            tempo_total = str(timedelta(seconds=int(total_duration)))
            cb(progress, 1, f"Transcrevendo... {tempo_atual}/{tempo_total}")
            
            segment_data = {
                "start": round(segment.start, 3),
                "end": round(segment.end, 3),
                "text": segment.text.strip()
            }
            all_segments_data.append(segment_data)
            
            # Backup contínuo por segmento
            safe_json_write(segment_data, backup_dir / f"segment_{segment.start:.3f}.json")
        
        cb(100, 2, "Gerando arquivos finais...")
        
        # Gerar arquivo JSON completo
        json_output_path = job_dir / "transcricao_completa.json"
        safe_json_write(all_segments_data, json_output_path)
        logging.info(f"Arquivo JSON da transcrição salvo em: {json_output_path}")

        # Gerar arquivo TXT completo
        txt_output_path = job_dir / "transcricao_completa.txt"
        with open(txt_output_path, 'w', encoding='utf-8') as f:
            for seg in all_segments_data:
                f.write(f"{seg['text']}\n")
        logging.info(f"Arquivo TXT da transcrição salvo em: {txt_output_path}")

        cb(100, 3, "Transcrição concluída!")

    except Exception as e:
        logging.error(f"ERRO NO PIPELINE DE TRANSCRIÇÃO (Job ID: {job_id}): {e}\n{traceback.format_exc()}")
        set_progress(job_id, 100, len(ETAPAS_TRANSCRICAO) - 1, start_time, ETAPAS_TRANSCRICAO, subetapa=f"Erro: {e}")
        status_path = job_dir / "job_status.json"
        status_data = safe_json_read(status_path) or {}
        status_data['status'] = 'failed'
        status_data['error'] = str(e)
        safe_json_write(status_data, status_path)
    finally:
        with active_jobs_lock:
            if job_id in active_jobs:
                active_jobs.remove(job_id)


def processar_conversao(job_dir, job_id, start_time):
    with active_jobs_lock:
        if len(active_jobs) >= MAX_CONCURRENT_JOBS:
            logging.warning(f"❌ [HARDWARE] Limite de {MAX_CONCURRENT_JOBS} job(s) atingido. Ignorando {job_id}.")
            return
        active_jobs.add(job_id)
    
    try:
        set_low_process_priority()
        def cb(p, etapa, s=None): set_progress(job_id, p, etapa, start_time, ETAPAS_CONVERSAO, s)
        
        status = safe_json_read(job_dir / "job_status.json") or {}
        file_format_map = status.get('file_format_map', {})

        referencia_dir = job_dir / "_1_referencia"
        para_converter_dir = job_dir / "_2_para_converter"
        convertidos_dir = job_dir / "_3_convertidos"
        convertidos_dir.mkdir(exist_ok=True)
        
        files_to_process = list(para_converter_dir.glob("*.*"))
        total_files = len(files_to_process)
        if total_files == 0:
            cb(100, 2, "Nenhum arquivo para converter.")
            return

        reference_files_map = {p.stem: p for p in referencia_dir.glob("*.*")}
        logging.info(f"Arquivos de referência encontrados: {list(reference_files_map.keys())}")


        cb(0, 1, f"Iniciando conversão para {total_files} arquivos...")
        
        sucesso_count = 0
        for i, file_to_convert in enumerate(files_to_process):
            cb((i / total_files) * 100, 1, f"Convertendo: {file_to_convert.name}")
            
            convert_stem = file_to_convert.stem
            base_ref_stem = convert_stem.replace("_dubbed", "")
            
            ref_path = None
            if base_ref_stem in reference_files_map:
                ref_path = reference_files_map[base_ref_stem]
                logging.info(f"Referência encontrada para '{file_to_convert.name}' -> '{ref_path.name}'.")
            elif convert_stem in reference_files_map:
                ref_path = reference_files_map[convert_stem]
                logging.info(f"Referência com nome exato (stem) encontrada para '{file_to_convert.name}' -> '{ref_path.name}'.")
            else:
                logging.warning(f"Nenhum arquivo de referência encontrado para '{file_to_convert.name}' (procurando por stem: '{base_ref_stem}'). Pulando.")
                continue

            try:
                sample_rate, channels, bitrate = get_audio_metadata(str(ref_path))
                final_path = convertidos_dir / ref_path.name
                threads = str(os.cpu_count() or 4)
                cmd = ['ffmpeg', '-threads', threads, '-y', '-i', str(file_to_convert), '-ar', str(sample_rate), '-ac', str(channels), '-af', 'dynaudnorm', str(final_path)]
                if bitrate and str(bitrate).isdigit():
                    cmd.extend(['-b:a', str(bitrate)])
                subprocess.run(cmd, check=True, capture_output=True, text=True)
                sucesso_count += 1
            except subprocess.CalledProcessError as e:
                logging.error(f"Erro ao converter {file_to_convert.name}: {e.stderr}")
            except Exception as e:
                logging.error(f"Erro inesperado ao processar {file_to_convert.name}: {e}")

        if sucesso_count > 0:
            cb(100, 2, f"Conversão concluída! {sucesso_count}/{total_files} arquivos processados.")
        else:
            cb(100, 2, "Concluído, mas nenhum arquivo foi convertido. Verifique os nomes dos arquivos de referência.")

    except Exception as e:
        logging.error(f"ERRO NO PIPELINE DE CONVERSÃO (Job ID: {job_id}): {e}\n{traceback.format_exc()}")
        set_progress(job_id, 100, len(ETAPAS_CONVERSAO) - 1, start_time, ETAPAS_CONVERSAO, subetapa=f"Erro: {e}")
        status_path = job_dir / "job_status.json"
        status_data = safe_json_read(status_path) or {}
        status_data['status'] = 'failed'
        status_data['error'] = str(e)
        safe_json_write(status_data, status_path)
    finally:
        with active_jobs_lock:
            if job_id in active_jobs:
                active_jobs.remove(job_id)


def try_reconstruct_project_from_all_backups(job_dir):
    """
    SISTEMA PHOENIX (Recuperação Avançada):
    Tenta reconstruir o project_data.json a partir dos backups fragmentados,
    caso o arquivo principal esteja corrompido ou vazio.
    """
    project_data_path = job_dir / "project_data.json"
    backup_transc_dir = job_dir / "_backup_transcricao"
    backup_texto_dir = job_dir / "_backup_texto_final"
    
    # Se já tem arquivo com dados, não mexe
    if project_data_path.exists():
        data = safe_json_read(project_data_path)
        if data and len(data) > 0:
            return
            
    logging.warning("=== ALERTA PHOENIX: INICIANDO RECUPERAÇÃO DE PROJETO ===")
    logging.warning("project_data.json ausente ou corrompido. Tentando reconstrução...")
    
    recovered_nodes = {}
    
    # Base: Transcrição (Tem os dados originais e metadados)
    if backup_transc_dir.exists():
        for bf in backup_transc_dir.glob("*.json"):
            try:
                data = safe_json_read(bf)
                if data and 'id' in data:
                    recovered_nodes[data['id']] = data
            except: pass

    # Override: Texto Final (Tem os textos traduzidos e editados pelo usuário)
    if backup_texto_dir.exists():
        for bf in backup_texto_dir.glob("*.json"):
            try:
                data = safe_json_read(bf)
                if data and 'id' in data:
                    if data['id'] in recovered_nodes:
                        # Atualiza apenas os campos importantes mantendo a base
                        recovered_nodes[data['id']].update(data)
                    else:
                        recovered_nodes[data['id']] = data
            except: pass
            
    if recovered_nodes:
        final_list = list(recovered_nodes.values())
        final_list.sort(key=lambda x: x.get('id', ''))
        safe_json_write(final_list, project_data_path)
        logging.info(f"PHOENIX SUCCESSO: {len(final_list)} blocos recuperados e salvos!")
    else:
        logging.error("PHOENIX FALHOU: Nenhum backup recuperável encontrado.")

def processar_dublagem_jogos(job_dir, job_id, start_time):
    with active_jobs_lock:
        if len(active_jobs) >= MAX_CONCURRENT_JOBS:
             logging.warning(f"❌ [HARDWARE] Limite de {MAX_CONCURRENT_JOBS} job(s) atingido. Ignorando {job_id}.")
             return
        active_jobs.add(job_id)
    
    try:
        set_low_process_priority()
        
        # [CUMULATIVE TIME] Lê o tempo acumulado de sessões anteriores
        status = safe_json_read(job_dir / "job_status.json") or {}
        accumulated_time = status.get('total_elapsed_secs', 0)
        # Ajusta o cronômetro para iniciar de onde parou (Puro Ouro!)
        virtual_start_time = time.time() - accumulated_time
        
        def cb(p, etapa, s=None): set_progress(job_id, p, etapa, virtual_start_time, ETAPAS_JOGOS, s)
        
        for dir_name in ["_1_MOVER_OS_FICHEIROS_DAQUI", "_2_PARA_AS_PASTAS_DE_VOZ", "_backup_transcricao", "_backup_texto_final", "_dubbed_audio", "_saida_final"]:
            (job_dir / dir_name).mkdir(parents=True, exist_ok=True)
            
        # [FEATURE] Manual Volume Boost - Garante que o arquivo existe
        boost_file = job_dir / "volume_boost.txt"
        if not boost_file.exists():
            try:
                # [v10.71] Detecção de Perfil para Valor Inicial Automático
                initial_boost = "0"
                status_temp = safe_json_read(job_dir / "job_status.json") or {}
                if status_temp.get('game_profile') == 'bioshock':
                    initial_boost = "12"
                    logging.info("[PROFILE] BioShock: Definindo volume_boost.txt inicial para +12dB.")
                elif status_temp.get('game_profile') == 'cod':
                    initial_boost = "10"
                    logging.info("[PROFILE] Call of Duty (MW3): Definindo volume_boost.txt inicial para +10dB.")
                
                with open(boost_file, "w") as f:
                    f.write(f"{initial_boost}\n# AVISO: 1 = +1dB. NAO coloque mais que 25.\n# CUIDADO: Volumes extremos podem DANIFICAR seus alto-falantes.")
            except Exception as e:
                logging.error(f"Erro ao criar volume_boost.txt no start: {e}")
        
        status = safe_json_read(job_dir / "job_status.json") or {}
        file_format_map = status.get('file_format_map', {})
        source_language = status.get('source_language', 'auto')
        diarization_dir = job_dir / "_2_PARA_AS_PASTAS_DE_VOZ"
        project_data_path = job_dir / "project_data.json"
        
        # [ALERTA DE VELOCIDADE] Verifica se a GPU está livre antes de começar
        import torch
        if torch.cuda.is_available():
            free_m, _ = torch.cuda.mem_get_info()
            free_gb = free_m / (1024**3)
            if free_gb < 1.0:
                logging.warning("⚠️  [AVISO DE VELOCIDADE] Sua Placa de Vídeo está quase cheia!")
                logging.info("👉 DICA: Feche o LM Studio agora. Você só vai precisar dele na ETAPA 5.")
                logging.info("👉 Isso vai acelerar a Transcrição e Diarização em até 20x.")
                cb(0, 1, "AVISO: Feche o LM Studio para acelerar o início.")
                time.sleep(2) # Pausa curta para ele ler o aviso

        cb(0, 1, "Iniciando Diarização Automática...")
        # [MODIFICADO] Substituído Manual por Auto Diarização
        run_auto_diarization_batch(job_dir, job_id, cb)
        # wait_for_diarization_manual(job_id, cb) # Desativado
        unify_speaker_files(job_dir, cb)

        all_files_to_process = [f for f in diarization_dir.rglob("*.wav") if not f.name.startswith("_REF_")]
        
        # [FEATURE] Calculo Dinâmico de Duração Total do Projeto
        try:
            total_duration_secs = sum(get_audio_duration(str(f)) for f in all_files_to_process)
            status['duracao_total_secs'] = total_duration_secs
            
            # Formatação amigável
            horas, resto = divmod(int(total_duration_secs), 3600)
            minutos, segundos = divmod(resto, 60)
            
            if horas > 0:
                 status['duracao_total_formatada'] = f"{horas}h {minutos}m {segundos}s"
            else:
                 status['duracao_total_formatada'] = f"{minutos}m {segundos}s"
                 
            safe_json_write(status, job_dir / "job_status.json")
            logging.info(f"Duração total do projeto calculada: {status['duracao_total_formatada']} ({len(all_files_to_process)} arquivos)")
        except Exception as e:
            logging.error(f"Falha ao calcular a duração total do projeto: {e}")
        transcription_backup_dir = job_dir / "_backup_transcricao"
        
        # [PHOENIX RECOVERY] Dispara a recuperação ANTES de tentar ler
        try_reconstruct_project_from_all_backups(job_dir)
        
        project_data = safe_json_read(project_data_path) or []
        # Normalização de Segurança (Trata dic do App_videos vs list do app_jogos)
        if isinstance(project_data, dict) and 'segments' in project_data:
            project_data = project_data['segments']
        elif isinstance(project_data, dict):
            project_data = [] # Fallback seguro se vier um dict esquisito
            
        project_data_map = {item['id']: item for item in project_data}
        files_needing_transcription = []
        
        cb(5, 2, "Verificando arquivos e backups existentes...")
        cb(5, 2, "Verificando arquivos e backups existentes...")
        for audio_file in all_files_to_process:
            file_id = audio_file.stem
            current_speaker = audio_file.parent.name
            
            # [FIX] Garante que o 'speaker' no JSON esteja atualizado com a pasta real
            # Útil se a Unificação de Vozes moveu arquivos após a primeira transcrição.
            updated_speaker = False
            
            if file_id in project_data_map:
                if project_data_map[file_id].get('speaker') != current_speaker:
                    project_data_map[file_id]['speaker'] = current_speaker
                    updated_speaker = True
                
                if project_data_map[file_id].get('original_text'):
                    # Se atualizou o speaker, salva no backup individual também para garantir consistência imediata
                    if updated_speaker:
                        safe_json_write(project_data_map[file_id], transcription_backup_dir / f"{file_id}.json")
                    continue
            
            backup_file = transcription_backup_dir / f"{file_id}.json"
            backup_data = safe_json_read(backup_file)
            
            if backup_data and backup_data.get('original_text'):
                project_data_map[file_id] = backup_data
                if project_data_map[file_id].get('speaker') != current_speaker:
                    project_data_map[file_id]['speaker'] = current_speaker 
                    safe_json_write(project_data_map[file_id], backup_file) # Salva imediatamente no disco
            else:
                files_needing_transcription.append(audio_file)
        
        project_data = list(project_data_map.values())
        project_data.sort(key=lambda x: x.get('id', ''))
        
        if files_needing_transcription:
            total_to_transcribe = len(files_needing_transcription)
            cb(10, 2, f"Iniciando transcrição para {total_to_transcribe} arquivos...")
            logging.info(f"[DEBUG] Arquivos que precisam de transcrição: {[f.name for f in files_needing_transcription]}") # [DEBUG]
            model = get_whisper_model()
            for i, audio_file in enumerate(files_needing_transcription):
                start_seg = time.time()
                try:
                    text_result = transcribe_audio(model, str(audio_file), source_lang=source_language)
                    sample_rate, channels, _ = get_audio_metadata(str(audio_file))
                    file_data = {
                        "id": audio_file.stem, 
                        "file_name": audio_file.name, 
                        "speaker": audio_file.parent.name, 
                        "original_text": text_result.get("text", ""), 
                        "detected_language": text_result.get("detected_language", ""),
                        "duration": get_audio_duration(str(audio_file)), 
                        "sample_rate": sample_rate, 
                        "channels": channels
                    }
                    project_data.append(file_data)
                    safe_json_write(file_data, transcription_backup_dir / f"{audio_file.stem}.json")
                except Exception as e: 
                    logging.error(f"FALHA AO TRANSCREVER {audio_file.name}: {e}")
                finally:
                    seg_time = time.time() - start_seg
                    now_str = time.strftime("%H:%M:%S")
                    cb(10 + (i / total_to_transcribe) * 85, 2, f"[{now_str}] Transcrevendo: {audio_file.name} ({seg_time:.1f}s)")
            project_data.sort(key=lambda x: x.get('id', ''))
        

        # [MEMORY] Libera Whisper imediatamente após o uso para dar espaço ao Gema/Chatterbox
        unload_whisper_model()

        safe_json_write(project_data, project_data_path)
        cb(100, 2, "Transcrição carregada e verificada.")
        
        logging.info("Sincronizando o progresso com os backups de texto final...")
        text_backup_dir = job_dir / "_backup_texto_final"
        project_data_map = {item['id']: item for item in project_data}
        updated_count = 0

        for backup_file in text_backup_dir.glob("*.json"):
            file_id = backup_file.stem
            if file_id in project_data_map:
                backup_data = safe_json_read(backup_file)
                if backup_data:
                    # [v2026.11 FIX] Fusão Inteligente: O backup NÃO pode apagar um Manual Edit preenchido na memória
                    current_manual = project_data_map[file_id].get('manual_edit_text', '').strip()
                    fresh_manual = backup_data.get('manual_edit_text', '').strip()
                    
                    for key, val in backup_data.items():
                        if key == 'manual_edit_text' and not val and current_manual:
                            # Se o backup está vazio mas a memória tem texto, não sobrescreve o Manual
                            continue
                        project_data_map[file_id][key] = val
                    
                    updated_count += 1
        
        if updated_count > 0:
            project_data = list(project_data_map.values())
            project_data.sort(key=lambda x: x.get('id', ''))
            logging.info(f"Dados do projeto sincronizados com {updated_count} backups. Edições manuais preservadas. 🛡️")
            safe_json_write(project_data, project_data_path)
        else:
            logging.info("Nenhum progresso novo encontrado nos backups para sincronizar.")
            
        logging.info("Verificando e limpando dados de execuções anteriores...")
        needs_resave = False
        for seg_data in project_data:
            if 'sanitized_text' in seg_data:
                original_sanitized = seg_data['sanitized_text']
                corrected_sanitized = sanitize_tts_text(original_sanitized)
                if original_sanitized != corrected_sanitized:
                    logging.warning(f"Corrigido texto antigo para '{seg_data['id']}': '{original_sanitized}' -> '{corrected_sanitized}'")
                    seg_data['sanitized_text'] = corrected_sanitized
                    needs_resave = True
            if 'manual_edit_text' not in seg_data:
                seg_data['manual_edit_text'] = ""
                needs_resave = True
            elif seg_data['manual_edit_text']:
                # [SEGURANÇA] Se o campo manual está preenchido, garantimos que ele não seja resetado aqui
                pass


        if needs_resave:
            logging.info("Salvando correções de dados antigos no project_data.json...")
            safe_json_write(project_data, project_data_path)
        
        files_to_process_gema = []
        files_to_copy_directly = []

        for seg_data in project_data:
            # [FIX] Se já foi marcado como "Não-Verbal", PULA.
            if seg_data.get('processing_status') == 'Copiado Diretamente (Som Não-Verbal)':
                continue

            # [NOVO - Filtro de Idioma] Pula tradução do que já está em Português
            if seg_data.get('detected_language') == 'pt':
                if not seg_data.get('sanitized_text'):
                    seg_data['sanitized_text'] = seg_data.get('original_text', '')
                
                # [FIX] Garante a existência do backup para não acionar o apagamento forçado (fallback manual)
                backup_path_pt = job_dir / "_backup_texto_final" / f"{seg_data['id']}.json"
                if not backup_path_pt.exists():
                    safe_json_write(seg_data, backup_path_pt)
                    
                logging.info(f"Segmento {seg_data['id']} preservado (já é Português).")

            # [v12.32 SINCRONIA DE DADOS]
            backup_path = job_dir / "_backup_texto_final" / f"{seg_data['id']}.json"
            
            # [REGRA DE OURO] Se existe texto manual na memória (project_data), ele é SAGRADO.
            # Se o backup sumiu, nós RECRIAMOS o backup a partir do manual, em vez de apagar o manual.
            if seg_data.get('manual_edit_text'):
                if not backup_path.exists():
                    logging.info(f"🛡️ [RESGATE] Recriando backup para '{seg_data['id']}' a partir da edição manual preservada.")
                    safe_json_write(seg_data, backup_path)
                continue # Pula qualquer lógica de "pop" ou limpeza para este arquivo

            # [PROTEÇÃO VITALÍCIA] Se NÃO tem manual, aí sim podemos limpar traduções antigas se o backup sumir
            if not backup_path.exists() and seg_data.get('sanitized_text'):
                seg_data.pop('translated_text', None)
                seg_data.pop('synced_text', None)
                seg_data.pop('sanitized_text', None)
                seg_data['translation_fallback'] = False

            if seg_data.get('sanitized_text') and not seg_data.get('translation_fallback'):
                continue

            original_text = seg_data.get('original_text', '').strip()
            clean_text = re.sub(r'[^\w\s]', '', original_text).lower()
            words = clean_text.split()
            if (words and all(word in SONS_A_IGNORAR for word in words)) or \
               len(original_text.replace(" ", "")) < 3 or \
               is_junk_text(original_text):
                files_to_copy_directly.append(seg_data)
                seg_data['processing_status'] = 'Copiado Diretamente (Som Não-Verbal)'
                
                # [FIX] Garante que esses arquivos também tenham backup em _backup_texto_final
                # para evitar discrepância de contagem e permitir edição manual se o usuário quiser.
                safe_json_write(seg_data, job_dir / "_backup_texto_final" / f"{seg_data['id']}.json")
                
                logging.info(f"O áudio '{seg_data['id']}' foi marcado como som não verbal ('{original_text}'). Será copiado, não dublado.")
            else:
                files_to_process_gema.append(seg_data)
                seg_data['processing_status'] = 'Processado para Dublagem'
        
        safe_json_write(project_data, project_data_path)

        if files_to_process_gema:
            cb(0, 3, f"Processando {len(files_to_process_gema)} textos com Gema...")
            wait_for_gema_service(lambda s: cb(0, 3, s))
            
            # [v12.70] Prioridade para o perfil escolhido pelo usuário no HTML/Status
            game_profile_id = status.get('game_profile', 'padrao').lower()
            
            # [v20.6] EXTRAÇÃO DO GLOSSÁRIO PERSONALIZADO
            # Transforma "Nome=Nome, Termo=Trad" em um dicionário real para o Gema
            user_glossary_raw = status.get('user_glossary', '')
            merged_glossary = {}
            if user_glossary_raw:
                parts = [p.strip() for p in user_glossary_raw.split(',') if p.strip()]
                for p in parts:
                    if '=' in p:
                        k, v = p.split('=', 1)
                        merged_glossary[k.strip()] = v.strip()
                    else:
                        merged_glossary[p.strip()] = p.strip()

            # Combina Perfil com Contexto Imediato das falas
            sample_ctx = " / ".join([s['original_text'] for s in files_to_process_gema[:3]])
            cenario_ctx = f"{game_profile_id.upper()} - Contexto: {sample_ctx}"
            cb(5, 3, f"Estilo: {game_profile_id.upper()}")
            
            # [v20.16] CACHE GRANULAR (WYSIWYG - What You See Is What You Get)
            # Se o arquivo individual .json existir na pasta de backup, usamos ele.
            # Se o usuário apagar o arquivo da pasta, a IA traduz novamente.
            backup_texto_dir = job_dir / "_backup_texto_final"
            backup_texto_dir.mkdir(parents=True, exist_ok=True)
            
            unique_texts_map = {}
            unique_files = []
            
            # [v21.15] MICRO-CACHE DINÂMICO (SEM ARQUIVO FÍSICO)
            # Agora o micro_cache é construído EM MEMÓRIA toda vez que você inicia o Job.
            # Isso evita ter que apagar um arquivo a mais quando você quer mudar uma tradução.
            micro_cache = {}
            
            # Passo 1: Popula o micro_cache com a "Prioridade das Prioridades" (Edição Manual)
            for f in files_to_process_gema:
                orig = f.get('original_text', '').strip()
                manual = f.get('manual_edit_text', '').strip()
                if orig and manual:
                    micro_cache[orig] = manual
                    micro_cache[orig.lower()] = manual

            for f in files_to_process_gema:
                orig_txt = f.get('original_text', '').strip()
                if not orig_txt: continue
                
                # [PRIORIDADE 1] Edição Manual (O usuário escreveu lá no HTML)
                # Se houver edição manual, ela anula qualquer tradução de IA ou Cache.
                if f.get('manual_edit_text', '').strip():
                    f['translated_text'] = f['manual_edit_text']
                    f['synced_text'] = f['manual_edit_text']
                    f['sanitized_text'] = gema_etapa_3_sanitizacao(f['manual_edit_text'])
                    f['_usar_cache'] = True
                    continue

                # [PRIORIDADE 2] Cache Granular
                individual_json = backup_texto_dir / f"{f['id']}.json"
                if individual_json.exists():
                    saved_data = safe_json_read(individual_json)
                    if saved_data and saved_data.get('translated_text'):
                        f.update(saved_data)
                        f['_usar_cache'] = True
                        continue

                # [PRIORIDADE 3] Repetição Interna
                if orig_txt in micro_cache or orig_txt.lower() in micro_cache:
                    f['_usar_cache_da_fila'] = True
                    continue

                # Sem cache: Vai para a fila de tradução da IA única
                if orig_txt.lower() not in unique_texts_map:
                    unique_texts_map[orig_txt.lower()] = True
                    unique_files.append(f)
                else:
                    f['_usar_cache_da_fila'] = True

            rus_files = [f for f in unique_files if re.search(r'[А-Яа-яЁё]', f.get('original_text', ''))]
            eng_files = [f for f in unique_files if not re.search(r'[А-Яа-яЁё]', f.get('original_text', ''))]
            
            # [v20.8 REVOLUÇÃO ATÔMICA]
            # Em vez de lotes cegos, processamos em paralelo com janela de contexto.
            total_items = len(unique_files)
            completed_atomic = 0
            
            def worker_traducao(idx, item_data):
                nonlocal completed_atomic
                start_seg = time.time()
                try:
                    # 1. Constrói Janela de Contexto Equilibrada (3 antes, 3 depois - Sprint Mode para i5)
                    # Reduzido de 10 para 3 para acelerar o 'Prefill' da CPU (menos texto para o i5 ler antes de traduzir).
                    start_ctx = max(0, idx - 3)
                    end_ctx = min(total_items, idx + 4)
                    context_lines = []
                    for j in range(start_ctx, end_ctx):
                        f_ctx = unique_files[j]
                        prio = ">>> ALVO >>>" if j == idx else "            "
                        speaker = f_ctx.get('speaker', 'Voz')
                        context_lines.append(f"{prio} {f_ctx['id']} ({speaker}): \"{f_ctx.get('original_text','')}\"")
                    
                    ctx_str = "\n".join(context_lines)
                    
                    # [v20.17] MODO TURBO: Agente Atômico Único (Optimized for G4B)
                    # Confiamos na inteligência do Gemma 4 para ajustar o limite de tempo (18 CPS) na primeira tentativa.
                    # Isso elimina a necessidade de um segundo agente de LQA, acelerando o processo em quase 50%.
                    final_text = gema_atomic_processor_v3(
                        item_data, ctx_str, 
                        glossary=merged_glossary, 
                        profile_id=game_profile_id, 
                        job_dir=job_dir
                    )
                    
                    # Trava de Segurança Final (Anti-Alucinação apenas)
                    orig_text = item_data.get('original_text', '')
                    nao_traduziu = (final_text.strip().lower() == orig_text.strip().lower()) and len(orig_text) > 3
                    
                    if nao_traduziu:
                        # Uma única tentativa de correção se ele insistir no Inglês
                        final_text = gema_etapa_correcao_master(orig_text, final_text, item_data.get('duration', 0), reason="qualidade")
                    
                    # Persiste resultados no objeto
                    item_data['translated_text'] = final_text
                    item_data['synced_text'] = final_text
                    item_data['sanitized_text'] = gema_etapa_3_sanitizacao(final_text)
                    
                    # [v20.15] Salvamento Granular: Cria um arquivo individual para cada segmento na pasta de backup dedicada
                    backup_dir = job_dir / "_backup_texto_final"
                    individual_backup_file = backup_dir / f"{item_data['id']}.json"
                    safe_json_write(item_data, individual_backup_file)
                    
                except Exception as ex_atomic:
                    logging.error(f"Falha atômica no item {idx}: {ex_atomic}")
                finally:
                    completed_atomic += 1
                    seg_time = time.time() - start_seg
                    now_str = time.strftime("%H:%M:%S")
                    
                    # Recibo limpo no terminal (como o Alexandre sugeriu)
                    logging.info(f"   ✅ [{now_str}] Segmento {idx} finalizado ({seg_time:.1f}s)")
                    
                    cb((completed_atomic / total_items) * 100, 3, f"[{now_str}] Traduzindo: {completed_atomic}/{total_items} ({seg_time:.1f}s)...")

            # Disparo em Paralelo (3 threads para 4-core i5 / 10 para GPU)
            # Deixa sempre 1 núcleo livre para o sistema não travar.
            device_hw = get_optimal_device()
            # [v20.15] Gemma 4 Optimization: 1 worker on CPU for 4B+ models.
            # Isso garante que a CPU foque 100% dos núcleos em um único pensamento complexo.
            max_pthreads = 1 if "cpu" in device_hw else 10
            logging.info(f"   -> 🚀 [PARALELISMO] Iniciando tradução atômica com {max_pthreads} workers (Safe Mode).")
            
            with ThreadPoolExecutor(max_workers=max_pthreads) as executor:
                futures = [executor.submit(worker_traducao, i, f) for i, f in enumerate(unique_files)]
                for future in as_completed(futures):
                    try: 
                        future.result()
                    except: 
                        pass

            # [v21.05] Popula o micro_cache com as traduções bem-sucedidas para clonar nas repetições
            for f in unique_files:
                orig_key = f.get('original_text', '').strip()
                if orig_key and f.get('translated_text'):
                    micro_cache[orig_key] = f['translated_text']

            # [v21.15] Fim da tradução: Micro-cache atualizado em memória. Sem gravação em disco necessária.
            
            # Aplica cache e clones para o resto da lista
            for f in files_to_process_gema:
                orig = f.get('original_text', '').strip()
                if f.get('_usar_cache') or f.get('_usar_cache_da_fila'):
                    trad_val = micro_cache.get(orig) or micro_cache.get(orig.lower())
                    if trad_val:
                        f['translated_text'] = trad_val
                        f['synced_text'] = trad_val
                        f['sanitized_text'] = gema_etapa_3_sanitizacao(trad_val)
                    else:
                        # Se falhou tudo, mantém original
                        f['translated_text'] = f.get('original_text', '')
                        f['synced_text'] = f.get('original_text', '')
                        f['sanitized_text'] = gema_etapa_3_sanitizacao(f.get('original_text', ''))

            # Finaliza e salva o status final dos arquivos
            for f in files_to_process_gema:
                safe_json_write(f, job_dir / "_backup_texto_final" / f"{f['id']}.json")

            cb(100, 5, "Processamento de texto concluído.")
            unload_gema_model() # [NEW] Libera RAM para o Chatterbox v2

        # --- LÓGICA ANTI-REPETIÇÃO ---
        logging.info("Otimizando a geração de áudio para evitar repetições...")
        text_to_audio_map = {}
        generation_queue = []

        for seg_data in project_data:
            if seg_data.get('processing_status') == 'Copiado Diretamente (Som Não-Verbal)':
                continue
            
            text_to_speak = seg_data.get('manual_edit_text', '').strip() or seg_data.get('sanitized_text', '')
            if not text_to_speak:
                continue

            # Agrupa por texto e locutor para gerar variações por personagem
            # [FIX] Fallback para 'Unknown' se por algum motivo o speaker não estiver definido
            speaker_id = seg_data.get('speaker', 'Unknown')
            unique_key = (text_to_speak, speaker_id)

            if unique_key not in text_to_audio_map:
                text_to_audio_map[unique_key] = []
            
            if len(text_to_audio_map[unique_key]) < 2:
                generation_queue.append(seg_data)
                text_to_audio_map[unique_key].append(seg_data['id'])
                seg_data['is_master_audio'] = True
            else:
                master_id = random.choice(text_to_audio_map[unique_key])
                seg_data['reuse_audio_from_id'] = master_id
                seg_data['is_master_audio'] = False
                logging.info(f"Marcando '{seg_data['id']}' para reutilizar áudio de '{master_id}'.")
        
        # [FIX CRÍTICO] - Forçar Consolidação dos Backups antes do TTS
        # Garante que, mesmo se o usuário pausou/editou/retomou, o Chatterbox pegue o texto MAIS RECENTE do disco.
        logging.info("Consolidando dados finais de texto antes do TTS...")
        backup_final_dir = job_dir / "_backup_texto_final"
        if backup_final_dir.exists():
            for seg in project_data:
                bkp_path = backup_final_dir / f"{seg['id']}.json"
                if bkp_path.exists():
                    try:
                        fresh_data = safe_json_read(bkp_path)
                        if fresh_data:
                            # [v2026.11 FIX] Prioridade Absoluta ao Manual Edit (Não deixa o backup apagar o que o usuário escreveu)
                            current_manual = seg.get('manual_edit_text', '').strip()
                            fresh_manual = fresh_data.get('manual_edit_text', '').strip()
                            
                            # Só atualizamos se o backup tiver algo e a memória estiver vazia, ou se ambos tiverem mas o backup for novo.
                            # Se o usuário ACABOU de editar na UI, a memória é mais nova que o backup em disco.
                            seg['sanitized_text'] = fresh_data.get('sanitized_text', seg.get('sanitized_text', ''))
                            
                            if fresh_manual and not current_manual:
                                seg['manual_edit_text'] = fresh_manual
                            elif current_manual:
                                # Se já tem na memória, mantemos o da memória (que veio do Clique do Usuário)
                                pass 
                            
                            # [FIX CRÍTICO] O speaker REAL vem da pasta física (que já atualizamos no início).
                            # O backup JSON pode estar desatualizado se houve unificação.
                            # Em vez de sobrescrever com o dado velho, vamos ATUALIZAR O BACKUP.
                            current_real_speaker = seg.get('speaker', 'Unknown')
                            backup_speaker = fresh_data.get('speaker')
                            
                            if backup_speaker != current_real_speaker:
                                logging.info(f"Corrigindo speaker no backup de '{seg['id']}': {backup_speaker} -> {current_real_speaker}")
                                fresh_data['speaker'] = current_real_speaker
                                safe_json_write(fresh_data, bkp_path) # Salva o JSON correto no disco!
                    except: pass
        
        safe_json_write(project_data, project_data_path) # Salva as marcações e consolidação
        
        # [PHOENIX VRAM SAFETY LOCK - v2026.5]
        # Inteligência Artificial: Detecta se o usuário PRECISA fechar o LM Studio ou não.
        import torch
        tem_gpu = torch.cuda.is_available()
        
        if tem_gpu:
            print("\n" + "!"*70)
            print(" 🚀 DETECTADA PLACA NVIDIA RTX!")
            print(" ⚠️  IMPORTANTE: FECHE O LM STUDIO AGORA PARA LIBERAR A VRAM.")
            print(" O motor de voz (Chatterbox) precisa da GPU livre para não dar erro.")
            print("!"*70 + "\n")
            
            while True:
                import requests
                lm_studio_vivo = False
                try:
                    res = requests.get("http://127.0.0.1:1234/v1/models", timeout=2)
                    if res.status_code == 200: lm_studio_vivo = True
                except: lm_studio_vivo = False
                
                if not lm_studio_vivo:
                    logging.info("✅ VRAM LIBERADA! Iniciando vozes na RTX...")
                    break
                
                cb(100, 5, "AGUARDANDO: Feche o LM Studio para liberar a GPU.")
                time.sleep(2) # Verifica a cada 2 segundos
        else:
            # SEU CASO (CPU): Apenas um lembrete rápido, sem travar o programa.
            print("\n" + "-"*70)
            print(" 💻 MODO CPU DETECTADO (16GB+ RAM)")
            print(" ℹ️  DICA: Você pode manter o LM Studio aberto se quiser.")
            print(" Se o PC ficar lento, feche o LM Studio para dar mais fôlego à voz.")
            print("-"*70 + "\n")
            cb(100, 5, "Tradução concluída. Continuando processo em modo CPU...")
            time.sleep(3) # Apenas 3 segundos para você ler a dica e ele segue sozinho!


        # --- ETAPA 6: GERAÇÃO TTS CHATTERBOX ---
        cb(0, 6, "Analisando hardware e VRAM...")
        try:
            current_device = get_optimal_device()
            if "cuda" in current_device:
                cb(2, 6, "🚀 Usando Placa de Vídeo (Modo Turbo)")
            else:
                cb(2, 6, "🐢 Usando Processador (Gemma 4 ativo ou sem GPU)")
        except:
            pass

        dubbed_audio_dir = job_dir / "_dubbed_audio"
        dubbed_audio_dir.mkdir(exist_ok=True)

        actual_generation_queue = []
        if generation_queue:
            for seg_data in generation_queue:
                output_path = dubbed_audio_dir / f"{seg_data['id']}_dubbed.wav"
                current_text = seg_data.get('manual_edit_text', '').strip() or seg_data.get('sanitized_text', '')
                force_regen = False
                individual_json = job_dir / "_backup_texto_final" / f"{seg_data['id']}.json"
                if individual_json.exists():
                    saved_val = safe_json_read(individual_json)
                    saved_text = saved_val.get('manual_edit_text', '').strip() or saved_val.get('sanitized_text', '')
                    if saved_text != current_text:
                        force_regen = True

                if not output_path.exists() or force_regen:
                    actual_generation_queue.append(seg_data)

        if actual_generation_queue:
            # [MEMORY SAFETY] Limpeza agressiva antes de carregar o motor de voz
            import gc
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            
            cb(0, 6, "Carregando Chatterbox...")
            tts_model = get_chatterbox_model()
            if tts_model is None: raise RuntimeError("Motor TTS indisponível.")

            global_fallback = Path("resources/base_speakers/pt/default_pt_speaker.wav")
            total_gen = len(actual_generation_queue)
            for i, seg_data in enumerate(actual_generation_queue):
                start_seg = time.time()
                output_path = dubbed_audio_dir / f"{seg_data['id']}_dubbed.wav"
                perc_total = 5 + (i / total_gen) * 95
                
                ref_path = diarization_dir / seg_data.get('speaker', 'Unknown') / "_REF_VOZ_UNIFICADA.wav"
                if not ref_path.exists():
                    ref_path = diarization_dir / seg_data.get('speaker', 'Unknown') / seg_data.get('file_name', '')
                if not ref_path.exists(): ref_path = global_fallback

                # Bypass e Hallucination Check
                text_clean = seg_data.get('original_text', '').strip().lower()
                is_hallucination = False
                if text_clean:
                    normal = len([c for c in text_clean if c.isalnum() or c.isspace()])
                    if (normal / len(text_clean)) < 0.5: is_hallucination = True

                if is_hallucination or seg_data.get('detected_language') == 'pt':
                    try:
                        orig = (diarization_dir / seg_data.get('speaker', 'Unknown') / seg_data.get('file_name', ''))
                        if orig.exists():
                            from pydub import AudioSegment
                            AudioSegment.from_file(str(orig)).set_frame_rate(24000).set_channels(1).export(output_path, format="wav")
                    except: pass
                    continue

                # Geração Real
                try:
                    text_to_speak = seg_data.get('manual_edit_text', '').strip() or seg_data.get('sanitized_text', '')
                    # [BR-FIX] Aplica o Corretor de Sotaque e Expansão Fonética
                    text_to_speak = corrigir_sotaque_pt_br(text_to_speak)
                    
                    text_to_speak = text_to_speak.replace(".", ",")
                    # [v22.10 FIX] Removido espaço forçado para evitar gaguejo inicial ("tê")
                    # [v2026.9 FIX] Estabilizador de Fôlego: Usamos vírgula para não confundir a IA com fim de frase
                    if "!" in text_to_speak and text_to_speak.find("!") < 15:
                        text_to_speak = text_to_speak.replace("!", ",", 1)
                    
                    is_shout = "!" in text_to_speak or text_to_speak.isupper()
                    logging.info(f"🎙️ Dublando [{seg_data.get('speaker', 'voz')}]: '{text_to_speak}'")

                    # Parâmetros recalibrados para evitar que a IA 'morra' no meio da frase
                    wav_tensor = tts_model.generate(
                        text=text_to_speak, language_id="pt", audio_prompt_path=str(ref_path),
                        exaggeration=0.32 if is_shout else 0.15, # Valor menor = mais fôlego
                        temperature=0.72 if is_shout else 0.45,
                        min_p=0.10, 
                        repetition_penalty=1.2 
                    )
                    import soundfile as sf
                    sf.write(str(output_path), wav_tensor.squeeze(0).cpu().numpy(), 24000)

                    try:
                        # [v22.30] SOFT CLEANUP (Rotina Ultra-Segura)
                        time.sleep(0.1) # Pequeno delay para garantir que o arquivo foi liberado pelo SO
                        from pydub import AudioSegment
                        from pydub.silence import detect_nonsilent
                        
                        audio = AudioSegment.from_wav(str(output_path))
                        # Detecta partes com som real
                        nonsilent_ranges = detect_nonsilent(audio, min_silence_len=100, silence_thresh=-45)
                        
                        if nonsilent_ranges:
                            start_trim = max(0, nonsilent_ranges[0][0] - 100) 
                            end_trim = min(len(audio), nonsilent_ranges[-1][1] + 200) 
                            
                            audio = audio[start_trim:end_trim]
                            audio = audio.fade_in(50).fade_out(50) 
                            audio.export(output_path, format="wav")
                    except Exception as clean_err:
                        # Se a limpeza falhar por qualquer motivo, o áudio original continua existindo, então ignoramos o erro
                        pass
                    finally:
                        seg_time = time.time() - start_seg
                        now_str = time.strftime("%H:%M:%S")
                        cb(perc_total, 6, f"[{now_str}] Gerando {i+1}/{total_gen}: {seg_data['id']} ({seg_time:.1f}s)")

                except Exception as e:
                    import traceback
                    tb_str = traceback.format_exc()
                    print(f"\n" + "="*50)
                    print(f"🚨 ERRO CRÍTICO NO CHATTERBOX ({seg_data.get('id', 'N/A')}) 🚨")
                    print(f"Detalhes do Erro: {e}")
                    print(f"Traceback Técnico para Debug:")
                    print(tb_str)
                    print("="*50 + "\n")
                    # Registra também no arquivo de log do sistema se o logging estiver ativo
                    logging.error(f"Chatterbox Falhou na geração de {seg_data.get('id', 'N/A')}:\n{tb_str}")
        else:
             cb(100, 6, "Pronto.")

        # --- ETAPA 7: FINALIZAÇÃO E MASTERIZAÇÃO ---
        cb(0, 7, "Iniciando finalização e masterização...")
        final_output_dir = job_dir / "_saida_final"
        mastering_cache_path = job_dir / "mastering_cache.json"
        mastering_cache = safe_json_read(mastering_cache_path) or {}
        durations_cache_path = job_dir / "durations_cache.json"
        durations_cache = safe_json_read(durations_cache_path) or {}

        # [FEATURE] Manual Volume Boost - Leitura da Configuração
        volume_boost_factor = 1.0
        try:
            boost_file = job_dir / "volume_boost.txt"
            if boost_file.exists():
                with open(boost_file, "r") as f:
                    content = f.read().split('\n')[0].split('#')[0].strip()
                    val_int = int(content) if content else 0
                    
                    # [SAFETY] Limite Duro de Segurança (Atômico)
                    # +30dB já é um absurdo (32x potêcia). Acima disso é risco real de dano físico.
                    if val_int > 30:
                        logging.warning(f"[SAFETY] Volume solicitado ({val_int}dB) excede o limite seguro. Ajustado para +30dB.")
                        val_int = 30
                    
                    # [MODIFIED] Removido limite de 100% a pedido do usuário (Bioshock 1)
                    # Agora o céu é o limite (Cuidado com distorção!)
                    if val_int < 0: val_int = 0
                    
                    if val_int > 0:
                        # [MODIFIED] Interpretação Direta em dB (COD Style)
                        # 1 = +1dB
                        # 15 = +15dB (Alto)
                        # 100 = +100dB (Explodido)
                        volume_boost_factor = float(val_int)
                        logging.info(f"Audio Compression Ativado: Master Boost + {val_int}dB de Ganho.")
                    else:
                        volume_boost_factor = 0
                        logging.info("Audio Compression: Desativado (0dB).")
        except Exception as e:
            logging.error(f"Erro ao ler volume_boost.txt: {e}")

        # [v12.70] Lógica Unificada de Perfis via Dicionário Dinâmico
        game_profile_id = status.get('game_profile', 'padrao')
        profile = load_game_profile(game_profile_id)
        audio_cfg = profile.get('audio_settings', {})
        profile_filters = []
        
        # 1. Normalização / Loudnorm
        if 'loudnorm' in audio_cfg:
             profile_filters.append(f"loudnorm={audio_cfg['loudnorm']}")
        
        # 2. Compressor
        if 'acompressor' in audio_cfg:
             profile_filters.append(f"acompressor={audio_cfg['acompressor']}")
             
        # 3. Equalizador (Bass/Treble)
        if 'bass' in audio_cfg:
             profile_filters.append(f"bass={audio_cfg['bass']}")
        if 'treble' in audio_cfg:
             profile_filters.append(f"treble={audio_cfg['treble']}")
        
        # 4. Volume Boost Default
        if volume_boost_factor <= 1.0: 
            volume_boost_factor = audio_cfg.get('volume_boost_default', 0)
            if volume_boost_factor > 0:
                 logging.info(f"[PROFILE] {profile['name']}: Aplicando ganho automático de +{volume_boost_factor}dB.")

        logging.info(f"[PROFILE] {profile['name']}: Ativando Otimização de Áudio Profissional.")

        # Define pastas (Etapa 7) - FORÇANDO REPROCESSAMENTO TOTAL (SEM CACHE)
        dubbed_audio_dir = job_dir / "_dubbed_audio"
        final_output_dir = job_dir / "_saida_final"
        diarization_dir = job_dir / "_2_PARA_AS_PASTAS_DE_VOZ"
        final_output_dir.mkdir(exist_ok=True)

        # [FORÇAR RESET] Limpa cache de masterização para este job para garantir reprocessamento total
        logging.info("🔥 AVISO: Forçando limpeza do cache de masterização para garantir áudio dublado.")
        for key in list(mastering_cache.keys()):
            if any(key.startswith(str(s['id'])) for s in project_data):
                del mastering_cache[key]

        logging.info("--- INICIANDO MASTERIZAÇÃO FINAL (MODO FORÇADO) ---")

        for i, seg_data in enumerate(project_data):
            file_id = seg_data['id']
            file_name = seg_data.get('file_name', f"{file_id}.wav")
            final_path = final_output_dir / f"{file_id}{file_format_map.get(file_id, '.wav')}"
            
            # [RESET] Sempre tenta re-processar para garantir que não fique inglês
            if final_path.exists(): 
                try: os.remove(final_path)
                except: pass
            
            speaker_id = seg_data.get('speaker', 'Unknown')
            cb((i / len(project_data)) * 100, 7, f"Finalizando: {file_name}")

            original_duration = seg_data.get('duration', 0)
            original_file_path = diarization_dir / speaker_id / file_name
            source_path = None
            is_fallback_copy = False
            dubbed_check_path = dubbed_audio_dir / f"{file_id}_dubbed.wav"

            if dubbed_check_path.exists():
                source_path = dubbed_check_path
                logging.info(f"Usando áudio dublado encontrado para '{file_id}'.")
            elif seg_data.get('reuse_audio_from_id'):
                master_id = seg_data['reuse_audio_from_id']
                source_path = dubbed_audio_dir / f"{master_id}_dubbed.wav"
                logging.info(f"Reutilizando áudio de '{master_id}' para '{file_id}'.")
            else: # Realmente não existe dublagem
                source_path = original_file_path
                is_fallback_copy = True

            # --- LÓGICA DE SELEÇÃO INTELIGENTE (SEM ENROLAÇÃO) ---
            is_non_verbal = (seg_data.get('processing_status') == 'Copiado Diretamente (Som Não-Verbal)')
            
            if is_fallback_copy and not is_non_verbal:
                # SE TEM TRADUÇÃO MAS NÃO ACHOU O ÁUDIO DUBLADO -> ERRO!
                logging.error(f"❌ ERRO CRÍTICO: Áudio dublado NÃO encontrado para '{file_id}' (Deveria estar dublado).")
                logging.error(f"   Verifique a pasta '_dubbed_audio'. Pulando para não gerar em inglês.")
                continue

            # Se for não-verbal, 'source_path' já aponta para o original e 'is_fallback_copy' é True. 
            # Isso é o esperado para gemidos/sons.

            try:
                # Medimos a duração do source
                source_duration = get_audio_duration(str(source_path))
                
                filters_to_apply = []
                speed_factor = 1.0
                TOLERANCE_SECONDS = 0.1

                # 1. Poda de Silêncio e Aceleração (Sincronia)
                if original_duration > 0:
                    filters_to_apply.append("silenceremove=start_periods=1:start_threshold=-50dB:stop_periods=-1:stop_threshold=-50dB")
                    if source_duration > (original_duration + TOLERANCE_SECONDS):
                        calculated_factor = source_duration / original_duration
                        speed_factor = min(calculated_factor, 1.30)
                        temp_factor = speed_factor
                        while temp_factor > 2.0:
                            filters_to_apply.append("atempo=2.0")
                            temp_factor /= 2.0
                        if temp_factor > 1.0: filters_to_apply.append(f"atempo={temp_factor:.4f}")

                # 2. Corrente de Masterização
                master_chain = ["dynaudnorm"]
                if volume_boost_factor > 0:
                    has_compressor = any("acompressor" in f for f in profile_filters)
                    if not has_compressor:
                        master_chain.append("acompressor=threshold=-12dB:ratio=4:attack=5:release=50:makeup=2")
                    master_chain.append(f"volume={volume_boost_factor}dB")
                    master_chain.append("alimiter=limit=0.966:level=disabled:attack=5:release=50")
                
                if profile_filters and seg_data.get('processing_status') != 'Copiado Diretamente (Som Não-Verbal)':
                    master_chain = profile_filters + master_chain

                cmd = ['ffmpeg', '-y', '-threads', str(os.cpu_count() or 4), '-i', str(source_path)]
                
                # Som de Fundo (Se houver)
                bg_file_path = None
                try:
                    if str(status.get('preserve_background', 'false')).lower() == 'true':
                        stem_bg_dir = job_dir / "_0b_SEPARACAO_FUNDO"
                        if stem_bg_dir.exists():
                            pb = list(stem_bg_dir.rglob(seg_data['file_name']))
                            if pb: bg_file_path = pb[0]
                except: pass

                if bg_file_path:
                    cmd.extend(['-i', str(bg_file_path)])
                    v_chain = ",".join(filters_to_apply + master_chain)
                    cmd.extend(['-filter_complex', f"[0:a]{v_chain}[v];[1:a]volume=0.4[b];[v][b]amix=inputs=2:duration=longest[out]", '-map', '[out]'])
                else:
                    all_filters = filters_to_apply + master_chain
                    if all_filters: cmd.extend(['-af', ",".join(all_filters)])

                # [v2026.8 FIX] Seleção Inteligente de Codec
                # Se o destino for .mp3, usamos libmp3lame. Se for .wav, usamos pcm_s16le.
                ext_final = str(final_path.suffix).lower()
                codec_final = 'libmp3lame' if ext_final == '.mp3' else 'pcm_s16le'
                
                output_profile = status.get('detected_profile', {})
                native_ar = str(output_profile.get('ar', '44100'))
                native_ac = str(output_profile.get('ac', '1'))
                
                cmd.extend([
                    '-c:a', codec_final, 
                    '-ar', native_ar, 
                    '-ac', native_ac,
                    '-map_metadata', '-1' # Limpa metadados corrompidos do jogo original
                ])
                
                # Para MP3, adicionamos o bitrate padrão de alta qualidade
                if codec_final == 'libmp3lame':
                    cmd.extend(['-b:a', '192k'])

                cmd.append(str(final_path))
                logging.info(f"🔊 Masterizando ({codec_final}): {file_id}")
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                     logging.error(f"❌ FFmpeg falhou para {file_id}: {result.stderr}")
                     continue 
                
                if file_id not in durations_cache: durations_cache[file_id] = {}
                durations_cache[file_id]['speed_factor'] = speed_factor

            except Exception as e:
                logging.error(f"❌ Erro grave em {file_id}: {e}")
                continue # Não copia original!

            # --- CAPTURA DE MÉTRICAS PÓS-PROCESSAMENTO ---
            # Agora que garantimos que o arquivo existe (criado agora ou já existia), vamos medir.
            if final_path.exists():
                try:
                    # 1. Duração Final
                    final_duration = get_audio_duration(str(final_path))
                    
                    # 2. Pico de Áudio (Mastering Check)
                    final_peak = get_audio_peak_dbfs(final_path)

                    # Atualiza Cache de Duração
                    if file_id not in durations_cache: durations_cache[file_id] = {}
                    durations_cache[file_id]['duration'] = final_duration
                    
                    # Se foi gerado pelo Chatterbox, salvamos a duração "pura" dele antes da masterização também
                    # Como já passamos dessa fase, se não tiver no cache, paciência. Mas podemos tentar inferir ou ignorar.
                    if source_path and source_path.exists() and not is_fallback_copy:
                         durations_cache[file_id]['Chatterbox_duration'] = get_audio_duration(str(source_path))

                    # Atualiza Cache de Masterização
                    mastering_status = 'fallback_copied' if is_fallback_copy else 'mastered'
                    
                    # Tenta pegar pico original para comparação
                    original_peak = None
                    if original_file_path.exists():
                         original_peak = get_audio_peak_dbfs(original_file_path)

                    mastering_cache[file_id] = {
                        'status': mastering_status,
                        'original_peak_dbfs': original_peak,
                        'final_peak_dbfs': final_peak,
                        'timestamp': datetime.now().isoformat()
                    }
                    if source_path and source_path.exists() and not is_fallback_copy:
                        mastering_cache[file_id]['dubbed_peak_before_mastering_dbfs'] = get_audio_peak_dbfs(source_path)

                    # --- PERSISTÊNCIA ATÔMICA ---
                    safe_json_write(durations_cache, durations_cache_path)
                    safe_json_write(mastering_cache, mastering_cache_path)
                    
                except Exception as e:
                    logging.error(f"Erro ao capturar métricas finais para {file_id}: {e}")

        cb(100, 7, "Finalização e masterização concluídas.")
        
        # --- ETAPA EXTRA: UNIR SEGMENTOS SEPARADOS ---
        # Se houve split de arquivos longos (ex: sample_seg001, sample_seg002...), precisamos juntá-los agora.
        logging.info("Verificando se há segmentos para unir...")
        final_output_dir = job_dir / "_saida_final"
        segment_groups = {}
        
        # Regex tripla para capturar todas as táticas de divisão de segmentos do sistema
        # 1. Vídeos: sample_0156_seg001_3s.wav
        # 2. Jogos (Silêncio): sample_0088_parte_004.wav
        # 3. Jogos (Orador/Cirúrgica v10.60): sample_0088_p01.wav
        seg_pattern_video = re.compile(r"(.+)_seg(\d{3})_(\d+)s(\..+)")
        seg_pattern_jogos_silence = re.compile(r"(.+)_parte_(\d{3})(\..+)")
        seg_pattern_jogos_speaker = re.compile(r"(.+)_p(\d{2})(\..+)")
        
        for file_path in final_output_dir.glob("*"):
            match_video = seg_pattern_video.match(file_path.name)
            match_jogos_s = seg_pattern_jogos_silence.match(file_path.name)
            match_jogos_p = seg_pattern_jogos_speaker.match(file_path.name)
            
            if match_video:
                base_name, idx, ext = match_video.group(1), int(match_video.group(2)), match_video.group(4)
                if base_name not in segment_groups: segment_groups[base_name] = []
                segment_groups[base_name].append((idx, file_path, ext))
            elif match_jogos_s:
                base_name, idx, ext = match_jogos_s.group(1), int(match_jogos_s.group(2)), match_jogos_s.group(3)
                if base_name not in segment_groups: segment_groups[base_name] = []
                segment_groups[base_name].append((idx, file_path, ext))
            elif match_jogos_p:
                base_name, idx, ext = match_jogos_p.group(1), int(match_jogos_p.group(2)), match_jogos_p.group(3)
                if base_name not in segment_groups: segment_groups[base_name] = []
                segment_groups[base_name].append((idx, file_path, ext))
        
        if segment_groups:
            segments_backup_dir = final_output_dir / "segmentos_individuais_backup"
            segments_backup_dir.mkdir(exist_ok=True)
            
            for base_name, segments in segment_groups.items():
                if not segments: continue
                segments.sort(key=lambda x: x[0])
                output_merged_path = final_output_dir / f"{base_name}{segments[0][2]}"
                list_path = final_output_dir / f"{base_name}_concat_list.txt"
                
                logging.info(f"Unindo {len(segments)} segmentos para criar: {output_merged_path.name}")
                
                # Se for apenas 1 segmento restante (ex: a parte 02 foi silenciada/apagada por erro)
                if len(segments) == 1:
                    logging.info(f"Reconstruindo arquivo único órfão: {segments[0][1].name}")
                    try:
                        shutil.copy(str(segments[0][1]), str(output_merged_path))
                        shutil.move(str(segments[0][1]), str(segments_backup_dir / segments[0][1].name))
                        continue
                    except Exception as e:
                        logging.error(f"Erro ao renomear arquivo órfão {base_name}: {e}")
                        continue
                
                concat_success = False
                # TENTATIVA 1: FFmpeg stream copy (Rápido e sem perda)
                try:
                    with open(list_path, 'w', encoding='utf-8') as f:
                        for _, seg_path, _ in segments:
                            f.write(f"file '{seg_path.name}'\n")
                    
                    subprocess.run([
                        'ffmpeg', '-y', '-f', 'concat', '-safe', '0', 
                        '-i', str(list_path), '-c', 'copy', str(output_merged_path)
                    ], check=True, capture_output=True)
                    concat_success = True
                except Exception as e:
                    logging.warning(f"FFmpeg copy falhou para {base_name}, tentando Fallback Pydub... Erro: {e}")
                
                # TENTATIVA 2: PyDub Concat (Robusto, recodifica mas ignora cabeçalhos corrompidos)
                if not concat_success:
                    try:
                        from pydub import AudioSegment
                        merged_audio = AudioSegment.empty()
                        for _, seg_path, _ in segments:
                            merged_audio += AudioSegment.from_file(str(seg_path))
                        merged_audio.export(str(output_merged_path), format=segments[0][2].replace('.', ''))
                        concat_success = True
                        logging.info(f"Fallback PyDub concluiu a união de {base_name}.")
                    except Exception as e2:
                        logging.error(f"Erro FATAL ao unir segmentos de {base_name} no fallback: {e2}")
                
                # Limpeza: Move fragmentos para backup e apaga lista
                if concat_success:
                    
                    for _, seg_path, _ in segments:
                        if seg_path.exists():
                            moved = False
                            for attempt in range(5):
                                try:
                                    shutil.move(str(seg_path), str(segments_backup_dir / seg_path.name))
                                    moved = True
                                    break
                                except Exception:
                                    time.sleep(1) # Aguarda liberação do Antivírus/Windows
                            
                            if not moved:
                                logging.warning(f"Aviso: Não foi possível mover {seg_path.name} para backup (Lock persistente do Windows).")
                                
                    if list_path.exists():
                        try:
                            os.remove(list_path)
                        except: pass
                        
        gerar_relatorio_final(job_dir, job_id, project_data, file_format_map)
        cb(100, 8, "Processo concluído! Arquivos finais em '_saida_final'.")

    except Exception as e:
        import traceback
        logging.error(f"ERRO NO PIPELINE (Job ID: {job_id}): {e}\n{traceback.format_exc()}")
        set_progress(job_id, 100, len(ETAPAS_JOGOS) - 1, start_time, ETAPAS_JOGOS, subetapa=f"Erro: {e}")
        status_path = job_dir / "job_status.json"
        status_data = safe_json_read(status_path) or {}
        status_data['status'] = 'failed'
        status_data['error'] = str(e)
        safe_json_write(status_data, status_path)
    finally:
        with active_jobs_lock:
            if job_id in active_jobs:
                active_jobs.remove(job_id)
                
        # [MEMORY RECOVERY] Limpeza agressiva no final do Job
        import gc
        import torch
        logging.info(" === INICIANDO LIMPEZA AGRESSIVA DE MEMÓRIA PÓS-JOB ===")
        unload_whisper_model()
        unload_chatterbox_model()
        unload_gema_model()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        logging.info(" === LIMPEZA DE MEMÓRIA CONCLUÍDA ===")

# --- FUNÇÃO DE REMOÇÃO DE RÁDIO/NOISE (SUBSTITUI OPENUNMIX) ---
def processar_separacao(job_dir, job_id, start_time):
    with active_jobs_lock:
        if len(active_jobs) >= MAX_CONCURRENT_JOBS:
             logging.warning(f"❌ [HARDWARE] Limite de {MAX_CONCURRENT_JOBS} job(s) atingido. Ignorando {job_id}.")
             return
        active_jobs.add(job_id)
    
    try:
        set_low_process_priority()
        def cb(p, etapa, s=None): set_progress(job_id, p, etapa, start_time, ETAPAS_SEPARACAO, s)
        
        input_dir = job_dir / "_input"
        output_dir = job_dir / "_saida_audio_restaurado"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        input_files = list(input_dir.glob("*.*"))
        total_files = len(input_files)

        if total_files == 0:
            cb(100, 3, "Nenhum arquivo encontrado para restaurar.")
            return

        cb(0, 1, f"Iniciando restauração de áudio (FFmpeg) para {total_files} arquivos...")
        
        for i, audio_file in enumerate(input_files):
            cb((i / total_files) * 100, 1, f"Processando: {audio_file.name}")
            
            # Verificar se é vídeo e extrair áudio
            suffix = audio_file.suffix.lower()
            if suffix in ['.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.webm']:
                logging.info(f"Arquivo de vídeo detectado: {audio_file.name}. Extraindo áudio...")
                extracted_audio_path = audio_file.with_name(f"{audio_file.stem}_extracted.wav")
                try:
                    # Extrai áudio para wav estéreo 44.1kHz
                    cmd_extract = [
                        'ffmpeg', '-y', '-i', str(audio_file),
                        '-vn', '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '2',
                        str(extracted_audio_path)
                    ]
                    subprocess.run(cmd_extract, check=True, capture_output=True, text=True)
                    audio_file = extracted_audio_path # Atualiza a variável para usar o áudio extraído
                    logging.info(f"Áudio extraído com sucesso: {audio_file.name}")
                except subprocess.CalledProcessError as e:
                    logging.error(f"Erro ao extrair áudio do vídeo {audio_file.name}: {e.stderr}")
                    continue
            
            # --- LÓGICA DE LIMPEZA / REMOÇÃO DE EFEITO DE RÁDIO (FFMPEG) ---
            output_path = output_dir / f"{audio_file.stem}.wav"
            
            # Filtros Explicados:
            # 1. afftdn=nf=-25: Remove ruído de banda larga (chiado/hiss) com força média (-25dB)
            # 2. highpass=f=80: Remove "rumble" e graves exagerados que sujam a clonagem
            # 3. lowpass=f=12000: Corta frequências super agudas inúteis que podem ter artefatos
            # 4. equalizer=f=3500...: Dá um leve ganho nas frequências de voz para tentar "abrir" o som abafado
            # [FIX] Removido compand agressivo que causava o efeito "bombear" / cortar a voz subitamente.
            
            cmd_clean = [
                'ffmpeg', '-y', '-i', str(audio_file),
                '-af', 'afftdn=nf=-25, highpass=f=80, lowpass=f=12000, equalizer=f=3500:t=h:w=2000:g=2',
                str(output_path)
            ]

            try:
                subprocess.run(cmd_clean, check=True, capture_output=True, text=True)
                logging.info(f"Áudio restaurado salvo em: {output_path}")
            except subprocess.CalledProcessError as e:
                logging.error(f"Erro ao limpar áudio {audio_file.name}: {e.stderr}")
                # Em caso de erro, tenta copiar o original como fallback
                shutil.copy(audio_file, output_path)

        cb(100, 2, "Finalizando processos...")
        cb(100, 3, "Restauração concluída!")

    except Exception as e:
        logging.error(f"ERRO NO PIPELINE DE RESTAURAÇÃO (Job ID: {job_id}): {e}\n{traceback.format_exc()}")
        set_progress(job_id, 100, len(ETAPAS_SEPARACAO) - 1, start_time, ETAPAS_SEPARACAO, subetapa=f"Erro: {e}")
        status_path = job_dir / "job_status.json"
        status_data = safe_json_read(status_path) or {}
        status_data['status'] = 'failed'
        status_data['error'] = str(e)
        safe_json_write(status_data, status_path)
    finally:
        with active_jobs_lock:
            if job_id in active_jobs:
                active_jobs.remove(job_id)


# --- ROTAS FLASK ---
def calculate_files_hash(files):
    hasher = hashlib.sha256()
    sorted_files = sorted(files, key=lambda f: f.filename)
    for f in sorted_files:
        file_info = f"{f.filename}:{f.seek(0, os.SEEK_END)}"
        hasher.update(file_info.encode('utf-8'))
        f.seek(0)
    return hasher.hexdigest()

# --- INTEGRAÇÃO COM MÓDULO DE VÍDEOS ---
try:
    from App_videos import pipeline_dublar_video
except ImportError:
    logging.error("Falha ao importar App_videos. O módulo de dublagem de vídeo não funcionará.")
    pipeline_dublar_video = None

@app.route('/dublar', methods=['POST'])
def dublar_video_route():
    start_time = time.time()
    if 'video' not in request.files:
        return jsonify({'error': 'Nenhum vídeo enviado.'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'Nenhum vídeo selecionado.'}), 400
        
    timestamp = int(time.time())
    datestamp = datetime.now().strftime('%d.%m.%Y')
    job_id = f"job_video_{datestamp}_{timestamp}"
    job_dir = Path(app.config['UPLOAD_FOLDER']) / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    
    filename = secure_filename(file.filename)
    # Salva como input_video (padrão do App_videos)
    extension = Path(filename).suffix
    file.save(job_dir / f"input_video{extension}")
    
    # Salva Params
    params = {
        'target_language': request.form.get('destino', 'pt'),
        'num_speakers': int(request.form.get('num_speakers', '1')),
        'original_filename': filename
    }
    safe_json_write(params, job_dir / "job_params.json")
    
    status_data = {'job_id': job_id, 'status': 'iniciando', 'mode': 'video_dubbing'}
    safe_json_write(status_data, job_dir / "job_status.json")
    
    if pipeline_dublar_video:
        threading.Thread(target=pipeline_dublar_video, args=(job_dir, job_id, start_time)).start()
        return jsonify({'status': 'processing', 'job_id': job_id})
    else:
        return jsonify({'error': 'Módulo de Vídeo não carregado.'}), 500


@app.route('/transcrever', methods=['POST'])
def transcrever_arquivo():
    start_time = time.time()
    if 'transcricao_file' not in request.files:
        return jsonify({'error': 'Nenhum ficheiro enviado.'}), 400
    
    file = request.files['transcricao_file']
    if file.filename == '':
        return jsonify({'error': 'Nenhum ficheiro selecionado.'}), 400

    timestamp = int(time.time())
    datestamp = datetime.now().strftime('%d.%m.%Y')
    job_id = f"job_transcricao_{datestamp}_{timestamp}"
    job_dir = Path(app.config['UPLOAD_FOLDER']) / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    filename = secure_filename(file.filename)
    extension = Path(filename).suffix
    input_path = job_dir / f"input{extension}"
    file.save(input_path)

    status_data = {'job_id': job_id, 'status': 'iniciando', 'original_filename': filename}
    safe_json_write(status_data, job_dir / "job_status.json")

    threading.Thread(target=processar_transcricao, args=(job_dir, job_id, start_time)).start()
    return jsonify({'status': 'processing', 'job_id': job_id})

# [REDE] A rota /dublar_jogos foi consolidada abaixo (ver linha 3750)
# @app.route('/dublar_jogos', ... ) 

@app.route('/converter', methods=['POST'])
def converter_arquivos():
    start_time = time.time()
    if 'arquivos_referencia' not in request.files or 'arquivos_para_converter' not in request.files:
        return jsonify({'error': 'Faltam os arquivos de referência ou os arquivos para converter.'}), 400

    ref_files = request.files.getlist('arquivos_referencia')
    conv_files = request.files.getlist('arquivos_para_converter')

    if not ref_files or not conv_files:
        return jsonify({'error': 'Nenhum arquivo enviado em uma das categorias.'}), 400

    timestamp = int(time.time())
    datestamp = datetime.now().strftime('%d.%m.%Y')
    job_id = f"job_conversor_{datestamp}_{timestamp}"
    job_dir = Path(app.config['UPLOAD_FOLDER']) / job_id
    
    ref_dir = job_dir / "_1_referencia"
    conv_dir = job_dir / "_2_para_converter"
    ref_dir.mkdir(parents=True, exist_ok=True)
    conv_dir.mkdir(parents=True, exist_ok=True)

    file_format_map = {}
    for file in ref_files:
        filename = secure_filename(file.filename)
        file.save(ref_dir / filename)
    
    for file in conv_files:
        filename = secure_filename(file.filename)
        base_filename, extension = Path(filename).stem, Path(filename).suffix
        file_format_map[base_filename] = extension
        file.save(conv_dir / filename)

    status_data = {'job_id': job_id, 'status': 'iniciando', 'file_format_map': file_format_map}
    safe_json_write(status_data, job_dir / "job_status.json")

    threading.Thread(target=processar_conversao, args=(job_dir, job_id, start_time)).start()
    return jsonify({'status': 'processing', 'job_id': job_id})

@app.route('/separar_audio', methods=['POST'])
def separar_audio():
    start_time = time.time()
    
    files = []
    if 'separacao_file' in request.files:
        files = request.files.getlist('separacao_file')
    
    # Fallback se não encontrar pela chave especifica, tenta pegar todos os arquivos enviados
    if not files:
        files = list(request.files.values())

    # Filtra arquivos vazios
    files = [f for f in files if f.filename != '']

    if not files:
        return jsonify({'error': 'Nenhum ficheiro selecionado.'}), 400

    timestamp = int(time.time())
    datestamp = datetime.now().strftime('%d.%m.%Y')
    job_id = f"job_separacao_{datestamp}_{timestamp}"
    job_dir = Path(app.config['UPLOAD_FOLDER']) / job_id
    input_dir = job_dir / "_input"
    input_dir.mkdir(parents=True, exist_ok=True)

    saved_filenames = []
    for file in files:
        filename = secure_filename(file.filename)
        file.save(input_dir / filename)
        saved_filenames.append(filename)

    status_data = {'job_id': job_id, 'status': 'iniciando', 'file_count': len(saved_filenames), 'files': saved_filenames}
    safe_json_write(status_data, job_dir / "job_status.json")

    threading.Thread(target=processar_separacao, args=(job_dir, job_id, start_time)).start()
    return jsonify({'status': 'processing', 'job_id': job_id})

def limpar_hallucinacoes_projeto(job_id):
    """
    Varre a pasta de saída final e aplica o Surgical Sync v2.0 em arquivos existentes.
    Útil para limpar projetos que já terminaram mas ficaram com 'eeee' no áudio.
    """
    project_dir = Path(app.config['UPLOAD_FOLDER']) / job_id
    output_dir = project_dir / "_saida_final"
    
    if not output_dir.exists():
        return False, "Pasta de saída final não encontrada."
    
    # Carregar project_data para saber as durações originais (para a Trava de Segurança)
    project_data_path = project_dir / "project_data.json"
    durations = {}
    if project_data_path.exists():
        try:
            with open(project_data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for seg in data:
                    durations[seg['id']] = seg.get('duration', 0)
        except: pass

    from pydub import AudioSegment
    from pydub.silence import detect_nonsilent
    
    count = 0
    files = list(output_dir.glob("*.wav"))
    for file_path in files:
        seg_id = file_path.stem
        
        try:
            clip_raw = AudioSegment.from_wav(str(file_path))
            # [v11.7] Relaxado para -50dB (COD Radio Style)
            nonsilent_ranges = detect_nonsilent(clip_raw, min_silence_len=150, silence_thresh=-50)
            
            if nonsilent_ranges:
                start_trim = max(0, nonsilent_ranges[0][0] - 20)
                end_trim = nonsilent_ranges[-1][1]
                # [v11.7] Margem de 150ms para evitar 'mova' -> 'mo'
                final_end_trim = min(len(clip_raw), end_trim + 150)
                clip_trimmed = clip_raw[start_trim:final_end_trim]
                
                # Trava de Segurança (+50%)
                original_dur = durations.get(seg_id, 0)
                if original_dur > 0 and len(clip_trimmed) > (original_dur * 1500):
                    clip_trimmed = clip_trimmed[:int(original_dur * 1400) + 100]
                
                if len(clip_trimmed) < len(clip_raw) - 50: # Se mudou mais de 50ms
                    clip_trimmed.export(str(file_path), format="wav")
                    count += 1
        except: continue
        
    return True, f"Limpeza concluída. {count} arquivos de áudio foram higienizados no projeto {job_id}."

@app.route('/limpar_artefatos/<job_id>')
def route_limpar_artefatos(job_id):
    success, message = limpar_hallucinacoes_projeto(job_id)
    return jsonify({"success": success, "message": message})

@app.route('/dublar_jogos', methods=['POST'])
def dublar_jogos():
    start_time = time.time()
    
    if 'job_id' in request.form and request.form.get('job_id'):
        job_id = request.form.get('job_id')
        job_dir = Path(app.config['UPLOAD_FOLDER']) / job_id
        if not job_dir.exists(): return jsonify({'error': f'Trabalho não encontrado.'}), 404
        threading.Thread(target=processar_dublagem_jogos, args=(job_dir, job_id, start_time)).start()
        return jsonify({'status': 'processing', 'job_id': job_id})

    elif 'wav_files' in request.files:
        files = request.files.getlist('wav_files')
        if not files or files[0].filename == '': return jsonify({'error': 'Nenhum ficheiro enviado.'}), 400

        files_hash = calculate_files_hash(files)
        if existing_job_id := find_existing_project(files_hash):
            job_dir = Path(app.config['UPLOAD_FOLDER']) / existing_job_id
            threading.Thread(target=processar_dublagem_jogos, args=(job_dir, existing_job_id, start_time)).start()
            return jsonify({'status': 'processing', 'job_id': existing_job_id})

        timestamp = int(time.time())
        datestamp = datetime.now().strftime('%d.%m.%Y')
        job_id = f"job_jogos_{datestamp}_{timestamp}"
        job_dir = Path(app.config['UPLOAD_FOLDER']) / job_id
        (job_dir / "_1_MOVER_OS_FICHEIROS_DAQUI").mkdir(parents=True, exist_ok=True)
        diarization_dir = job_dir / "_2_PARA_AS_PASTAS_DE_VOZ"
        diarization_dir.mkdir(parents=True, exist_ok=True)

        game_profile = request.form.get('game_profile', 'padrao')
        user_glossary = request.form.get('glossary', '')

        first_file_data = files[0].read()
        files[0].seek(0)
        
        best_profile = find_best_audio_profile(first_file_data, job_dir)

        if not best_profile:
            logging.error("NÃO FOI POSSÍVEL DETECTAR UM FORMATO DE ÁUDIO VÁLIDO.")
            return jsonify({'error': 'Não foi possível detectar um formato de áudio válido.'}), 400

        if 'wav_files' not in request.files:
            return jsonify({'error': 'Nenhum arquivo enviado.'}), 400
            
        files = request.files.getlist('wav_files')
        
        file_format_map = {}
        # [v10.71 CONSOLIDATED] Suporte a ZIP e arquivos soltos
        for file in files:
            if not file.filename: continue
            filename = secure_filename(file.filename)
            extension = Path(filename).suffix.lower()
            
            if extension == '.zip':
                temp_zip_path = job_dir / filename
                file.save(temp_zip_path)
                try:
                    with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
                        for member in zip_ref.namelist():
                            if member.startswith('__MACOSX') or member.startswith('.'): continue
                            m_path = Path(member)
                            if not member.endswith('/'):
                                target_path = (job_dir / "_1_MOVER_OS_FICHEIROS_DAQUI") / m_path.name
                                file_format_map[m_path.stem] = m_path.suffix
                                with zip_ref.open(member) as source, open(target_path, "wb") as target:
                                    shutil.copyfileobj(source, target)
                    os.remove(temp_zip_path)
                    logging.info(f"ZIP extraído com sucesso: {filename}")
                except Exception as e:
                    logging.error(f"Falha ao extrair ZIP: {e}")
            else:
                # Arquivo WAV comum
                file_data = file.read()
                file.seek(0)
                base_filename = Path(filename).stem
                file_format_map[base_filename] = extension
                
                if best_profile:
                    try:
                        base_filename = Path(filename).stem
                        output_path = job_dir / "_1_MOVER_OS_FICHEIROS_DAQUI" / f"{base_filename}.wav"
                        cmd = ['ffmpeg', '-y']
                        profile_params = {k: v for k, v in best_profile.items() if k != 'name'}
                        for key, value in profile_params.items():
                            cmd.extend([f'-{key}', value])
                        cmd.extend(['-i', 'pipe:0', '-c:a', 'pcm_s16le', '-ar', '44100', '-ac', '1', str(output_path)])
                        subprocess.run(cmd, input=file_data, check=True, capture_output=True)
                    except Exception as e:
                        logging.error(f"Falha ao converter {filename}: {e}")
                else:
                    file.save(job_dir / "_1_MOVER_OS_FICHEIROS_DAQUI" / filename)

        source_dir = job_dir / "_1_MOVER_OS_FICHEIROS_DAQUI"
        for audio_file in list(source_dir.glob("*.wav")):
            # [v10.55 INTELLIGENT DIARIZATION SPLIT]
            # O usuário solicitou "escutar todos os áudios" e cortar se a voz mudar.
            # Também mantemos o split por duração se > 25s.
            
            dur = get_audio_duration(audio_file)
            dur_f = round(dur, 2)
            logging.info(f"Auditando '{audio_file.name}' ({dur_f}s) em busca de trocas de orador...")
            
            # 1. Tenta Split por Orador (Diarização Cirúrgica baseada em VAD)
            if dur > 1.8: # Tamanho mínimo para análise estatística
                 if split_audio_by_speaker(audio_file, job_dir):
                      continue # Já foi splitado e movido
            
            # [v10.86 REMOVIDO] Split por Silêncio foi desativado.
            # O corte bruto por silêncio estava cortando o final das palavras e limitando
            # o tamanho dos espaços de dublagem da IA. Agora deixamos o Whisper gerenciar
            # arquivos grandes de forma inteligente e segura (Micro-chunking lógico).

        for i in range(1, int(request.form.get('num_speakers_jogos', 1)) + 1):
            (diarization_dir / f"voz{i}").mkdir(exist_ok=True)

        status_data = {
            'job_id': job_id, 'status': 'iniciando', 'files_hash': files_hash, 
            'file_format_map': file_format_map, 'detected_profile': best_profile,
            'game_profile': game_profile, 'user_glossary': user_glossary
        }
        safe_json_write(status_data, job_dir / "job_status.json")

        threading.Thread(target=processar_dublagem_jogos, args=(job_dir, job_id, start_time)).start()
        return jsonify({'status': 'processing', 'job_id': job_id})
    else:
        return jsonify({'error': 'Requisição inválida.'}), 400


@app.route('/update_text', methods=['POST'])
def update_text():
    data = request.json
    job_id, file_id, new_text = data.get('job_id'), data.get('file_id'), data.get('text')

    if not all([job_id, file_id, new_text is not None]):
        return jsonify({'error': 'Dados incompletos.'}), 400

    job_dir = Path(app.config['UPLOAD_FOLDER']) / job_id
    if not job_dir.is_dir(): return jsonify({'error': 'Job não encontrado.'}), 404

    project_data_path = job_dir / "project_data.json"
    project_data = safe_json_read(project_data_path)
    if project_data is None: return jsonify({'error': 'Falha ao ler dados do projeto.'}), 500

    target_segment = next((seg for seg in project_data if seg.get('id') == file_id), None)
    if not target_segment: return jsonify({'error': f'Arquivo com id {file_id} não encontrado.'}), 404

    target_segment['manual_edit_text'] = new_text
    logging.info(f"Texto atualizado para '{file_id}' no job '{job_id}': '{new_text}'")
    safe_json_write(project_data, project_data_path)
    safe_json_write(target_segment, job_dir / "_backup_texto_final" / f"{file_id}.json")
    
    try:
        dubbed_audio_path = job_dir / "_dubbed_audio" / f"{file_id}_dubbed.wav"
        if dubbed_audio_path.exists():
            os.remove(dubbed_audio_path)
            logging.info(f"Áudio temporário removido para '{file_id}'.")

        status = safe_json_read(job_dir / "job_status.json") or {}
        file_format_map = status.get('file_format_map', {})
        extension = file_format_map.get(file_id, '.wav')
        final_audio_path = job_dir / "_saida_final" / f"{file_id}{extension}"
        if final_audio_path.exists():
            os.remove(final_audio_path)
            logging.info(f"Áudio final removido para '{file_id}'.")
            
    except Exception as e:
        logging.error(f"Erro ao remover áudios antigos para '{file_id}': {e}")

    return jsonify({'status': 'success', 'message': f'Texto para {file_id} atualizado.'})

@app.route('/recent_jobs')
def recent_jobs():
    jobs = []
    # Force absolute path resolution
    upload_path = Path(app.config['UPLOAD_FOLDER']).resolve()
    
    # logging.debug(f"[RECENT_JOBS] Verificando pasta uploads: {upload_path}")
    
    if not upload_path.exists():
        logging.warning(f"[RECENT_JOBS] Pasta de uploads NÃO encontrada: {upload_path}")
        return jsonify([])

    try:
        # List directories sorted by mtime descending
        dirs = sorted([d for d in upload_path.iterdir() if d.is_dir()], 
                      key=lambda x: x.stat().st_mtime, reverse=True)
        
        # logging.info(f"[RECENT_JOBS] Encontrados {len(dirs)} diretórios candidatos.")

        for d in dirs:
            status_file = d / "job_status.json"
            # logging.debug(f"[RECENT_JOBS] Checando: {d.name}")
            
            if status_file.exists():
                try:
                    data = safe_json_read(status_file)
                    if data:
                        # logging.info(f"[RECENT_JOBS] PROJETO VÁLIDO: {d.name} | Status: {data.get('status')}")
                        jobs.append({
                            'id': data.get('job_id', d.name),
                            'status': data.get('status', 'unknown'),
                            'progress': data.get('progress', 0),
                            'etapa': data.get('etapa', ''),
                            'file_count': data.get('file_count', 0),
                            'date': datetime.fromtimestamp(d.stat().st_mtime).strftime('%d/%m %H:%M')
                        })
                    else:
                        # logging.warning(f"[RECENT_JOBS] JSON vazio ou inválido em {d.name}")
                        pass
                except Exception as read_err:
                     logging.error(f"[RECENT_JOBS] Erro lendo JSON {d.name}: {read_err}", exc_info=True)
            else:
                # logging.debug(f"[RECENT_JOBS] Ignorado (sem JSON): {d.name}")
                pass

            if len(jobs) >= 10: break
            
    except Exception as e:
        logging.error(f"[RECENT_JOBS] ERRO CRÍTICO AO LISTAR: {e}", exc_info=True)
        
    # logging.info(f"[RECENT_JOBS] Retornando {len(jobs)} projetos.")
    return jsonify(jobs)

@app.route('/resume_job/<job_id>', methods=['POST'])
def resume_job(job_id):
    """Retoma um trabalho parado ou retorna status se já estiver rodando."""
    logging.info(f"[RESUME] Pedido de retomada recebido para: {job_id}")
    upload_path = Path(app.config['UPLOAD_FOLDER'])
    job_dir = upload_path / job_id
    
    if not job_dir.exists():
        logging.error(f"[RESUME] Pasta não encontrada: {job_dir}")
        return jsonify({'error': 'Job não encontrado'}), 404
    
    with active_jobs_lock:
        if job_id in active_jobs:
            logging.info(f"[RESUME] Job {job_id} já está em active_jobs. Ignorando.")
            return jsonify({'status': 'running', 'message': 'O projeto já está em execução.'})

    # Se não estiver rodando, reinicia a thread
    logging.info(f"[RESUME] Iniciando Thread para: {job_id}")
    
    # [FIX] Força atualização visual imediata
    status_file = job_dir / "job_status.json"
    status_data = safe_json_read(status_file) or {}
    status_data['status'] = 'retomando'
    status_data['etapa'] = 'Reinicializando Processos...'
    safe_json_write(status_data, status_file)
    
    # [FIX] Recupera o tempo já gasto anteriormente no projeto para não zerar o relogio
    tempo_previo = status_data.get('total_elapsed_secs', 0)
    start_time = time.time() - tempo_previo 
    try:
        if "job_dublagem" in job_id or "job_traduz_srt" in job_id or "job_limpeza" in job_id:
            import App_videos
            # Direciona de volta pro motor principal de vídeos se for job de vídeo
            if "job_dublagem" in job_id:
                t = threading.Thread(target=App_videos.pipeline_dublagem, args=(job_dir, job_id, start_time))
            elif "job_limpeza" in job_id:
                 t = threading.Thread(target=App_videos.pipeline_limpar_audio, args=(job_dir, job_id, start_time, status_data.get('level', 'leve')))
            else:
                 t = threading.Thread(target=App_videos.pipeline_traduzir_legendas, args=(job_dir, job_id, start_time, status_data.get('target_lang', 'en')))
        else:
            t = threading.Thread(target=processar_dublagem_jogos, args=(job_dir, job_id, start_time))
            
        t.start()
        logging.info(f"[RESUME] Thread disparada com sucesso: {t.name}")
    except Exception as thread_err:
        logging.critical(f"[RESUME] FALHA AO INICIAR THREAD: {thread_err}", exc_info=True)
        return jsonify({'error': 'Falha interna ao iniciar processo'}), 500

    return jsonify({'status': 'resumed', 'message': 'Processamento retomado com sucesso.'})

@app.route('/progress/<job_id>')
def progress(job_id):
    with progress_lock:
        if job_id in progress_dict: return jsonify(progress_dict[job_id])
        status_path = Path(app.config['UPLOAD_FOLDER']) / job_id / "job_status.json"
        if status_data := safe_json_read(status_path):
             return jsonify({ 'progress': status_data.get('progress', 0), 'etapa': status_data.get('etapa', 'Pronto'),
                              'subetapa': status_data.get('subetapa'), 'tempo_decorrido': status_data.get('tempo_decorrido', '0:00:00') })
    return jsonify({})

@app.errorhandler(Exception)
def handle_exception(e):
    logging.error(f"Erro não tratado na rota: {request.url}\n{traceback.format_exc()}")
    return jsonify({"error": "Ocorreu um erro interno no servidor.", "details": str(e)}), 500

@app.route('/')
def index(): return send_from_directory(app.template_folder, 'Editor de Vídeo Inteligente.html')

@app.route('/favicon.ico')
def favicon(): return make_response('', 204)

@app.route('/uploads/<path:path>')
def send_upload(path): return send_from_directory(app.config['UPLOAD_FOLDER'], path)

# [v2026.9 FIX] Aquecimento de Motor e Patch de Emergência
def prewarm_audio_engines():
    try:
        import sys
        import types
        import logging
        import os
        
        logging.info("🔥 Iniciando Pre-warm e Proteção de Memória...")
        
        # [NEW] Previne o erro 'partially initialized module' garantindo a ordem global
        import speechbrain
        try:
            import speechbrain.utils.quirks
            import speechbrain.utils.importutils
        except: pass

        # [PATCH] Neutraliza o erro do SpeechBrain 1.1.0 (transducer_loss)
        stubs = ['speechbrain.integrations.numba', 'speechbrain.integrations.numba.transducer_loss']
        for stub in stubs:
            if stub not in sys.modules:
                sys.modules[stub] = types.ModuleType(stub)
        
        os.environ["SB_DISABLE_QUIRKS"] = "1"
        import speechbrain.inference
        
        logging.info("✅ Motores blindados e prontos.")
    except Exception as e:
        logging.warning(f"⚠️ Aviso no Pre-warm: {e}")

if __name__ == "__main__":
    prewarm_audio_engines()
    check_ffmpeg()
    check_lm_studio()
    host, port = "0.0.0.0", 5001
    url = f"http://127.0.0.1:{port}"
    import logging
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    def open_browser():
        time.sleep(1.5)
        import webbrowser
        webbrowser.open_new(url)
    import threading
    threading.Thread(target=open_browser).start()
    print("\n" + "="*80)
    print(f"Servidor NEXUS (Dublagem de Jogos) [v0.09] iniciado.")
    print(f"Acesse a aplicação em: {url}")
    print("O progresso detalhado de todos os jobs será exibido aqui no CMD.")
    print("="*80 + "\n")
    app.run(host=host, port=port, debug=False, threaded=True)
