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
import gc # [NEW] Gerenciamento de Memória
import torch # [NEW] Para limpeza agressiva
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
from pydub.silence import split_on_silence

# --- CONFIGURAÇÕES DE AMBIENTE ---
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

# --- DIARIZAÇÃO INTELIGENTE (v10.55) ---
class SimpleDiarizer:
    def __init__(self, source="speechbrain/spkrec-ecapa-voxceleb", device="cpu"):
        try:
            from speechbrain.inference.speaker import EncoderClassifier
            self.encoder = EncoderClassifier.from_hparams(source=source, run_opts={"device": device})
            self.device = device
        except:
            from speechbrain.pretrained import EncoderClassifier
            self.encoder = EncoderClassifier.from_hparams(source=source, run_opts={"device": device})

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
            
            # Se o trecho for muito curto (<0.5s), ignoramos para estabilidade do embedding
            if (end_ms - start_ms) < 500: continue
            
            try:
                # Extrai embedding do bloco de fala inteiro
                chunk = signal[:, s_start:s_end]
                emb = self.encoder.encode_batch(chunk)
                embeddings.append(emb.squeeze().cpu().numpy())
                valid_ranges.append((start_ms, end_ms))
            except: continue
            
        if len(embeddings) < 2: return []
        
        splits_ms = []
        for i in range(len(embeddings) - 1):
            dist = cosine(embeddings[i], embeddings[i+1])
            if dist > 0.5: # Mudança de voz entre blocos detectada
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
    # [TUNING 13/02] User Report: "Extra Voices" (Over-segmentation).
    # Previous: Sim=0.55 / Hyst=0.45 (Too strict, splitting same person).
    # New: Sim=0.42 / Hyst=0.35 (More lenient/sticky).
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
                    # logging.info(f"Hysteresis Active: Kept {self.last_speaker_id} (Score: {last_score:.2f})")
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
                    # logging.info(f"Merging {id_b} into {id_a} (Similarity: {score:.2f})")
                    
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
    def __init__(self, source="speechbrain/spkrec-ecapa-voxceleb", device="cpu"):
        try:
            import torchaudio
            if not hasattr(torchaudio, 'list_audio_backends'):
                def _list_audio_backends(): return ['soundfile']
                torchaudio.list_audio_backends = _list_audio_backends
            
            from speechbrain.inference.speaker import EncoderClassifier
            self.encoder = EncoderClassifier.from_hparams(source=source, run_opts={"device": device})
            self.device = device
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
            
            return embeddings[0, 0].numpy()
            
        except Exception as e:
            logging.error(f"Erro ao gerar embedding: {e}")
            return None

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

# --- VARIÁVEIS GLOBAIS E LOCKS ---
whisper_model = None
chatterbox_model = None
model_lock = Lock()
progress_dict, progress_lock = {}, Lock()
active_jobs_lock = Lock()
active_jobs = set()

# --- INICIALIZAÇÃO DO FLASK ---
app = Flask(__name__, template_folder='client', static_folder='client')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 * 1024 
app.config['MAX_FORM_PARTS'] = 10000
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- FUNÇÕES DE SEGURANÇA ---
def safe_json_write(data, path, indent=4, ensure_ascii=False, retries=5, delay=0.2):
    path = Path(path)
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

def get_whisper_model():
    global whisper_model
    with model_lock:
        if whisper_model is None:
            import torch
            from faster_whisper import WhisperModel
            
            if torch.cuda.is_available():
                logging.info("GPU Detectada! Carregando Whisper ('small') em CUDA (float16)...")
                whisper_model = WhisperModel("small", device="cuda", compute_type="float16")
            else:
                num_cores = os.cpu_count() or 4
                whisper_threads = max(1, num_cores // 2)
                logging.info(f"GPU não encontrada. Carregando Whisper ('small') em CPU (int8) com {whisper_threads} threads...")
                whisper_model = WhisperModel("small", device="cpu", compute_type="int8", cpu_threads=whisper_threads)
                
            logging.info("Modelo faster-whisper carregado.")
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

def get_chatterbox_model():
    """
    Singleton para o modelo Chatterbox (Siri Local), portado do App_videos.
    Otimizado para CPU com PyTorch 2.5+.
    """
    global chatterbox_model
    with model_lock:
        if chatterbox_model is None:
            import torch
            num_cores = os.cpu_count() or 4
            torch.set_num_threads(num_cores)
            logging.info(f"Carregando modelo Chatterbox TTS (usando {num_cores} threads de CPU)...")
            
            try:
                from chatterbox import ChatterboxMultilingualTTS
                device = "cuda" if torch.cuda.is_available() else "cpu"
                
                # [v10.15] RAM/VRAM OPTIMIZATION
                logging.info(f"Carregando Chatterbox Multilíngue em {device}...")
                chatterbox_model = ChatterboxMultilingualTTS.from_pretrained(device=device)
                
                logging.info(f"Modelo Chatterbox carregado com sucesso em '{device}'.")
            except Exception as e:
                logging.critical(f"Falha ao carregar o modelo Chatterbox: {e}\n{traceback.format_exc()}")
                logging.critical("Instrução: 'pip install chatterbox-tts'")
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
    source_lang = status_data.get('source_language', 'auto')

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
                logging.info(f"Mesclando {current_folder.name} -> {target['folder'].name} (Sim: {dist:.2f})")
                
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
            logging.info(f"[SMART MERGE] Fundindo {q['folder'].name} -> {best_match['folder'].name} (Sim: {best_score:.2f})")
            for f in q['folder'].glob("*.wav"):
                try: shutil.move(str(f), str(best_match['folder'] / f.name))
                except: pass
            try: q['folder'].rmdir()
            except: pass
            count_merged += 1
        else:
            logging.info(f"[SMART KEEP] Mantendo {q['folder'].name} (Sim Máx: {best_score:.2f} < 0.60)")
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
            # [HEURISTIC] Se o unificado existe, checa se a pasta tem arquivos novos de merge (mtime).
            folder_mtime = voice_folder.stat().st_mtime
            ref_mtime = output_ref_path.stat().st_mtime
            if ref_mtime < folder_mtime:
                logging.info(f"[REF REFRESH] Pasta '{voice_folder.name}' atualizada (Merge detectado). Regenerando referência...")
                try: output_ref_path.unlink()
                except: pass
            else:
                continue
        if not wav_files: continue
        
        valid_wavs = []
        for w in wav_files:
            if w.name.startswith("_REF_"): continue
            if w.stat().st_size < 32000: continue 

            fid = w.stem
            if fid in project_text_map:
                text = project_text_map[fid]
                if len(text) < 20 and any(bad in text for bad in BAD_REF_WORDS): continue
                if len(text) < 4: continue
            
            valid_wavs.append(w)

        if not valid_wavs: 
            valid_wavs = [w for w in wav_files if not w.name.startswith("_REF_") and w.stat().st_size > 10000]

        valid_wavs.sort(key=lambda x: x.stat().st_size, reverse=True)
        top_files = valid_wavs[:20] 
        
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
                
                combined_audio += seg
                total_dur += len(seg)
                if total_dur > 24000: break # [v10.66] Optimum duration for stable cloning
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

# --- GEMA LOCAL (LLAMA-CPP) SINGLETON ---
gema_instance = None
gema_lock = Lock()

def get_gema_model():
    """
    Singleton robusto para o modelo Gema Local (GGUF). Unificado com App_videos.
    """
    global gema_instance
    with gema_lock:
        if gema_instance is None:
            model_path = r"C:\IA_dublagem\models\gemma-3n-E4B-it-Q4_K_M.gguf"
            if not os.path.exists(model_path):
                logging.error(f"Modelo Gema não encontrado em: {model_path}")
                return None
            
            try:
                logging.info(f"Carregando Gema Local: {model_path}...")
                from llama_cpp import Llama
                gema_instance = Llama(
                    model_path=model_path,
                    n_ctx=2048,
                    n_threads=6,
                    n_gpu_layers=0, # Garante CPU para não competir com Chatterbox (VRAM)
                    verbose=False
                )
                logging.info("Gema Local carregado com sucesso.")
            except Exception as e:
                logging.error(f"Falha ao carregar Gema Local: {e}")
                return None
    return gema_instance

def unload_gema_model():
    """
    Libera memória RAM/VRAM ocupada pelo Gema.
    """
    global gema_instance
    with gema_lock:
        if gema_instance is not None:
            logging.info("Descarregando Gema Local para liberar memória...")
            del gema_instance
            gema_instance = None
            gc.collect()
            import torch
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

def wait_for_gema_service(progress_callback):
    """
    Adaptado: Garante que o modelo local esteja pronto em vez de esperar API externa.
    """
    progress_callback("Carregando Gema Local (Direct GGUF)...")
    model = get_gema_model()
    if model:
        logging.info(">>> Modelo Gema Local pronto no app_jogos. <<<")
    else:
        logging.error(">>> Falha ao carregar Gema Local no app_jogos. <<<")

def make_gema_request_with_retries(payload, timeout=3600, retries=5, backoff_factor=2):
    """
    Refatorado: Utiliza o modelo Local em vez de POST Request ao LM Studio.
    """
    model = get_gema_model()
    if not model:
        raise Exception("Modelo Gema não carregado localmente.")

    messages = payload.get("messages", [])
    temp = payload.get("temperature", 0.3)
    max_tk = payload.get("max_tokens", 2048)

    try:
        # Inferência direta
        response_raw = model.create_chat_completion(
            messages=messages,
            temperature=temp,
            max_tokens=max_tk
        )
        # Mock de objeto de resposta do 'requests'
        class ResponseMock:
            def __init__(self, data): self.data = data
            def json(self): return self.data
            def raise_for_status(self): pass
        
        return ResponseMock(response_raw)
        
    except Exception as e:
        logging.error(f"Erro na inferência do Gema Local (app_jogos): {e}")
        raise e

def gema_batch_processor_v2(segments, context, glossary={}, batch_size=5):
    """
    Processa traduções em lote usando o Gema Local 9B.
    Mantém contexto da cena e força o dicionário de termos do usuário.
    """
    logging.info(f"Iniciando lote Gema V2: {len(segments)} segmentos. (Contexto: '{context}')")
    
    # Monta o prompt de lote (Protegendo contra injeção de prompt)
    prompt = f"""Você é o tradutor.

REGRAS:
Traduza cada bloco separadamente para PT-BR.
Cada bloco é independente.
Não use informação de outros blocos.
Se a frase estiver incompleta, mantenha incompleta.

FORMATO OBRIGATÓRIO DE SAÍDA:
ID => Tradução
ID => Tradução
"""

    if glossary:
        termos = ", ".join([f"'{k}' -> '{v}'" for k, v in glossary.items()])
        prompt += f"\nTERMOS OBRIGATÓRIOS NESTA CENA:\n{termos}\n"
    
    prompt += "\nATENÇÃO: Cada bloco abaixo é de cenas diferentes e não possuem continuidade.\n\n"
    
    # [BATCH RANDOMIZATION] - Desvinculação forçada de contexto cronológico.
    # Evita que LLMs menores (4B) tentem "emendar" frases picotadas da mesma cena.
    import random
    segments_to_prompt = list(segments) # Cria uma cópia
    random.shuffle(segments_to_prompt)  # Embaralha a ordem de leitura da IA
    
    for seg in segments_to_prompt:
        prompt += f"### BLOCO {seg['id']} ###\n{seg['original_text']}\n\n"
        
    payload = {
        "messages": [{"role": "user", "content": prompt}], 
        "temperature": 0.45,  # Criatividade controlada
        "max_tokens": 2000
    }
    
    # Processa via rede
    try:
        response = make_gema_request_with_retries(payload)
        content = response.json()['choices'][0]['message']['content']
        
        # Parse baseado em Regex para slots
        results_dict = {}
        # Captura ID => Tradução multilinha até encontrar o próximo ID ou fim do texto
        matches = re.finditer(r'(sample_[a-zA-Z0-9_]+)\s*=>(.*?)(?=\nsample_|\n###|$)', content, re.DOTALL)
        for match in matches:
            uid = match.group(1).strip()
            val = match.group(2).strip().replace('\n', ' ')
            # Limpeza bruta caso o LLM mande aspas no começo/fim
            if val.startswith('"') and val.endswith('"'): val = val[1:-1]
            results_dict[uid] = val
            
        if not results_dict:
             logging.error(f"Gema V2: A IA não retornou nenhum slot formatado corretamente! Texto lido: {content[:200]}...")
             return {}
             
        return results_dict
        
    except Exception as e:
        import traceback
        logging.error(f"Erro Crítico Batch Gema V2: {e}")
        return {}


def select_best_sync_option(original_duration, options_list, original_text):
    """
    Seleciona a melhor opção de sincronização baseada em critérios matemáticos e linguísticos.
    """
    best_opt = None
    best_score = float('inf')
    target_rate = 16.0 # [UPDATED] Aumentado de 13 para 16 a pedido do usuário (Fala mais rápida)
    
    # Validação básica
    valid_options = [opt.strip() for opt in options_list if opt and len(opt.strip()) > 0]
    if not valid_options: return None

    logging.info(f"Avaliando {len(valid_options)} candidatos para duração {original_duration:.2f}s...")

    for opt in valid_options:
        # Limpeza básica
        clean_opt = re.sub(r'^\d+[\.\-\)]\s*', '', opt).strip('"').strip()
        if not clean_opt: continue

        if not clean_opt: continue
        
        # Limpeza de Vírgulas Duplas e Pontuação excessiva HALLUCINATED
        clean_opt = re.sub(r',+', ',', clean_opt) # ,, -> ,
        clean_opt = re.sub(r'[\.,;]+$', '', clean_opt) # Remove pontuação final redundante na contagem
        
        # [MATH] Custo Real: Letras (Vírgulas têm custo ZERO pois serão removidas)
        # O usuário pediu para o Gema escrever natural (com vírgulas) e o sistema limpar depois.
        commas = 0 
        pauses = 0 
        effective_char_count = len(clean_opt) # Apenas letras contam
        
        cps = effective_char_count / original_duration if original_duration > 0 else 0
        
        # [NOVA LÓGICA V3] - Ajuste para fala rápida (15-20 CPS)
        # Se o CPS for <= 20, consideramos aceitável (já que permitimos aceleração de 1.35x)
        if cps <= 20:
             # Alvo ideal agora padronizado com target_rate (16)
            score = 0.0 + (abs(cps - target_rate) * 0.1) 
        else:
            # Penalidade apenas se passar de 20 CPS
            score = (cps - 20) * 10.0
 

        # 3. Regra de Ouro para Áudios Curtos (< 1.2s)
        if original_duration < 1.2:
            words = clean_opt.split()
            # Penaliza severamente 1 palavra isolada, a menos que o original seja curto também
            if len(words) < 2 and len(original_text.split()) > 1:
                score += 50 
            # Bônus para frases nominais completas (ex: "Perímetro perdido")
            if len(words) >= 2:
                score -= 5

        logging.info(f"   - Candidato: '{clean_opt}' | CPS: {cps:.1f} | Score: {score:.1f}")
        if original_duration < 1.2:
            words = clean_opt.split()
            # Penaliza severamente 1 palavra isolada, a menos que o original seja curto também
            if len(words) < 2 and len(original_text.split()) > 1:
                score += 50 
            # Bônus para frases nominais completas (ex: "Perímetro perdido")
            if len(words) >= 2:
                score -= 5

        logging.info(f"   - Candidato: '{clean_opt}' | CPS: {cps:.1f} | Score: {score:.1f}")

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

def gema_etapa_2_sincronizacao(original_text, translated_text, duration, previous_context=None):
    # Prompt aprimorado para Single Shot + Modos de Edição
    target_chars = int(duration * 16) # [UPDATED] Alvo de 16 chars/s
    duration_seconds = duration
    
    # 1. Definição de MODO e TEMPERATURA (Ajuste Final)
    # Segmentos MUITO curtos (< 1.2s) exigem DETERMINISMO TOTAL
    if duration_seconds <= 1.2:
        mode = "CORTE"
        temperature = 0.0
        instruction_prompt = """
        [MODO: CORTE DRÁSTICO]
        - Tempo CRÍTICO (< 1.2s).
        - TAREFA: Mantenha APENAS o essencial.
        - Prioridade: VERBO > SUJEITO. Jogue fora o resto.
        """

    elif duration_seconds <= 1.8:
        mode = "SIMPLIFICACAO_AGRESSIVA"
        temperature = 0.1
        instruction_prompt = """
        [MODO: SIMPLIFICAÇÃO AGRESSIVA]
        - Tempo: MUITO APERTADO (1.2s - 1.8s).
        - TAREFA: O texto TEM que caber. Priorize velocidade.
        - ESTRATÉGIAS:
          1. **Ataque o Verbo:** "Take this!" -> "Pega!".
          2. **Remova Sujeitos:** "You need this" -> "Precisa disso".
          3. **Junte Frases:** "No! Don't!" -> "Não! Para!".
          4. **Meta:** ~18 a 20 caracteres por segundo (Fala rápida).
        """
    elif duration_seconds <= 3.2:
        mode = "SIMPLIFICACAO_MODERADA"
        temperature = 0.15
        instruction_prompt = """
        [MODO: SIMPLIFICAÇÃO MODERADA]
        - Tempo: RAZOÁVEL (1.8s - 3.2s).
        - TAREFA: Encaixe a frase sem perder a elegância.
        - ESTRATÉGIAS:
          1. **Tente a Frase Completa:** Muitas vezes cabe!
          2. **Corte Leve:** Se sobrar, tire advérbios ("Realmente", "Muito").
          3. **Refraseie:** "It is more blessed to give" -> "É melhor dar do que receber" (Cabe bem).
          4. **EVITE NONSENSE:** Não transforme frases bonitas em índio ("Dá, é feliz").
          5. **Meta:** ~16 caracteres por segundo (Ritmo normal/rápido).
        """
    else:
        mode = "ADAPTACAO"
        temperature = 0.2
        instruction_prompt = """
        [MODO: ADAPTAÇÃO]
        - Tempo: Confortável (> 3.0s).
        - TAREFA: Dublagem natural.
        - REGRAS:
          1. TOM DE DUBLAGEM (BR): Use linguagem falada, coloquial.
             - PROIBIDO: "Local de habitação", "Efetuou", "Aguardar".
             - USE: "Lugar de morar", "Fez", "Esperar".
          2. **Fluidez**: Priorize frases que soem naturais.
        """

    prompt = f"""[ESPECIALISTA EM LIP-SYNC]
    Original: "{original_text}"
    Tradução: "{translated_text}"
    Contexto Anterior: "{previous_context or ''}"
    
    [CRÍTICO - LIMITE EXATO DE CARACTERES]:
    - O limite máximo aproximado para a tradução caber na dublagem é de {target_chars} caracteres.
    - A sua frase atual tem {len(translated_text)} caracteres e precisa ser reescrita para se adequar a este limite matemático, preservando o núcleo semântico.
    
    {instruction_prompt}
    
    [DIRETIVA ÚNICA]:
    Escreva APENAS a melhor adaptação possível. UMA ÚNICA LINHA.
    Sem listas. Sem "Opção 1". Sem explicações.
    
    RESPOSTA DEFINITIVA:
    """

    try:
        payload = {"messages": [{"role": "user", "content": prompt}], "temperature": temperature, "max_tokens": 600}
        response = make_gema_request_with_retries(payload)
        content = response.json()['choices'][0]['message']['content'].strip()
        
        # [FIX] Normaliza quebra de linha
        content = re.sub(r'(\d+[\.\-\)])', r'\n\1', content)
        
        options = []
        for line in content.split('\n'):
             line = line.strip()
             if not line: continue
             
             # Limpeza básica (Números e Letras)
             clean_line = re.sub(r'^[\(\[]?\s*[0-9a-zA-Z]+\s*[\)\]\.]\s*', '', line)
             
             # Remove labels (Opção 1, Natural, etc)
             clean_line = re.sub(r'^(\[.*?\]|\(.*?\)|\b(Conservadora|Natural|Expressiva|Opção \d)\b)\s*[-:]?\s*', '', clean_line, flags=re.IGNORECASE)
             clean_line = re.sub(r'\s*[\(\[].*?[\)\]]$', '', clean_line) # Remove (Texto) no fim
             clean_line = clean_line.replace('**', '').replace('"', '').replace("'", "").strip()
             
             if clean_line:
                 options.append(clean_line)

        # Fallback Parsing
        if not options:
             options = [l for l in content.split('\n') if len(l.strip()) > 3]

        # [CRITICAL FIX] Forçar inclusão da tradução original (LITERAL) como candidata
        # Muitas vezes a tradução direta é a melhor opção se couber no tempo.
        if translated_text and len(translated_text.strip()) > 2:
             t_clean = translated_text.strip()
             if t_clean not in options:
                 options.append(t_clean)

        # [NOVO] Expansão de Candidatos (Sistema assume controle das vírgulas)
        # Gera versões sem vírgula para competir no Leilão do Judge
        expanded_options = list(options)
        for opt in options:
            if ',' in opt:
                no_comma = opt.replace(',', '')
                expanded_options.append(no_comma)
        
        options = expanded_options

        # Seleção Inteligente via Judge
        best_choice = select_best_sync_option(duration, options, original_text)
        
        if best_choice:
            logging.info(f"Melhor sync: '{best_choice}'")
            return apply_string_fallback(best_choice, target_chars or 999), 1
        else:
            logging.warning(f"Sem opção válida. Usando tradução base.")
            return apply_string_fallback(translated_text, target_chars or 999), 2

    except Exception as e:
        logging.error(f"Erro na API Gema (Etapa 2): {e}")
        return translated_text, 2

def sanitize_tts_text(text):
    """
    Remove pontuação e caracteres que o Chatterbox não gosta.
    Mantém vírgulas pois o usuário pediu naturalidade, mas remove excessos.
    """
    if not text: return ""
    
    # 1. Normalização Básica
    text = text.replace("...", ",").replace("..", ",").replace("—", ", ")
    
    # [INJEÇÃO DE ENERGIA] Frases curtas de ação ganham exclamação no Chatterbox (Call of Duty vibes).
    if len(text) > 0 and len(text) < 15 and not text.endswith('!') and not text.endswith('?'):
        text = text + "!"
        
    # 2. Remove caracteres estranhos (exceto pontuação básica)
    text = re.sub(r'[^\w\s,\.\!\?áéíóúâêîôûãõàèìòùçÁÉÍÓÚÂÊÎÔÛÃÕÀÈÌÒÙÇ]', '', text)
    
    # 3. Limpa espaços duplos
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Alias para compatibilidade
def gema_etapa_3_sanitizacao(text):
    return sanitize_tts_text(text)

def gema_etapa_3_adaptacao_tts(synced_text, is_retry=False):
    prompt_normal = f"""Você é um editor de roteiros para um motor de Texto-para-Fala (TTS). Adapte o texto a seguir para uma leitura natural, mas seja económico com as pausas.
**REGRAS CRÍTICAS:**
1.  **ECONOMIZE VÍRGULAS:** Use o mínimo de vírgulas possível, apenas o essencial para a clareza.
2.  **SUBSTITUA PONTOS:** Troque pontos finais (.) por vírgulas (,) apenas se a pausa for necessária.
3.  **MANTENHA '!' E '?':** Exclamações e interrogações devem ser mantidas.
4.  **NÃO TERMINE COM PAUSA:** O texto final nunca deve terminar com ponto ou vírgula.
5.  **FORMATO:** Responda **APENAS** com o texto adaptado.
**Texto Original:** "{synced_text}"
**Texto Adaptado para TTS (com o mínimo de vírgulas):**"""
    prompt_retry = f"""Ajuste a pontuação deste texto para um robô de voz ler. Use poucas vírgulas. Mantenha '!' e '?'.
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
                val_str = boost_file.read_text().strip()
                logging.info(f"Lendo volume_boost.txt para relatório: '{val_str}'")
                if val_str.isdigit():
                    val = int(val_str)
                    if val > 0:
                        f.write(f"--- CONFIGURAÇÃO ESPECIAL ---\n")
                        f.write(f"Volume Boost Manual: ATIVADO (+{val}%)\n")
                        f.write(f"---------------------------\n")
                else:
                    logging.warning(f"Valor inválido em volume_boost.txt: {val_str}")
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

                dubbed_audio_exists = (job_dir / "_audio_dublado_temp" / f"{file_id}_dubbed.wav").exists()
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

            f.write(f"  - Duração Original:   {original_duration:.2f}s\n")
            
            if seg_data.get('processing_status') != 'Copiado Diretamente (Som Não-Verbal)':
                if Chatterbox_duration > 0: f.write(f"  - Duração Chatterbox:       {Chatterbox_duration:.2f}s\n")
                if final_duration > 0: f.write(f"  - Duração Final:      {final_duration:.2f}s\n")
                else: f.write("  - Duração Final:      N/A (Erro ou não processado)\n")

                if speed_factor: f.write(f"  - Ajuste de Velocidade: Sim (fator {speed_factor:.2f}x)\n")
                elif Chatterbox_duration > 0: f.write("  - Ajuste de Velocidade: Não necessário\n")
                
                mastering_info = mastering_cache.get(file_id, {})
                if mastering_info.get('status') == 'mastered':
                    f.write("  - Masterização:       Aplicada (dynaudnorm)\n")
                    original_peak = mastering_info.get('original_peak_dbfs')
                    dubbed_peak = mastering_info.get('dubbed_peak_before_mastering_dbfs')
                    if original_peak is not None: f.write(f"    - Pico Original:    {original_peak:.2f} dBFS\n")
                    if dubbed_peak is not None: f.write(f"    - Pico Dublado:     {dubbed_peak:.2f} dBFS (antes da masterização)\n")
                elif mastering_info.get('status') == 'fallback_copied':
                    f.write("  - Masterização:       Falhou. Cópia do original utilizada.\n")

            else: f.write(f"  - Duração Final:      {original_duration:.2f}s (Cópia do original)\n")
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
        if job_id in active_jobs:
            logging.warning(f"Job de transcrição '{job_id}' já em execução. Ignorando.")
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
        if job_id in active_jobs:
            logging.warning(f"Job de conversão '{job_id}' já em execução. Ignorando.")
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
        if job_id in active_jobs:
            logging.warning(f"Tentativa de iniciar o trabalho '{job_id}' que já está em execução. A nova tentativa foi ignorada.")
            return
        active_jobs.add(job_id)
    
    try:
        set_low_process_priority()
        def cb(p, etapa, s=None): set_progress(job_id, p, etapa, start_time, ETAPAS_JOGOS, s)
        
        for dir_name in ["_1_MOVER_OS_FICHEIROS_DAQUI", "_2_PARA_AS_PASTAS_DE_VOZ", "_backup_transcricao", "_backup_texto_final", "_audio_dublado_temp", "_saida_final"]:
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
                
                with open(boost_file, "w") as f:
                    f.write(f"{initial_boost}\n# AVISO: 1 = +1dB. NAO coloque mais que 25.\n# CUIDADO: Volumes extremos podem DANIFICAR seus alto-falantes.")
            except Exception as e:
                logging.error(f"Erro ao criar volume_boost.txt no start: {e}")
        
        status = safe_json_read(job_dir / "job_status.json") or {}
        file_format_map = status.get('file_format_map', {})
        source_language = status.get('source_language', 'auto')
        diarization_dir = job_dir / "_2_PARA_AS_PASTAS_DE_VOZ"
        project_data_path = job_dir / "project_data.json"

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
                cb(10 + (i / total_to_transcribe) * 85, 2, f"Transcrevendo: {audio_file.name}")
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
                except Exception as e: logging.error(f"FALHA AO TRANSCREVER {audio_file.name}: {e}")
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
                # [FIX UPDATED] Sempre confia no backup, pois é onde o usuário salva edições manuais
                backup_data = safe_json_read(backup_file)
                if backup_data:
                    project_data_map[file_id].update(backup_data)
                    updated_count += 1
                    if backup_data and backup_data.get('sanitized_text'):
                        project_data_map[file_id].update(backup_data)
                        updated_count += 1
        
        if updated_count > 0:
            project_data = list(project_data_map.values())
            project_data.sort(key=lambda x: x.get('id', ''))
            logging.info(f"Dados do projeto atualizados com {updated_count} backups. Salvando progresso.")
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


        if needs_resave:
            logging.info("Salvando correções de dados antigos no project_data.json...")
            safe_json_write(project_data, project_data_path)
        
        files_to_process_gema = []
        files_to_copy_directly = []

        for seg_data in project_data:
            # [FIX] Se já foi marcado como "Não-Verbal", PULA. Não precisa de texto sanitizado.
            if seg_data.get('processing_status') == 'Copiado Diretamente (Som Não-Verbal)':
                continue

            # [NOVO - Filtro de Idioma] Pula tradução do que já está em Português
            if seg_data.get('detected_language') == 'pt':
                if not seg_data.get('sanitized_text'):
                    seg_data['sanitized_text'] = seg_data.get('original_text', '')
                logging.info(f"Segmento {seg_data['id']} preservado (já é Português).")

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
            
            # [PERFORMANCE] Processamento Paralelo (Multithread)
            # [V2 BATCH PROCESSING]
            # Usa o novo processador em lote para acelerar e dar contexto
            
            # Obtém contexto da cena (simplificado para jogos, pega as primeiras frases)
            cenario_ctx = " / ".join([s['original_text'] for s in files_to_process_gema[:3]])
            
            # Divide em lotes de 10
            batch_size = 10
            batches = [files_to_process_gema[i:i + batch_size] for i in range(0, len(files_to_process_gema), batch_size)]
            
            total_batches = len(batches)
            completed_items = 0
            total_items = len(files_to_process_gema)
            
            for b_idx, batch in enumerate(batches):
                cb((b_idx / total_batches) * 100, 3, f"Traduzindo lote {b_idx+1}/{total_batches}...")
                
                # 1. Tradução em Lote
                resultados_lote = gema_batch_processor_v2(batch, cenario_ctx, glossary={})
                
                for file_data in batch:
                    f_id = file_data['id']
                    original_text = file_data.get('original_text', '').strip()
                    if not original_text: continue
                    
                    translated_text = DICIONARIO_TRADUCOES.get(original_text.lower())
                    if not translated_text:
                         translated_text = resultados_lote.get(f_id)
                    
                    file_data['translation_fallback'] = False
                    
                    if not translated_text or "FALHA" in translated_text:
                         logging.warning(f"Atenção: A IA ignorou o ID {f_id} no lote. Forçando tradução individual (Retry)...")
                         
                         # Retry individual no mesmo processador
                         retry_result = gema_batch_processor_v2([file_data], cenario_ctx, glossary={})
                         translated_text = retry_result.get(f_id)
                         
                         if not translated_text or "FALHA" in translated_text:
                             with model_lock:
                                  log_error_to_file(job_dir, f_id, original_text, "Tradução Lote", translated_text or "VAZIO", 2)
                             translated_text = original_text # Fallback Final
                             file_data['translation_fallback'] = True
                             file_data['translated_text'] = translated_text
                             file_data['synced_text'] = translated_text
                             file_data['sanitized_text'] = gema_etapa_3_sanitizacao(translated_text)
                         else:
                             # Sucesso no Retry
                             logging.info(f"Sucesso no Retry individual para o ID {f_id}!")
                             file_data['translation_fallback'] = False
                             file_data['translated_text'] = translated_text
                             
                             original_dur = file_data.get('duration', 0)
                             clean_trans = re.sub(r'^\d+[\.\-\)]\s*', '', translated_text).strip()
                             char_count = len(clean_trans)
                             cps = char_count / original_dur if original_dur > 0 else 99
                             
                             if cps <= 18.0:
                                 file_data['synced_text'] = clean_trans
                             else:
                                 synced_text, _ = gema_etapa_2_sincronizacao(original_text, translated_text, original_dur)
                                 file_data['synced_text'] = synced_text if "FALHA" not in synced_text else translated_text
                             
                             file_data['sanitized_text'] = gema_etapa_3_sanitizacao(file_data.get('synced_text', translated_text))
                    else:
                         file_data['translated_text'] = translated_text
                         file_data['translation_fallback'] = False
                         
                         original_dur = file_data.get('duration', 0)
                         clean_trans = re.sub(r'^\d+[\.\-\)]\s*', '', translated_text).strip()
                         char_count = len(clean_trans)
                         cps = char_count / original_dur if original_dur > 0 else 99
                         
                         if cps <= 18.0:
                             logging.info(f"[SYNC SKIP] Trad {f_id} cabe no tempo (CPS {cps:.1f}).")
                             file_data['synced_text'] = clean_trans
                         else:
                             # Sincronização individual (pois depende da duração exata)
                             synced_text, _ = gema_etapa_2_sincronizacao(original_text, translated_text, original_dur)
                             if "FALHA" not in synced_text:
                                  file_data['synced_text'] = synced_text
                             else:
                                  file_data['synced_text'] = translated_text # Fallback
                         
                         file_data['sanitized_text'] = gema_etapa_3_sanitizacao(file_data.get('synced_text', translated_text))
                    
                    file_data['manual_edit_text'] = "" 
                    safe_json_write(file_data, job_dir / "_backup_texto_final" / f"{f_id}.json")
                    completed_items += 1
                    cb((completed_items / total_items) * 100, 3, f"Processado: {completed_items}/{total_items}")

            # Salva o progresso final do lote
            safe_json_write(project_data, project_data_path)
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
                            # Preserva campos vitais, atualiza textos
                            seg['sanitized_text'] = fresh_data.get('sanitized_text', seg.get('sanitized_text', ''))
                            seg['manual_edit_text'] = fresh_data.get('manual_edit_text', seg.get('manual_edit_text', ''))
                            
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


        # --- ETAPA 6: GERANDO ÁUDIOS (Chatterbox) ---
        dubbed_audio_dir = job_dir / "_audio_dublado_temp"
        
        # Otimização: Filtrar o que realmente precisa ser gerado ANTES de carregar o modelo
        actual_generation_queue = []
        if generation_queue:
            for seg_data in generation_queue:
                output_path = dubbed_audio_dir / f"{seg_data['id']}_dubbed.wav"
                if not output_path.exists():
                    actual_generation_queue.append(seg_data)
                else:
                    logging.info(f"Áudio para '{seg_data['id']}' já existe. Será pulado.")

        if actual_generation_queue:
            cb(0, 6, "A carregar modelo Chatterbox...")
            tts_model = get_chatterbox_model()
            
            if tts_model is None:
                raise RuntimeError("Falha Crítica: Memória insuficiente ou erro ao carregar o motor TTS Chatterbox (NoneType).")
                
            cb(5, 6, f"Gerando {len(actual_generation_queue)} áudios únicos...")
            
            # Referência de Voz: Carrega mapa de referências
            global_fallback = Path("resources/base_speakers/pt/default_pt_speaker.wav")
            if (job_dir / "voices" / "vocals_speaker_default.wav").exists():
                global_fallback = job_dir / "voices" / "vocals_speaker_default.wav"
            elif (job_dir / "vocals.wav").exists():
                global_fallback = job_dir / "vocals.wav"
            
            for i, seg_data in enumerate(actual_generation_queue):
                output_path = dubbed_audio_dir / f"{seg_data['id']}_dubbed.wav"
                
                cb(5 + (i / len(actual_generation_queue)) * 95, 6, f"Gerando áudio para {seg_data['id']}")
                ref_path = diarization_dir / seg_data['speaker'] / "_REF_VOZ_UNIFICADA.wav"

                if not ref_path.exists():
                     # Tenta fallback global
                     ref_path = global_fallback
                     
                if not ref_path.exists():
                    logging.error(f"Nenhum áudio de referência VÁLIDO encontrado para {seg_data['id']} em '{ref_path}'. Pulando.")
                    continue
                # [SKIP TTS]
                # Verifica se o texto é apenas um NOME ou SOM NÃO-VERBAL que deve ser mantido original.
                # [UPDATED] Lista unificada e completa (referenciada acima mentalmente)
                NOMES_A_IGNORAR = [
                    "jolly!", "jolly", "ahh!", "ahh", "ah!", "ah", "oh!", "oh", "uh!", "uh", "hmm!", "hmm", "wow!", "wow",
                    "argh", "tsk", "ugh", "screams", "gasps", "moans", "chokes", "grita", "geme", "laughs", "chuckles", 
                    "sobs", "cries", "sighs", "eh", "heh", "hum", "ha", "haha", "hah", "whoa", "ooh", "aw", "ouch", "ow",
                    "psst", "shh", "yikes", "yay", "ew", "ick", "boo", "hiss", "growl", "snarl", "roar", "bark", "meow",
                    "purr"
                ]
                
                text_clean_check = seg_data.get('original_text', '').strip().lower()
                
                # Check 1: Está na lista explícita OU contém palavras de ruído (Lista Expandida)
                # Expandimos a verificação para incluir a lista BAD_REF_WORDS conceitual
                # [FIX] Removido 'run', 'walk', 'step' para evitar falsos positivos em diálogos.
                BAD_REF_WORDS_TTS = ['argh', 'ah', 'oh', 'uh', 'hmm', 'wow', 'tsk', 'ugh', 'screams', 'gasps', 'moans', 'chokes', 'grita', 'geme', 'laughs', 'chuckles', 'sobs', 'cries', 'sighs', 'eh', 'heh', 'hum', 'ha', 'haha', 'hah', 'whoa', 'ooh', 'aw', 'ouch', 'ow', 'psst', 'shh', 'yikes', 'yay', 'ew', 'ick', 'boo', 'hiss', 'growl', 'snarl', 'roar', 'bark']
                
                is_exception = False
                if any(exc in text_clean_check for exc in NOMES_A_IGNORAR if len(text_clean_check) < len(exc) + 5):
                     is_exception = True
                elif len(text_clean_check) < 25 and any(bad in text_clean_check for bad in BAD_REF_WORDS_TTS):
                     is_exception = True
                
                # Check 2: É alucinação do Whisper? (Ex: ᗴᗴᗴᗸᗼᗶ)
                # Se tiver muitos caracteres estranhos ou não-latinos repetidos.
                is_hallucination = False
                if len(text_clean_check) > 0:
                    # Conta caracteres alfabéticos normais
                    normal_chars = len(re.findall(r'[a-zA-Z\s]', text_clean_check))
                    # Se menos de 50% for normal, provavelmente é lixo do Whisper (gemidos/gritos mal interpretados)
                    if (normal_chars / len(text_clean_check)) < 0.5:
                        is_hallucination = True

                # [v10.44] BYPASS TTS (PRESERVAÇÃO DO PORTUGUÊS)
                is_preserved_pt = (
                    seg_data.get('detected_language') == 'pt' or 
                    (seg_data.get('sanitized_text') == seg_data.get('original_text') and not seg_data.get('manual_edit_text'))
                )

                if is_exception or is_hallucination or is_preserved_pt:
                    if is_preserved_pt:
                        reason = "já em PT"
                    else:
                        reason = "Nome/Interjeição/Ruído" if is_exception else "Alucinação/Som"
                    
                    logging.info(f"Pulando TTS para '{seg_data['id']}' ({reason}: '{text_clean_check}'). Copiando original.")
                    
                    try:
                        from pydub import AudioSegment
                        original_speaker_dir = diarization_dir / seg_data.get('speaker', 'Unknown')
                        original_audio_path = original_speaker_dir / seg_data.get('file_name', f"{seg_data['id']}.wav")
                        
                        if original_audio_path.exists():
                            # Usa a voz original do jogo se for uma frase idêntica (nomes) ou ruído.
                            # OBRIGATÓRIO: Resample para 24kHz para não quebrar a velocidade do concat.
                            preserved_audio = AudioSegment.from_file(str(original_audio_path))
                            preserved_audio = preserved_audio.set_frame_rate(24000).set_channels(1)
                            preserved_audio.export(output_path, format="wav")
                            logging.info(f"Sucesso: Áudio original copiado para '{seg_data['id']}' (24kHz).")
                        else:
                            # Fallback extremo só se o arquivo não existir
                            dur_ms = int(seg_data['duration']*1000)
                            AudioSegment.silent(duration=dur_ms).export(output_path, format="wav")
                    except Exception as copy_e:
                        logging.error(f"Erro ao copiar original para {seg_data['id']}: {copy_e}")
                    continue

                try: 
                    language = 'pt'
                    text_to_speak = seg_data.get('manual_edit_text', '').strip() or seg_data.get('sanitized_text', '')
                    
                    # [v10.88] ANTI-CORTE (TRUQUE DA VÍRGULA)
                    # O usuário notou que o Chatterbox "grita" ou corta o fôlego abruptamente quando lê 
                    # um ponto final isolado. Trocamos tudo por vírgula para manter a fluidez natural,
                    # mantendo reticências (...) intactas caso o usuário tenha digitado.
                    text_to_speak = re.sub(r'(?<!\.)\.(?!\.)', ',', text_to_speak)
                    
                    # [DEBUG] Confirma o texto exato que vai para o TTS
                    logging.info(f"Gerando Chatterbox para {seg_data['id']} (Text: '{text_to_speak}')")
                    
                    try:
                        wav_tensor = tts_model.generate(
                            text=text_to_speak,
                            language_id="pt",
                            audio_prompt_path=str(ref_path),
                            exaggeration=0.5,    # [v10.82] Emoção original restaurada a pedido do usuário
                            temperature=0.55,    # [v10.82] Menos "Criatividade" (Era 0.72, Default 0.8). Evita artefatos/alucinação.
                            min_p=0.15,          # Poda de artefatos ainda maior
                            repetition_penalty=2.0 # Força o modelo a não gaguejar ou distorcer no fim da frase
                        )
                        
                        import soundfile as sf
                        wav_data = wav_tensor.squeeze(0).cpu().numpy()
                        sf.write(str(output_path), wav_data, 24000)
                        
                        # [v10.51 SURGICAL SYNC v2.0]
                        # Trim silence/hallucinated humming ("eeee"/"uuu") from TTS start and end
                        # Agressividade aumentada para -42dB para ignorar chiado Chatterbox
                        try:
                            from pydub import AudioSegment
                            from pydub.silence import detect_nonsilent
                            clip_raw = AudioSegment.from_wav(str(output_path))
                            
                            # VAD Inteligente: Ignora ruídos de até -42dB (padrão era -50)
                            nonsilent_ranges = detect_nonsilent(clip_raw, min_silence_len=150, silence_thresh=-42)
                            
                            if nonsilent_ranges:
                                start_trim = max(0, nonsilent_ranges[0][0] - 15) # Margem pequena 15ms
                                end_trim = nonsilent_ranges[-1][1]
                                final_end_trim = min(len(clip_raw), end_trim + 30)
                                clip_trimmed = clip_raw[start_trim:final_end_trim]
                                
                                # [SAFETY VALVE] Trava de Duração v2
                                # Se mesmo após o trim o áudio for +50% maior que o original, 
                                # cortamos matematicamente com uma margem de segurança de 200ms
                                original_dur_ms = int(seg_data.get('duration', 0) * 1000)
                                if original_dur_ms > 0 and len(clip_trimmed) > (original_dur_ms * 1.5):
                                    logging.warning(f"[SAFETY VALVE] {seg_data['id']}: Áudio excessivo ({len(clip_trimmed)}ms vs {original_dur_ms}ms). Forçando corte de segurança.")
                                    clip_trimmed = clip_trimmed[:int(original_dur_ms * 1.4) + 100]
                                
                                clip_trimmed.export(str(output_path), format="wav")
                                saved_ms = len(clip_raw) - len(clip_trimmed)
                                if saved_ms > 100:
                                    logging.info(f"[SURGICAL SYNC v2] {seg_data['id']}: Higienizado (limiar -42dB). Removidos {saved_ms}ms.")
                        except Exception as e_sync:
                            logging.warning(f"Aviso no Surgical Sync do segmento {seg_data['id']}: {e_sync}")

                        logging.info(f"Chatterbox: Áudio salvo e higienizado: {output_path.name}")
                    except Exception as e_chat:
                         logging.error(f"ERRO CRÍTICO NO CHATTERBOX ({seg_data['id']}): {e_chat}.")
                         # Fallback
                         from pydub import AudioSegment
                         dur_ms = int(seg_data['duration']*1000)
                         AudioSegment.silent(duration=dur_ms).export(output_path, format="wav")
                except Exception as e:
                    logging.error(f"Erro no pipeline de áudio para {seg_data['id']}: {e}\n{traceback.format_exc()}")
        else:
             cb(100, 6, "Todos os áudios já existem. Pulando geração Chatterbox.")
             logging.info("Todos os arquivos de áudio já existem. Carregamento do modelo Chatterbox pulado.")

        cb(100, 6, "Geração de áudios concluída (ou verificada).")

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
                        logging.info(f"Audio Compression Ativado: Compressor + {val_int}dB de Ganho.")
                    else:
                        volume_boost_factor = 0
                        logging.info("Audio Compression: Desativado (0dB).")
        except Exception as e:
            logging.error(f"Erro ao ler volume_boost.txt: {e}")

        # [FEATURE] Perfil de Jogo - Lógica de MASTERIZAÇÃO Profissional v10.74
        game_profile = status.get('game_profile', 'padrao')
        profile_filters = []
        
        if game_profile == 'bioshock':
            # FMOD Optimization: Loudnorm + high fidelity
            if volume_boost_factor <= 1.0: 
                volume_boost_factor = 12.0
                logging.info("[PROFILE] BioShock: Aplicando ganho automático de +12dB.")
            profile_filters.append("loudnorm=I=-14:TP=-1.0:LRA=11")
            logging.info("[PROFILE] BioShock: Ativando Normalização Profissional (Standard LUFS -14).")
            
        elif game_profile == 'cod':
            # Wall of Sound: Heavy Compression + EQ (MW3 Style)
            if volume_boost_factor <= 1.0: 
                volume_boost_factor = 10.0 # MW3 é muito alto, subimos o ganho base
                logging.info("[PROFILE] Call of Duty (MW3): Aplicando ganho automático de +10dB.")
            
            # 1. Normalização agressiva para -12 LUFS (Máxima presença)
            profile_filters.append("loudnorm=I=-12:TP=-1.0:LRA=11")
            # 2. Compressor de rádio/cinema (puxa os detalhes sem estourar)
            profile_filters.append("acompressor=threshold=-18dB:ratio=4:attack=5:release=50:makeup=2")
            # 3. Exciter/EQ: Bass 100Hz (Peso) + Treble 3.5kHz (Clareza da voz sob tiros)
            profile_filters.append("bass=g=3:f=100[bassout];[bassout]treble=g=2:f=3500")
            logging.info("[PROFILE] Call of Duty: Ativando Otimização MW3 (Wall of Sound + EQ).")
            
        elif game_profile == 'rpg':
            # Natural Dynamics: Dynamic Range preservation
            if volume_boost_factor <= 1.0: 
                volume_boost_factor = 4.0
                logging.info("[PROFILE] RPG: Aplicando ganho automático de +4dB.")
            profile_filters.append("loudnorm=I=-20:TP=-1.5:LRA=7") # Mais dinâmico, alvo -20 LUFS
            logging.info("[PROFILE] RPG: Ativando Dinâmica Natural (Diálogos Limpos).")

        for i, seg_data in enumerate(project_data):
            file_id = seg_data['id']
            final_path = final_output_dir / f"{file_id}{file_format_map.get(file_id, '.wav')}"
            
            # --- VERIFICAÇÃO INTELIGENTE DE CACHE ---
            # Se o arquivo existe, verifica se já temos os metadados dele.
            # Se não tivermos (cache vazio ou incompleto), forçamos a "revisão" (não reprocessa o áudio, só le os dados).
            file_exists = final_path.exists()
            duration_cached = (file_id in durations_cache and durations_cache[file_id].get('duration', 0) > 0)
            mastering_cached = (file_id in mastering_cache and mastering_cache[file_id].get('status') is not None)
            
            if file_exists and duration_cached and mastering_cached:
                logging.info(f"Arquivo final e metadados já existem para '{file_id}'. Pulando.")
                continue
            
            # [FIX] Fallback seguro para file_name e speaker para evitar crash final
            file_name = seg_data.get('file_name', f"{seg_data.get('id', 'unknown')}.wav")
            speaker_id = seg_data.get('speaker', 'Unknown')
            
            cb((i / len(project_data)) * 100, 7, f"Finalizando: {file_name}")

            original_file_path = diarization_dir / speaker_id / file_name

            # Define o áudio de origem (ou é o próprio, ou é um reutilizado)
            source_path = None
            is_fallback_copy = False

            if seg_data.get('reuse_audio_from_id'):
                master_id = seg_data['reuse_audio_from_id']
                source_path = dubbed_audio_dir / f"{master_id}_dubbed.wav"
                logging.info(f"Finalizando '{file_id}' usando o áudio de '{master_id}'.")
            elif seg_data.get('is_master_audio', False):
                source_path = dubbed_audio_dir / f"{file_id}_dubbed.wav"
            else: # Copia direto
                source_path = original_file_path
                is_fallback_copy = True

            # Se o arquivo final NÃO existe, precisamos criar
            if not file_exists:
                if not source_path or not source_path.exists():
                    logging.error(f"Arquivo de origem não encontrado para '{file_id}'. Copiando original.")
                    if original_file_path.exists(): 
                        shutil.copy(original_file_path, final_path)
                        is_fallback_copy = True
                    else: continue

                try:
                    original_duration = seg_data.get('duration', 0)
                    # Medimos a duração do source para calcular speed_factor SE for dublagem
                    source_duration = get_audio_duration(str(source_path)) if not is_fallback_copy else original_duration
                    
                    af_filters = ["dynaudnorm"]
                    atempo_filters = []
                    speed_factor = None
                    TOLERANCE_SECONDS = 0.5

                    if not is_fallback_copy and original_duration > 0 and source_duration > (original_duration + TOLERANCE_SECONDS):
                        calculated_factor = source_duration / original_duration
                        # [SPEED UPDATE v10.51] Permitir aceleração de até 1.30x (Solicitação Usuário BioShock)
                        speed_factor = min(calculated_factor, 1.30)
                        logging.info(f"Áudio '{file_id}' ({source_duration:.2f}s) é mais longo que o original ({original_duration:.2f}s). Acelerando por {speed_factor:.2f}x.")
                        
                        temp_factor = speed_factor
                        while temp_factor > 2.0:
                            atempo_filters.append("atempo=2.0")
                            temp_factor /= 2.0
                        if temp_factor > 1.0: atempo_filters.append(f"atempo={temp_factor:.4f}")
                    
                    if atempo_filters: af_filters = atempo_filters + af_filters
                    
                    filters_to_apply = list(af_filters) # Cópia para não modificar original

                    # [MODIFIED] Se for 'Copiado Diretamente' (Som Não-Verbal), NÃO aplica filtros.
                    if seg_data.get('processing_status') == 'Copiado Diretamente (Som Não-Verbal)':
                        filters_to_apply = [] # Sem filtros, copia limpa
                        logging.info(f"Arquivo '{file_id}' é som não-verbal. Copiando sem filtros de áudio.")
                    else:
                        # [v10.74] Injeção de Filtros de Perfil de Jogo
                        if profile_filters:
                            # Filtros de Perfil antes da compressão/volume manual
                            filters_to_apply.extend(profile_filters)
                    
                    cmd = ['ffmpeg', '-y', '-threads', str(os.cpu_count() or 4), '-i', str(source_path)]
                    
                    # [FEATURE] OpenUnmix: Prepara Mixagem do Fundo
                    bg_file_path = None
                    try:
                        use_openunmix = str(status.get('preserve_background', 'false')).lower() == 'true'
                        if use_openunmix: # Só tenta se o usuário pediu
                            stem_bg_dir = job_dir / "_0b_SEPARACAO_FUNDO"
                            if stem_bg_dir.exists():
                               # Tenta encontrar o arquivo de fundo correspondente (busca recursiva)
                               potential_bg = list(stem_bg_dir.rglob(seg_data['file_name']))
                               if potential_bg:
                                   bg_file_path = potential_bg[0]
                                   logging.info(f"Fundo encontrado para mixagem: {bg_file_path.name}")
                    except Exception as e_bg:
                        logging.warning(f"Erro ao buscar fundo para {file_id}: {e_bg}")
                    
                    if bg_file_path:
                        cmd.extend(['-i', str(bg_file_path)])
                        
                        # [MIXING] Aplica filtros na VOZ e mistura com FUNDO
                        filter_complex = []
                        
                        # 1. Filtros da Voz (dynaudnorm, speed, etc)
                        if filters_to_apply:
                            # Aplica na entrada 0 (voz)
                            filter_complex.append(f"[0:a]{','.join(filters_to_apply)}[voice_processed]")
                            voice_input = "[voice_processed]"
                        else:
                            voice_input = "[0:a]"
                            
                        # 2. Preparação do Fundo (Volume Ducking)
                        # Reduz volume do fundo para 40% para a voz sobressair (Pedido do Usuário)
                        filter_complex.append(f"[1:a]volume=0.4[bg_low]")
                        bg_input = "[bg_low]"

                        # 3. Mistura (amix)
                        filter_complex.append(f"{voice_input}{bg_input}amix=inputs=2:duration=longest:dropout_transition=2[out]")
                        
                        cmd.extend(['-filter_complex', ";".join(filter_complex), '-map', '[out]'])
                         # [v10.67 SPECTRAL BALANCE] 
                        # Se o orador original era "Rádio/Sujeito", aplicamos o "Meio Termo"
                        # para colar a voz dublada na ambiência do jogo sem alucinar no TTS.
                        speaker_id = seg_data.get('speaker', 'voz_unknown')
                        profile_path = job_dir / "_2_PARA_AS_PASTAS_DE_VOZ" / speaker_id / "acoustic_profile.json"
                        if profile_path.exists():
                            profile = safe_json_read(profile_path)
                            if profile.get('is_noisy'):
                                # Filtro que "suja" levemente a voz limpa para o modo rádio
                                # Lowpass 4k + EQ Boost 3k (Dá aquele som 'nasal' de radinho)
                                filters_to_apply.append("lowpass=f=4000,equalizer=f=3000:t=h:w=1000:g=2")
                        
                        cmd.extend(['-af', ",".join(filters_to_apply)])

                    # [FEATURE] Manual Volume Boost - Aplicação (COD STYLE / AGGRESSIVE)
                    # 1. Compressor: Achata a dinâmica (Sussurro = Grito).
                    # 2. Volume: Aplica ganho bruto em dB (1% = +1dB).
                    # 3. Limiter: REMOVIDO a pedido (Pode distorcer/ estourar).
                    
                    if volume_boost_factor > 0:
                        # Compressor Agressivo para nivelar tudo ("Wall of Sound")
                        # ratio=4 (forte), attack=5ms (pega picos rápidos), makeup=2 (+2dB nativo)
                        compressor_filter = "acompressor=threshold=-12dB:ratio=4:attack=5:release=50:makeup=2"
                        
                        # Ganho do Usuário (Raw dB)
                        # Ex: 10% -> volume=10dB
                        boost_filter = f"volume={volume_boost_factor}dB"
                        
                        # [TRUE PEAK LIMITER] Proteção contra Ducking da Engine (FMOD/Wwise)
                        # Se bater no 0dB, a engine abaixa o volume.
                        # Alvo: -1.0 dB (limit=0.89)
                        limiter_filter = "alimiter=limit=0.89:level=disabled:attack=5:release=50"
                        
                        full_filter_chain = f"{compressor_filter},{boost_filter},{limiter_filter}"
                        
                        if '-filter_complex' in cmd:
                            idx = cmd.index('-filter_complex')
                            # Injeta na cadeia complexa
                            cmd[idx+1] = cmd[idx+1].replace("[out]", f"[pre_boost];[pre_boost]{full_filter_chain}[out]")
                        elif '-af' in cmd:
                            idx = cmd.index('-af')
                            cmd[idx+1] += f",{full_filter_chain}"
                        else:
                            cmd.extend(['-af', full_filter_chain])
                    
                    # Codificação de Saída Inteligente
                    output_profile = status.get('detected_profile', {})
                    if output_profile.get('f') == 'mp3':
                        cmd.extend(['-c:a', 'libmp3lame', '-b:a', '128k', '-ar', '44100', '-ac', '1'])
                    elif output_profile.get('c:a') == 'adpcm_ms':
                        cmd.extend(['-c:a', 'adpcm_ms', '-ar', output_profile.get('ar', '22050'), '-ac', output_profile.get('ac', '1')])
                    else:
                        # [v10.86] FORÇA 44.1kHz UNIFORME PARA TODOS OS ÁUDIOS (WAV PADRÃO)
                        # Isso previne que as peças _parte_001 e _parte_002 tenham Sample Rates diferentes (ex: 24hKz da IA vs 48kHz do original)
                        cmd.extend(['-ar', '44100', '-ac', '1'])
                    
                    cmd.append(str(final_path))
                    subprocess.run(cmd, check=True, capture_output=True, text=True)
                    
                    # Salva info de speed factor
                    if file_id not in durations_cache: durations_cache[file_id] = {}
                    if speed_factor: durations_cache[file_id]['speed_factor'] = speed_factor

                except Exception as e:
                    logging.error(f"Erro ao finalizar/masterizar {file_id}: {e}. Copiando original como fallback.")
                    if final_path.exists(): os.remove(final_path)
                    if original_file_path.exists(): shutil.copy(original_file_path, final_path)
                    is_fallback_copy = True

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
                    import time
                    
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
        if job_id in active_jobs:
            logging.warning(f"Job de remoção de rádio '{job_id}' já em execução. Ignorando.")
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
            # Threshold agressivo de -42dB
            nonsilent_ranges = detect_nonsilent(clip_raw, min_silence_len=150, silence_thresh=-42)
            
            if nonsilent_ranges:
                start_trim = max(0, nonsilent_ranges[0][0] - 15)
                end_trim = nonsilent_ranges[-1][1]
                final_end_trim = min(len(clip_raw), end_trim + 30)
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
            logging.info(f"Auditando '{audio_file.name}' ({dur:.2f}s) em busca de trocas de orador...")
            
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
            'game_profile': game_profile
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
        dubbed_audio_path = job_dir / "_audio_dublado_temp" / f"{file_id}_dubbed.wav"
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

if __name__ == "__main__":
    check_ffmpeg()
    host, port = "127.0.0.1", 5001
    url = f"http://{host}:{port}"
    
    def open_browser():
        webbrowser.open_new(url)

    # Abre o navegador após um pequeno atraso para dar tempo ao servidor de iniciar
    Timer(1, open_browser).start()
@app.route('/test_local_gema', methods=['GET'])
def test_local_gema():
    """Rota de diagnóstico para validar o Gema Local sem LM Studio."""
    try:
        model = get_gema_model()
        if not model:
            return jsonify({'status': 'error', 'message': 'Modelo não encontrado ou falha no carregamento.'}), 500
        
        response = model.create_chat_completion(
            messages=[{"role": "user", "content": "Olá, responda apenas: 'Gema Local nos Jogos Ativo!'"}],
            max_tokens=15
        )
        content = response['choices'][0]['message']['content'].strip()
        return jsonify({
            'status': 'success',
            'model_response': content,
            'info': 'Gema Local (Jogos) funcionando corretamente.'
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    logging.info("=============================================================================")
    logging.info("Servidor Nexus (Dublagem de Jogos) iniciado.")
    logging.info(f"Acesse a aplicação em: http://127.0.0.1:5001")
    logging.info("O progresso detalhado de todos os jobs será exibido aqui no CMD.")
    logging.info("================================================================================")
    
    # [FIX] Força host 0.0.0.0 para acesso via rede local/Termux se necessário
    app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)
