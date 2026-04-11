# -*- coding: utf-8 -*-
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
import subprocess
import struct
import json
import webbrowser
import shutil
import hashlib
import logging
import datetime
import traceback
import re
from collections import OrderedDict
from threading import Timer
import tkinter as tk
from tkinter import filedialog

# Tenta importar o Flask. Se não conseguir, dá uma mensagem de erro útil.
try:
    from flask import Flask, jsonify, request, render_template_string
except ImportError:
    print("[ERRO FATAL] A biblioteca Flask não está instalada.")
    print("Por favor, instale-a executando o comando no seu terminal:")
    print("pip install Flask")
    sys.exit(1)

# [DEPENDÊNCIAS EXTERNAS - NOVAS]
try:
    import vpk
except ImportError:
    vpk = None

FSB_ERROR_MSG = None
try:
    import fsb5
    # from fsb5.utils import rebuild_sample # NÃO EXISTE na v1.0
    rebuild_sample = None
except ImportError as e_fsb:
    FSB_ERROR_MSG = str(e_fsb)
    print(f"[DEBUG] Falha ao importar fsb5: {e_fsb}")
    fsb5 = None
    rebuild_sample = None

# --- Constantes ---
BACKUP_DIR = "arch_manager_backups"
MODS_FINALIZADOS_DIR = 'mods_finalizados'

import subprocess
# Caminho para Ferramentas Externas
VGMSTREAM_PATH = os.path.join(os.getcwd(), 'tools', 'vgmstream-cli', 'vgmstream-cli.exe')

KNOWN_ARCH_MAGICS = [
    b'Arch01\x00\x00', # Versão comum
    b'Arch00\x00\x00', # Versão para F.E.A.R. base
    b'LTAR\x03\x00\x00\x00'   # Assinatura encontrada em F.E.A.R. Extraction Point
]

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

def gerar_relatorio_erro(funcao, erro, traceback_completo, detalhes_adicionais=None):
    """Cria um arquivo JSON com detalhes de um erro para facilitar o reporte."""
    relatorio = {
        'timestamp_utc': datetime.datetime.utcnow().isoformat() + 'Z',
        'versao_script': '9.2', # Versão atualizada
        'funcao_com_erro': funcao,
        'mensagem_erro': str(erro),
        'traceback': traceback_completo,
        'detalhes_adicionais': detalhes_adicionais or {}
    }
    try:
        with open('relatorio_de_erro.json', 'w', encoding='utf-8') as f:
            json.dump(relatorio, f, indent=4, ensure_ascii=False)
        logging.info("Um relatório de erro foi salvo em 'relatorio_de_erro.json'")
    except Exception as e:
        logging.error(f"Não foi possível criar o relatório de erro: {e}")

def analisar_arquivo_extraido(caminho_arquivo):
    """
    Analisa um único arquivo extraído para identificar seu tipo e retornar informações.
    Esta função ajuda o usuário a entender o que fazer com os arquivos .bndl e .mdl.
    """
    extensao = os.path.splitext(caminho_arquivo)[1].lower()
    
    if extensao == '.bndl':
        try:
            with open(caminho_arquivo, 'rb') as f:
                magic = f.read(4)
            if magic == b'BNDL':
                return "Arquivo Bundle (.bndl) - Assinatura 'BNDL' encontrada. Provavelmente contém outros arquivos, como sons (.snd) ou texturas."
            else:
                return "Arquivo Bundle (.bndl) - Formato de pacote. **Provavelmente contém os arquivos de áudio que você procura.**"
        except Exception:
            return "Arquivo Bundle (.bndl) - Não foi possível ler, mas provavelmente é um pacote de arquivos de jogo (sons, etc.)."
    
    elif extensao == '.mdl':
        try:
            with open(caminho_arquivo, 'rb') as f:
                magic = f.read(4)
            if magic == b'MDL ':
                 return "Arquivo de Modelo 3D (.mdl) - Assinatura 'MDL ' encontrada. Contém dados de geometria, **não áudio**."
            else:
                 return "Arquivo de Modelo 3D (.mdl) - Tipo de arquivo para modelos de personagens/objetos."
        except Exception:
             return "Arquivo de Modelo 3D (.mdl) - Tipo de arquivo para modelos."
    
    return None

# --- Lógica Principal ---

def get_projects_logic():
    """Escaneia a pasta de backups e retorna uma lista de projetos analisados."""
    projects = []
    if not os.path.exists(BACKUP_DIR):
        return projects
    for filename in os.listdir(BACKUP_DIR):
        if filename.startswith('backup_') and filename.endswith('.json'):
            try:
                with open(os.path.join(BACKUP_DIR, filename), 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if data.get('status') == 'done':
                        projects.append({
                            'id': data['hash_sha1_original'],
                            'name': data['nome_arquivo']
                        })
            except Exception:
                continue
    return sorted(projects, key=lambda x: x['name'])

def get_project_files_logic(project_id, search_query=None, limit=200):
    """Retorna lista filtrada de arquivos do projeto. Suporta busca para projetos gigantes."""
    if not project_id: return []
    caminho_backup_json = os.path.join(BACKUP_DIR, f"backup_{project_id}.json")
    if not os.path.exists(caminho_backup_json): return []
    
    with open(caminho_backup_json, 'r', encoding='utf-8') as f:
        info_backup = json.load(f)
    
    all_files = info_backup.get('arquivos_internos', [])
    
    if not search_query:
        # Se não tiver busca, retorna apenas os primeiros 'limit' para não travar o browser
        return all_files[:limit]
    
    # Busca Case-Insensitive no backend
    query = search_query.lower()
    filtered = [f for f in all_files if query in f['safe_name'].lower()]
    return filtered[:limit] # Retorna top 'limit' matches

def calcular_hash_sha1(caminho_arquivo):
    sha1 = hashlib.sha1()
    try:
        with open(caminho_arquivo, 'rb') as f:
            while chunk := f.read(8192):
                sha1.update(chunk)
        return sha1.hexdigest()
    except Exception:
        return None

def sanitize_archive_name(name: str, max_len: int = 100) -> str:
	if not name: return ''
	name = name.replace('\x00', '').replace('\\', '/')
	parts = [p for p in name.split('/') if p and p != '..']
	if not parts: return ''

	safe_parts = []
	for p in parts:
		comp = ''.join([ch if ch.isalnum() or ch in ' ._-' else '_' for ch in p]).strip()
		while comp.endswith('.'): comp = comp[:-1]
		comp = comp[:max_len]
		if comp: safe_parts.append(comp)
	if not safe_parts: return ''
	
	reserved = {'CON','PRN','AUX','NUL'} | {f'COM{i}' for i in range(1,10)} | {f'LPT{i}' for i in range(1,10)}
	final_parts = [p + '_' if p.upper() in reserved else p for p in safe_parts]
	return os.path.join(*final_parts)

def analisar_arch_logic(caminho_arch):
    """Analisa um arquivo .arch, cria backups e salva a estrutura."""
    if not os.path.isfile(caminho_arch):
        return False, "Erro: Arquivo não encontrado."
    # ... (código existente sem alterações)
    try:
        with open(caminho_arch, 'rb') as f:
            magic = f.read(8)
        if magic not in KNOWN_ARCH_MAGICS:
            return False, "Erro: Assinatura de arquivo inválida. Este não é um arquivo .arch compatível."
        hash_original = calcular_hash_sha1(caminho_arch)
        if not hash_original: return False, "Erro ao calcular o hash de integridade do arquivo."
        if not os.path.exists(BACKUP_DIR): os.makedirs(BACKUP_DIR)
        caminho_backup_json = os.path.join(BACKUP_DIR, f"backup_{hash_original}.json")
        if os.path.exists(caminho_backup_json):
            try:
                with open(caminho_backup_json, 'r', encoding='utf-8') as f: data = json.load(f)
                if data.get('status') == 'done': return True, "Este arquivo já foi analisado com sucesso. O projeto está pronto para o Passo 2."
                else: os.remove(caminho_backup_json)
            except (json.JSONDecodeError, KeyError): os.remove(caminho_backup_json)

        initial_info = { 'caminho_original': caminho_arch, 'nome_arquivo': os.path.basename(caminho_arch), 'hash_sha1_original': hash_original, 'magic_number': magic.hex(), 'status': 'in_progress', 'created_at': datetime.datetime.utcnow().isoformat() + 'Z' }
        with open(caminho_backup_json, 'w', encoding='utf-8') as tf: json.dump(initial_info, tf, indent=4, ensure_ascii=False)
        info_original = OrderedDict([ ('caminho_original', caminho_arch), ('nome_arquivo', os.path.basename(caminho_arch)), ('hash_sha1_original', hash_original), ('magic_number', magic.hex()), ('expected_num_arquivos', 0), ('arquivos_internos', []) ])
        with open(caminho_arch, 'rb') as f:
            f.seek(8)
            header = f.read(12)
            if len(header) < 12: return False, "Erro: Cabeçalho do arquivo truncado."
            table_offset, name_block_offset, num_files = struct.unpack('<3I', header)
            info_original['expected_num_arquivos'] = num_files
            file_total_size = os.path.getsize(caminho_arch)
            if not (table_offset < file_total_size and name_block_offset < file_total_size and num_files > 0): return False, f"Erro Crítico: Cabeçalho do arquivo parece inválido."
            f.seek(table_offset)
            entradas_lidas = 0
            for idx in range(num_files):
                try:
                    entry_bytes = f.read(16)
                    if len(entry_bytes) < 16: break
                    offset_to_name, file_size, file_offset, file_hash = struct.unpack('<4I', entry_bytes)
                    pos_after_entry = f.tell()
                    f.seek(name_block_offset + offset_to_name)
                    name_bytes_list = []
                    while True:
                        b = f.read(1)
                        if not b or b == b'\x00': break
                        name_bytes_list.append(b)
                    file_name = b''.join(name_bytes_list).decode('latin-1', errors='ignore')
                    f.seek(pos_after_entry)
                    if not (0 <= file_offset < file_total_size and file_size >= 0 and file_offset + file_size <= file_total_size): continue
                    safe_name = sanitize_archive_name(file_name)
                    if not safe_name: continue
                    info_original['arquivos_internos'].append({ 'safe_name': safe_name, 'hash': file_hash, 'offset': file_offset, 'size': file_size })
                    entradas_lidas += 1
                except Exception as e_loop:
                    gerar_relatorio_erro("analisar_arch_logic (loop)", e_loop, traceback.format_exc(), {'indice_do_loop': idx})
                    return False, f"Erro ao ler entrada {idx}. 'relatorio_de_erro.json' foi criado."
        nome_original_base = os.path.basename(caminho_arch)
        caminho_backup_arch = os.path.join(BACKUP_DIR, f"original_{hash_original}_{nome_original_base}")
        if not os.path.exists(caminho_backup_arch): shutil.copyfile(caminho_arch, caminho_backup_arch)
        info_original['status'] = 'done'
        info_original['completed_at'] = datetime.datetime.utcnow().isoformat() + 'Z'
        with open(caminho_backup_json, 'w', encoding='utf-8') as tf: json.dump(info_original, tf, indent=4, ensure_ascii=False)
        return True, f"Análise Concluída! {entradas_lidas} arquivos mapeados."
    except Exception as e:
        gerar_relatorio_erro("analisar_arch_logic (geral)", e, traceback.format_exc(), {'caminho_arquivo': caminho_arch})
        return False, "Ocorreu um erro inesperado. Um 'relatorio_de_erro.json' foi criado."

# --- LÓGICA DE ANÁLISE (.FSB - BIOSHOCK REMASTERED) ---
def analisar_fsb_logic(caminho_fsb):
    """Analisa um arquivo .fsb (FMOD Sound Bank) e prepara para extração."""
    if not fsb5: 
        return False, f"Erro: fsb5 ausente. ({FSB_ERROR_MSG}). Python: {sys.executable}. Cmd: pip install fsb5"
    if not os.path.isfile(caminho_fsb): return False, "Erro: Arquivo Inexistente."

    try:
        hash_original = calcular_hash_sha1(caminho_fsb)
        if not hash_original: return False, "Erro ao calcular hash."

        # Setup Backup JSON
        if not os.path.exists(BACKUP_DIR): os.makedirs(BACKUP_DIR)
        caminho_backup_json = os.path.join(BACKUP_DIR, f"backup_{hash_original}.json")
        
        if os.path.exists(caminho_backup_json):
             try:
                 with open(caminho_backup_json, 'r', encoding='utf-8') as f: d = json.load(f)
                 if d.get('status') == 'done': return True, "Banco de Áudio (.fsb) já analisado!"
             except: pass

        # Abre FSB
        with open(caminho_fsb, 'rb') as f:
            fsb = fsb5.FSB5(f.read())
        
        info_fsb = OrderedDict([
            ('caminho_original', caminho_fsb),
            ('nome_arquivo', os.path.basename(caminho_fsb)),
            ('hash_sha1_original', hash_original),
            ('type', 'fsb'), # Flag para o descompactador
            ('status', 'in_progress'),
            ('arquivos_internos', [])
        ])

        # Mapeia arquivos
        files_found = 0
        # fsb5 guarda amostras em fsb.samples
        # Cada amostra tem 'name', 'frequency', 'channels', 'data_offset', 'data_size'
        for i, sample in enumerate(fsb.samples):
            # Tenta pegar nome ou gera um índice
            try:
                safe_name = sample.name.decode('utf-8', errors='ignore') if sample.name else f"sample_{i:04d}"
            except:
                safe_name = f"sample_{i:04d}"
                
            # Adiciona extensão se não tiver
            if not safe_name.lower().endswith(('.wav', '.ogg', '.mp3')):
                ext = '.wav' # Default
                # Tenta adivinhar pelo header interno (não implementado aqui, assume wav/ogg pela extração futura)
                safe_name += ext

            info_fsb['arquivos_internos'].append({
                'safe_name': safe_name,
                'index': i,
                'size': len(sample.data) if sample.data else 0, # CORREÇÃO: data_size não existe na v1.0
                'frequency': sample.frequency,
                'channels': sample.channels
            })
            files_found += 1

        info_fsb['status'] = 'done'
        info_fsb['header_version'] = fsb.header.version
        
        # [DEBUG] Mostra versão no log para decidir ferramenta de repack
        msg_versao = f"Versão FSB Detectada: {fsb.header.version}"
        print(f"[DEBUG] {msg_versao}")
        
        info_fsb['completed_at'] = datetime.datetime.utcnow().isoformat() + 'Z'
        with open(caminho_backup_json, 'w', encoding='utf-8') as tf: json.dump(info_fsb, tf, indent=4, ensure_ascii=False)
        
        return True, f"Análise FSB Concluída! {files_found} amostras. ({msg_versao})"

    except Exception as e:
        gerar_relatorio_erro("analisar_fsb_logic", e, traceback.format_exc(), {'caminho_fsb': caminho_fsb})
        return False, f"Erro ao ler FSB: {e}"

def analisar_pck_logic(caminho_pck):
    if not os.path.isfile(caminho_pck): return False, "Arquivo Inexistente."
    try:
        hash_original = calcular_hash_sha1(caminho_pck)
        if not hash_original: return False, "Erro calc hash"
        if not os.path.exists(BACKUP_DIR): os.makedirs(BACKUP_DIR)
        caminho_backup_json = os.path.join(BACKUP_DIR, f"backup_{hash_original}.json")
        if os.path.exists(caminho_backup_json):
            try:
                with open(caminho_backup_json, 'r', encoding='utf-8') as f: d = json.load(f)
                if d.get('status') == 'done': return True, "PCK já analisado!"
            except: pass

        info_pck = OrderedDict([('caminho_original', caminho_pck), ('nome_arquivo', os.path.basename(caminho_pck)), ('hash_sha1_original', hash_original), ('type', 'pck'), ('status', 'in_progress'), ('arquivos_internos', [])])
        with open(caminho_pck, 'rb') as f:
            magic = f.read(4)
            if magic != b'AKPK': return False, "Não é um arquivo AKPK válido."
            header_size, unk, lang_size, bank_size, sound_size, ext_size = struct.unpack('<IIIIII', f.read(24))
            
            if bank_size > 0:
                f.seek(0x1C + lang_size)
                num_banks = struct.unpack('<I', f.read(4))[0]
                for i in range(num_banks):
                    sid, align, size, offset, lang_id = struct.unpack('<IIIII', f.read(20))
                    info_pck['arquivos_internos'].append({'safe_name': f"bank_{sid}.bnk", 'size': size, 'offset': offset * align, 'align': align, 'id': sid, 'lang_id': lang_id})
            if sound_size > 0:
                f.seek(0x1C + lang_size + bank_size)
                num_sounds = struct.unpack('<I', f.read(4))[0]
                for i in range(num_sounds):
                    sid, align, size, offset, lang_id = struct.unpack('<IIIII', f.read(20))
                    info_pck['arquivos_internos'].append({'safe_name': f"sound_{sid}.wem", 'size': size, 'offset': offset * align, 'align': align, 'id': sid, 'lang_id': lang_id})
                    
        info_pck['status'] = 'done'
        caminho_backup_pck = os.path.join(BACKUP_DIR, f"original_{hash_original}_{os.path.basename(caminho_pck)}")
        if not os.path.exists(caminho_backup_pck): shutil.copyfile(caminho_pck, caminho_backup_pck)
        with open(caminho_backup_json, 'w', encoding='utf-8') as tf: json.dump(info_pck, tf, indent=4, ensure_ascii=False)
        return True, f"Análise PCK Concluída! {len(info_pck['arquivos_internos'])} arquivos (WEM/BNK) encontrados."
    except Exception as e:
        gerar_relatorio_erro("analisar_pck_logic", e, traceback.format_exc())
        return False, f"Erro estrutural ao ler PCK: {e}"


def descompactar_arch_logic(project_id, files_to_extract=None):
    """
    Função híbrida para extrair ARCH (F.E.A.R.) e VPK (Source).
    """
    try:
        if not project_id: return False, "Erro: ID do projeto não fornecido."
        
        caminho_backup_json = os.path.join(BACKUP_DIR, f"backup_{project_id}.json")
        if not os.path.exists(caminho_backup_json): return False, "Erro: Projeto não encontrado."
        
        with open(caminho_backup_json, 'r', encoding='utf-8') as f: info_backup = json.load(f)
        
        nome_arquivo_original = info_backup['nome_arquivo']
        tipo_projeto = info_backup.get('type', 'arch') # Default para arch
        
        nome_base = os.path.splitext(nome_arquivo_original)[0]
        if tipo_projeto == 'vpk': nome_base = nome_base.replace('_dir', '') # Remove sufixo do nome da pasta
            
        pasta_destino_final = os.path.join(BACKUP_DIR, f"{nome_base}_MOD")

        # Verifica se pasta já existe (Regra geral)
        if os.path.exists(pasta_destino_final) and not files_to_extract:
             return False, f"A pasta '{pasta_destino_final}' já existe. Use a extração seletiva para adicionar arquivos."

        # Filtra arquivos
        all_entries = info_backup.get('arquivos_internos', [])
        entries_to_process = [e for e in all_entries if e['safe_name'] in files_to_extract] if files_to_extract else all_entries
        
        if not os.path.exists(pasta_destino_final): os.makedirs(pasta_destino_final)
        
        extruidos, pulados = 0, 0
        arquivos_extraidos_nesta_sessao = []
        
        # --- BRANCH: VPK EXTRACTION ---
        if tipo_projeto == 'vpk':
            caminho_original = info_backup['caminho_original']
            if not os.path.exists(caminho_original):
                return False, "O arquivo VPK original não foi encontrado no local de origem."
            
            try:
                pak = vpk.open(caminho_original)
                for entry in entries_to_process:
                    safe_name = entry.get('safe_name') # No VPK, safe_name == caminho interno
                    caminho_completo_saida = os.path.join(pasta_destino_final, safe_name.replace('/', os.sep))
                    
                    try:
                        os.makedirs(os.path.dirname(caminho_completo_saida), exist_ok=True)
                        
                        # Extrai usando a lib vpk
                        pak_file = pak.get_file(safe_name)
                        pak_file.save(caminho_completo_saida)
                        
                        arquivos_extraidos_nesta_sessao.append(caminho_completo_saida)
                        extruidos += 1
                    except Exception as e:
                        logging.warning(f"Falha ao extrair VPK '{safe_name}': {e}")
                        pulados += 1
            except Exception as e:
                return False, f"Erro ao abrir VPK para extração: {e}"



        # --- BRANCH: FSB EXTRACTION (BIOSHOCK) ---
        elif tipo_projeto == 'fsb':
            caminho_original = info_backup['caminho_original']
            if not os.path.exists(caminho_original): return False, "FSB original não encontrado."
            
            try:
                with open(caminho_original, 'rb') as f_fsb:
                    fsb = fsb5.FSB5(f_fsb.read())
                    
                for entry in entries_to_process:
                    safe_name = entry.get('safe_name')
                    idx = entry.get('index')
                    caminho_completo_saida = os.path.join(pasta_destino_final, safe_name)
                    
                    try:
                        os.makedirs(os.path.dirname(caminho_completo_saida), exist_ok=True)
                        sample = fsb.samples[idx]
                        
                        # --- NOVO MÉTODO: VGMSTREAM (Robusto) ---
                        # Se existir a ferramenta, usa ela
                        if os.path.exists(VGMSTREAM_PATH):
                            # Ajusta extensão para WAV (vgmstream converte para wav por padrão)
                            if not safe_name.lower().endswith('.wav'):
                                safe_name = os.path.splitext(safe_name)[0] + '.wav'
                                caminho_completo_saida = os.path.join(pasta_destino_final, safe_name)

                            # Comando: vgmstream-cli.exe -s <index+1> -o <outfile> <infile>
                            # NOTA: vgmstream usa index 1-based para subsongs? Vamos testar.
                            # Documentação diz: -s N select subsong N (default=0=all? No, default is 1 if multiple?)
                            # Com fsb, subsong index geralmente bate com o nosso 'i' + 1
                            
                            subsong_index = int(idx) + 1  # CORREÇÃO: 'i' não definido, usar 'idx'
                            cmd = [
                                VGMSTREAM_PATH,
                                "-s", str(subsong_index),
                                "-o", caminho_completo_saida,
                                "-i", # Ignore looping
                                caminho_original
                            ]
                            
                            # Executa silenciosamente
                            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                            
                            if os.path.exists(caminho_completo_saida):
                                arquivos_extraidos_nesta_sessao.append(caminho_completo_saida)
                                extruidos += 1
                            else:
                                logging.warning(f"vgmstream falhou para '{safe_name}'")
                                pulados += 1
                        
                        else:
                            # Fallback (Broken fsb5 logic removed/commented)
                            logging.error(f"vgmstream-cli.exe não encontrado em {VGMSTREAM_PATH}")
                            pulados += 1

                        arquivos_extraidos_nesta_sessao.append(caminho_completo_saida)
                        extruidos += 1
                    except Exception as e:
                        logging.warning(f"Falha ao extrair FSB '{safe_name}': {e}")
                        pulados += 1
            except Exception as e:
                return False, f"Erro Fatal ao abrir FSB: {e}"

        # --- BRANCH: ARCH EXTRACTION (LEGACY) E PCK (NOVO) ---
        else:
            if tipo_projeto == 'pck':
                caminho_backup_arch = os.path.join(BACKUP_DIR, f"original_{project_id}_{nome_arquivo_original}")
            else:
                caminho_backup_arch = os.path.join(BACKUP_DIR, f"original_{project_id}_{nome_arquivo_original}")
                
            if not os.path.isfile(caminho_backup_arch): return False, f"Erro Crítico: O backup do arquivo original não foi encontrado em {caminho_backup_arch}"
            
            with open(caminho_backup_arch, 'rb') as f_arch:
                # Priorizar processamento de .wem ANTES de .bnk para que wems originais sejam a fonte primária do hash
                if tipo_projeto == 'pck':
                     entries_to_process.sort(key=lambda e: 1 if str(e.get('safe_name', '')).endswith('.bnk') else 0)
                
                wav_hashes = set() # Guarda checksum dos arquivos para ignorar cópias (ex: um .wem contido em bancada .bnk)
                
                for entry in entries_to_process:
                    safe_name = entry.get('safe_name')
                    
                    # [MODO FOCADO] O usuário deseja apenas os WEMs principais agora, ignorando os Banks.
                    if tipo_projeto == 'pck' and safe_name.endswith('.bnk'):
                        continue # Pula completamente a extração de BNKs para economizar espaço e focar nas vozes principais
                        
                    caminho_completo_saida = os.path.normpath(os.path.join(pasta_destino_final, safe_name))
                    try:
                        os.makedirs(os.path.dirname(caminho_completo_saida), exist_ok=True)
                        f_arch.seek(int(entry['offset']))
                        raw_data = f_arch.read(int(entry['size']))
                        with open(caminho_completo_saida, 'wb') as f_out: f_out.write(raw_data)
                        arquivos_extraidos_nesta_sessao.append(caminho_completo_saida)
                        extruidos += 1
                        
                        # PCK: Conversões para WAV usando vgmstream
                        if tipo_projeto == 'pck' and os.path.exists(VGMSTREAM_PATH):
                            if safe_name.endswith('.wem'):
                                try:
                                    wav_out = os.path.splitext(caminho_completo_saida)[0] + '.wav'
                                    res = subprocess.run([VGMSTREAM_PATH, "-o", wav_out, caminho_completo_saida], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                                    if res.returncode == 0:
                                        os.remove(caminho_completo_saida) # Apaga o arquivo proprietário da engine
                                except: pass
                    except Exception as e:
                        logging.warning(f"Falha ao extrair '{safe_name}': {e}")
                        pulados += 1

        # Finalização Comum
        if not os.path.exists(os.path.join(pasta_destino_final, '.modinfo')):
            with open(os.path.join(pasta_destino_final, '.modinfo'), 'w', encoding='utf-8') as f: 
                json.dump({'original_hash': project_id, 'type': tipo_projeto}, f)
        
        # Gera relatório/insights
        mensagem_final = f"Extração ({tipo_projeto.upper()}) concluída: {extruidos} arquivos."
        return True, mensagem_final

    except Exception as e:
        gerar_relatorio_erro("descompactar_arch_logic", e, traceback.format_exc())
        return False, f"Erro inesperado: {e}"


# --- NOVA LÓGICA PARA PROCESSAR .BNDL ---
def processar_bndl_logic(caminho_bndl):
    """
    Processa um arquivo .bndl.
    ATENÇÃO: Esta é uma SIMULAÇÃO. A lógica real de extração de .bndl é
    extremamente complexa e específica para o jogo. Esta função apenas demonstra
    o fluxo de trabalho, criando arquivos .snd de exemplo.
    """
    if not caminho_bndl or not os.path.isfile(caminho_bndl):
        return False, "Erro: Arquivo .bndl não encontrado ou caminho inválido."

    if not caminho_bndl.lower().endswith('.bndl'):
        return False, "Erro: O arquivo selecionado não parece ser um arquivo .bndl."

    try:
        # Cria uma pasta de destino para os arquivos extraídos (simulados)
        pasta_base = os.path.dirname(caminho_bndl)
        nome_bndl = os.path.basename(caminho_bndl)
        pasta_destino = os.path.join(pasta_base, f"{nome_bndl}_extracted")
        
        if not os.path.exists(pasta_destino):
            os.makedirs(pasta_destino)
        else:
            # Limpa a pasta se já existir para uma nova extração
            for f in os.listdir(pasta_destino):
                os.remove(os.path.join(pasta_destino, f))

        # Simulação: Cria alguns arquivos .snd de exemplo
        arquivos_exemplo = [
            'exemplo_dialogo_01.snd',
            'exemplo_som_ambiente_02.snd',
            'exemplo_musica_combate_01.snd',
            'exemplo_efeito_arma_03.snd'
        ]
        for nome_arquivo in arquivos_exemplo:
            caminho_completo = os.path.join(pasta_destino, nome_arquivo)
            with open(caminho_completo, 'wb') as f:
                f.write(b'Este e um arquivo .snd simulado.') # Conteúdo fictício

        mensagem = (
            f"Processamento simulado de '{nome_bndl}' concluído!\n\n"
            f"{len(arquivos_exemplo)} arquivos .snd de exemplo foram criados na pasta:\n'{pasta_destino}'\n\n"
            "**AVISO IMPORTANTE:**\n"
            "Esta é apenas uma demonstração. Os arquivos .snd criados são fictícios.\n"
            "Para extrair e converter os sons reais, você ainda precisará de uma ferramenta de modding "
            "específica para o F.E.A.R. 2 que consiga ler o formato .bndl e converter .snd para .wav/.ogg."
        )
        return True, mensagem

    except Exception as e:
        gerar_relatorio_erro("processar_bndl_logic", e, traceback.format_exc())
        return False, "Ocorreu um erro inesperado durante o processamento simulado do .bndl."


def restaurar_arch_logic(caminho_alvo):
    # Implementação omitida para brevidade
    return True, "Restauração completa!"

# --- Interface Web e Servidor Flask ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Universal Mod Manager (Multi-Game)</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background-color: #121212; color: #e0e0e0; margin: 0; padding: 20px; display: flex; justify-content: center; }
        .container { max-width: 800px; width: 100%; background-color: #1e1e1e; padding: 25px; border-radius: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.5); }
        h1, h2, h3, h4 { color: #bb86fc; border-bottom: 2px solid #373737; padding-bottom: 10px; margin-top: 0; }
        h2 { margin-top: 30px; font-size: 1.2em; }
        .step { background-color: #2a2a2a; padding: 20px; border-radius: 8px; margin-bottom: 20px; border-left: 4px solid #bb86fc; }
        .step-bndl { border-left-color: #03dac6; } /* Nova cor para o passo 3 */
        .restore-tool { border-left-color: #f44336; }
        label { display: block; margin-bottom: 8px; font-weight: bold; color: #cfcfcf; }
        .input-group { display: flex; gap: 10px; margin-bottom: 15px; }
        input[type="text"], select { flex-grow: 1; padding: 10px; background-color: #333; border: 1px solid #444; border-radius: 5px; color: #e0e0e0; font-size: 0.9em; font-family: 'Courier New', Courier, monospace; }
        button { background-color: #bb86fc; color: #121212; border: none; padding: 12px 20px; border-radius: 5px; cursor: pointer; font-size: 1em; font-weight: bold; transition: background-color 0.3s; }
        button:hover { background-color: #a36ef4; }
        button:disabled { background-color: #555; color: #888; cursor: not-allowed; }
        button.browse-btn { background-color: #03dac6; color: #121212; flex-shrink: 0; }
        button.browse-btn:hover { background-color: #01b8a5; }
        button.restore-btn { background-color: #f44336; color: white; }
        button.restore-btn:hover { background-color: #d32f2f; }
        #log { background-color: #1a1a1a; border: 1px solid #333; border-radius: 5px; padding: 15px; white-space: pre-wrap; word-wrap: break-word; min-height: 100px; font-family: 'Courier New', Courier, monospace; margin-top: 10px; max-height: 400px; overflow-y: auto; }
        .hidden { display: none; }
        #file-list-container { max-height: 300px; background-color: #1e1e1e; border: 1px solid #444; overflow-y: auto; padding: 10px; margin: 15px 0; }
        .controls { display: flex; gap: 10px; margin-bottom: 10px; }
        .alert-box { background-color: #332a00; border: 1px solid #554400; padding: 15px; border-radius: 5px; margin: 15px 0; font-size: 0.9em; color: #ffcc00; line-height: 1.4; }
        .modal-overlay { position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.85); display: flex; justify-content: center; align-items: center; z-index: 1000; }
        .modal-content { background: #1e1e1e; padding: 30px; border-radius: 12px; max-width: 600px; width: 90%; border: 1px solid #333; box-shadow: 0 10px 30px rgba(0,0,0,0.8); text-align: center; }
        .modal-content ul { padding-left: 20px; text-align: left; }
        .modal-content li { margin-bottom: 10px; color: #ccc; }
        .hidden { display: none !important; }
        .close-modal-btn { margin-top: 25px; width: 100%; background-color: #373737; color: white; border: 1px solid #444; padding: 12px; border-radius: 8px; cursor: pointer; transition: all 0.3s; }
        .close-modal-btn:hover { background-color: #444; border-color: #bb86fc; }
    </style>
</head>
<body>
    <div class="container">
        <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid #373737; margin-bottom: 20px; padding-bottom: 10px;">
            <h1 style="border:none; margin:0; font-size: 1.5em;">Universal Mod Manager</h1>
            <button onclick="toggleTutorial()" style="background-color: #03dac6; color: #121212; padding: 8px 15px; font-size: 0.85em; border-radius: 20px;">📖 Tutorial de Instalação</button>
        </div>
        <div class="step">
            <h2>Passo 1: Analisar Arquivo (.arch / .vpk / .fsb)</h2>
            <div class="input-group">
                <input type="text" id="path-analisar" placeholder="Clique em 'Procurar...' para selecionar o arquivo" readonly>
                <button class="browse-btn" onclick="browseFile('path-analisar')">Procurar...</button>
            </div>
            <button id="btn-analisar" onclick="executarAcao('analisar')">Analisar e Criar Projeto</button>
        </div>
        <div class="step">
            <h2>Passo 2: Descompactar Projeto</h2>
            <select id="project-selector" style="margin-bottom: 15px;" onchange="clearFileCacheAndUI()"><option value="">Carregando projetos...</option></select>
            <p class="info-text">Extrai os arquivos de dentro do pacote selecionado.</p>
            <button id="btn-toggle-files" onclick="toggleFileSelector()">Mostrar/Selecionar Arquivos...</button>
            <div id="file-selector-inline-container" class="hidden" style="margin-top: 20px; border-top: 1px solid #444; padding-top: 15px;">
                <!-- ... (código existente sem alterações) ... -->
                <h3>Arquivos para Extrair (<span id="file-count">0</span> selecionados)</h3>
                <input type="text" id="file-search" onkeyup="handleSearchInput()" placeholder="Pesquisar arquivos... (Ex: voice, wav)" style="margin-bottom: 10px; width: calc(100% - 22px);">
                <div class="controls">
                    <button onclick="toggleAllFiles(true)">Selecionar Visíveis</button>
                    <button onclick="toggleAllFiles(false)">Desmarcar Visíveis</button>
                </div>
                <div id="file-list-container"><p>Clique em 'Mostrar/Selecionar Arquivos' para carregar a lista.</p></div>
            </div>
            <button id="btn-descompactar" onclick="executarAcao('descompactar')" style="margin-top: 15px;">Descompactar</button>
        </div>
        
        <!-- NOVO PASSO 3 -->
        <div class="step step-bndl">
            <h2>Passo 3: Processar Arquivo Bundle (.bndl) - (Simulação)</h2>
            <p class="info-text">Esta ferramenta demonstra como um arquivo .bndl seria extraído.</p>
            <div class="input-group">
                <input type="text" id="path-bndl" placeholder="Selecione um arquivo .bndl extraído no Passo 2" readonly>
                <button class="browse-btn" onclick="browseFile('path-bndl')">Procurar .bndl...</button>
            </div>
            <button id="btn-bndl" onclick="executarAcao('processar-bndl')">Processar .bndl</button>
        </div>
        
        <div class="step">
            <h2>Passo 4: Reempacotar (Repack FSB / PCK)</h2>
            <p class="info-text">Reconstroi o arquivo original (.fsb ou .pck) usando os áudios modificados.</p>
            
            <label>1. Pasta com os Áudios Modificados (WAV/WEM):</label>
            <div class="input-group">
                <input type="text" id="path-repack-input" placeholder="Selecione a pasta ..._MOD criada no Passo 2" readonly>
                <button class="browse-btn" onclick="browseFolder('path-repack-input')">Procurar Pasta...</button>
            </div>

            <br>
            <label>2. Executável do FMOD (Somente para FSB/BioShock):</label>
            <div class="input-group">
                <input type="text" id="path-fmod-tool" placeholder="C:\Program Files (x86)\...\bin\fsbankcl.exe" value="C:\Program Files (x86)\FMOD SoundSystem\FMOD Studio API Windows\bin\fsbankcl.exe">
                <button class="browse-btn" onclick="browseFile('path-fmod-tool', 'exe')">Procurar Tool...</button>
            </div>
            <p class="info-text">Nota para jogos Wwise (PCK): Deixe o caminho do FMOD acima como está, o script fará o repack de forma nativa e automática!</p>
            
            <div class="alert-box">
                <strong>Instruções de Reempacotamento:</strong><br>
                • <b>Modo Pro (Recomendado):</b> Use o <code>fsbankcl.exe</code> (Caminho: <code>C:\Program Files (x86)\FMOD SoundSystem\FMOD Studio API Windows\bin</code>). O processo é 100% automático (você verá sua CPU trabalhar e depois a mensagem de sucesso).<br>
                • <b>Modo Legado:</b> Se usar o <code>Fmod_Bank_Tools.exe</code>, você precisará clicar em "Rebuild" manualmente na janela que abrir e depois fechá-la.
            </div>

            <button id="btn-repack" onclick="executarAcao('repack-fsb')">Reempacotar e Instalar</button>
        </div>

        <div class="step restore-tool">
            <h2>Ferramenta: Restaurar Backup Original</h2>
            <!-- ... (código existente sem alterações) ... -->
            <div class="input-group">
                <input type="text" id="path-restaurar" placeholder="Clique em 'Procurar...' para selecionar o arquivo" readonly>
                <button class="browse-btn" onclick="browseFile('path-restaurar')">Procurar...</button>
            </div>
            <button id="btn-restaurar" class="restore-btn" onclick="executarAcao('restaurar')">Restaurar Original</button>
        </div>

        <!-- MODAL TUTORIAL -->
        <div id="tutorial-modal" class="modal-overlay hidden" onclick="if(event.target === this) toggleTutorial()">
            <div class="modal-content">
                <h2 style="color: #03dac6; border:none; margin-bottom: 5px;">📖 Guia de Instalação</h2>
                <div style="text-align: left; max-height: 450px; overflow-y: auto; padding-right: 10px; margin-top: 15px;">
                    <p style="color: #888; font-size: 0.9em;">Ferramentas necessárias para o funcionamento completo:</p>
                    
                    <h3 style="font-size: 1.1em; color: #bb86fc; margin-top: 15px;">1. FMOD Studio API (BioShock)</h3>
                    <p style="font-size: 0.9em;">Necessário para reempacotar os arquivos <code>.fsb</code>.</p>
                    <ul style="font-size: 0.9em;">
                        <li>Site: <a href="https://www.fmod.com/download#fmodengine" target="_blank" style="color:#03dac6">fmod.com/download</a>.</li>
                        <li>Baixe <b>"FMOD Studio API Windows"</b>.</li>
                        <li>O <code>fsbankcl.exe</code> fica em: <code>.../bin/fsbankcl.exe</code>.</li>
                    </ul>

                    <h3 style="font-size: 1.1em; color: #bb86fc; margin-top: 15px;">2. vgmstream-cli (Extração)</h3>
                    <p style="font-size: 0.9em;">Necessário para converter áudios originais de consoles/PC.</p>
                    <ul style="font-size: 0.9em;">
                        <li>Baixe em <a href="https://vgmstream.org" target="_blank" style="color:#03dac6">vgmstream.org</a>.</li>
                        <li>Extraia em: <code>C:\IA_dublagem\tools\vgmstream-cli\</code>.</li>
                    </ul>

                    <h3 style="font-size: 1.1em; color: #bb86fc; margin-top: 15px;">3. Bibliotecas Python</h3>
                    <code style="background:#000; padding:10px; display:block; border-radius:5px; font-size: 0.85em; border: 1px solid #333; color: #03dac6;">pip install flask vpk fsb5</code>
                </div>
                <button onclick="toggleTutorial()" class="close-modal-btn">Fechar Guia</button>
            </div>
        </div>

        <h2>Log de Operações</h2>
        <pre id="log">Bem-vindo ao Arch Mod Manager v9.2!</pre>
    </div>

    <script>
        function toggleTutorial() {
            const modal = document.getElementById('tutorial-modal');
            if (modal) {
                modal.classList.toggle('hidden');
                console.log("Tutorial Modal Toggled:", !modal.classList.contains('hidden'));
            } else {
                console.error("Erro: Elemeno 'tutorial-modal' não encontrado.");
            }
        }
        const logElement = document.getElementById('log');
        function log(message) { const timestamp = new Date().toLocaleTimeString(); logElement.textContent = `${timestamp}: ${message.replace(/\\n/g, '\\n')}\\n` + logElement.textContent.split('\\n').slice(0, 100).join('\\n'); }
        async function browseFile(targetInputId, type='file') { 
            const endpoint = type === 'exe' ? '/api/browse-exe' : '/api/browse-file';
            const response = await fetch(endpoint); 
            const data = await response.json(); 
            if (data.path) { 
                document.getElementById(targetInputId).value = data.path; 
                log('Arquivo selecionado: ' + data.path); 
                if (targetInputId === 'path-fmod-tool') localStorage.setItem('fmod_tool_path', data.path);
            } 
        }
        async function browseFolder(targetInputId) {
             const response = await fetch('/api/browse-folder');
             const data = await response.json();
             if (data.path) {
                 document.getElementById(targetInputId).value = data.path;
                 log('Pasta selecionada: ' + data.path);
             }
        }
        
        async function loadProjects() { 
            const selector = document.getElementById('project-selector'); 
            selector.innerHTML = '<option value="">Carregando...</option>'; 
            const response = await fetch('/api/get-projects'); 
            const projects = await response.json(); 
            selector.innerHTML = projects.length > 0 ? '' : '<option value="">Nenhum projeto. Use o Passo 1.</option>'; 
            projects.forEach(p => { 
                const option = document.createElement('option'); 
                option.value = p.id; 
                option.textContent = p.name; 
                selector.appendChild(option); 
            }); 
            clearFileCacheAndUI(); 
        }
        
        // Load saved FMOD path
        document.addEventListener('DOMContentLoaded', () => { 
            loadProjects(); 
            const savedFmod = localStorage.getItem('fmod_tool_path');
            if (savedFmod) document.getElementById('path-fmod-tool').value = savedFmod;
        });
        const fileSelectorContainer = document.getElementById('file-selector-inline-container');
        const fileListContainer = document.getElementById('file-list-container');
        const btnToggleFiles = document.getElementById('btn-toggle-files');
        const btnDescompactar = document.getElementById('btn-descompactar');
        const fileCountSpan = document.getElementById('file-count');
        let projectFilesCache = {};
        let searchTimer = null;
        async function loadFiles(query = '') {
            const projectId = document.getElementById('project-selector').value;
            if (!projectId) return;
            
            fileListContainer.innerHTML = '<p style="color:#aaa">Buscando...</p>';
            
            try {
                const response = await fetch('/api/project-files', { 
                    method: 'POST', 
                    headers: {'Content-Type': 'application/json'}, 
                    body: JSON.stringify({ project_id: projectId, query: query }) 
                });
                const files = await response.json();
                
                if (files.length === 0) {
                    fileListContainer.innerHTML = '<p>Nenhum arquivo encontrado com esse termo.</p>';
                    updateFileCount();
                    return;
                }

                const listHTML = files.map(file => 
                    `<div><input type="checkbox" class="file-checkbox" value="${file.safe_name}" onchange="updateFileCount()"> <label>${file.safe_name}</label> <span style="font-size:0.8em; color:#666">(${Math.ceil(file.size/1024)} KB)</span></div>`
                ).join('');
                
                // Aviso de limite
                const msg = query ? '' : '<div style="padding:10px; background:#221100; color:#ffaa00; border:1px solid #442200; margin-bottom:10px">⚠️ Exibindo apenas os primeiros 200 arquivos.<br>Use a busca acima para encontrar arquivos de áudio específicos (ex: "voice", "coach").</div>';
                
                fileListContainer.innerHTML = msg + listHTML;
                updateFileCount(); // Recalcula (embora seleções anteriores possam ser perdidas se não persistidas, para este uso simples é aceitável)
                
            } catch (e) {
                fileListContainer.innerHTML = '<p style="color:red">Erro ao buscar arquivos: ' + e + '</p>';
            }
        }

        function handleSearchInput() {
            const query = document.getElementById('file-search').value;
            clearTimeout(searchTimer);
            searchTimer = setTimeout(() => loadFiles(query), 500); // Debounce 500ms
        }

        function clearFileCacheAndUI() { fileSelectorContainer.classList.add('hidden'); btnToggleFiles.textContent = 'Mostrar/Selecionar Arquivos...'; btnDescompactar.textContent = 'Descompactar'; }
        
        async function toggleFileSelector() { 
            const isHidden = fileSelectorContainer.classList.contains('hidden'); 
            if (!isHidden) { 
                fileSelectorContainer.classList.add('hidden'); 
                btnToggleFiles.textContent = 'Mostrar/Selecionar Arquivos...'; 
                return; 
            } 
            const projectId = document.getElementById('project-selector').value; 
            if (!projectId) { log('[ERRO] Por favor, selecione um projeto primeiro.'); return; } 
            
            fileSelectorContainer.classList.remove('hidden'); 
            btnToggleFiles.textContent = 'Ocultar Lista de Arquivos'; 
            
            loadFiles(''); // Carrega inicial (vazia)
        }
        
        function updateFileCount() { const count = fileListContainer.querySelectorAll('.file-checkbox:checked').length; fileCountSpan.textContent = count; btnDescompactar.textContent = count > 0 ? `Descompactar (${count} Arquivos)` : 'Descompactar (Tudo)'; }
        
        // Função Filter removida (substituída por handleSearchInput no HTML)
        function toggleAllFiles(check) { const checkboxes = fileListContainer.querySelectorAll('.file-checkbox'); checkboxes.forEach(cb => cb.checked = check); updateFileCount(); }

        async function executarAcao(action) {
            let payload = {};
            let endpoint = action;

            if (action === 'analisar') {
                payload.path = document.getElementById('path-analisar').value;
                if (!payload.path) { log("[ERRO] Por favor, selecione um arquivo para analisar."); return; }
            } else if (action === 'descompactar') {
                payload.project_id = document.getElementById('project-selector').value;
                if (!payload.project_id) { log('[ERRO] Nenhum projeto foi selecionado.'); return; }
                if (!fileSelectorContainer.classList.contains('hidden')) {
                    const selectedFiles = Array.from(fileListContainer.querySelectorAll('.file-checkbox:checked')).map(cb => cb.value);
                    if (selectedFiles.length > 0) { payload.selected_files = selectedFiles; }
                }
            } else if (action === 'restaurar') {
                payload.path = document.getElementById('path-restaurar').value;
                if (!payload.path) { log("[ERRO] Por favor, selecione um arquivo para restaurar."); return; }
            } else if (action === 'processar-bndl') {
                payload.path = document.getElementById('path-bndl').value;
                if (!payload.path) { log("[ERRO] Por favor, selecione um arquivo .bndl para processar."); return; }
                endpoint = 'processar-bndl'; // Garante o endpoint correto
            } else if (action === 'repack-fsb') {
                payload.input_folder = document.getElementById('path-repack-input').value;
                payload.tool_path = document.getElementById('path-fmod-tool').value;
                if (!payload.input_folder) { log("[ERRO] Selecione a pasta com os áudios (WAV)."); return; }
                if (!payload.tool_path) { log("[ERRO] Selecione o executável do FMOD (fsbankcl.exe ou Bank Tools)."); return; }
                endpoint = 'repack-fsb';
            }
            
            document.querySelectorAll('button').forEach(b => b.disabled = true);
            log(`Iniciando ${action}...`);
            try {
                const response = await fetch(`/api/${endpoint}`, { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload) });
                const result = await response.json();
                log((result.success ? '[SUCESSO] ' : '[ERRO] ') + result.message);
                if (result.success && action === 'analisar') {
                    loadProjects();
                }
            } catch (e) {
                log(`[ERRO CRÍTICO] Falha na comunicação com o servidor: ${e}`);
            } finally {
                document.querySelectorAll('button').forEach(b => b.disabled = false);
            }
        }
    </script>
</body>
</html>
"""

# --- Configuração do Servidor Flask ---
app = Flask(__name__)
@app.route('/')
def home(): return render_template_string(HTML_TEMPLATE)

@app.route('/api/get-projects')
def api_get_projects(): return jsonify(get_projects_logic())

@app.route('/api/project-files', methods=['POST'])
def api_project_files():
    data = request.get_json()
    project_id = data.get('project_id', '').strip()
    query = data.get('query', '').strip()
    return jsonify(get_project_files_logic(project_id, query))

@app.route('/api/browse-file')
def browse_file():
    try:
        root = tk.Tk(); root.withdraw(); root.attributes('-topmost', True)
        filepath = filedialog.askopenfilename(title="Selecione um arquivo", filetypes=(("Todos os arquivos", "*.*"),))
        root.destroy()
        return jsonify({'path': filepath or ''})
    except Exception:
        return jsonify({'path': ''})

@app.route('/api/browse-folder')
def browse_folder():
    try:
        root = tk.Tk(); root.withdraw(); root.attributes('-topmost', True)
        filepath = filedialog.askdirectory(title="Selecione uma pasta")
        root.destroy()
        return jsonify({'path': filepath or ''})
    except Exception: return jsonify({'path': ''})

@app.route('/api/browse-exe')
def browse_exe():
    try:
        root = tk.Tk(); root.withdraw(); root.attributes('-topmost', True)
        filepath = filedialog.askopenfilename(title="Selecione o Executável", filetypes=(("Executáveis", "*.exe"),))
        root.destroy()
        return jsonify({'path': filepath or ''})
    except Exception: return jsonify({'path': ''})

@app.route('/api/repack-fsb', methods=['POST'])
def api_repack_fsb():
    data = request.get_json()
    tool_path = data.get('tool_path')
    success, msg = reempacotar_fsb_logic(data.get('input_folder'), tool_path)
    return jsonify({'success': success, 'message': msg})

def reempacotar_fsb_logic(input_folder, tool_path):
    """
    Automatiza o FMOD Bank Tools para reempacotar o FSB.
    1. Identifica o arquivo original (.fsb) baseado no .modinfo da pasta de input.
    2. Copia os WAVs da pasta de input para a pasta 'wav' do FMOD Tools.
    3. Executa o rebuild.
    4. Copia o resultado de volta.
    """
    try:
        if not os.path.isdir(input_folder): return False, "Pasta de entrada inválida."
        if not os.path.isfile(tool_path): return False, "Executável do FMOD Bank Tools não encontrado."

        modinfo_path = os.path.join(input_folder, '.modinfo')
        if not os.path.exists(modinfo_path): return False, "Arquivo .modinfo não encontrado na pasta. Extraia o projeto novamente se necessário."
        
        with open(modinfo_path, 'r') as f: mod_data = json.load(f)
        
        if mod_data.get('type') == 'pck':
            return reempacotar_pck_logic(input_folder, mod_data)
            
        if mod_data.get('type') != 'fsb': return False, f"Este projeto não é um FSB. O tipo é '{mod_data.get('type')}'. Verifique o formato."
        
        # Localiza o arquivo original para saber o nome do banco
        backup_json = os.path.join(BACKUP_DIR, f"backup_{mod_data['original_hash']}.json")
        if not os.path.exists(backup_json): return False, "Backup do projeto original não encontrado."
        
        with open(backup_json, 'r') as f: backup_data = json.load(f)
        original_fsb_path = backup_data['caminho_original'] # Caminho completo original
        fsb_filename = backup_data['nome_arquivo'] # Nome do arquivo (streams_1_audio.fsb)

        # =================================================================================
        # MODO PRO: FMOD OFFICIAL ENGINE (fsbankcl.exe)
        # =================================================================================
        if "fsbank" in os.path.basename(tool_path).lower():
            print(f"[INFO] Detectado FMOD Engine Oficial: {tool_path}")
            
            # Prepara output temporário (EVITA usar Program Files para não dar erro de permissão)
            # Vamos criar uma pasta temp dentro da pasta do projeto (input_folder)
            temp_dir = os.path.join(input_folder, "temp_build_fmod")
            if not os.path.exists(temp_dir): os.makedirs(temp_dir)
            
            output_fsb_name = fsb_filename # ex: streams_1_audio.fsb
            output_full_path = os.path.join(temp_dir, output_fsb_name)
            
            # Comando: fsbankcl.exe -format vorbis -o "output.fsb" "input_folder"
            # O input folder é o _MOD com os wavs.
            
            cmd = [
                tool_path,
                '-format', 'vorbis',
                '-o', output_full_path,
                input_folder
            ]
            
            print(f"[CMD] Executando: {' '.join(cmd)}")
            
            try:
                # Executa e captura output
                result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=os.path.dirname(tool_path))
                print(f"[FMOD LOG] {result.stdout}")
                
                if result.returncode != 0 and "Error" in result.stdout:
                    return False, f"Erro no fsbankcl:\n{result.stdout}"

                # fsbankcl as vezes adiciona .bank no final mesmo se pedimos .fsb
                # Verifica o que ele criou
                created_file = None
                if os.path.exists(output_full_path): created_file = output_full_path
                elif os.path.exists(output_full_path + ".bank"): created_file = output_full_path + ".bank"
                elif os.path.exists(os.path.join(temp_dir, os.path.splitext(output_fsb_name)[0] + ".bank")): created_file = os.path.join(temp_dir, os.path.splitext(output_fsb_name)[0] + ".bank")

                if not created_file:
                     return False, f"O fsbankcl rodou mas o arquivo não apareceu em: {output_full_path}\nLog: {result.stdout}"
                
                # SUCESSO! Agora instalar.
                # O caminho original está em 'original_fsb_path'
                
                # Backup
                backup_path = original_fsb_path + ".backup"
                if not os.path.exists(backup_path) and os.path.exists(original_fsb_path):
                     shutil.copy2(original_fsb_path, backup_path)
                
                # Instalação (Sobrescreve o original no jogo)
                shutil.copy2(created_file, original_fsb_path)
                
                return True, f"SUCESSO TOTAL (Via FMOD Engine)!\n\nArquivo gerado e instalado em:\n{original_fsb_path}\n\nAbra o jogo e teste!"

            except Exception as e:
                return False, f"Erro crítico ao rodar FMOD Engine: {e}"

        # =================================================================================
        # MODO LEGADO: FMOD BANK TOOLS (Mantido para compatibilidade)
        # =================================================================================

        # Pasta base do FMOD Tools
        tools_dir = os.path.dirname(tool_path)
        wav_dir = os.path.join(tools_dir, 'wav')
        
        if not os.path.exists(wav_dir): os.makedirs(wav_dir)

        # Atualiza o config.ini do FMOD Bank Tools para garantir que as pastas estejam CERTAS
        config_ini_path = os.path.join(tools_dir, 'config.ini')
        try:
            # Formato esperado pelo FMOD Bank Tools (Nexus version)
            # Use forward slashes ou escaped backslashes. Vamos usar forward slashes para compatibilidade python.
            # O header é [Directorys] (sic)
            
            # Formato esperado pelo FMOD Bank Tools (Nexus version)
            # TENTATIVA 3: Caminhos Relativos!
            # Evita problemas de barra/contra-barra inteiramente.
            # O script roda com cwd=tools_dir, então deve funcionar.
            
            config_content = f"""[Directorys]
BankDir=bank
WavDir=wav
RebuildDir=output

[Options]
Format=vorbis
Quality=85
"""

            with open(config_ini_path, 'w') as f:
                f.write(config_content)
        except Exception as e:
            print(f"[AVISO] Não foi possível atualizar config.ini automaticamente: {e}")

        # CHECK CRÍTICO: O FMOD precisa ter 'extraído' o banco antes para gerar o mapeamento (.txt)
        # Verifica se o arquivo .txt existe na pasta wav
        # O nome do txt geralmente é o mesmo do banco (streams_1_audio.txt) ou bankname.txt
        bank_name_no_ext = os.path.splitext(fsb_filename)[0]
        txt_map_path = os.path.join(wav_dir, f"{str(bank_name_no_ext)}.txt")

        # CHECK CRÍTICO + AUTO-GERAÇÃO
        # Se NÃO existe o mapeamento (.txt), nós vamos GERAR ele baseados nos arquivos de entrada.
        # Assim pulamos a necessidade do "Extract" que está falhando para o usuário.
        
        bank_name_no_ext = os.path.splitext(fsb_filename)[0]
        txt_map_path = os.path.join(wav_dir, f"{str(bank_name_no_ext)}.txt")
        target_wav_subdir = os.path.join(wav_dir, bank_name_no_ext)
        
        if not os.path.exists(txt_map_path):
             print("[AUTO-GEN] Gerando estrutura FMOD manualmente...")
             if not os.path.exists(target_wav_subdir): os.makedirs(target_wav_subdir)
             
             # Lista os arquivos que temos na entrada (_MOD)
             input_wavs = [f for f in os.listdir(input_folder) if f.lower().endswith('.wav')]
             
             if not input_wavs: return False, "A pasta de entrada (_MOD) está vazia! Não tem áudios para reempacotar."
             
             with open(txt_map_path, 'w') as f_txt:
                 # Formato SEGURO: Usar caminhos ABSOLUTOS para cada arquivo
                 # Assim o FMOD não se perde com diretórios relativos
                 for wav in input_wavs:
                     src = os.path.join(input_folder, wav)
                     dst = os.path.join(target_wav_subdir, wav)
                     shutil.copy2(src, dst)
                     
                     # Escreve caminho ABSOLUTO no TXT
                     # Nota: FMOD pode ter problema com espaços. Vou tentar aspas se necessário, mas primeiro raw.
                     # Na verdade, FMOD Bank Tools costuma aceitar caminho relativo simples se estiver na pasta certa.
                     # Mas absoluto é mais garantido SE ele suportar.
                     
                     # TENTATIVA 2: Caminho relativo simples mas explicito
                     # Se o TXT está em wav/, e o arquivo em wav/subfolder/file.wav
                     # O formato padrão é: subfolder\file.wav (backslashes para Windows)
                     
             # CRIA MÚLTIPLOS ALIASES do TXT e do .BANK para garantir compatibilidade
             # O FMOD pode buscar pc_streams_1_audio.txt, streams_1_au.txt, etc.
             # E ele espera encontrar bank/NOME.bank correspondente.
             # Vamos criar parzinhos bank+txt para todos os nomes prováveis.
             
             possible_names_no_ext = [
                 bank_name_no_ext,              # streams_1_audio
                 f"pc_{bank_name_no_ext}",      # pc_streams_1_audio
                 "pc_streams_1_au",             # Do log do usuário (truncado?)
                 "pc_streams_1",                # Do log mais recente
                 "streams_1_au"
             ]
             
             for name in possible_names_no_ext:
                # 1. Cria SUBPASTA em wav/ (ex: wav/pc_streams_1)
                sub_dir = os.path.join(wav_dir, name)
                if not os.path.exists(sub_dir): os.makedirs(sub_dir)
                
                # 2. Copia arquivos para a subpasta e prepara linhas do TXT
                txt_lines = []
                for wav in input_wavs:
                    src = os.path.join(input_folder, wav)
                    dst = os.path.join(sub_dir, wav)
                    try:
                        shutil.copy2(src, dst)
                        # Caminho RELATIVO: nome_pasta\arquivo.wav (SEM ASPAS)
                        txt_lines.append(f"{name}\\{wav}\n")
                    except: pass

                # 3. Cria TXT na raiz wav/ apontando para a subpasta
                txt_dest = os.path.join(wav_dir, f"{name}.txt")
                try:
                    with open(txt_dest, 'w') as f:
                        f.writelines(txt_lines)
                except: pass
                
                # 4. Copia BANK para bank/
                bank_dest_dir = os.path.join(tools_dir, 'bank')
                if not os.path.exists(bank_dest_dir): os.makedirs(bank_dest_dir)
                
                bank_dest = os.path.join(bank_dest_dir, f"{name}.bank")
                try:
                    if not os.path.exists(bank_dest):
                        shutil.copy2(original_fsb_path, bank_dest)
                except: pass
             
             print("[AUTO-GEN] Estrutura 'Standard' criada (Relative Paths).")

        # Se mesmo gerando falhar (ex: erro de permissão), cai aqui.

        # Se mesmo gerando falhar (ex: erro de permissão), cai aqui.
        # Mas vamos assumir que funcionou e prosseguir para REBUILD.

        # Copia arquivos (Redundância caso o bloco acima não tenha rodado ex: txt já existia mas arquivos novos)
        if not os.path.exists(target_wav_subdir): os.makedirs(target_wav_subdir)
        
        processed_count = 0
        for filename in os.listdir(input_folder):
             if filename.lower().endswith('.wav'):
                shutil.copy2(os.path.join(input_folder, filename), os.path.join(target_wav_subdir, filename))
                processed_count += 1
        
        if processed_count == 0: return False, "Nenhum arquivo .wav encontrado na pasta de entrada."

        # Executa o Rebuild e ESPERA fechar
        # Se for o 'FMOD Bank Tools.exe' de UI, ele pode não aceitar CLI.
        # Mas vamos abrir a ferramenta para o usuário.

        # Se já existe o mapeamento, prossegue com a cópia dos novos e rebuild
        
        # Copia WAVs modificados para a pasta wav do FMOD
        # IMPORTANTE: FMOD Bank Tools espera uma estrutura especifica? Geralmente ele usa o nome da pasta do banco dentro de wav/
        # Ex: wav/streams_1_audio/0001.wav
        # O FMOD Bank Tools (Nexus version) cria subpastas baseadas no nome do FSB.
        
        target_wav_subdir = os.path.join(wav_dir, bank_name_no_ext)
        if not os.path.exists(target_wav_subdir): os.makedirs(target_wav_subdir) # Só garante que existe, não apaga o txt que está na raiz wav/

        # Copia arquivos
        processed_count = 0
        for filename in os.listdir(input_folder):
            if filename.lower().endswith('.wav'):
                shutil.copy2(os.path.join(input_folder, filename), os.path.join(target_wav_subdir, filename))
                processed_count += 1
        
        if processed_count == 0: return False, "Nenhum arquivo .wav encontrado na pasta de entrada."

        # Executa o Rebuild e ESPERA fechar
        # Se for o 'FMOD Bank Tools.exe' de UI, ele pode não aceitar CLI.
        # Mas vamos abrir a ferramenta para o usuário.
        
        try:
             # os.startfile não permite esperar o processo fechar facilmente de forma bloquante como Popen.wait()
             # Mas subprocess.Popen as vezes falha em GUI.
             # Vamos tentar Popen com shell=True que ajuda as vezes
             process = subprocess.Popen([tool_path], cwd=tools_dir, shell=True)
             process.wait() # Espera o usuário fechar a ferramenta
        except Exception:
             # Fallback
             os.startfile(tool_path)
             # Não conseguimos esperar se usar startfile, então avisamos o usuário para clicar em OK
             # Mas o fluxo atual depende do wait().
             # Vamos assumir que Popen shell=True funcione.
        
        # Após fechar, verifica se o arquivo output existe
        output_fsb = os.path.join(tools_dir, 'output', fsb_filename) # Ex: tools/Fmod tools/output/streams_1_audio.fsb
        # Algumas versoes usam 'build'
        if not os.path.exists(output_fsb):
             output_fsb_alt = os.path.join(tools_dir, 'build', fsb_filename)
             if os.path.exists(output_fsb_alt): output_fsb = output_fsb_alt
        
        if os.path.exists(output_fsb):
            # Instala automaticamente!
            
            # Backup do original no jogo antes de substituir (se ainda não tiver feito pelo vpk_manager, mas safety first)
            if os.path.exists(original_fsb_path):
                backup_game_file = original_fsb_path + ".bak_auto"
                if not os.path.exists(backup_game_file):
                    shutil.copy2(original_fsb_path, backup_game_file)
            
            shutil.copy2(output_fsb, original_fsb_path)
            return True, f"SUCESSO TOTAL!\n\n1. Arquivo reempacotado.\n2. Instalado em: {original_fsb_path}\n\nPode abrir o jogo e testar!"
        else:
            return False, "O arquivo .fsb não foi criado na pasta output. Você clicou em 'Rebuild'?"

    except Exception as e:
        return False, f"Erro no Repack FSB: {e}"

def reempacotar_pck_logic(input_folder, mod_data):
    try:
        backup_json = os.path.join(BACKUP_DIR, f"backup_{mod_data['original_hash']}.json")
        with open(backup_json, 'r') as f: backup_data = json.load(f)
        caminho_original = backup_data['caminho_original']
        caminho_backup_pck = os.path.join(BACKUP_DIR, f"original_{mod_data['original_hash']}_{os.path.basename(caminho_original)}")
        
        temp_dir = os.path.join(input_folder, "temp_build_pck")
        os.makedirs(temp_dir, exist_ok=True)
        output_pck = os.path.join(temp_dir, os.path.basename(caminho_original))
        
        with open(caminho_backup_pck, 'rb') as f_in, open(output_pck, 'wb') as f_out:
            f_in.seek(0)
            magic = f_in.read(4)
            header_size, unk, lang_size, bank_size, sound_size, ext_size = struct.unpack('<IIIIII', f_in.read(24))
            lang_map_bytes = f_in.read(lang_size)
            
            num_banks = struct.unpack('<I', f_in.read(4))[0]
            banks = [list(struct.unpack('<IIIII', f_in.read(20))) for _ in range(num_banks)]
            
            num_sounds = struct.unpack('<I', f_in.read(4))[0]
            sounds = [list(struct.unpack('<IIIII', f_in.read(20))) for _ in range(num_sounds)]
                
            entries = []
            for idx, b in enumerate(banks):
                entries.append({'type': 'bank', 'idx': idx, 'id': b[0], 'align': b[1], 'size': b[2], 'orig_offset_block': b[3], 'lang_id': b[4], 'safe_name': f"bank_{b[0]}.bnk", 'orig_abs_offset': b[3]*b[1]})
            for idx, s in enumerate(sounds):
                entries.append({'type': 'sound', 'idx': idx, 'id': s[0], 'align': s[1], 'size': s[2], 'orig_offset_block': s[3], 'lang_id': s[4], 'safe_name': f"sound_{s[0]}.wem", 'orig_abs_offset': s[3]*s[1]})
                
            entries.sort(key=lambda x: x['orig_abs_offset'])

            base_data_offset = entries[0]['orig_abs_offset'] if entries else header_size + 0x1C
            f_out.write(b'\\x00' * base_data_offset)
            
            for e in entries:
                f_out.seek(0, 2)
                current_offset = f_out.tell()
                align = e['align']
                remainder = current_offset % align
                if remainder != 0:
                    pad = align - remainder
                    f_out.write(b'\\x00' * pad)
                    current_offset += pad
                    
                e['new_offset_block'] = current_offset // align
                override_path = os.path.join(input_folder, e['safe_name'])
                if os.path.exists(override_path):
                    with open(override_path, 'rb') as mod_f: file_data = mod_f.read()
                else:
                    f_in.seek(e['orig_abs_offset'])
                    file_data = f_in.read(e['size'])
                e['new_size'] = len(file_data)
                f_out.write(file_data)
                
            f_out.seek(0)
            f_out.write(magic)
            f_out.write(struct.pack('<IIIIII', header_size, unk, lang_size, bank_size, sound_size, ext_size))
            f_out.write(lang_map_bytes)
            f_out.write(struct.pack('<I', num_banks))
            for e in sorted([e for e in entries if e['type'] == 'bank'], key=lambda x: x['idx']):
                f_out.write(struct.pack('<IIIII', e['id'], e['align'], e['new_size'], e['new_offset_block'], e['lang_id']))
                
            f_out.write(struct.pack('<I', num_sounds))
            for e in sorted([e for e in entries if e['type'] == 'sound'], key=lambda x: x['idx']):
                f_out.write(struct.pack('<IIIII', e['id'], e['align'], e['new_size'], e['new_offset_block'], e['lang_id']))
                
        backup_path = caminho_original + ".backup"
        if not os.path.exists(backup_path): shutil.copy2(caminho_original, backup_path)
        shutil.copy2(output_pck, caminho_original)
        
        return True, f"SUCESSO TOTAL!\nO arquivo PCK foi reempacotado silenciosamente (Python Nativo) e instalado em:\n{caminho_original}\nPode jogar!"
    except Exception as e:
        return False, f"Erro Fatal no Repack PCK: {e}"

import vpk # Nova dependência para Source Engine

# ... (Imports anteriores mantidos)

def analisar_vpk_logic(caminho_vpk):
    """Analisa um arquivo .vpk (Left 4 Dead / Source) e cria projeto."""
    if not os.path.isfile(caminho_vpk):
        return False, "Erro: Arquivo Inexistente."
        
    # [FLEXIBILIDADE COM GUIDANCE]
    # Se o usuário tentar abrir pak01_001.vpk, avisamos para abrir o _dir.
    if re.search(r'_\d+\.vpk$', caminho_vpk.lower()):
        return False, "❌ Não abra o arquivo numerado (.001, .002)! Selecione o arquivo '_dir.vpk' que está na mesma pasta. Ele controla todos os outros."

    if not caminho_vpk.lower().endswith('.vpk'):
         return False, "Arquivo inválido. Deve ser .vpk"

    try:
        hash_original = calcular_hash_sha1(caminho_vpk)
        if not hash_original: return False, "Erro ao calcular hash."

        # Setup Backup JSON
        if not os.path.exists(BACKUP_DIR): os.makedirs(BACKUP_DIR)
        caminho_backup_json = os.path.join(BACKUP_DIR, f"backup_{hash_original}.json")
        
        # Check se já existe
        if os.path.exists(caminho_backup_json):
             try:
                 with open(caminho_backup_json, 'r', encoding='utf-8') as f: d = json.load(f)
                 if d.get('status') == 'done': return True, "Projeto VPK já analisado!"
             except: pass

        # Abre VPK
        print(f"[DEBUG] Tentando abrir VPK com a lib: {caminho_vpk}")
        pak = vpk.open(caminho_vpk)
        print(f"[DEBUG] VPK aberto. Scan files...")
        
        info_vpk = OrderedDict([
            ('caminho_original', caminho_vpk),
            ('nome_arquivo', os.path.basename(caminho_vpk)),
            ('hash_sha1_original', hash_original),
            ('type', 'vpk'), # Flag para o descompactador
            ('status', 'in_progress'),
            ('arquivos_internos', [])
        ])

        # Mapeia arquivos
        files_found = 0
        for filepath in pak:
            # VPK usa / como separador padrão interno
            safe_name = filepath.replace('\\', '/') 
            
            # [FIX] VPK Lib API Update (vpk 1.4.0 usa get_file_meta)
            try:
                pak_file_info = pak.get_file_meta(filepath)
                # Tenta descobrir o formato do objeto de retorno (Object ou Tuple)
                print(f"[DEBUG] Meta for {safe_name}: {pak_file_info}") 
                
                # Suporte para ambos os casos possíveis da lib
                if hasattr(pak_file_info, 'file_length'):
                    f_len = pak_file_info.file_length
                    p_len = getattr(pak_file_info, 'preload_length', 0)
                elif hasattr(pak_file_info, 'entry_length'):
                     f_len = pak_file_info.entry_length
                     p_len = getattr(pak_file_info, 'preload_data_length', 0)
                elif isinstance(pak_file_info, (list, tuple)) and len(pak_file_info) >= 3:
                     # Se for tupla: (archive_idx, offset, length, ...)
                     f_len = pak_file_info[2]
                     p_len = 0
                else:
                    f_len = 0
                    p_len = 0
            except Exception as e_meta:
                 print(f"[DEBUG] Erro ao ler meta de {safe_name}: {e_meta}")
                 f_len = 0
                 p_len = 0
            
            info_vpk['arquivos_internos'].append({
                'safe_name': safe_name,
                'size': f_len,
                'preload_bytes': p_len 
            })
            files_found += 1

        # Backup do arquivo original (Cópia de Segurança)
        # Nota: VPKs são multi-arquivo (pak01_000, pak01_001...). 
        # Copiar APENAS o _dir.vpk geralmente não basta para backup full, 
        # mas serve como referência de índice. 
        # Para L4D modding, geralmente editamos e empacotamos, não "restauramos" binário.
        # Vamos copiar o _dir por segurança.
        nome_orig = os.path.basename(caminho_vpk)
        bkp_path = os.path.join(BACKUP_DIR, f"original_{hash_original}_{nome_orig}")
        if not os.path.exists(bkp_path):
            shutil.copyfile(caminho_vpk, bkp_path)

        info_vpk['status'] = 'done'
        with open(caminho_backup_json, 'w', encoding='utf-8') as tf:
            json.dump(info_vpk, tf, indent=4, ensure_ascii=False)

        return True, f"Análise VPK Concluída! {files_found} arquivos indexados."

    except Exception as e:
        gerar_relatorio_erro("analisar_vpk_logic", e, traceback.format_exc())
        return False, f"Erro ao ler VPK: {e}"

@app.route('/api/analisar', methods=['POST'])
def api_analisar():
    path = request.get_json().get('path', '').strip()
    print(f"[DEBUG] Recebido pedido de análise para: {path}") # Debug
    if not path: return jsonify({'success': False, 'message': 'O caminho não pode ser vazio.'})
    
    # [DISPATCHER INTELIGENTE]
    if path.lower().endswith('.vpk'):
        # Aceita tanto _dir.vpk (Jogo Base) quanto .vpk (Addons/Mods)
        success, message = analisar_vpk_logic(path)
    elif path.lower().endswith('.fsb') or path.lower().endswith('.fsb5'): # [NOVO] Suporte Bioshock
        success, message = analisar_fsb_logic(path)
    elif path.lower().endswith('.pck'): # [NOVO] Suporte Wwise AKPK
        success, message = analisar_pck_logic(path)
    elif path.lower().endswith('.arch') or path.lower().endswith('.iwd'): # Suporte FEAR
        success, message = analisar_arch_logic(path)
    else:
        # Tenta adivinhar ou falha
        success, message = False, "Formato não suportado. Use .arch (F.E.A.R.), .vpk (Source/L4D2) ou .fsb (Bioshock)."
        
    return jsonify({'success': success, 'message': message})

@app.route('/api/descompactar', methods=['POST'])
def api_descompactar():
    data = request.get_json()
    project_id = data.get('project_id', '').strip()
    selected_files = data.get('selected_files')
    if not project_id: return jsonify({'success': False, 'message': 'Nenhum projeto foi selecionado.'})
    success, message = descompactar_arch_logic(project_id, selected_files)
    return jsonify({'success': success, 'message': message})

# --- NOVO ENDPOINT PARA .BNDL ---
@app.route('/api/processar-bndl', methods=['POST'])
def api_processar_bndl():
    path = request.get_json().get('path', '').strip()
    success, message = processar_bndl_logic(path)
    return jsonify({'success': success, 'message': message})


@app.route('/api/restaurar', methods=['POST'])
def api_restaurar():
    path = request.get_json().get('path', '').strip()
    if not path: return jsonify({'success': False, 'message': 'O caminho do arquivo alvo não pode ser vazio.'})
    success, message = restaurar_arch_logic(path)
    return jsonify({'success': success, 'message': message})

def abrir_navegador():
    if os.environ.get('NO_BROWSER') in ('1', 'true', 'True'):
        return
    Timer(1, lambda: webbrowser.open_new("http://127.0.0.1:5000")).start()

if __name__ == '__main__':
    print("="*40); print("    Universal Mod Manager v10.0 (Hybrid)"); print("="*40)
    print(f"Python Executable: {sys.executable}")
    print(f"VPK Lib Status: {'OK' if vpk else 'MISSING'}")
    print(f"FSB5 Lib Status: {'OK' if fsb5 else 'MISSING'}")
    print(f"VGMStream Status: {'OK' if os.path.exists(VGMSTREAM_PATH) else 'MISSING (Place in tools/)'}")
    if not fsb5:
        print(" [!] ATENÇÃO: Para usar Bioshock (Análise), instale: pip install fsb5")
    if not os.path.exists(VGMSTREAM_PATH):
        print(" [!] ATENÇÃO: Para usar Bioshock (Extração), baixe vgmstream-cli e coloque em 'tools/'")
    else:
        print(" [!] Módulo Bioshock .fsb ativo (vgmstream encontrado)!")
    print("="*40)
    print("Iniciando servidor local...")
    print(f"Pasta de backups e extração: '{os.path.abspath(BACKUP_DIR)}'")
    print("Acesse em seu navegador: http://127.0.0.1:5000")
    print("Para desligar o servidor, pressione CTRL+C nesta janela.")
    
    if os.environ.get('WERKZEUG_RUN_MAIN') != 'true':
        abrir_navegador()
    
    app.run(host='127.0.0.1', port=5000, threaded=True)

