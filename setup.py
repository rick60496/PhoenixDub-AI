import os
import subprocess
import sys
import shutil

# Detecta o diretório atual do script para ser flexível com renomeios de pasta
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, "env")

def print_header():
    print("\n" + "="*65)
    print(" 🛠️  PHOENIXDUB AI - GERENCIADOR DE AMBIENTE (v2026.1) 🛠️ ")
    print("="*65)

def run_cmd(cmd):
    subprocess.run(cmd, check=True, shell=True)

def check_nvidia_gpu():
    """Detecta se existe uma placa NVIDIA com pelo menos 4GB de VRAM."""
    try:
        # Pergunta o total de memória da GPU em MB
        cmd = "nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits"
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, text=True)
        if res.returncode == 0:
            vram = int(res.stdout.strip())
            if vram >= 4000: # Exige no mínimo 4GB para o Modo Turbo
                return True
    except:
        pass
    return False

def check_conda():
    try:
        subprocess.run(["conda", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    except Exception:
        print("Instale o Anaconda/Miniconda e marque 'Add to PATH' durante a instalação.")
        sys.exit(1)

def check_git():
    """Verifica se o Git está instalado, necessário para baixar pacotes do GitHub."""
    try:
        subprocess.run(["git", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        return True
    except Exception:
        print("\n" + "!"*65)
        print(" ⚠️ ALERTA: GIT NÃO ENCONTRADO! ⚠️")
        print(" O Chatterbox exige o Git para ser instalado.")
        print(" Baixe e instale agora em: https://git-scm.com/download/win")
        print(" Após instalar, feche e abra este terminal novamente.")
        print("!"*65)
        sys.exit(1)

def check_espeak_ng():
    """Verifica se o eSpeak-NG está instalado, essencial para o motor de voz."""
    # Tenta encontrar no PATH ou em locais padrão do Windows
    paths_to_check = [
        shutil.which("espeak-ng"),
        os.path.join(os.environ.get("ProgramFiles", "C:\\Program Files"), "eSpeak NG", "espeak-ng.exe"),
        os.path.join(os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)"), "eSpeak NG", "espeak-ng.exe")
    ]
    
    if any(p and os.path.exists(p) for p in paths_to_check):
        return True
        
    print("\n" + "!"*65)
    print(" ⚠️ ALERTA: eSpeak-NG NÃO ENCONTRADO! ⚠️")
    print(" O Chatterbox exige o eSpeak-NG para funcionar.")
    print("-" * 65)
    
    choice = input(" [?] Deseja tentar a INSTALAÇÃO AUTOMÁTICA via winget? (s/n): ").strip().lower()
    
    if choice == 's':
        print("\n[+] Iniciando instalação do eSpeak-NG via winget...")
        print("[!] Uma janela de permissão do Windows (UAC) pode aparecer.")
        try:
            subprocess.run(["winget", "install", "eSpeak-NG.eSpeak-NG", "--accept-source-agreements", "--accept-package-agreements"], check=True, shell=True)
            print("[+] Instalação concluída com sucesso!")
            return True
        except Exception as e:
            print(f"❌ Falha na instalação automática: {e}")
            print(" Por favor, instale manualmente o 'espeak-ng-X64.msi' em:")
            print(" https://github.com/espeak-ng/espeak-ng/releases")
    else:
        print("\n[!] Instalação manual necessária.")
        print(" Baixe o MSI em: https://github.com/espeak-ng/espeak-ng/releases")
    
    input("\n Pressione Enter para CONTINUAR após a instalação...")
    return False

def accept_conda_tos():
    """Tenta aceitar os termos de serviço do Conda automaticamente."""
    print("[+] Aceitando termos de serviço do Conda...")
    channels = [
        "https://repo.anaconda.com/pkgs/main",
        "https://repo.anaconda.com/pkgs/r",
        "https://repo.anaconda.com/pkgs/msys2"
    ]
    for channel in channels:
        try:
            subprocess.run(f"conda tos accept --override-channels --channel {channel}", 
                           shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except:
            pass

def download_onnx_model():
    """Baixa o modelo Chatterbox ONNX automaticamente do Hugging Face."""
    target_dir = os.path.join(BASE_DIR, "models", "chatterbox_onnx")
    tokenizer_check = os.path.join(target_dir, "tokenizer.json")
    
    if os.path.exists(tokenizer_check):
        print("\n[+] Motor ONNX já encontrado em 'models/chatterbox_onnx/'. Pulando download.")
        return

    print("\n" + "*"*65)
    print(" 🧬 BAIXANDO MOTOR DE IA ACELERADO (ONNX - v2026) 🧬")
    print(" Isso vai garantir a performance máxima na sua RTX.")
    print("⏳ Download de aprox. 2GB iniciado... Por favor, aguarde.")
    print("*"*65)
    
    try:
        # Tenta usar o python do ambiente para baixar (pois já tem huggingface-hub)
        python_exe = os.path.join(ENV_PATH, "python.exe")
        repo_id = "onnx-community/chatterbox-multilingual-ONNX"
        
        # [FIX] Desativa o acelerador Rust que causa erro de Hardware em algumas CPUs
        env_vars = os.environ.copy()
        env_vars["HF_HUB_DISABLE_FAST_HF_TRANSFER"] = "1"
        
        # Script inline para usar o snapshot_download
        cmd = f'"{python_exe}" -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id=\'{repo_id}\', local_dir=\'{target_dir}\', local_dir_use_symlinks=False)"'
        subprocess.run(cmd, shell=True, check=True, env=env_vars)
        print("\n✅ Motor ONNX baixado e configurado com sucesso!")
    except Exception as e:
        print(f"\n⚠️ Falha no download automático do ONNX: {e}")
        print(" [!] O erro fatal de hardware foi contornado, mas o download falhou por outro motivo.")
        print(" O programa ainda funcionará no modo padrão (PyTorch).")
        print(" Você pode baixar manualmente em: https://huggingface.co/onnx-community/chatterbox-multilingual-ONNX")

def download_chatterbox_official():
    """Baixa o modelo Chatterbox Padrão/Oficial automaticamente do Hugging Face."""
    target_dir = os.path.join(ENV_PATH, "models", "chatterbox_official")
    
    # Se existe algum arquivo lá (ex: config.json or pytorch_model.bin), já pulamos
    if os.path.exists(target_dir) and len(os.listdir(target_dir)) > 2:
        print("\n[+] Vozes Oficiais já encontradas em 'env/models/chatterbox_official/'. Pulando download.")
        return

    print("\n" + "*"*65)
    print(" 🔊 BAIXANDO VOZES OFICIAIS DO CHATTERBOX 🔊")
    print(" Isso vai baixar os arquivos de dublagem necessários.")
    print("⏳ Por favor, aguarde...")
    print("*"*65)
    
    try:
        os.makedirs(target_dir, exist_ok=True)
        python_exe = os.path.join(ENV_PATH, "python.exe")
        repo_id = "ResembleAI/chatterbox"
        
        env_vars = os.environ.copy()
        env_vars["HF_HUB_DISABLE_FAST_HF_TRANSFER"] = "1"
        
        cmd = f'"{python_exe}" -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id=\'{repo_id}\', local_dir=r\'{target_dir}\', local_dir_use_symlinks=False)"'
        subprocess.run(cmd, shell=True, check=True, env=env_vars)
        print("\n✅ Vozes oficiais do Chatterbox baixadas com sucesso!")
    except Exception as e:
        print(f"\n⚠️ Falha no download automático do Chatterbox Oficial: {e}")
        print(f" Você terá que baixar ou copiar a pasta manualmente para: {target_dir}")

def pre_install_fix():
    """Corrige problemas de packaging e pip antes da instalação pesada."""
    print("\n[+] Aplicando Correções de Base (Pip, Packaging, Wheel)...")
    python_exe = os.path.join(ENV_PATH, "python.exe")
    trusted_flags = "--trusted-host pypi.org --trusted-host files.pythonhosted.org --trusted-host download.pytorch.org"
    try:
        run_cmd(f'"{python_exe}" -m pip install --upgrade pip setuptools wheel {trusted_flags}')
    except:
        pass

def repair_env(has_gpu_selected=True):
    """Repara um ambiente existente instalando apenas o que falta."""
    print("\n" + "="*65)
    print(" 🛠️  MODO REPARO RÁPIDO ATIVADO 🛠️ ")
    print("="*65)
    
    pre_install_fix()
    
    req_file = "requirements_RTX.txt" if has_gpu_selected else "requirements_CPU.txt"
    trusted_flags = "--trusted-host pypi.org --trusted-host files.pythonhosted.org --trusted-host download.pytorch.org"
    python_exe = os.path.join(ENV_PATH, "python.exe")
    
    print(f"\n[+] Atualizando dependências de: {req_file}")
    run_cmd(f'"{python_exe}" -m pip install -r {req_file} {trusted_flags}')

    # [BYPASS] Motor de IA via LM Studio (Evita erros de compilação no PC do colega)
    print("\n[+] Configurando ponte para LM Studio (Gemma 4)...")

    # [FIX 2026] Trava do Numpy para evitar quebra de áudio e adição do Numba
    print("\n[+] Aplicando trava de compatibilidade (Numpy + Numba)...")
    run_cmd(f'"{python_exe}" -m pip install "numpy<2.0.0" numba')
    
    print("\n[+] Re-instalando Motor de Voz (Modo de Convivência)...")
    # Forçamos a convivência pacífica dos pacotes que causam conflitos na internet
    pkgs = "chatterbox-tts resemble-perth s3tokenizer diffusers==0.29.0 conformer==0.3.2 spacy-pkuseg pykakasi==2.3.0 pyloudnorm omegaconf safetensors==0.7.0"
    repair_cmd = f'"{python_exe}" -m pip install {pkgs} --force-reinstall --no-deps {trusted_flags}'
    run_cmd(repair_cmd)
    
    download_chatterbox_official()
    if has_gpu_selected:
        download_onnx_model()
        
    print("\n✅ REPARO CONCLUÍDO COM SUCESSO!")
    print(" O ambiente está atualizado e pronto para o Gemma 4.")

def install_env(has_gpu_selected=True):
    check_git() # Verificação proativa do Git
    check_espeak_ng() # Verificação proativa do eSpeak-NG
    accept_conda_tos()
    
    print("\n" + "-"*65)
    if has_gpu_selected:
        print(" ✨ CONFIGURAÇÃO SELECIONADA: MODO TURBO (NVIDIA RTX) ✨")
        print(" [MODO TURBO] O sistema será instalado para ALTA PERFORMANCE.")
        req_file = "requirements_RTX.txt"
    else:
        print(" 💻 CONFIGURAÇÃO SELECIONADA: MODO PADRÃO (CPU) 💻")
        print(" [MODO ESTÁVEL] O sistema será instalado para PROCESSADORES.")
        req_file = "requirements_CPU.txt"
    print("-"*65)

    if not os.path.exists(req_file):
        print(f"❌ ERRO: Arquivo {req_file} não encontrado na pasta!")
        return

    print(f"\n[+] Criando ambiente virtual em: {ENV_PATH}")
    try:
        # Força Python 3.10 para evitar conflitos conhecidos
        run_cmd(f"conda create --prefix {ENV_PATH} python=3.10 -y")
        
        # [FIX] Aplica correções de base antes de mais nada
        pre_install_fix()
        
        print("\n[+] Instalando Ferramentas de Base (FFmpeg & TK)...")
        run_cmd(f"conda install --prefix {ENV_PATH} ffmpeg tk -y")
        
        print("\n[+] Liberando comandos de Login (HuggingFace)...")
        # Instala uma versão compatível com transformers 4.46.3 (>= 1.0.0)
        trusted_flags = "--trusted-host pypi.org --trusted-host files.pythonhosted.org --trusted-host download.pytorch.org"
        run_cmd(f'"{ENV_PATH}\\python.exe" -m pip install huggingface-hub --upgrade {trusted_flags}')
        
        print("\n[+] Instalando Motor de Voz (Isolado)...")
        pkgs = "chatterbox-tts resemble-perth s3tokenizer diffusers==0.29.0 conformer==0.3.2 spacy-pkuseg pykakasi==2.3.0 pyloudnorm omegaconf safetensors==0.7.0 sentencepiece"
        run_cmd(f'"{ENV_PATH}\\python.exe" -m pip install {pkgs} --no-deps {trusted_flags}')
        
        if has_gpu_selected:
            download_onnx_model()
            
        print("\n" + "="*65)
        print(" 🔑 AGORA VOCÊ PODE FAZER O LOGIN! 🔑")
        print(" O comando 'huggingface-cli login' já estará disponível.")
        print("*"*65)

        print(f"\n[+] ✨ INSTALANDO CÉREBRO DE IA (Versão 2026 Modern) ✨")
        print(f"[Arquivo: {req_file}]")
        print("⏳ Este passo é largo. A barra de progresso será exibida abaixo...")
        run_cmd(f'"{ENV_PATH}\\python.exe" -m pip install -r {req_file} {trusted_flags}')
        
        # [CONFLITO RESOLVIDO] Instala o Chatterbox separadamente
        print("\n[+] 🔊 Finalizando: Instalando Motor de Voz (Chatterbox v0.1.7)...")
        chatterbox_cmd = f'"{ENV_PATH}\\python.exe" -m pip install git+https://github.com/resemble-ai/chatterbox.git --no-deps {trusted_flags}'
        run_cmd(chatterbox_cmd)

        download_chatterbox_official()
        if has_gpu_selected:
            # Baixa o modelo ONNX automaticamente para Modo Turbo
            download_onnx_model()
        
        print("\n" + "="*65)
        print("🤖 FASE FINAL: CONFIGURAÇÃO DO GEMA 4 (LM STUDIO) 🤖")
        print("="*65)
        print("\n1. Baixe o LM Studio em: https://lmstudio.ai")
        print("2. Pesquise por: unsloth/gemma-4-E4B-it-GGUF")
        print("3. Vá em 'Local Server' e clique em 'Start Server' (Porta 1234)")
        
        print("\n⚡ [OPCIONAL] ACELERAÇÃO ONNX ATIVADA! ⚡")
        print(" Para usar a versão ONNX (Vantagem para RTX 2060):")
        print(" Baixe os arquivos em: https://huggingface.co/onnx-community/chatterbox-multilingual-ONNX")
        print(" Coloque os arquivos .onnx e tokenizer.json na pasta 'models/chatterbox_onnx/'")
        
        print("\n" + "="*65)
        print("✅ AMBIENTE 2026 PRONTO! ✅")
        print("="*65)
        print(f"\nPara rodar:")
        print(f"👉 1. conda activate {ENV_PATH}")
        print(f"👉 2. python app_jogos.py  ou  python App_videos.py\n")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Erro crítico na instalação: {e}")

def delete_env():
    print(f"\n[!] Removendo o ambiente virtual em: {ENV_PATH}")
    try:
        if os.path.exists(ENV_PATH):
            shutil.rmtree(ENV_PATH, ignore_errors=True)
            print("[+] Ambiente deletado com sucesso.")
        else:
            print("[?] O ambiente já não existia no local indicado.")
    except Exception as e:
        print(f"❌ Erro ao deletar: {e}")

def hf_login():
    """Executa o login do HuggingFace de dentro do ambiente virtual."""
    # Procura scripts tanto em /bin quanto em /Scripts (Windows)
    hf_cli = os.path.join(ENV_PATH, "Scripts", "huggingface-cli.exe")
    if not os.path.exists(hf_cli):
        print("\n❌ ERRO: O ambiente ainda não foi instalado ou o hf-cli sumiu.")
        input("\nPressione Enter para voltar...")
        return
        
    print("\n" + "="*65)
    print(" 🔑 LOGIN HUGGINGFACE 🔑")
    print(" 1. Acesse: https://huggingface.co/settings/tokens")
    print(" 2. Crie um token (tipo READ) e copie.")
    print(" 3. Cole o token abaixo (não aparecerá nada ao colar, é normal).")
    print("="*65)
    try:
        subprocess.run(f'"{hf_cli}" login', shell=True)
        print("\n✅ Login processado!")
    except Exception as e:
        print(f"❌ Erro ao tentar login: {e}")
    input("\nPressione Enter para voltar ao menu...")

def main():
    check_conda()
    # Detecta GPU apenas uma vez para performance
    has_gpu = check_nvidia_gpu()
    status_gpu = "🚀 NVIDIA RTX DETECTADA" if has_gpu else "💻 NVIDIA NÃO ENCONTRADA (Usar CPU)"

    while True:
        # Limpa o console para o menu ficar sempre organizado
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print_header()
        print(f"  Detecção de Hardware: {status_gpu}")
        print("-" * 35)
        print("  [ 1 ] ✅ REPARO RÁPIDO (Conserta erros e falta de arquivos)")
        print("  [ 2 ] 🚀 Instalar NOVO Ambiente (MODO TURBO/RTX)")
        print("  [ 3 ] 🏠 Instalar NOVO Ambiente (MODO PADRÃO/CPU)")
        print("  [ 4 ] 🔑 Fazer Login no HuggingFace")
        print("  [ 5 ] 📖 Ver Instruções do LM Studio")
        print("  [ 6 ] ❌ Desinstalar Ambiente (Apagar do HD)")
        print("  [ 7 ] 🚪 Sair")
        
        choice = input("\n👉 Escolha uma opção (1-7): ").strip()
        
        if not choice:
            continue

        if choice == '1':
            if not os.path.exists(ENV_PATH):
                print("\n❌ ERRO: Ambiente não encontrado. Instale primeiro (Opção 2 ou 3).")
            else:
                repair_env(has_gpu_selected=has_gpu)
            input("\nPressione Enter para continuar...")
        elif choice == '2':
            if os.path.exists(ENV_PATH):
                print("\n⚠️ O ambiente já existe. Use o REPARO (Opção 1) ou Apague (Opção 6) primeiro.")
            else:
                install_env(has_gpu_selected=True)
            input("\nPressione Enter para continuar...")
        elif choice == '3':
            if os.path.exists(ENV_PATH):
                print("\n⚠️ O ambiente já existe. Use o REPARO (Opção 1) ou Apague (Opção 6) primeiro.")
            else:
                install_env(has_gpu_selected=False)
            input("\nPressione Enter para continuar...")
        elif choice == '4':
            hf_login()
        elif choice == '5':
            print("\n- URL: http://localhost:1234\n- Modelo Recomendado: gemma-4-E4B-it-GGUF")
            print("- Vá em 'Local Server' e ative 'Start Server'.")
            input("\nPressione Enter para voltar...")
        elif choice == '6':
            conf = input("\n⚠️ Tem certeza que deseja apagar o ambiente? (s/n): ").strip().lower()
            if conf == 's':
                delete_env()
                input("\nAmbiente removido. Pressione Enter...")
        elif choice == '7':
            print("\nSaindo... Use 'conda activate " + ENV_PATH + "' para rodar o app.")
            break
        else:
            print(f"\n❌ Opção '{choice}' inválida.")
            import time
            time.sleep(1)

if __name__ == "__main__":
    main()
