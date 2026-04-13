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
    """Detecta se existe uma placa NVIDIA no sistema para ativar CUDA."""
    try:
        res = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        if res.returncode == 0:
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
    print(" O Chatterbox exige o eSpeak-NG para converter texto em voz.")
    print(" Baixe e instale o 'espeak-ng-X64.msi' em:")
    print(" https://github.com/espeak-ng/espeak-ng/releases")
    print(" Após instalar, você pode continuar a instalação.")
    print("!"*65)
    input(" Pressione Enter para CONTINUAR mesmo assim (ou instale e reinicie)...")
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
        
        print("\n[+] Instalando Ferramentas de Base (FFmpeg & TK)...")
        run_cmd(f"conda install --prefix {ENV_PATH} ffmpeg tk -y")
        
        print("\n[+] Liberando comandos de Login (HuggingFace)...")
        # Instala uma versão compatível com transformers 4.40 (< 1.0.0)
        trusted_flags = "--trusted-host pypi.org --trusted-host files.pythonhosted.org --trusted-host download.pytorch.org"
        run_cmd(f'"{ENV_PATH}\\python.exe" -m pip install "huggingface-hub<1.0.0" --upgrade {trusted_flags}')
        
        print("\n" + "*"*65)
        print(" 🔑 AGORA VOCÊ PODE FAZER O LOGIN! 🔑")
        print(" O comando 'huggingface-cli login' já está disponível.")
        print("*"*65)

        print(f"\n[+] ✨ INSTALANDO CÉREBRO DE IA (Versão 2026 Modern) ✨")
        print(f"[Arquivo: {req_file}]")
        print("⏳ Este passo é largo. A barra de progresso será exibida abaixo...")
        run_cmd(f'"{ENV_PATH}\\python.exe" -m pip install -r {req_file} {trusted_flags}')
        
        # [CONFLITO RESOLVIDO] Instala o Chatterbox separadamente
        print("\n[+] 🔊 Finalizando: Instalando Motor de Voz (Chatterbox v0.1.7)...")
        chatterbox_cmd = f'"{ENV_PATH}\\python.exe" -m pip install git+https://github.com/resemble-ai/chatterbox.git --no-deps {trusted_flags}'
        run_cmd(chatterbox_cmd)
        
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

def main():
    check_conda()
    # Detecta GPU apenas uma vez para performance
    has_gpu = check_nvidia_gpu()
    status_gpu = "🚀 NVIDIA RTX DETECTADA" if has_gpu else "💻 NVIDIA NÃO ENCONTRADA (Usar CPU)"

    while True:
        # Limpa o console para o menu ficar sempre organizado
        # No Windows usamos 'cls', no Linux/Mac 'clear'
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print_header()
        print(f"  Detecção de Hardware: {status_gpu}")
        print("-" * 35)
        print("  [ 1 ] 🚀 Instalar MODO TURBO (Para quem tem Placa RTX)")
        print("  [ 2 ] 🏠 Instalar MODO PADRÃO (Para o meu PC / CPU)")
        print("  [ 3 ] 📖 Ver Instruções do LM Studio")
        print("  [ 4 ] ❌ Desinstalar Ambiente (Apagar do HD)")
        print("  [ 5 ] 🚪 Sair")
        
        choice = input("\n👉 Escolha uma opção (1-5): ").strip()
        
        if not choice:
            continue

        if choice == '1':
            if os.path.exists(ENV_PATH):
                conf = input("\n⚠️ O ambiente já existe. Deseja re-instalar? (s/n): ").strip().lower()
                if conf == 's':
                    delete_env()
                    install_env(has_gpu_selected=True)
            else:
                install_env(has_gpu_selected=True)
            break
        elif choice == '2':
            if os.path.exists(ENV_PATH):
                conf = input("\n⚠️ O ambiente já existe. Deseja re-instalar? (s/n): ").strip().lower()
                if conf == 's':
                    delete_env()
                    install_env(has_gpu_selected=False)
            else:
                install_env(has_gpu_selected=False)
            break
        elif choice == '3':
            print("\n- URL: http://localhost:1234\n- Modelo: gemma-4-E4B-it-Q4_K_M.gguf")
            input("\nEnter para voltar...")
        elif choice == '4':
            conf = input("\n⚠️ Tem certeza que deseja apagar? (s/n): ").strip().lower()
            if conf == 's':
                delete_env()
            break
        elif choice == '5':
            print("Até logo!")
            break
        else:
            print(f"\n❌ Opção '{choice}' inválida. Tente novamente.")
            import time
            time.sleep(1.5)

if __name__ == "__main__":
    main()
