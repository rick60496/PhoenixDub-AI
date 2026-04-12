import os
import subprocess
import sys
import shutil

ENV_PATH = r"C:\IA_Dublagem_Files\env"

def print_header():
    print("\n" + "="*65)
    print(" 🛠️  PHOENIXDUB AI - GERENCIADOR DE AMBIENTE (v18.6) 🛠️ ")
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
        print("\n❌ ERRO FATAL: Conda não encontrado no PATH.")
        print("Instale o Anaconda/Miniconda e marque 'Add to PATH' durante a instalação.")
        sys.exit(1)

def install_env():
    has_gpu = check_nvidia_gpu()
    
    print("\n" + "-"*65)
    if has_gpu:
        print(" ✨ DETECTADO: Hardware NVIDIA/CUDA presente! ✨")
        print(" [MODO TURBO] O sistema será instalado para ALTA PERFORMANCE.")
        req_file = "requirements_RTX.txt"
    else:
        print(" 💻 DETECTADO: Processador apenas (CPU Mode). 💻")
        print(" [MODO ESTÁVEL] O sistema será instalado para máxima compatibilidade.")
        req_file = "requirements.txt"
    print("-"*65)

    if not os.path.exists(req_file):
        print(f"❌ ERRO: Arquivo {req_file} não encontrado na pasta!")
        return

    print(f"\n[+] Criando ambiente virtual em: {ENV_PATH}")
    try:
        # Força Python 3.10 para evitar conflitos conhecidos
        run_cmd(f"conda create --prefix {ENV_PATH} python=3.10 -y")
        
        print("\n[+] Instalando FFmpeg & dependências básicas...")
        if has_gpu:
            # [MODO TURBO] Instala o CUDA Toolkit oficial para garantir as DLLs da placa
            print("[+] Instalando motor NVIDIA (CUDA Toolkit 11.8)...")
            run_cmd(f"conda install --prefix {ENV_PATH} -c conda-forge ffmpeg tk -y")
            run_cmd(f"conda install --prefix {ENV_PATH} -c nvidia cudatoolkit=11.8 -y")
        else:
            run_cmd(f"conda install --prefix {ENV_PATH} -c conda-forge ffmpeg tk -y")
        
        print(f"\n[+] ✨ INSTALANDO CÉREBRO DE IA ({req_file}) ✨")
        print("⏳ Esse passo pode levar alguns minutos. Vá tomar um café...")
        run_cmd(f"conda run --prefix {ENV_PATH} pip install -r {req_file}")
        
        print("\n" + "="*65)
        print("🤖 FASE FINAL: CONFIGURAÇÃO DO GEMA 4 (LM STUDIO) 🤖")
        print("="*65)
        print("\n1. Baixe o LM Studio em: https://lmstudio.ai")
        print("2. Pesquise por: unsloth/gemma-4-E4B-it-GGUF")
        print("3. Vá em 'Local Server' e clique em 'Start Server' (Porta 1234)")
        
        print("\n" + "="*65)
        print("✅ AMBIENTE PRONTO! ✅")
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
    while True:
        print_header()
        has_gpu = check_nvidia_gpu()
        status_gpu = "🚀 RTX ATIVADA" if has_gpu else "💻 CPU APENAS"
        
        print(f"  Detecção de Hardware: {status_gpu}")
        print("-" * 30)
        print("  [ 1 ] 🟢 Instalar/Reparar Ambiente (Recomendado)")
        print("  [ 2 ] 📖 Ver Instruções do LM Studio")
        print("  [ 3 ] ❌ Desinstalar Ambiente (Apagar do HD)")
        print("  [ 4 ] 🚪 Sair")
        
        choice = input("\n👉 Escolha uma opção (1-4): ").strip()
        
        if choice == '1':
            if os.path.exists(ENV_PATH):
                conf = input("\n⚠️ O ambiente já existe. Deseja re-instalar por cima? (s/n): ").strip().lower()
                if conf == 's':
                    delete_env()
                    install_env()
            else:
                install_env()
            break
        elif choice == '2':
            print("\n- URL: http://localhost:1234\n- Modelo: gemma-4-E4B-it-Q4_K_M.gguf")
            input("\nEnter para voltar...")
        elif choice == '3':
            conf = input("\n⚠️ Tem certeza que deseja apagar? (s/n): ").strip().lower()
            if conf == 's':
                delete_env()
            break
        elif choice == '4':
            print("Até logo!")
            break
        else:
            print("\n❌ Opção inválida.")

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
