import os
import subprocess
import sys
import shutil

ENV_PATH = r"C:\IA_Dublagem_Files\env"

def print_header():
    print("\n" + "="*65)
    print(" 🛠️  PHOENIXDUB AI - GERENCIADOR AVANÇADO DE AMBIENTE 🛠️ ")
    print("="*65)

def run_cmd(cmd):
    # No Windows, shell=True ajuda a encontrar o conda
    subprocess.run(cmd, check=True, shell=True)

def check_conda():
    try:
        subprocess.run(["conda", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    except Exception:
        print("\n❌ ERRO FATAL: Conda não encontrado no PATH.")
        print("Instale o Anaconda/Miniconda e marque 'Add to PATH' durante a instalação.")
        sys.exit(1)

def install_env():
    print(f"\n[+] Criando ambiente virtual limpo em: {ENV_PATH}")
    try:
        # Força Python 3.10 para evitar os conflitos relatados com PyTorch/Chatterbox
        run_cmd(f"conda create --prefix {ENV_PATH} python=3.10 -y")
        print("[+] Ambiente Python 3.10 criado com sucesso!")
        
        print("\n[+] Baixando e instalando programas externos essenciais (FFmpeg)...")
        # Instala dependências via Conda Forge que não ficam no requirements
        run_cmd(f"conda install --prefix {ENV_PATH} -c conda-forge ffmpeg -y")
        
        print("\n[+] ✨ MÁGICA EXTRA: Instalando automaticamente as bibliotecas do PIP! ✨")
        print("⏳ Esse passo instala o Torch, Whisper e TTS. Pode ir tomar um café...")
        run_cmd(f"conda run --prefix {ENV_PATH} pip install -r requirements.txt")
        
        print("\n" + "="*65)
        print("🤖 FASE FINAL: DOWNLOAD DO CÉREBRO DE TRADUÇÃO (GEMA) 🤖")
        print("="*65)
        
        models_dir = os.path.join(os.getcwd(), "models")
        os.makedirs(models_dir, exist_ok=True)
        gema_path = os.path.join(models_dir, "gemma-4-E4B-it-Q4_K_M.gguf")
        
        if os.path.exists(gema_path):
            print("✅ Cérebro Gema 4 (gemma-4-E4B-it-Q4_K_M) já encontrado na pasta 'models'!")
            print("Pulo inteligente ativado. Evitando redownload de 5GB.")
        else:
            print("Precisamos da sua chave do HuggingFace para baixar o modelo de 5GB.")
            hf_token = input("👉 Cole seu Token do HuggingFace (Read) e aperte Enter: ").strip()
            
            safe_models_dir = models_dir.replace('\\', '\\\\')
            
            # Cria um script temporário para baixar o Gema via Python do ambiente
            dl_script = f"""
import os
from huggingface_hub import login, hf_hub_download
import sys

token = "{hf_token}"
if token:
    print("\\n[!] Autenticando no HuggingFace...")
    login(token=token, add_to_git_credential=False)

print("\\n⏳ INICIANDO DOWNLOAD DO GEMA 4 (gemma-4-E4B-it-Q4_K_M.gguf) ⏳")
print("Isso tem ~5GB. A barrinha vai aparecer abaixo. Aguarde...")

try:
    path = hf_hub_download(
        repo_id="unsloth/gemma-4-E4B-it-GGUF", 
        filename="gemma-4-E4B-it-Q4_K_M.gguf", 
        local_dir="{safe_models_dir}", 
        local_dir_use_symlinks=False
    )
    print(f"\\n✅ CÉREBRO BAIXADO E INSTALADO EM: {{path}}")
except Exception as e:
    print(f"\\n❌ Erro durante o download: {{e}}")
"""
            with open("download_temp_hf.py", "w", encoding="utf-8") as f:
                f.write(dl_script)
                
            run_cmd(f"conda run --prefix {ENV_PATH} python download_temp_hf.py")
            if os.path.exists("download_temp_hf.py"):
                os.remove("download_temp_hf.py")
        
        print("\n" + "="*65)
        print("✅ INSTALAÇÃO SUPREMA 100% CONCLUÍDA! ✅")
        print("="*65)
        print("Tudo foi automatizado na sua máquina. O modelo está na pasta models/")
        print("Para usar o programa no dia a dia, basta abrir o terminal e digitar:\n")
        print(f"👉 1. conda activate {ENV_PATH}")
        print(f"👉 2. python app_jogos.py")
        print("\nCaso seu ambiente quebre no futuro, você pode rodar 'python setup.py' e escolher Reparar.")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Erro crítico na instalação: {e}")

def delete_env():
    print(f"\n[!] Apagando o ambiente virtual corrompido em: {ENV_PATH}")
    try:
        run_cmd(f"conda env remove --prefix {ENV_PATH} -y")
        # Força a deleção da pasta física caso sobrem resquícios
        if os.path.exists(ENV_PATH):
            shutil.rmtree(ENV_PATH, ignore_errors=True)
        print("[+] Ambiente completamente deletado do disco.")
    except Exception as e:
        print(f"❌ Erro ao tentar deletar a pasta: {e}")

def reinstall_env():
    print("\n🔄 Iniciando o modo de Auto-Reparo...")
    delete_env()
    install_env()

def main():
    check_conda()
    while True:
        print_header()
        print("Escolha uma opção de Gerenciamento:")
        print("  [ 1 ] 🟢 Instalar Novo Ambiente (Python 3.10 + Softwares Externos)")
        print("  [ 2 ] 🔄 Reinstalar/Reparar Ambiente (Apaga e refaz do zero)")
        print("  [ 3 ] ❌ Excluir Ambiente Definitivamente")
        print("  [ 4 ] 🚪 Sair do Instalador")
        
        choice = input("\n👉 Digite o número da opção (1-4): ").strip()
        
        if choice == '1':
            if os.path.exists(ENV_PATH):
                print("\n⚠️ O ambiente já existe! Use a Opção 2 para reparar/reinstalar.")
            else:
                install_env()
            break
            
        elif choice == '2':
            print("\n⚠️ ATENÇÃO: Isso vai excluir todas as bibliotecas atuais e recriar o ambiente limpo.")
            conf = input("Tem certeza que deseja prosseguir com a reinstalação? (s/n): ").strip().lower()
            if conf == 's':
                reinstall_env()
                break
                
        elif choice == '3':
            conf = input("\n⚠️ Tem certeza absoluta que deseja destruir a pasta do ambiente? (s/n): ").strip().lower()
            if conf == 's':
                delete_env()
            break
            
        elif choice == '4':
            print("Saindo do instalador. Até logo!")
            break
        else:
            print("\n❌ Opção inválida. Digite 1, 2, 3 ou 4.")

if __name__ == "__main__":
    main()
