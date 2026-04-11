import os
import subprocess
import sys
import shutil

ENV_PATH = r"C:\IA_Dublagem_Files\env"

def print_header():
    print("\n" + "="*65)
    print(" 🛠️  PHOENIXDUB AI - GERENCIADOR DE AMBIENTE (v3.0) 🛠️ ")
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
        
        print("\n[+] Baixando e instalando softwares externos (FFmpeg & TK)...")
        # tk é necessário para as janelas de arquivos do vpk_manager
        run_cmd(f"conda install --prefix {ENV_PATH} -c conda-forge ffmpeg tk -y")
        
        print("\n[+] ✨ INSTALANDO BIBLIOTECAS DE IA (PyTorch, Whisper, TTS, etc) ✨")
        print("⏳ Esse passo pode levar alguns minutos. Vá tomar um café...")
        run_cmd(f"conda run --prefix {ENV_PATH} pip install -r requirements.txt")
        
        print("\n" + "="*65)
        print("🤖 FASE FINAL: CONFIGURAÇÃO DO CÉREBRO (GEMA 4 via LM STUDIO) 🤖")
        print("="*65)
        print("\nEsta versão do sistema utiliza o LM STUDIO para processamento local.")
        print("Siga os passos abaixo para concluir a instalação:")
        print("\n1. Baixe o LM Studio em: https://lmstudio.ai")
        print("2. Dentro do LM Studio, pesquise por: unsloth/gemma-4-E4B-it-GGUF")
        print("3. Baixe a versão 'Q4_K_M' ou superior.")
        print("4. Vá na aba 'Local Server' (ícone de <-> no lado esquerdo).")
        print("5. Carregue o modelo baixado no topo da tela.")
        print("6. Clique em 'Start Server'.")
        print("7. Garanta que a porta é a '1234'.")
        
        print("\n" + "="*65)
        print("✅ AMBIENTE 100% PRONTO! ✅")
        print("="*65)
        print("\nPara usar o programa, mantenha o LM Studio aberto com o Server ligado e digite:\n")
        print(f"👉 1. conda activate {ENV_PATH}")
        print(f"👉 2. python app_jogos.py  (ou App_videos.py)")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Erro crítico na instalação: {e}")

def delete_env():
    print(f"\n[!] Apagando o ambiente virtual corrompido em: {ENV_PATH}")
    try:
        run_cmd(f"conda env remove --prefix {ENV_PATH} -y")
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
        print("  [ 1 ] 🟢 Instalar Ambiente de IA (Python 3.10 + Softwares)")
        print("  [ 2 ] 🔄 Reinstalar/Reparar Ambiente (Apaga e refaz)")
        print("  [ 3 ] 📖 Ver Instruções do LM Studio (Gema 4)")
        print("  [ 4 ] ❌ Excluir Ambiente Definitivamente")
        print("  [ 5 ] 🚪 Sair")
        
        choice = input("\n👉 Digite o número da opção (1-5): ").strip()
        
        if choice == '1':
            if os.path.exists(ENV_PATH):
                print("\n⚠️ O ambiente já existe!")
            else:
                install_env()
            break
        elif choice == '2':
            conf = input("Tem certeza que deseja reinstalar? (s/n): ").strip().lower()
            if conf == 's':
                reinstall_env()
                break
        elif choice == '3':
            print("\n" + "-"*30)
            print("CONFIGURAÇÃO DO LM STUDIO:")
            print("- URL Server: http://localhost:1234")
            print("- Modelo Recomendado: gemma-4-E4B-it-Q4_K_M.gguf")
            print("- Repo: unsloth/gemma-4-E4B-it-GGUF")
            print("-"*30)
            input("\nAperte Enter para voltar...")
        elif choice == '4':
            conf = input("\n⚠️ Tem certeza que deseja destruir o ambiente? (s/n): ").strip().lower()
            if conf == 's':
                delete_env()
            break
        elif choice == '5':
            print("Até logo!")
            break
        else:
            print("\n❌ Opção inválida.")

if __name__ == "__main__":
    main()
