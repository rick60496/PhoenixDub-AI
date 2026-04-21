# PhoenixDub AI 🚀🔥

[Português](#português) | [English](#english)

---

### 🗨️ Participe da nossa Comunidade! / Join our Community!
**Queremos ouvir você!** Se você baixou o projeto, por favor, deixe seu feedback, sugestões ou poste seus resultados na nossa aba de [**Discussões (Discussions)**](https://github.com/rick60496/PhoenixDub-AI/discussions). Sua opinião é fundamental para a evolução do PhoenixDub!

---

## Português

**PhoenixDub AI** é uma solução completa de **edição de vídeo e dublagem automatizada** de nível profissional. Projetado para alta precisão, fluxo natural e extrema resiliência, o PhoenixDub utiliza IAs de última geração para processar, editar e sincronizar vídeos e jogos em Português (PT-BR) de forma inteligente.

> [!IMPORTANTE]
> - **Processador**: Otimizado para **Intel** (6ª geração ou superior). Compatibilidade total para rodar apenas no processador.
> - **Placa de Vídeo**: Recomendado **NVIDIA RTX 2060** (6GB) para o "Modo Turbo". A placa de vídeo entrega muito mais velocidade, mas o sistema funciona 100% apenas com o processador.
> - **Memória**: Mínimo 16 GB de RAM.

### 🌟 O que há de novo na Versão 0.10 (TURBO-PATCH-UNLEASHED)
*   **Modo Turbo Ativado**: Desativamos o raciocínio interno da IA para aumentar a velocidade em até 10x em CPUs i5/i7.
*   **Universal Hardware Support**: Detecção inteligente e automática de NVIDIA (CUDA), AMD (ROCm), Apple (MPS) e CPU.
*   **AI Video Editing**: O motor `App_videos` agora conta com o **Magic Cut**, que remove silências e falhas de gravação automaticamente via IA.
*   **Thinking Scrubber**: Filtro que elimina "vazamentos" de pensamentos da IA nas traduções.
*   **Web Interface Auto-Detect**: Opções de idioma e vozes agora começam em "Auto-Detectar" por padrão.

### 🚀 Tutorial de Instalação (Passo a Passo)

#### Passo 1: Ferramentas de Base (Obrigatório)
1.  **Git para Windows**: [Baixe Aqui](https://git-scm.com/download/win). (Essencial para baixar a IA).
2.  **Anaconda (ou Miniconda)**: [Baixe Aqui](https://www.anaconda.com/download). **Marque "Add to PATH"** durante a instalação.
3.  **eSpeak-NG**: [Baixe o .msi X64 Aqui](https://github.com/espeak-ng/espeak-ng/releases). **VITAL**: O motor de voz não funciona no Windows sem ele.

#### Passo 2: O Cérebro de Tradução (LM Studio)
1.  Instale o **LM Studio** ([lmstudio.ai](https://lmstudio.ai)).
2.  Pesquise e baixe: `unsloth/gemma-4-E4B-it-GGUF` (Recomendado: Q4_K_M).
3.  Em **Local Server**, clique em **Start Server** na porta **1234**.
4.  > [!CAUTION]
    > **USUÁRIOS NVIDIA RTX**: Após a fase de tradução, o programa pedirá para você **FECHAR O LM STUDIO**. Isso é obrigatório para liberar a memória (VRAM) para o motor de voz.

#### Passo 3: Token HuggingFace (Opcional, mas Recomendado)
1.  No terminal (Anaconda Prompt), digite: `huggingface-cli login`.
2.  Cole seu Token de acesso (gerado no site huggingface.co).

#### Passo 4: Rodando o Instalador/Reparador
Na pasta do projeto, execute:
```bash
python setup.py
```
*   **Opção [ 1 ] (REPARO RÁPIDO)**: Use se algo der erro ou se estiver atualizando. (Recomendado).
*   **Opção [ 2 ] (MODO TURBO/RTX)**: Use se você tiver uma placa NVIDIA RTX.
*   **Opção [ 3 ] (MODO PADRÃO/CPU)**: Use se você for rodar apenas no processador.

#### Passo 5: Como Rodar e Usar (A Hora da Verdade) 🎮
Agora que tudo está instalado, veja como abrir o painel de controle pelo **Anaconda Prompt**:

1. **Ative o ambiente virtual** (VITAL):
   ```bash
   conda activate C:\IA_dublagem\env
   ```
2. Entre na pasta do projeto:
   ```bash
   cd C:\IA_dublagem
   ```
3. Digite o comando para o que você deseja fazer:
   - Para dublar **Jogos**: `python app_jogos.py`
   - Para dublar **Vídeos**: `python app_videos.py`
4. O terminal mostrará que o servidor Flask está ativo.
5. **Abra o seu navegador** (Chrome ou Edge) e digite: `http://localhost:5000`

---

### 🛠️ Configuração do FFmpeg FULL (Obrigatório)

Diferente do FFmpeg comum, você precisa da versão **completa** para gerar arquivos MP3 e vídeos de alta qualidade:

1. Acesse: [Gyan.dev (FFmpeg Full)](https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-full.7z)
2. Baixe o arquivo `ffmpeg-release-full.7z`.
3. Extraia e copie o arquivo `ffmpeg.exe` (da pasta `bin`) para:
   - `C:\IA_dublagem\env\Library\bin\ffmpeg.exe`
   - (Substitua o arquivo que já estiver lá).

---

### 🕹️ Como Usar (App Jogos vs App Vídeos)

> [!WARNING]
> **⚠️ ESTADO DO DESENVOLVIMENTO**: 
> O motor **`app_jogos.py`** é o foco principal das atualizações atuais e está em sua versão mais estável e inteligente. 
> O motor **`app_videos.py`** está **desatualizado** e pode apresentar falhas ou comportamentos inesperados. Uma reconstrução completa para o motor de vídeos está nos planos para as próximas versões!

O PhoenixDub possui dois motores independentes:

*   **Dublagem de Jogos (`app_jogos.py`)**: Para arquivos de áudio extraídos de games (WAV/MP3). Possui sistema agêntico que respeita a duração original e usa o estilo do personagem.
*   **Dublagem de Vídeos (`app_videos.py`)**: Para trailers e cutscenes. Possui o **Magic Cut** (corte automático de silêncios) e redublagem sincronizada.

---

### 🛠️ Solução de Problemas (FAQ)

| Problema | Solução |
| :--- | :--- |
| **"Invalid audio stream" ou erro MP3** | Você está usando o FFmpeg básico. Instale o **FFmpeg FULL** como descrito acima. |
| **"Out of Memory" or Error 1455** | Você não fechou o LM Studio quando o programa pediu. Feche-o para liberar VRAM. |
| **"espeak-ng not found"** | Você esqueceu o Passo 1. Instale o eSpeak-NG e reinicie o PC. |
| **O som da dublagem sai mudo** | Verifique se o eSpeak-NG está instalado corretamente. |
| **Erro 1234 (Connection Refused)** | O LM Studio não está com o "Start Server" ligado. |

### 🎖️ Créditos e Agradecimentos
Para conhecer todas as pessoas e tecnologias envolvidas no PhoenixDub, veja o arquivo [CREDITS.md](CREDITS.md).

---

## English

**PhoenixDub AI** is a complete **AI Video Editing and automated dubbing** solution for professional-grade media projects. Designed for high precision and natural flow, it uses state-of-the-art AI to edit and synchronize videos and games into Portuguese (PT-BR).

> [!IMPORTANT]
> - **Processor**: Optimized for **Intel** (6th Gen or newer). Full compatibility to run on the processor alone.
> - **Video Card**: **NVIDIA RTX 2060** (6GB) recommended for "Turbo Mode". A video card provides much higher speeds, but the system is 100% functional on the processor alone.
> - **Memory**: Minimum 16 GB RAM.

### 🌟 What's New in v0.10 (TURBO-PATCH-UNLEASHED)
*   **Turbo Mode Activated**: Disabled internal AI reasoning for up to 10x speed boost on i5/i7 CPUs.
*   **Universal Hardware Support**: Smart auto-detection for NVIDIA (CUDA), AMD (ROCm), Apple (MPS), and CPU.
*   **AI Video Editing**: The `App_videos` engine now features **Magic Cut**, automatically removing silences and recording errors.
*   **Thinking Scrubber**: Filter to eliminate AI "thought leaks" in translations.
*   **Web Interface Auto-Detect**: Language and speaker options now default to "Auto-Detect".

### 🚀 Installation Tutorial (Step-by-Step)

#### Step 1: Base Tools (Mandatory)
1.  **Git for Windows**: [Download Here](https://git-scm.com/download/win). (Essential for downloading the AI models).
2.  **Anaconda (or Miniconda)**: [Download Here](https://www.anaconda.com/download). **Check "Add to PATH"** during installation.
3.  **eSpeak-NG**: [Download .msi X64 Here](https://github.com/espeak-ng/espeak-ng/releases). **VITAL**: The voice engine will NOT work on Windows without it.

#### Step 2: The Translation Brain (LM Studio)
1.  Install **LM Studio** ([lmstudio.ai](https://lmstudio.ai)).
2.  Search and download: `unsloth/gemma-4-E4B-it-GGUF` (Recommended: Q4_K_M).
3.  In **Local Server**, click **Start Server** on port **1234**.
4.  > [!CAUTION]
    > **NVIDIA RTX USERS**: After the translation phase, the program will ask you to **CLOSE LM STUDIO**. This is mandatory to free up VRAM for the voice engine.

#### Step 3: HuggingFace Token (Optional, but Recommended)
1.  In your terminal (Anaconda Prompt), type: `huggingface-cli login`.
2.  Paste your access Token (generated at huggingface.co).

#### Step 4: Running the Installer/Repairer
In the project folder, execute:
```bash
python setup.py
```
*   **Option [ 1 ] (QUICK REPAIR)**: Use this if you encounter errors or are updating. (Recommended).
*   **Option [ 2 ] (TURBO/RTX MODE)**: Use if you have an NVIDIA RTX card.
*   **Option [ 3 ] (STANDARD/CPU MODE)**: Use if you are running on the processor only.

#### Step 5: How to Run and Use 🎮
Now that everything is installed, here's how to launch the control panel via **Anaconda Prompt**:

1. **Activate the virtual environment** (VITAL):
   ```bash
   conda activate C:\IA_dublagem\env
   ```
2. Navigate to the project folder:
   ```bash
   cd C:\IA_dublagem
   ```
3. Type the command for your desired task:
   - For **Game** dubbing: `python app_jogos.py`
   - For **Video** dubbing: `python app_videos.py`
4. The terminal will show that the Flask server is active.
5. **Open your browser** (Chrome or Edge) and type: `http://localhost:5000`
6. **Done!** The intelligent panel will appear, and you can start dubbing.


---

### 🛠️ FFmpeg FULL Setup (Mandatory)

Unlike basic FFmpeg, you need the **Full** build to support MP3 encoding and high-quality video:

1. Visit: [Gyan.dev (FFmpeg Full)](https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-full.7z)
2. Download `ffmpeg-release-full.7z`.
3. Extract and copy the `ffmpeg.exe` file (from the `bin` folder) to:
   - `C:\IA_dublagem\env\Library\bin\ffmpeg.exe`
   - (Overwrite the existing file).

---

### 🕹️ How to Use (Game App vs Video App)

> [!WARNING]
> **⚠️ DEVELOPMENT STATUS**: 
> The **`app_jogos.py`** engine is the primary focus of current updates and is in its most stable and intelligent state. 
> The **`app_videos.py`** engine is currently **outdated** and may experience bugs or unexpected behavior. A complete overhaul for the video engine is planned for future releases!

PhoenixDub features two independent engines:

*   **Game Dubbing (`app_jogos.py`)**: For audio assets extracted from games. Agentic system that respects timing and character style.
*   **Video Dubbing (`app_videos.py`)**: For trailers and cutscenes. Features **Magic Cut** and synchronized re-dubbing.

---

### 🛠️ Troubleshooting (FAQ)

| Issue | Solution |
| :--- | :--- |
| **"Invalid audio stream" or MP3 error** | You are using basic FFmpeg. Install **FFmpeg FULL** as described above. |
| **"Out of Memory" or Error 1455** | You didn't close LM Studio when prompted. Close it to free up VRAM. |
| **"espeak-ng not found"** | You missed Step 1. Install eSpeak-NG and restart your terminal/PC. |
| **Dubbed audio is silent** | Ensure eSpeak-NG is correctly installed. |
| **Error 1234 (Connection Refused)** | LM Studio "Start Server" is not toggled on. |

---

### 🎖️ Credits and Acknowledgments
To meet the incredible people and technologies behind PhoenixDub, check the [CREDITS.md](CREDITS.md) file.

---
*Developed with ❤️ by Paulo Henrik Carvalho de Araújo.*
