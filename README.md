# PhoenixDub AI 🚀🔥

[Português](#português) | [English](#english)

---

## Português

**PhoenixDub AI** é um pipeline de dublagem automatizada de nível profissional, projetado para alta precisão, fluxo natural e extrema resiliência. Sincroniza vídeos e jogos em Português (PT-BR) usando IAs de última geração.

> [!IMPORTANTE]
> **COMPATIBILIDADE DE HARDWARE**:
> - **Processador**: Otimizado para **Intel** (6ª geração ou superior).
> - **Placa de Vídeo**: Recomendado **NVIDIA RTX 2060** (6GB) para o "Modo Turbo".
> - **Memória**: Mínimo 16 GB de RAM.

### 🌟 O que há de novo na Versão 2026 (Phoenix STABLE)
*   **Motor Dual Chatterbox (ONNX + Torch)**: Suporte a aceleração ONNX para GPUs RTX e CPUs antigas.
*   **Setup Inteligente**: Instalador `setup.py` que detecta drivers e componentes faltantes.
*   **Conexão LM Studio**: Tradução via Gemma 4 (Host Local) para evitar falhas de API externa.

---

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

O PhoenixDub possui dois motores independentes:

*   **Dublagem de Jogos (`app_jogos.py`)**: Para arquivos de áudio extraídos de games (WAV/MP3). Possui sistema agêntico que respeita a duração original e usa o estilo do personagem.
*   **Dublagem de Vídeos (`app_videos.py`)**: Para trailers e cutscenes. Possui o **Magic Cut** (corte automático de silêncios) e redublagem sincronizada.

---

### 🛠️ Solução de Problemas (FAQ)

| Problema | Solução |
| :--- | :--- |
| **"Invalid audio stream" ou erro MP3** | Você está usando o FFmpeg básico. Instale o **FFmpeg FULL** como descrito acima. |
| **"Out of Memory" ou Erro 1455** | Você não fechou o LM Studio quando o programa pediu. Feche-o para liberar VRAM. |
| **"espeak-ng not found"** | Você esqueceu o Passo 1. Instale o eSpeak-NG e reinicie o PC. |
| **O som da dublagem sai mudo** | Verifique se o eSpeak-NG está instalado corretamente. |
| **Erro 1234 (Connection Refused)** | O LM Studio não está com o "Start Server" ligado. |

---

## English

**PhoenixDub AI** is a professional-grade automated dubbing pipeline designed for high precision, natural flow, and extreme resilience. It synchronizes videos and games into Portuguese (PT-BR) using state-of-the-art AI.

> [!IMPORTANT]
> **HARDWARE COMPATIBILITY**:
> - **Processor**: Optimized for **Intel** (6th Gen or newer).
> - **Video Card**: **NVIDIA RTX 2060** (6GB) recommended for "Turbo Mode".
> - **Memory**: Minimum 16 GB RAM.

### 🌟 What's New in 2026 (Phoenix STABLE)
*   **Dual Chatterbox Engine (ONNX + Torch)**: ONNX acceleration support for RTX GPUs and legacy CPUs.
*   **Smart Setup**: `setup.py` installer that detects drivers and missing components.
*   **LM Studio Integration**: Translation via Gemma 4 (Local Host) to avoid external API failures.

---

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
*Developed with ❤️ by Paulo Henrik Carvalho de Araújo.*
