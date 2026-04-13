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

#### Passo 3: Token HuggingFace (Opcional, mas Recomendado)
1.  No terminal (Anaconda Prompt), digite: `huggingface-cli login`.
2.  Cole seu Token de acesso (gerado no site huggingface.co).

#### Passo 4: Rodando o Instalador
Na pasta do projeto, execute:
```bash
python setup.py
```
*   **Opção 1 (Modo Turbo)**: Use se você tiver uma placa NVIDIA RTX.
*   **Opção 2 (Modo Padrão)**: Use se você for rodar apenas no processador (CPU).

---

### ⚡ Aceleração ONNX (Como ativar o Turbo)
Para ganhar até 2x mais velocidade na RTX 2060 ou no i5:
1.  Baixe os modelos em: [ONNX Community Chatterbox](https://huggingface.co/onnx-community/chatterbox-multilingual-ONNX).
2.  Crie a pasta `models/chatterbox_onnx/` no projeto.
3.  Coloque os arquivos `.onnx` e o `tokenizer.json` lá dentro. O programa detectará automaticamente no próximo início.

---

### 🛠️ Solução de Problemas (FAQ)

| Problema | Solução |
| :--- | :--- |
| **"espeak-ng not found"** | Você esqueceu o Passo 1. Instale o eSpeak-NG e reinicie o PC. |
| **Erro de versão "huggingface-hub"** | Corrigido no setup.py atual. Rode a instalação novamente com a opção 4 (deletar) e depois a 1 ou 2. |
| **O som da dublagem sai mudo** | Verifique se o eSpeak-NG está instalado corretamente e se os Drivers de Som estão OK. |
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

#### Step 3: HuggingFace Token (Optional, but Recommended)
1.  In your terminal (Anaconda Prompt), type: `huggingface-cli login`.
2.  Paste your access Token (generated at huggingface.co).

#### Step 4: Running the Installer
In the project folder, execute:
```bash
python setup.py
```
*   **Option 1 (Turbo Mode)**: Use this if you have an NVIDIA RTX card.
*   **Option 2 (Standard Mode)**: Use this if you are running on the processor (CPU) only.

---

### ⚡ ONNX Acceleration (Activate Turbo Mode)
To gain up to 2x speed on RTX 2060 or i5 CPUs:
1.  Download models from: [ONNX Community Chatterbox](https://huggingface.co/onnx-community/chatterbox-multilingual-ONNX).
2.  Create a `models/chatterbox_onnx/` folder in the project directory.
3.  Place the `.onnx` files and `tokenizer.json` inside. The program will auto-detect them on the next run.

---

### 🛠️ Troubleshooting (FAQ)

| Issue | Solution |
| :--- | :--- |
| **"espeak-ng not found"** | You missed Step 1. Install eSpeak-NG and restart your terminal/PC. |
| **"huggingface-hub" version error** | Fixed in current setup.py. Re-run installation with Option 4 (delete) then 1 or 2. |
| **Dubbed audio is silent** | Ensure eSpeak-NG is correctly installed and your system Audio Drivers are OK. |
| **Error 1234 (Connection Refused)** | LM Studio "Start Server" is not toggled on. |

---
*Developed with ❤️ by Paulo Henrik Carvalho de Araújo.*
