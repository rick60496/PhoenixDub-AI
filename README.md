# PhoenixDub AI 🚀🔥

[Português](#português) | [English](#english)

---

## Português

**PhoenixDub AI** é um pipeline de dublagem automatizada de nível profissional e código aberto, projetado para alta precisão, fluxo natural e extrema resiliência.

Nascido da necessidade de uma solução local robusta que rivalize com os gigantes da nuvem, o PhoenixDub utiliza um pipeline de IA de vários estágios para transformar vídeos e jogos em experiências naturalmente sincronizadas em Português (PT-BR).

> [!WARNING]
> **Compatibilidade & Hardware**: Este sistema foi testado exclusivamente em arquiteturas **Intel**. Devido às otimizações específicas, ele **não funcionará em processadores AMD**.
> 
> **Requisitos Mínimos**:
> - Processador: Intel Core i5 (6ª Geração) ou superior.
> - Memória: 16 GB de RAM.
> 
> **Performance Estimada**: Este é um pipeline de alta fidelidade e processamento intenso. Em um hardware com os requisitos mínimos, um vídeo/áudio de **50 minutos** leva em média **25 horas** para ser totalmente processado.

### 🌟 Principais Recursos

*   **Smart Split (v10.0)**: Diarização avançada ao nível de palavra e divisão de segmentos para isolamento perfeito do locutor.
*   **Sincronia Labial Elástica**: Um motor proprietário de "Empréstimo de Silêncios" (Smart Gap Borrowing) que adapta a velocidade da fala e os silêncios para manter a sincronia perfeita.
*   **Lógica de Recuperação Phoenix**: Resiliência integrada que pode reconstruir projetos a partir de estados corrompidos ou quedas de energia.
*   **Ghost Vocal Filler**: Preserva perfeitamente a ambiência original em segmentos que não requerem dublagem.
*   **Precisão de Duas Passagens**: Transcrição de alta precisão via Faster-Whisper com verificação linguística secundária baseada em NLP.
*   **Fades Cinemáticos**: Transições de áudio suaves (200ms) para uma experiência de audição premium.

### 🛠️ Stack Tecnológica e Créditos

O PhoenixDub AI se apóia nos ombros de gigantes. Utilizamos modelos de código aberto de última geração:

*   **Transcrição**: Faster-Whisper.
*   **Tradução**: Google Gema (via Llama-cpp).
*   **Separação Vocal**: OpenUnmix (UMX) / Demucs.
*   **Motor de TTS**: Chatterbox.
*   **Núcleo de Áudio**: PyDub, Librosa & FFmpeg.

Uma lista completa de bibliotecas e licenças pode ser encontrada em [CREDITS.md](./CREDITS.md).

### 🚀 Como Instalar e Rodar (Tutorial para Iniciantes)

Este projeto foi construído para rodar localmente utilizando o Anaconda. Siga o passo a passo abaixo para garantir que os modelos de IA funcionem de primeira:

#### Passo 1: Preparando o Ambiente (Anaconda)
1. Baixe e instale o **Miniconda** ou **Anaconda** no seu computador.
2. ⚠️ **Atenção na Instalação:** Durante o instalador, na tela de "Advanced Options" (Opções Avançadas), marque a caixinha **"Add Anaconda to my PATH environment variable"** (Adicionar ao PATH). Isso é vital para que o programa se comunique com o Windows.
3. Após instalar, abra o aplicativo **Anaconda Prompt** no menu iniciar.

#### Passo 2: O Token de IA do HuggingFace
O nosso sistema baixa as inteligências e vozes de alta qualidade direto do HuggingFace (O GitHub das IAs). Você precisa autorizar seu PC para baixar:
1. Crie uma conta gratuita em [huggingface.co](https://huggingface.co/).
2. Após logar, acesse a página de Tokens: [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
3. Clique em **"New token"** (Criar novo token). Escolha a permissão "Read" (Leitura) e gere a chave secreta. Mande copiar esse código gigante.
4. Volte na tela preta do seu **Anaconda Prompt** e digite o seguinte comando:
   ```bash
   huggingface-cli login
   ```
5. Quando ele pedir o token ("Token:"), cole o código que você copiou (Dica: no CMD/Prompt, ao clicar com o botão direito para colar, o código ficará "invisível" na tela por segurança. É normal. Apenas aperte Enter). 
6. Quando perguntar sobre *Add token as git credential (Y/n)*, aperte `n` e depois Enter. Pronto! Seu PC tem a chave do hangar.

#### Passo 3: O Cérebro de Tradução (Gema 4 via LM Studio)
A inteligência artificial de tradução e sincronia (Gema 4) agora roda via **LM Studio** para maior performance e facilidade de uso:
1. Baixe e instale o **LM Studio** em: [https://lmstudio.ai/](https://lmstudio.ai/)
2. Abra o LM Studio e, na barra de pesquisa, procure por: `unsloth/gemma-4-E4B-it-GGUF`.
3. Escolha o arquivo que termina em `Q4_K_M.gguf` (ou superior) e clique em **Download**.
4. Após baixar, vá no ícone de **Local Server** (lado esquerdo, ícone de <->).
5. Selecione o modelo carrgado no topo da tela.
6. Clique em **Start Server** e certifique-se de que a porta está configurada como `1234`.
7. **Importante:** Mantenha o LM Studio aberto com o servidor ligado enquanto utiliza o PhoenixDub!

#### Passo 4: Instalação Automática (Setup)
O nosso sistema possui um instalador interativo robusto que fará todo o trabalho pesado para você (Criar a arquitetura com Python 3.10, baixar softwares externos como FFmpeg, e instalar as Inteligências Artificiais via PIP):
1. Abra o **Anaconda Prompt** e navegue até a pasta central do PhoenixDub:
   ```bash
   cd C:\IA_dublagem
   ```
2. Inicie o menu do instalador digitando:
   ```bash
   python setup.py
   ```
3. Digite `1` e pressione **Enter** (Instalar Novo Ambiente). A tela começará a descer uma cachoeira de logs de download (pode ir tomar um café, são muitos arquivos pesados).
4. Quando a mensagem verde de sucesso aparecer, seu sistema estará pronto!

#### Passo 5: Abrindo a Interface PhoenixDub
Sempre que for usar o programa no seu dia a dia, abra o Anaconda Prompt e ative o ambiente:
```bash
conda activate C:\IA_Dublagem_Files\env
cd C:\IA_dublagem
```

Escolha o robô que deseja iniciar:
*   **Dublagem de Jogos:** `python app_jogos.py`
*   **Editor de Vídeo Inteligente:** `python App_videos.py`
*   **Gerenciador de Arquivos (Mod Manager):** `python vpk_manager.py`

A interface abrirá magicamente no seu navegador!

### 📁 Ferramentas de Modding (Pasta `tools`)
Para que o **Universal Mod Manager** (vpk_manager.py) consiga extrair áudios complexos de jogos, utilizamos ferramentas externas:
*   **vgmstream-cli**: Localizado em `tools/vgmstream-cli/`. É responsável por converter arquivos proprietários (como `.wem` do Wwise ou `.fsb` do FMOD) em áudio `.wav` padrão.
*   **Instalação**: Esta pasta já deve acompanhar o repositório. Caso precise atualizar, baixe a versão CLI em [vgmstream.org](https://vgmstream.org) e extraia exatamente nesta pasta.

### 👤 Autor e Criador Original

**Paulo Henrik Carvalho de Araújo**
*Arquiteto original e desenvolvedor principal.*

### 📄 Licença

Este projeto é licenciado sob a **Apache License 2.0**.

Você é livre para usar, modificar e distribuir este software para qualquer finalidade, desde que mantenha os créditos de autoria original.

---

## English

**PhoenixDub AI** is a professional-grade, open-source automated dubbing pipeline designed for high precision, natural flow, and extreme resilience.

Born from the need for a robust local solution that rivals cloud giants, PhoenixDub uses a multi-stage AI pipeline to transform videos and games into naturally synchronized Portuguese (PT-BR) experiences.

> [!WARNING]
> **Compatibility & Hardware**: This system has been tested exclusively on **Intel** architectures. Due to specific optimizations, it **will not work on AMD processors**.
> 
> **Minimum Requirements**:
> - Processor: Intel Core i5 (6th Gen) or higher.
> - RAM: 16 GB.
> 
> **Estimated Performance**: This is a high-fidelity, CPU-intensive pipeline. On hardware meeting the minimum specs, a **50-minute** video/audio takes approximately **25 hours** to process.

### 🌟 Key Features

*   **Smart Split (v10.0)**: Advanced word-level diarization and segment cleaving for perfect speaker isolation.
*   **Elastic Lip-Sync**: A proprietary "Smart Gap Borrowing" engine that adapts speech speed and silences to maintain perfect synchronization.
*   **Phoenix Recovery Logic**: Built-in resilience that can reconstruct projects from corrupted states or power outages.
*   **Ghost Vocal Filler**: Seamlessly preserves original background ambiance in segments that don't require dubbing.
*   **Dual-Pass Precision**: High-accuracy transcription via Faster-Whisper with secondary NLP-based language verification.
*   **Cinematic Fades**: Smooth audio transitions (200ms) for a premium listening experience.

### 🛠️ Technology Stack & Credits

PhoenixDub AI stands on the shoulders of giants. We utilize state-of-the-art open-source models:

*   **Transcription**: Faster-Whisper.
*   **Translation**: Google Gema (via Llama-cpp).
*   **Vocal Separation**: OpenUnmix (UMX) / Demucs.
*   **TTS Engine**: Chatterbox.
*   **Audio Core**: PyDub, Librosa & FFmpeg.

A full list of libraries and licenses can be found in [CREDITS.md](./CREDITS.md).

### 🚀 How to Install and Run (Beginner's Guide)

This project leverages heavy AI models running locally. Follow the step-by-step Anaconda guide below to ensure a smooth first run:

#### Step 1: Preparing the Anaconda Environment
1. Download and install **Miniconda** or **Anaconda** on your computer.
2. ⚠️ **Installation Warning:** During the installer, on the "Advanced Options" screen, make sure to check the box **"Add Anaconda to my PATH environment variable"**. This is essential for the system to access the correct python executable.
3. After installing, open the **Anaconda Prompt** via your start menu.

#### Step 2: The HuggingFace Magic Token
Our system downloads high-quality voices and translation AI models directly from HuggingFace. You need to authorize your local PC mapping:
1. Create a free account at [huggingface.co](https://huggingface.co/).
2. Once logged in, visit the tokens settings: [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
3. Click on **"New token"**. Select the "Read" permission and generate the secret key. Copy the giant token code completely.
4. Go back to your **Anaconda Prompt** black screen and type the following command:
   ```bash
   huggingface-cli login
   ```
5. When it prompts for "Token:", paste the code you just copied. (Note: when you right-click to paste in the prompt, the characters remain "invisible" for security purposes. This is normal. Just press Enter).
6. When asked about *Add token as git credential (Y/n)*, press `n` and then Enter. Done! Connection established.

#### Step 3: The Translation Brain (Gema 4 via LM Studio)
The translation and sync AI (Gema 4) now runs via **LM Studio** for better performance and ease of use:
1. Download and install **LM Studio** at: [https://lmstudio.ai/](https://lmstudio.ai/)
2. Open LM Studio and, in the search bar, look for: `unsloth/gemma-4-E4B-it-GGUF`.
3. Choose the file ending in `Q4_K_M.gguf` (or higher) and click **Download**.
4. Once downloaded, go to the **Local Server** icon (left side, <-> icon).
5. Select the model at the top of the screen.
6. Click **Start Server** and ensure the port is set to `1234`.
7. **Important:** Keep LM Studio open with the server running while using PhoenixDub!

#### Step 4: Automatic Installation (Setup)
Our system features a robust interactive installer that will do all the heavy lifting for you (Creating the Python 3.10 architecture, downloading external software like FFmpeg, and injecting PIP dependencies automatically):
1. Inside the **Anaconda Prompt**, navigate to the project's root folder:
   ```bash
   cd C:\IA_dublagem
   ```
2. Run the interactive setup menu:
   ```bash
   python setup.py
   ```
3. Type `1` and press **Enter** (Install New Environment). The screen will start cascading down with download logs (grab a coffee, it downloads massive AI PyTorch weights).
4. Once the green success message pops up, your system is fully built!

#### Step 5: Booting PhoenixDub AI
Every day, whenever you want to use the software, open your Anaconda Prompt and activate the universal environment:
```bash
conda activate C:\IA_Dublagem_Files\env
cd C:\IA_dublagem
```

Choose which tool you want to start:
*   **Game Dubbing:** `python app_jogos.py`
*   **Smart Video Editor:** `python App_videos.py`
*   **File Manager (Mod Manager):** `python vpk_manager.py`

The web interface will magically pop up in your browser!

### 📁 Modding Tools (`tools` folder)
To enable the **Universal Mod Manager** (vpk_manager.py) to extract complex game audio, we use external utilities:
*   **vgmstream-cli**: Located in `tools/vgmstream-cli/`. It handles the conversion of proprietary formats (like Wwise `.wem` or FMOD `.fsb`) into standard `.wav` audio.
*   **Installation**: This folder should be included in the repository. If you need to update it, download the CLI version from [vgmstream.org](https://vgmstream.org) and extract it exactly into this directory.

### 👤 Author & Original Creator

**Paulo Henrik Carvalho de Araújo**
*Original architect and lead developer.*

### 📄 License

This project is licensed under the **Apache License 2.0**.

You are free to use, modify, and distribute this software for any purpose, provided that you maintain the original authorship credits.

---

*Made with ❤️ for the AI Dubbing Community by Paulo Henrik.*
