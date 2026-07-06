# InoLabel

Ferramenta de anotação de imagens e vídeos desenvolvida pelo **Laboratório de Visão Computacional — Inovisão**.
Suporta cinco modos de trabalho: tracking, detecção padrão, detecção orientada (OBB), keypoint detection (pose) e classificação de imagens.

---

## Instalação — Linux (Ubuntu/Debian)

### 1. Dependências do sistema

```bash
sudo apt-get update
sudo apt-get install -y \
    python3 python3-pip python3-tk \
    build-essential python3-dev cmake \
    git
```

> `python3-tk` é obrigatório para a interface gráfica.
> `build-essential`, `python3-dev` e `cmake` são necessários para compilar `lap` e `cython-bbox`.

### 2. Instalar Miniconda (recomendado)

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# Feche e reabra o terminal após a instalação
```

### 3. Criar ambiente Python

```bash
conda create -n inolabel python=3.9 -y
conda activate inolabel
```

### 4. Instalar dependências do projeto

```bash
git clone <url-do-repositorio>
cd tracking-anotator
pip install -r requirements.txt
```

### 5. Rodar

```bash
python main.py
```

---

## Instalação — Windows 11

### 1. Instalar Python 3.9

1. Acesse [python.org/downloads](https://www.python.org/downloads/) e baixe o Python **3.9.x** (64-bit)
2. No instalador, marque **"Add Python to PATH"** antes de clicar em Install
3. Após instalar, abra o **Prompt de Comando** e confirme:
   ```cmd
   python --version
   ```

### 2. Instalar Visual C++ Build Tools

Necessário para compilar `lap` e `cython-bbox`.

1. Acesse [visualstudio.microsoft.com/visual-cpp-build-tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
2. Baixe e execute o instalador
3. Selecione **"Desenvolvimento para desktop com C++"** e clique em Instalar
4. Aguarde (pode demorar alguns minutos)

### 3. Instalar CMake

1. Acesse [cmake.org/download](https://cmake.org/download/) e baixe o instalador `.msi`
2. Durante a instalação, selecione **"Add CMake to the system PATH"**

### 4. Instalar Git (opcional, para clonar o repositório)

1. Acesse [git-scm.com](https://git-scm.com/) e instale com as opções padrão

### 5. Clonar e instalar o projeto

Abra o **Prompt de Comando** ou **PowerShell**:

```cmd
git clone <url-do-repositorio>
cd tracking-anotator
pip install -r requirements.txt
```

### 6. Rodar

```cmd
python main.py
```

> **Problema com Tkinter no WSL?** Configure um X server (VcXsrv ou X410) e a variável `DISPLAY`. Em ambiente Windows nativo (sem WSL) o Tkinter funciona sem configuração adicional.

---

## Gerar executável (build)

O script `build.sh` detecta o sistema operacional automaticamente e gera uma pasta autocontida com o executável.

### Linux

```bash
bash build.sh
```

### Windows (Git Bash ou WSL)

```bash
bash build.sh
```

### Windows (Prompt de Comando / PowerShell)

```cmd
pip install pyinstaller
python -m PyInstaller --noconfirm --onedir --windowed --name InoLabel ^
    --add-data "assets;assets" ^
    --hidden-import PIL._tkinter_finder ^
    --hidden-import cv2 ^
    --hidden-import ultralytics ^
    --collect-all ultralytics ^
    main.py
```

O executável gerado fica em:

```
dist/InoLabel-linux/InoLabel/InoLabel        # Linux
dist/InoLabel-windows/InoLabel/InoLabel.exe  # Windows
```

> **Importante:** coloque o arquivo `model.pt` e a pasta `dataset/` ao lado do executável antes de rodar. O executável não inclui o modelo nem os dados — apenas o código da aplicação.

> **Tamanho esperado:** entre 1.5 GB e 3 GB (Ultralytics/PyTorch são pesados).

---

## Como usar

```bash
python main.py   # ou execute o binário gerado pelo build
```

O wizard de configuração abrirá pedindo:

1. **Modo** — escolha entre os cinco modos abaixo
2. **Dataset** — pasta, vídeo, imagem única ou lista `.txt`/`.lst`
3. **Estado de saída** — continuar saída anterior, usar como template ou criar novo. Por padrão os estados são salvos em **`state_saved/`** na raiz da aplicação (a pasta-pai é editável no wizard)
4. **Modelo e classes** — adicione um ou mais pesos YOLO `.pt` (opcional) e configure as classes da sessão

---

## Modos de anotação

| Modo | Descrição |
|------|-----------|
| **Tracking** | Mantém identidade dos objetos entre frames via BYTETracker por classe |
| **Detecção padrão** | Caixas independentes por frame, sem `track_id` |
| **Detecção orientada (OBB)** | Caixas rotacionadas com ângulo, exportáveis no formato YOLO OBB |
| **Keypoint detection** | Pontos-chave ordenados por instância, exportáveis em COCO Keypoints e YOLO Pose |
| **Classificação** | Copia imagens para subpastas por classe ao pressionar o atalho da classe |

O modelo YOLO é **opcional** em todos os modos — é possível anotar inteiramente de forma manual.

---

## Modo Keypoint detection (pose)

Cada instância de objeto possui uma **classe** e um conjunto de **keypoints com ordem fixa** (a ordem é semântica e nunca é reordenada automaticamente durante a anotação).

### Configuração

No wizard, ao escolher o modo **Keypoint detection**, defina por classe a lista de keypoints **em ordem**, separados por vírgula (ex: `top_left, top_right, bottom_right, bottom_left`). A configuração é obrigatória — o início é bloqueado se alguma classe não tiver keypoints definidos — e fica salva em `categories[].keypoints` para reabrir e continuar depois.

### Como anotar

- Clique para posicionar cada ponto na ordem definida. Ao completar o último ponto da classe, a instância **fecha automaticamente**.
- Para classes **sem lista fixa** (modo livre), a forma fecha ao clicar novamente sobre o **primeiro ponto** (a partir de 3 pontos) — um anel verde e o texto `fechar` indicam quando o clique vai fechar.
- A **bounding box** é calculada automaticamente a partir dos pontos com visibilidade `> 0`.

Visibilidade (convenção COCO): `0` ausente, `1` oculto/anotável, `2` visível.

| Tecla / Mouse | Ação (modo keypoint) |
|-------|----------------------|
| `F` | Finalizar instância atual |
| `X` | Marcar o ponto atual como ausente (`v=0`) — durante a colocação |
| `C` | Com um ponto **selecionado**, cicla a visibilidade **desse** ponto (`2 → 1 → 0`); sem seleção, define a visibilidade do **próximo** ponto a colocar |
| **Clique direito** | Sobre um ponto já anotado: alterna **visível (2) ↔ oculto (1)** |
| `Backspace` | Remover o último ponto (ou cancelar a instância em construção) |
| `S` | Selecionar/mover pontos ou instâncias |
| `Esc` | Cancela a operação atual (instância em construção ou seleção) — não fecha o app |

Para corrigir a visibilidade de pontos **já anotados** (ex.: marcar como oclusos os que estão na imagem mas encobertos): clique direito sobre o ponto para alternar `2 ↔ 1`, ou selecione com `S` e use `C` para passar por todos os estados. Pontos oclusos (`v=1`) aparecem como **círculo vazado**.

### Exportação

O botão **Exportar dataset** abre a **mesma tela de exportação** dos detectores (seleção de pasta de saída, split train/val/test e data augmentation). Formatos:

- **COCO Keypoints** — `keypoints: [x,y,v,...]`, `num_keypoints`, `bbox` e `categories[].keypoints`.
- **YOLO Pose** — `class cx cy w h x1 y1 v1 ...` (tudo normalizado) e `kpt_shape` no `data.yaml`. Todas as linhas têm o mesmo número de keypoints; pontos ausentes saem como `0 0 0`.

O **data augmentation** transforma também os keypoints (flip, rotação, etc., além das operações fotométricas) e é aplicado apenas na pasta `train`.

### Conserto de anotações antigas

`utils/fix_keypoint_coco.py` repara um COCO Keypoints inconsistente: preenche `categories[].keypoints`, remove ponto de fechamento duplicado e recalcula `bbox`/`num_keypoints`. Por **padrão**, instâncias de 4 pontos são ordenadas em **TL → TR → BR → BL** (ideal para documentos); use `--no-sort-corners` para preservar a ordem original.

```bash
python utils/fix_keypoint_coco.py outputs/.../annotations_keypoints.coco.json
```

---

## Atalhos principais

A maioria dos atalhos é **remapeável** pelo editor visual (botão **Atalhos** na barra superior). Os valores abaixo são os padrões do perfil `arrows`. As teclas `1–9` e `Esc` são fixas e não aparecem no editor.

| Tecla | Ação |
|-------|------|
| `Enter` | Validar / salvar frame atual |
| `Espaço` | Rejeitar / avançar frame |
| `→` / `←` | Navegar entre frames salvos (perfil `arrows`) |
| `D` / `A` | Navegar entre frames salvos (perfil `wasd`) |
| `K` | Liga/desliga anotação manual |
| `S` | Modo de seleção de anotação |
| `H` | Liga/desliga modo mover imagem (pan) |
| `R` | Redefinir ROI |
| `E` | Editar ID de tracking (apenas modo tracking) |
| `Ctrl+Z` | Desfazer última ação |
| `Ctrl+0` | Ajustar imagem na tela |
| `1–9` | Trocar classe ativa |
| `Scroll` | Zoom centrado no cursor |
| `Esc` | Sair |

### Editor de atalhos

Clique no botão **Atalhos: arrows** (topbar) para abrir o editor visual. Nele é possível:

- Remapear qualquer ação clicando no botão da tecla e pressionando a nova tecla
- Criar perfis personalizados ou alternar entre `arrows` e `wasd`
- Restaurar os padrões de fábrica por perfil
- Detectar conflitos em tempo real (aviso laranja, não bloqueante)

O perfil ativo é salvo em `.local/keybinds.json` e restaurado automaticamente na próxima sessão.

---

## Rotação visual da imagem

Os botões **↺ Girar** e **Girar ↻** na barra lateral rotacionam a exibição em 90° sem alterar a imagem salva nem as coordenadas das bounding boxes. A rotação é desfeita automaticamente ao avançar para o próximo frame. Atalhos de teclado podem ser atribuídos via editor de atalhos (grupo **Imagem**).

---

## Fluxo de ROI (Tracking / Detecção / OBB)

1. Ao abrir cada fonte, clique 4 pontos para definir o ROI (ordem livre; o código ordena automaticamente).
2. A homografia é calculada e `warpPerspective` é aplicado internamente.
3. A detecção ocorre na imagem retificada; as caixas são mapeadas de volta ao frame original.
4. Pressione `R` a qualquer momento para redefinir o ROI sem perder anotações já salvas.

---

## Exportação de dataset

Clique em **Exportar dataset** na barra lateral para abrir a tela de exportação. As opções disponíveis são:

| Opção | Descrição |
|-------|-----------|
| **Destino / Nome da pasta** | Caminho e nome da pasta de saída |
| **YOLO** | Exporta imagens + labels `.txt` e `data.yaml` |
| **COCO (.json)** | Exporta `annotations.coco.json` + pasta `images/` com as imagens |
| **Split train/val/test** | Divide as imagens em proporções configuráveis |
| **Data augmentation** | Gera cópias aumentadas por imagem (flip, brilho, ruído, etc.) |

A exportação roda em **background** — a interface permanece responsiva. Uma barra de progresso exibe o avanço imagem por imagem; ao concluir, ela some automaticamente.

---

## Saídas geradas

```
outputs/<tarefa>_<DD.MM.HH-MM>/   (ex: detecção_25.05.14-30)
├── images/                         # frames salvos (originais ou retificados)
├── annotations.coco.json           # COCO com track_id (tracking) ou bbox simples
├── annotations_obb.coco.json       # COCO OBB (modo OBB)
├── annotations_keypoints.coco.json # COCO Keypoints (modo keypoint)
├── annotations_detection.coco.json # COCO detecção padrão exportado pelo botão
├── yolo_dataset/                   # dataset YOLO exportado pelo botão
│   ├── data.yaml
│   └── images/ labels/ {train,val,test}/
└── homography.json                 # homografias por fonte (tracking/detecção)
```

Exportação manual via botão cria uma pasta separada (nunca sobrescreve `outputs/`):

```
<destino>/<nome>/
├── annotations.coco.json    # formato COCO
├── images/                  # imagens (cópia)
└── (ou estrutura YOLO acima)
```

---

## Utilitários

### Converter COCO → YOLO

```bash
python utils/convert_coco_to_yolo_dataset.py outputs/.../annotations.coco.json \
    --image-root outputs/.../images \
    --output-root outputs/.../yolo_dataset \
    --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1
```

### Consolidar splits YOLO em train único

```bash
python utils/merge_yolo_splits.py outputs/.../yolo_dataset \
    --output-root outputs/.../yolo_dataset_train_only
```

### Converter anotações de tracking → detecção

```bash
python utils/convert_coco_tracking_to_detection.py outputs/.../annotations.coco.json
```

---

## Configurações em `app/config.py`

| Variável | Descrição |
|----------|-----------|
| `CONF_THRESHOLD` | Limiar de confiança do YOLO (padrão `0.40`) |
| `SAVE_RECTIFIED_FRAMES` | `True` salva frames com warpPerspective; `False` salva originais |
| `MANUAL_IOU_THRESHOLD` | IoU mínimo para fundir anotação manual com detecção existente |

Os caminhos de dataset, modelo e saída são configurados no wizard — os valores em `config.py` servem apenas como sugestão inicial.

---

## Resolução de problemas

| Problema | Solução |
|----------|---------|
| Tkinter não abre no WSL | Configure um X server e a variável `DISPLAY`, ou rode em ambiente gráfico nativo |
| Tkinter ausente no Linux | `sudo apt-get install python3-tk` |
| `lap`/`cython_bbox` falhando no Linux | Instale `build-essential python3-dev cmake` e tente novamente |
| `lap`/`cython_bbox` falhando no Windows | Instale o Visual C++ Build Tools e CMake conforme descrito acima |
| Logo não aparece na tela inicial | Verifique se `assets/inovisao.png` existe e se `Pillow` está instalado |
| Atalhos não respondem após remapear | Verifique conflitos no editor de atalhos (aviso laranja) |

---

## Créditos

- BYTETracker retirado de [FoundationVision/ByteTrack](https://github.com/FoundationVision/ByteTrack)
- Detecção e OBB via [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
