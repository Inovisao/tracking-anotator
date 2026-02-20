# Ferramenta de Anotacao com ROI, Homografia e Tracking (YOLO)

Interface Tkinter para validar e anotar deteccoes de multiplos videos em sequencia, aplicando ROI via 4 cliques, homografia (warpPerspective), tracking persistente (YOLO `model.track`) e salvando em formato COCO/MOT com `track_id`.

Bytetracker retirado de: https://github.com/FoundationVision/ByteTrack

## Requisitos de ambiente
- Python 3.9+ via `conda`.
- Tkinter instalado (Ubuntu/Debian: `sudo apt-get install python3-tk`).
- Toolchain para pacotes nativos (`build-essential`, `python3-dev`, `cmake`) — necessario para `lap`/`cython_bbox`.
- (Opcional) CUDA instalado caso queira acelerar o YOLO na GPU.
- Peso YOLOv11 `yolo11l.pt` na raiz do projeto (baixe da Ultralytics).

## Instalacao rapida
```bash
conda create -n tracking-anotator python=3.9
conda activate tracking-anotator
conda install -c conda-forge --file requirements.txt
```

Coloque seus videos em `videos/` (subpastas sao suportadas) e deixe `yolo11l.pt` na raiz do repo. A pasta `output_dataset/` sera criada automaticamente.

## Funcionalidades principais
- Percorre todos os videos em `videos/` (subpastas incluidas) e processa um por vez.
- Selecao de ROI por 4 cliques; calcula homografia (M e M_inv) e aplica warpPerspective.
- Detecta na imagem retificada, mapeia caixas de volta ao frame original e descarta deteccoes fora do ROI.
- Tracking persistente via `model.track(persist=True)` com `track_id` global.
- Anotacao manual com `track_id` exclusivo e reuso opcional por IoU.
- Salva COCO/MOT em `output_dataset/annotations.coco.json`, frames em `output_dataset/images/` e homografias em `output_dataset/homography.json`.
- Opcao de salvar frames retificados ou originais (`SAVE_RECTIFIED_FRAMES` em `main.py`).

## Estrutura esperada
```
tracking-anotator/
├── main.py
├── main_test.py
├── yolo11l.pt
└── videos/
    ├── Classe 1
    │   └── video1.mp4
    └── Classe 2
        └── video2.avi
```

## Como rodar
- Interface de anotacao:
```bash
python main.py
```
  1. Ao abrir cada video, clique 4 pontos para definir o ROI (ordem livre; o codigo ordena).  
  2. Interface apos ROI:
     - Enter: validar/salvar frame atual  
     - Espaco: pular frame  
     - K: liga/desliga modo anotacao manual  
     - Botao esquerdo + arrastar: desenhar caixa manual (quando anotacao ON)  
     - Botao "Remover anotacao": liga/desliga modo remocao (clique sobre uma caixa)  
     - R: redefinir ROI (nao altera anotacoes ja salvas do video atual)  
  3. Ao fim de cada video, o proximo e aberto automaticamente.

- Teste rapido (sem UI):
```bash
python main_test.py
```
Roda 10 frames do primeiro video encontrado em `videos/` e imprime contagens/track_ids (nao abre a interface).

Saidas geradas:
- `output_dataset/images/{video}_frame_00001.jpg` (originais ou retificados, conforme `SAVE_RECTIFIED_FRAMES`)
- `output_dataset/annotations.coco.json` (inclui `track_id` e `video`)
- `output_dataset/homography.json` (lista de homografias por video)

## Configuracoes uteis em `main.py`
- `SAVE_RECTIFIED_FRAMES`: False (salva originais) ou True (salva warpPerspective).
- `CONF_THRESHOLD`: limiar de confianca do YOLO.
- `TARGET_CLASS`: classe alvo (padrao `car`).

## Resolucao de problemas
- Tkinter nao abre no WSL: precisa de X server e variavel DISPLAY; ou rode em ambiente grafico nativo.
- Tkinter ausente no Linux: `sudo apt-get install python3-tk`.
- `lap`/`cython_bbox` falhando ao compilar: instale `build-essential python3-dev cmake` e tente novamente.
- Pesos nao encontrados: confirme `yolo11l.pt` na raiz ou ajuste `WEIGHTS_PATH` em `main.py`.
