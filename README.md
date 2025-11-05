# Annotation Tool

Ferramenta interativa para validar detecoes da classe `car` usando um modelo YOLO treinado (`../best.pt`) e registrar resultados em um arquivo `annotations.coco.json` no formato COCO.

## Requisitos
- Python 3.9+
- Dependencias instaladas no ambiente ativo:
  - `ultralytics`
  - `opencv-python`
  - `numpy`
  - `pillow`
  - `tqdm` (opcional, mas recomendada se usar o pipeline de extracao de frames)

> Observacao: `tkinter` ja acompanha a instalacao oficial do Python em Windows e macOS. No Linux pode ser necessario instalar `python3-tk` via gerenciador de pacotes.

## Estrutura esperada
```
tracker/
├── best.pt
├── videos/
│   └── ... video.mp4
└── testa_tracking/
    ├── main.py
    └── README.md
```

## Como usar
1. Ajuste `VIDEO_PATH` em `main.py` para apontar para o video a ser validado (caminho absoluto ou relativo).
2. Certifique-se de que `best.pt` esteja no diretorio pai (`../best.pt` a partir de `testa_tracking`).
3. Ative o ambiente virtual e execute:
   ```bash
   python main.py
   ```
4. Para cada frame processado com deteccoes (confidencia >= 90% para `car`), uma janela mostrara as caixas e a confianca.

## Controles
- `Validar (Enter)`: mantem as deteccoes exibidas e grava no `annotations.coco.json`.
- `Rejeitar (Espaco)`: ignora o frame atual e segue para o proximo.
- `Sair (Esc)`: encerra o processo; se houver anotacoes em memoria, sao persistidas antes de fechar.

## Saida COCO
O arquivo `annotations.coco.json` e atualizado continuamente com a estrutura:
- `categories`: lista contendo a classe `car` (id 1).
- `images`: um item por frame validado (`file_name`, `width`, `height`, `id`).
- `annotations`: cada deteccao aprovada com `bbox` no formato `[x, y, largura, altura]`, `score`, `image_id` e `category_id`.

Os campos `image_id` e `annotation_id` sao gerados sequencialmente e começam em 1. Cada frame aprovado recebe um nome padrao `VIDEO_FRAME_xxxxx.jpg` (sem gerar arquivo fisico; serve como referencia para integracoes posteriores).

## Dicas
- Se quiser trabalhar com outro limiar de confianca ou classe, ajuste `CONF_THRESHOLD` e `TARGET_CLASS` no topo do `main.py`.
- Caso o video seja longo, considere interromper com `Esc`; o progresso ate o momento sera mantido no JSON.
