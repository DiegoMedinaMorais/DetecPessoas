# 🎥 Sistema de Monitoramento com YOLO + Tracking

Detecção e contagem de pessoas em tempo real com rastreamento (ByteTrack), API REST, WebSocket e interface gráfica.

## 📦 Instalação

```bash
# 1. Clone ou baixe o projeto

git clone https://github.com/seu-usuario/seu-repositorio.git
cd seu-repositorio
```


# 2. Crie ambiente virtual (opcional mas recomendado)
``` bash
python -m venv venv
```

# Windows: 
``` bash
venv\Scripts\activate
```

# Linux/Mac: 
``` bash
source venv/bin/activate
```

# 3. Instale as dependências
``` bash
pip install -r requirements.txt
```
## 📥 Baixar modelos (para detecção facial)
Se quiser detecção facial, baixe os arquivos:

```bash
mkdir models
curl -L -o models/deploy.prototxt https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/deploy.prototxt
curl -L -o models/res10_300x300_ssd_iter_140000_fp16.caffemodel https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000_fp16.caffemodel
```
Se não quiser detecção facial, defina "enable_face_detection": false no config.json.

## ⚙️ Configuração (opcional)
Crie um arquivo config.json na raiz com seus parâmetros (exemplo mínimo):

```json
{
  "camera_sources": ["0"],
  "confidence": 0.5,
  "line_position": 0.6,
  "line_orientation": "horizontal",
  "enable_face_detection": true,
  "save_snapshots": true
}
camera_sources: ["0"] (webcam), ["rtsp://..."] (IP), ["video.mp4"] (arquivo)

line_position: posição da linha (0 a 1)

line_orientation: "horizontal" ou "vertical"

roi: [x1, y1, x2, y2] (opcional)
```

## ▶️ Como executar
```bash
python app.py
```

A interface gráfica abre. Selecione a fonte, digite o endereço e clique em Iniciar.

## 🌐 API e WebSocket
API: http://localhost:8000

GET /status – estado completo

GET /people – pessoas na cena

GET /total – total acumulado

WebSocket: ws://localhost:8000/ws (atualizações a cada 500ms)

## 📁 Estrutura de pastas
text
projeto/
├── app.py                  # Interface gráfica
├── config.py               # Configurações
├── camera_manager.py
├── detector.py
├── face_detector.py
├── counter.py
├── draw.py
├── utils.py
├── api.py
├── requirements.txt
├── config.json             # (opcional)
├── models/                 # pesos dos modelos
├── logs/                   # logs em CSV
├── snapshots/              # imagens salvas
└── data/                   # counter.json (persistência)

## 💾 Persistência
Contagem total salva em data/counter.json

Logs diários em logs/YYYY-MM-DD.csv

Snapshots em snapshots/YYYY-MM-DD/person_XXXXXX.jpg (se ativado)

## ⚠️ Problemas comuns
Erro no MediaPipe: desative enable_face_detection no config.json ou use OpenCV DNN.

GPU não detectada: instale PyTorch com CUDA.

Erro de serialização: certifique-se de usar a versão atualizada do counter.py.

## 📄 Licença
MIT
```
---
```
