# Brain Tumor MRI Classifier

Aplicação para classificar imagens de ressonância magnética de tumor cerebral.

O repositório está dividido em 3 partes:

- `frontend/`: interface web em React + Vite
- `backend/`: API em FastAPI para inferência
- `training-model/`: treinamento e teste do modelo, com README próprio

## Visão geral

O fluxo normal é:

1. ter um modelo treinado em `backend/models/best_model.pth`
2. subir a API do backend
3. subir o frontend e enviar uma imagem para análise

Se você quiser treinar o modelo do zero, siga as instruções em `training-model/README.md`.

## Requisitos

- Python 3
- Node.js e npm
- Modelo treinado disponível para o backend

## Rodando localmente

### 1) Backend

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Se necessário, ajuste o arquivo `.env`:

```env
MODEL_PATH=models/best_model.pth
HOST=0.0.0.0
PORT=8000
```

O backend procura o modelo em `backend/models/best_model.pth` por padrão. Se o arquivo vier do treino em `training-model/best_model.pth`, copie-o para esse caminho ou altere `MODEL_PATH`.

Inicie a API com:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

A documentação automática fica em:

- http://localhost:8000/docs
- http://localhost:8000/redoc

### 2) Frontend

Em outro terminal:

```bash
cd frontend
npm install
cp .env.example .env
```

Confirme que o `.env` aponta para a API:

```env
VITE_API_URL=http://localhost:8000
```

Depois rode:

```bash
npm run dev
```

O app fica disponível em:

- http://localhost:5173

### 3) Treinamento do modelo

O treino e a predição offline estão documentados em `training-model/README.md`.

Resumo rápido:

```bash
cd training-model
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python training/train.py
python testing/predict.py
```

## Estrutura principal

```text
backend/
frontend/
training-model/
```

## Observação

O frontend chama a API com `POST /predict` enviando a imagem no campo `image`.