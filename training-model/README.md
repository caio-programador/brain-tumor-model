# Brain Tumor MRI Images - Treinamento e Predição

Este projeto treina um modelo ResNet18 para classificar imagens de ressonância magnética de tumores cerebrais usando o dataset do Kaggle:

https://www.kaggle.com/datasets/fernando2rad/brain-tumor-mri-images-30-classes?resource=download

O fluxo esperado é:

1. baixar e extrair o dataset na pasta `dataset/` na raiz do projeto
2. executar `train.py` para treinar o modelo
3. usar `predict.py` com uma imagem externa ao dataset para fazer a previsão

## Estrutura esperada

Depois de extrair o dataset, a raiz do projeto deve ficar mais ou menos assim:

```text
training-model/
├── dataset/
│   ├── Astrocytoma T1/
│   ├── Astrocytoma T1C+/
│   ├── Astrocytoma T2/
│   ├── Ependymoma T1/
│   ├── Ependymoma T1C+/
│   ├── Ependymoma T2/
│   └── ...
├── train.py
├── predict.py
├── requirements.txt
└── best_model.pth
```

O `train.py` usa apenas as pastas com `T1C+` para montar automaticamente a pasta `dataset_t1c/` e treinar o modelo. O `predict.py` carrega esse modelo treinado para prever uma imagem nova.

## Preparação do ambiente

Recomendo usar um ambiente virtual:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Se você já estiver usando o `venv/` que existe no projeto, basta ativá-lo antes de rodar os comandos.

## Baixar e organizar o dataset

1. Acesse a página do Kaggle do dataset.
2. Faça o download e extraia o arquivo compactado.
3. Copie a pasta extraída para dentro da raiz do projeto com o nome `dataset/`.
4. Verifique se dentro de `dataset/` existem pastas como `Astrocytoma T1C+`, `Meningioma T1C+`, `Normal T1C+` e assim por diante.

Se quiser, a estrutura mínima precisa ser a mesma que o `train.py` espera: várias classes, cada uma com as imagens organizadas em subpastas.

## Treinar o modelo

Com o ambiente ativado e o dataset já colocado na pasta correta, rode:

```bash
python train.py
```

Durante a execução, o script:

1. cria a pasta `dataset_t1c/` se ela ainda não existir
2. copia apenas as imagens das pastas `T1C+` para `dataset_t1c/`
3. treina a ResNet18 por 10 épocas
4. salva o melhor modelo em `best_model.pth`

No fim do treino, o arquivo `best_model.pth` deve estar na raiz do projeto.

## Fazer predição com uma imagem fora do dataset

O `predict.py` espera uma imagem chamada `image.png` na raiz do projeto.

Para prever com uma imagem externa ao dataset:

1. escolha uma imagem nova, que não esteja dentro das pastas do dataset
2. salve essa imagem como `image.png` na raiz do projeto, ou ajuste a variável `IMAGE_PATH` dentro de `predict.py`
3. execute:

```bash
python predict.py
```

O script vai imprimir:

1. as classes encontradas em `dataset_t1c/`
2. a classe prevista
3. a confiança da predição

## Observações importantes

- O treino usa somente as imagens `T1C+`.
- O `predict.py` precisa do arquivo `best_model.pth` gerado pelo treino.
- Se você trocar a ordem ou o conteúdo das pastas em `dataset_t1c/`, a lista de classes usada na predição pode mudar.
- Para testar a predição, use uma imagem externa ao dataset e no mesmo estilo de entrada esperado pelo modelo, de preferência uma imagem em tons de cinza de ressonância magnética.

## Comandos rápidos

```bash
source venv/bin/activate
pip install -r requirements.txt
python train.py
python predict.py
```
