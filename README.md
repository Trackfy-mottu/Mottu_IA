# 🏍️ Classificação de Motos da Mottu com Visão Computacional

Este projeto utiliza visão computacional com redes neurais convolucionais (CNNs) para identificar e classificar três tipos de motos utilizadas pela empresa Mottu: `mottu_e`, `mottu_pop` e `mottu_sport`.

---

## 📁 Estrutura do Dataset

O conjunto de dados é composto apenas por imagens, organizadas da seguinte forma:

```
dataset/
├── mottu_e/
│   ├── mottu_e_1.jpg
│   ├── ...
├── mottu_pop/
│   ├── mottu_pop_1.jpg
│   ├── ...
├── mottu_sport/
    ├── mottu_sport_1.jpg
    ├── ...
```

As imagens foram aumentadas utilizando técnicas de **data augmentation** como rotação, inversão e variações de brilho, para melhorar a generalização do modelo.

---

## 🧠 Arquitetura do Modelo

O modelo foi desenvolvido com Keras e TensorFlow. A arquitetura segue uma CNN básica com as seguintes camadas principais:

- `Conv2D` + `MaxPooling2D`
- `Dropout` para evitar overfitting
- `Flatten` e `Dense` para classificação final

Modelo salvo em:  
```bash
modelo_mottu.keras
```

---

## ⚙️ Pré-processamento e Predição

Cada imagem é:

1. Carregada com OpenCV
2. Redimensionada para `224x224`
3. Normalizada (valores entre 0 e 1)
4. Passada para o modelo para predição

```python
img = cv2.imread(path)
img_resized = cv2.resize(img, (224, 224)) / 255.0
pred = model.predict(img_resized.reshape((1, 224, 224, 3)))
```

---

## ✅ Resultados

Após o treinamento e validação, o desempenho foi avaliado com as seguintes métricas:

### 📊 Relatório de Classificação:

| Classe       | Precision | Recall | F1-score | Suporte |
|--------------|-----------|--------|----------|---------|
| `mottu_e`    | 0.86      | 0.40   | 0.54     | 48      |
| `mottu_pop`  | 0.66      | 0.89   | 0.76     | 64      |
| `mottu_sport`| 0.81      | 0.87   | 0.84     | 60      |

**Acurácia total:** `0.74`  
**Macro média (média simples entre as classes):**  
- Precision: 0.78  
- Recall: 0.72  
- F1-score: 0.71  

**Média ponderada (ajustada ao tamanho de cada classe):**  
- Precision: 0.77  
- Recall: 0.74  
- F1-score: 0.73  

---

### 🧠 Interpretação dos Resultados

- A classe `mottu_sport` foi a mais bem identificada pelo modelo, com F1-score de 0.84.
- A `mottu_pop` teve uma ótima **revocação (recall)** de 0.89, ou seja, o modelo consegue detectar bem quando é realmente uma `mottu_pop`.
- A `mottu_e`, por outro lado, teve um desempenho mais fraco, especialmente em recall (0.40), indicando que muitas `mottu_e` reais foram classificadas como outra coisa.

---

## 📦 Requisitos

Instale as dependências com:

```bash
pip install -r requirements.txt
```

Principais pacotes utilizados:
- `tensorflow`
- `opencv-python`
- `numpy`
- `Pillow`

---

## ▶️ Como Executar

1. Execute o script para treinamento do modelo:
```bash
python treinar_modelo.py
```
2. Certifique-se de ter o modelo salvo como `modelo_mottu.keras`
3. Execute o script principal:

```bash
python main.py
```

Ele fará a predição de todas as imagens no diretório `dataset`, exibindo os resultados por imagem.
