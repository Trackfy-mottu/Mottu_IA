import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report

# Constantes
IMG_SIZE = (224, 224)
CLASSES = ["mottu_e", "mottu_pop", "mottu_sport"]
MODEL_PATH = "modelo_mottu.keras"
DATASET_DIR = "dataset"  # Pasta contendo as subpastas mottu_e, mottu_pop etc.

def carregar_modelo(caminho_modelo: str):
    if not os.path.exists(caminho_modelo):
        raise FileNotFoundError(f"Modelo não encontrado: {caminho_modelo}")
    return load_model(caminho_modelo)

def carregar_imagens_do_dataset(dataset_path):
    imagens, rotulos_reais, caminhos = [], [], []
    for classe in CLASSES:
        classe_path = os.path.join(dataset_path, classe)
        if not os.path.isdir(classe_path):
            continue

        for nome_arquivo in os.listdir(classe_path):
            if nome_arquivo.lower().endswith((".jpg", ".jpeg", ".png")):
                caminho_img = os.path.join(classe_path, nome_arquivo)
                imagem = cv2.imread(caminho_img)
                imagem = cv2.resize(imagem, IMG_SIZE)
                imagem = imagem.astype("float32") / 255.0
                imagens.append(imagem)
                rotulos_reais.append(classe)
                caminhos.append(caminho_img)

    return np.array(imagens), rotulos_reais, caminhos

def prever_classes(modelo, imagens):
    predicoes = modelo.predict(imagens)
    indices_preditos = np.argmax(predicoes, axis=1)
    rotulos_preditos = [CLASSES[i] for i in indices_preditos]
    return rotulos_preditos

def mostrar_resultado(imagem, predito, real, caminho):
    imagem_copia = (imagem * 255).astype(np.uint8)
    imagem_copia = cv2.resize(imagem_copia, (400, 400))
    cor = (0, 255, 0) if predito == real else (0, 0, 255)
    cv2.putText(imagem_copia, f"Predito: {predito}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, cor, 2)
    cv2.putText(imagem_copia, f"Real: {real}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, cor, 2)
    cv2.imshow(f"{os.path.basename(caminho)}", imagem_copia)
    cv2.waitKey(500)  # Exibe por 500 ms
    cv2.destroyAllWindows()

def main():
    modelo = carregar_modelo(MODEL_PATH)
    imagens, rotulos_reais, caminhos = carregar_imagens_do_dataset(DATASET_DIR)
    imagens_input = imagens.reshape((-1, 224, 224, 3))

    print(f"Total de imagens carregadas: {len(imagens)}")

    rotulos_preditos = prever_classes(modelo, imagens_input)

    # Exibir resultados por imagem
    for i in range(len(imagens)):
        mostrar_resultado(imagens[i], rotulos_preditos[i], rotulos_reais[i], caminhos[i])

    # Relatório de classificação
    print("\n📊 Relatório de Classificação:")
    print(classification_report(rotulos_reais, rotulos_preditos, target_names=CLASSES))

if __name__ == "__main__":
    main()