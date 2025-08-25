import cv2
import numpy as np
import os
import sys

def compare_parameters(image_path=None):
    # Diretório do arquivo atual
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Caminho para o classificador Haar Cascade
    haar_cascade_path = os.path.join(current_dir, 'haarcascade_frontalface_default.xml')
    
    # Verificar se o arquivo do classificador existe, caso contrário, baixá-lo
    if not os.path.exists(haar_cascade_path):
        print("Baixando o classificador Haar Cascade...")
        import urllib.request
        url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
        urllib.request.urlretrieve(url, haar_cascade_path)
        print("Classificador baixado com sucesso!")
    
    # Carregar o classificador Haar Cascade para detecção facial
    face_cascade = cv2.CascadeClassifier(haar_cascade_path)
    
    # Se nenhuma imagem for fornecida, usar a webcam
    if image_path is None:
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        if not ret:
            print("Erro ao capturar o frame!")
            sys.exit()
        cap.release()
        image = frame
    else:
        # Carregar a imagem
        image = cv2.imread(image_path)
        if image is None:
            print(f"Erro ao carregar a imagem: {image_path}")
            sys.exit()
    
    # Converter a imagem para escala de cinza
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Definir diferentes combinações de parâmetros para comparação
    parameters = [
        {"scale_factor": 1.1, "min_neighbors": 3, "min_size": (30, 30), "label": "SF: 1.1, MN: 3"},
        {"scale_factor": 1.3, "min_neighbors": 5, "min_size": (30, 30), "label": "SF: 1.3, MN: 5"},
        {"scale_factor": 1.5, "min_neighbors": 3, "min_size": (30, 30), "label": "SF: 1.5, MN: 3"},
        {"scale_factor": 1.3, "min_neighbors": 8, "min_size": (30, 30), "label": "SF: 1.3, MN: 8"}
    ]
    
    # Criar uma grade de imagens para comparação
    rows = 2
    cols = 2
    result_height = image.shape[0] * rows
    result_width = image.shape[1] * cols
    result = np.zeros((result_height, result_width, 3), dtype=np.uint8)
    
    # Processar cada combinação de parâmetros
    for i, params in enumerate(parameters):
        # Copiar a imagem original
        img_copy = image.copy()
        
        # Detectar faces com os parâmetros atuais
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=params["scale_factor"],
            minNeighbors=params["min_neighbors"],
            minSize=params["min_size"]
        )
        
        # Desenhar retângulos ao redor das faces detectadas
        for (x, y, w, h) in faces:
            cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Adicionar texto com os parâmetros e número de faces detectadas
        cv2.putText(img_copy, params["label"], (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(img_copy, f'Faces: {len(faces)}', (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Calcular a posição na grade
        row = i // cols
        col = i % cols
        
        # Colocar a imagem processada na grade
        result[row*image.shape[0]:(row+1)*image.shape[0], 
               col*image.shape[1]:(col+1)*image.shape[1]] = img_copy
    
    # Exibir a grade de comparação
    cv2.imshow('Parameter Comparison', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Se uma imagem for fornecida como argumento, usá-la; caso contrário, usar a webcam
    if len(sys.argv) > 1:
        compare_parameters(sys.argv[1])
    else:
        compare_parameters()