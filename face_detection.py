import cv2
import numpy as np
import os
import sys

class FaceDetector:
    def __init__(self):
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
        self.face_cascade = cv2.CascadeClassifier(haar_cascade_path)
        
        # Parâmetros iniciais para detecção facial
        self.scale_factor = 1.3
        self.min_neighbors = 5
        self.min_size = (30, 30)
        
        # Inicializar a webcam
        self.cap = cv2.VideoCapture(0)
        
        # Verificar se a webcam foi aberta corretamente
        if not self.cap.isOpened():
            print("Erro ao abrir a webcam!")
            sys.exit()
    
    def detect_faces(self, frame):
        # Converter o frame para escala de cinza (melhora a performance da detecção)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detectar faces usando o classificador Haar Cascade
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size
        )
        
        return faces
    
    def draw_faces(self, frame, faces):
        # Desenhar retângulos ao redor das faces detectadas
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        return frame
    
    def create_parameter_window(self):
        # Criar janela para controles deslizantes
        cv2.namedWindow('Parameters')
        
        # Criar controles deslizantes para ajustar os parâmetros
        cv2.createTrackbar('Scale Factor x10', 'Parameters', int(self.scale_factor * 10), 20, self.update_scale_factor)
        cv2.createTrackbar('Min Neighbors', 'Parameters', self.min_neighbors, 20, self.update_min_neighbors)
        cv2.createTrackbar('Min Size', 'Parameters', self.min_size[0], 100, self.update_min_size)
    
    def update_scale_factor(self, value):
        # Atualizar o valor do scale factor (dividido por 10 para obter valores decimais)
        self.scale_factor = max(1.1, value / 10)
    
    def update_min_neighbors(self, value):
        # Atualizar o valor de min neighbors
        self.min_neighbors = max(1, value)
    
    def update_min_size(self, value):
        # Atualizar o valor de min size
        self.min_size = (max(20, value), max(20, value))
    
    def display_parameters(self, frame):
        # Exibir os valores atuais dos parâmetros na tela
        cv2.putText(frame, f'Scale Factor: {self.scale_factor:.1f}', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f'Min Neighbors: {self.min_neighbors}', (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f'Min Size: {self.min_size[0]}', (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return frame
    
    def run(self):
        # Criar janela para controles deslizantes
        self.create_parameter_window()
        
        while True:
            # Capturar frame da webcam
            ret, frame = self.cap.read()
            
            if not ret:
                print("Erro ao capturar o frame!")
                break
            
            # Detectar faces no frame
            faces = self.detect_faces(frame)
            
            # Desenhar retângulos ao redor das faces detectadas
            frame = self.draw_faces(frame, faces)
            
            # Exibir os valores atuais dos parâmetros na tela
            frame = self.display_parameters(frame)
            
            # Exibir o número de faces detectadas
            cv2.putText(frame, f'Faces detectadas: {len(faces)}', (10, 120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Exibir o frame
            cv2.imshow('Face Detection', frame)
            
            # Sair se a tecla 'q' for pressionada
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Liberar recursos
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Criar e executar o detector facial
    detector = FaceDetector()
    detector.run()