# Aplicação de Reconhecimento Facial

## Objetivo
Esta aplicação realiza reconhecimento e identificação facial do usuário utilizando OpenCV e Haar Cascade. O sistema detecta rostos em tempo real através da webcam, destacando-os com retângulos e permitindo ajustes de parâmetros para melhorar a detecção.

## Tecnologias Utilizadas
- Python 3.8+
- OpenCV (cv2)
- Haar Cascade Classifier

## Video Demonstrativo 
- https://www.youtube.com/watch?v=IulTRE_GrRY

## Dependências
```
python -m pip install opencv-python numpy
```

## Execução
1. Clone este repositório
2. Instale as dependências
3. Execute o arquivo principal:
```
python face_detection.py
```

## Parâmetros Ajustáveis
A aplicação permite ajustar os seguintes parâmetros em tempo real:

- **Scale Factor**: Controla o fator de escala para detecção de rostos de diferentes tamanhos (valores típicos: 1.1 - 1.5)
- **Min Neighbors**: Define o número mínimo de vizinhos que cada retângulo candidato deve ter (valores típicos: 3 - 6)
- **Min Size**: Tamanho mínimo possível do objeto (em pixels)

## Impacto dos Parâmetros
- **Scale Factor**: Valores menores aumentam a chance de detecção, mas também aumentam falsos positivos e o tempo de processamento
- **Min Neighbors**: Valores maiores resultam em menos detecções, mas com maior qualidade/confiança
- **Min Size**: Ajuda a eliminar detecções muito pequenas que provavelmente são falsos positivos

## Nota Ética sobre Uso de Dados Faciais
Esta aplicação processa dados biométricos (imagens faciais) localmente, sem armazenamento ou transmissão. Ao utilizar esta ferramenta, considere:

1. **Privacidade**: Obtenha consentimento explícito antes de usar em imagens de terceiros
2. **Segurança**: Os dados são processados apenas localmente, sem persistência
3. **Transparência**: Informe aos usuários sobre o funcionamento da detecção facial
4. **Limitações**: A tecnologia de reconhecimento facial pode apresentar vieses e falhas

Esta aplicação foi desenvolvida apenas para fins educacionais e de demonstração.
