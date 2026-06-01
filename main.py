from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import torch
import io
from PIL import Image
import torchvision.transforms as transforms
import cv2
import numpy as np
import os
import sys

# Añadir el directorio actual al path para importar Paso3_Modelo
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Paso3_Modelo import HybridCNNViT

app = FastAPI(title="Clasificador de Enfermedades Pulmonares")

# Configurar CORS por si acaso
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Crear carpeta static si no existe
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Cargar el modelo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = HybridCNNViT(num_classes=4).to(device)
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modelo_hibrido_pulmonar.pth")

try:
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Modelo cargado exitosamente.")
except Exception as e:
    print(f"Error cargando el modelo: {e}")

# Clases según el dataset
CLASES = ['Normal', 'COVID', 'Lung_Opacity', 'Viral Pneumonia']

# Transformaciones
transformaciones = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="El archivo no es una imagen válida.")
    
    try:
        # Leer la imagen subida
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        
        # NOTA: En el entrenamiento original (Paso2_Dataset), se aplicaba una máscara de 
        # segmentación pulmonar (Lung Masking) mediante bitwise_and.
        # Como en inferencia no tenemos esa máscara del usuario, usaremos la imagen completa.
        # Esto podría afectar ligeramente la precisión respecto a las métricas del paper.
        
        # Convertir a RGB (necesario por el backbone ResNet preentrenado)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img_pil = Image.fromarray(img_rgb)
        
        # Aplicar transformaciones
        tensor_img = transformaciones(img_pil).unsqueeze(0).to(device)
        
        # Inferencia
        with torch.no_grad():
            outputs = model(tensor_img)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            confianza_pct = round(confidence.item() * 100, 2)
            
            # Si la confianza es menor al 60%, lo clasificamos como 'Otro'
            if confianza_pct < 60.0:
                clase_predicha = "Otro (Baja confianza)"
            else:
                clase_predicha = CLASES[predicted.item()]
            
            # Obtener probabilidades para todas las clases
            todas_probabilidades = {CLASES[i]: round(probabilities[0][i].item() * 100, 2) for i in range(len(CLASES))}
            
        return JSONResponse(content={
            "prediction": clase_predicha,
            "confidence": f"{confianza_pct}%",
            "all_probabilities": todas_probabilidades
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
