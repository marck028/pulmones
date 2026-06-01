# Paso5_Inferencia.py
import torch
from torchvision import transforms
from PIL import Image
import cv2
import os
from Paso3_Modelo import HybridCNNViT

RUTA_MODELO    = r"D:\Documentos\UNIVERSIDAD\INVESTIGACION PULMONES\Codigo\modelo_hibrido_pulmonar.pth"
NOMBRES_CLASES = ['Normal', 'COVID', 'Lung_Opacity', 'Viral Pneumonia']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = HybridCNNViT(num_classes=4).to(device)
model.load_state_dict(torch.load(RUTA_MODELO, map_location=device))
model.eval()
print(f"[OK] Modelo cargado — ejecutando en: {device}\n")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def clasificar_radiografia(ruta_imagen):
    img = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print(f"[ERROR] No se pudo leer: {ruta_imagen}")
        return

    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img_pil  = Image.fromarray(img_rgb)
    tensor   = transform(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs        = model(tensor)
        probabilidades = torch.softmax(outputs, dim=1)[0]
        clase_predicha = probabilidades.argmax().item()

    print(f"{'='*45}")
    print(f"  Archivo : {os.path.basename(ruta_imagen)}")
    print(f"  Resultado: >>> {NOMBRES_CLASES[clase_predicha]} <<<")
    print(f"{'='*45}")
    print("  Probabilidades:")
    for i, nombre in enumerate(NOMBRES_CLASES):
        barra = '█' * int(probabilidades[i].item() * 30)
        print(f"    {nombre:<20} {probabilidades[i]*100:5.1f}%  {barra}")
    print()

    return NOMBRES_CLASES[clase_predicha], probabilidades

if __name__ == '__main__':
    # Pon aquí las rutas a tus radiografías de prueba
    clasificar_radiografia(r"D:\Documentos\UNIVERSIDAD\INVESTIGACION PULMONES\Codigo\pruebas\Normal\images\Normal-208.png")
    clasificar_radiografia(r"D:\Documentos\UNIVERSIDAD\INVESTIGACION PULMONES\Codigo\pruebas\Viral Pneumonia\images\Viral Pneumonia-85.png")