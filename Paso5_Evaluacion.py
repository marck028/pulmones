# Paso6_Evaluacion.py
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader
from torchvision import transforms
import os

from paso2_dataset import RealChestXRayDataset
from Paso3_Modelo import HybridCNNViT

def evaluar_hipotesis_clinica():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ============================================================
    # 1. RECONSTRUIR TEST LOADER (sin máscaras, igual que entrenamiento)
    # ============================================================
    RUTA_DATASET_PROCESADO = r"D:\Documentos\UNIVERSIDAD\INVESTIGACION PULMONES\Codigo\Dataset_Procesado"

    transformaciones_evaluacion = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset_evaluacion = RealChestXRayDataset(
        os.path.join(RUTA_DATASET_PROCESADO, 'test'),
        transform=transformaciones_evaluacion
    )
    test_loader = DataLoader(dataset_evaluacion, batch_size=32, shuffle=False, num_workers=0)

    # ============================================================
    # 2. CARGAR MODELO
    # ============================================================
    model = HybridCNNViT(num_classes=4).to(device)
    ruta_pesos = r"D:\Documentos\UNIVERSIDAD\INVESTIGACION PULMONES\Codigo\modelo_hibrido_pulmonar.pth"
    model.load_state_dict(torch.load(ruta_pesos, map_location=device))
    model.eval()
    print(f"[OK] Modelo cargado — ejecutando en: {device}")

    # ============================================================
    # 3. INFERENCIA SOBRE TEST
    # ============================================================
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)

    # ============================================================
    # 4. MÉTRICAS
    # ============================================================
    NOMBRES_CLASES = ['Normal', 'COVID', 'Lung_Opacity', 'Viral Pneumonia']
    cm = confusion_matrix(all_labels, all_preds)

    print("\n==============================================")
    print("      AUDITORÍA DE MÉTRICAS CLÍNICAS (TEST)   ")
    print("==============================================")

    print("\nMatriz de Confusión Estructurada:")
    print(f"{'':20}", end="")
    for nombre in NOMBRES_CLASES:
        print(f"{nombre:>18}", end="")
    print()
    for i, nombre in enumerate(NOMBRES_CLASES):
        print(f"{nombre:20}", end="")
        for j in range(len(NOMBRES_CLASES)):
            print(f"{cm[i, j]:>18}", end="")
        print()

    print("\nReporte General de Sensibilidad (Recall):")
    print(classification_report(all_labels, all_preds, target_names=NOMBRES_CLASES))

    print("Especificidad por clase (One-vs-Rest):")
    total = cm.sum()
    for i, nombre in enumerate(NOMBRES_CLASES):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total - tp - fn - fp
        especificidad = tn / (tn + fp) if (tn + fp) > 0 else 0
        print(f"  [{nombre}]: {especificidad:.4f}")

    macro_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
    print(f"\nÁrea Bajo la Curva (Macro AUC-ROC): {macro_auc:.4f}")

if __name__ == '__main__':
    evaluar_hipotesis_clinica()