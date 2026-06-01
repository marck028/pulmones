# Paso7_CurvasROC.py
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader
from torchvision import transforms

from paso2_dataset import RealChestXRayDataset
from Paso3_Modelo import HybridCNNViT

def graficar_curvas_roc():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Generando curvas ROC en: {device}")

    # ============================================================
    # 1. TEST LOADER (sin máscaras, igual que entrenamiento)
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
    ruta_pesos = r"D:\Documentos\UNIVERSIDAD\INVESTIGACION PULMONES\Codigo\modelo_hibrido_pulmonar.pth"

    if not os.path.exists(ruta_pesos):
        print(f"[ERROR] No se encontraron los pesos en: {ruta_pesos}")
        return

    model = HybridCNNViT(num_classes=4).to(device)
    model.load_state_dict(torch.load(ruta_pesos, map_location=device))
    model.eval()
    print("[OK] Modelo cargado correctamente.")

    # ============================================================
    # 3. EXTRAER PROBABILIDADES
    # ============================================================
    all_labels = []
    all_probs  = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs   = torch.softmax(outputs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)

    # ============================================================
    # 4. CURVAS ROC
    # ============================================================
    NOMBRES_CLASES = ['Normal', 'COVID', 'Lung_Opacity', 'Viral Pneumonia']
    y_test_bin = label_binarize(all_labels, classes=[0, 1, 2, 3])
    colors     = ['navy', 'crimson', 'forestgreen', 'darkorange']

    plt.figure(figsize=(9, 7))

    for i in range(4):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], all_probs[:, i])
        roc_auc     = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[i], lw=2,
                 label=f'Curva ROC de {NOMBRES_CLASES[i]} (AUC = {roc_auc:.4f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=1.5)
    plt.xlim([-0.02, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos (1 - Especificidad)', fontsize=12, fontweight='bold')
    plt.ylabel('Tasa de Verdaderos Positivos (Sensibilidad / Recall)', fontsize=12, fontweight='bold')
    plt.title('Análisis de Desempeño Diagnóstico: Curvas ROC por Clase (CAD Híbrido)',
              fontsize=13, fontweight='bold', pad=15)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)

    # ============================================================
    # 5. GUARDAR Y MOSTRAR
    # ============================================================
    ruta_grafico = r"D:\Documentos\UNIVERSIDAD\INVESTIGACION PULMONES\Codigo\curvas_roc_pulmonar.png"
    plt.savefig(ruta_grafico, dpi=300, bbox_inches='tight')
    print(f"[OK] Gráfico exportado en alta resolución (300 DPI)")
    print(f"     Ubicación: {ruta_grafico}")
    plt.show()

if __name__ == '__main__':
    graficar_curvas_roc()