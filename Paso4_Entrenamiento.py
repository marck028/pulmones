# Paso4_Entrenamiento.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
import os
from collections import Counter
from paso2_dataset import RealChestXRayDataset
from Paso3_Modelo import HybridCNNViT

def entrenar_pipeline_corregido():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Entrenando en: {device}")

    RUTA_DATASET_PROCESADO = r"D:\Documentos\UNIVERSIDAD\INVESTIGACION PULMONES\Codigo\Dataset_Procesado"

    # ============================================================
    # 1. TRANSFORMACIONES CON DATA AUGMENTATION
    # ============================================================
    transformaciones_entrenamiento = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transformaciones_evaluacion = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # ============================================================
    # 2. DATASETS
    # IMPORTANTE: Las clases en paso2_dataset son:
    # idx 0 = Normal, idx 1 = COVID, idx 2 = Lung_Opacity, idx 3 = Viral Pneumonia
    # ============================================================
    NOMBRES_CLASES = ['Normal', 'COVID', 'Lung_Opacity', 'Viral Pneumonia']

    dataset_entrenamiento = RealChestXRayDataset(
        os.path.join(RUTA_DATASET_PROCESADO, 'train'),
        transform=transformaciones_entrenamiento
    )
    dataset_evaluacion = RealChestXRayDataset(
        os.path.join(RUTA_DATASET_PROCESADO, 'test'),
        transform=transformaciones_evaluacion
    )

    # ============================================================
    # 3. DIAGNÓSTICO DE DISTRIBUCIÓN
    # ============================================================
    etiquetas = [dataset_entrenamiento[i][1] for i in range(len(dataset_entrenamiento))]
    conteo = Counter(etiquetas)

    print("\n=== Distribución del Dataset de Entrenamiento ===")
    for clase_idx, cantidad in sorted(conteo.items()):
        print(f"  {NOMBRES_CLASES[clase_idx]}: {cantidad} imágenes ({cantidad/len(etiquetas)*100:.1f}%)")

    # ============================================================
    # 4. SAMPLER BALANCEADO (doble protección aunque el dataset esté balanceado)
    # ============================================================
    conteo_clases = [conteo[i] for i in range(len(NOMBRES_CLASES))]
    peso_por_clase = 1.0 / torch.tensor(conteo_clases, dtype=torch.float)
    peso_por_muestra = [peso_por_clase[etiqueta] for etiqueta in etiquetas]

    sampler = WeightedRandomSampler(
        weights=peso_por_muestra,
        num_samples=len(peso_por_muestra),
        replacement=True
    )

    train_loader = DataLoader(
        dataset_entrenamiento,
        batch_size=32,
        sampler=sampler,
        num_workers=0
    )
    test_loader = DataLoader(
        dataset_evaluacion,
        batch_size=32,
        shuffle=False,
        num_workers=0
    )

    # ============================================================
    # 5. MODELO — REINICIALIZACIÓN DE CAPA FINAL PARA ELIMINAR SESGO HEREDADO
    # El ResNet18 preentrenado en ImageNet tiene sesgos que pueden
    # favorecer patrones que accidentalmente se parecen a una clase.
    # Reinicializar la cabeza clasificadora elimina ese sesgo.
    # ============================================================
    model = HybridCNNViT(num_classes=4).to(device)

    # Reinicializar SOLO la capa de clasificación final (self.head)
    nn.init.xavier_uniform_(model.head.weight)
    nn.init.zeros_(model.head.bias)
    print("\n[OK] Capa clasificadora (head) reinicializada desde cero.")

    # Reinicializar también cls_token y pos_embed para no heredar sesgos
    nn.init.trunc_normal_(model.cls_token, std=0.02)
    nn.init.trunc_normal_(model.pos_embed, std=0.02)
    print("[OK] CLS token y embeddings de posición reinicializados.")

    # ============================================================
    # 6. LOSS CON PESOS + OPTIMIZER + SCHEDULER
    # ============================================================
    pesos_loss = peso_por_clase / peso_por_clase.sum()
    criterion = nn.CrossEntropyLoss(weight=pesos_loss.to(device))

    # Estrategia de dos fases:
    # Fase 1 (épocas 1-5): Solo entrenamos la cabeza ViT, el backbone CNN se congela
    # Fase 2 (épocas 6-10): Descongelamos todo con LR muy baja (fine-tuning)
    for param in model.patch_embed.backbone.parameters():
        param.requires_grad = False
    print("[OK] Backbone CNN congelado para fase 1 (épocas 1-5).\n")

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4,
        weight_decay=0.02
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

    # ============================================================
    # 7. LOOP DE ENTRENAMIENTO
    # ============================================================
    print("=== Iniciando Entrenamiento ===")
    for epoch in range(1, 11):

        # --- Descongelar backbone en época 6 ---
        if epoch == 6:
            print("\n[FASE 2] Descongelando backbone CNN para fine-tuning completo...")
            for param in model.patch_embed.backbone.parameters():
                param.requires_grad = True
            optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.02)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
            print("[OK] Todos los parámetros activos. LR reducida a 5e-5.\n")

        # --- Entrenamiento ---
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        predicciones_por_clase = Counter()

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            for p in predicted.cpu().tolist():
                predicciones_por_clase[p] += 1

        perdida_promedio = running_loss / total
        precision_train = (correct / total) * 100
        scheduler.step(perdida_promedio)

        # --- Evaluación en test ---
        model.eval()
        correct_test, total_test = 0, 0
        correctos_por_clase = Counter()
        totales_por_clase = Counter()

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                total_test += labels.size(0)
                correct_test += predicted.eq(labels).sum().item()

                for p, r in zip(predicted.cpu().tolist(), labels.cpu().tolist()):
                    totales_por_clase[r] += 1
                    if p == r:
                        correctos_por_clase[r] += 1

        precision_test = (correct_test / total_test) * 100

        print(f"Época [{epoch}/10] | Pérdida: {perdida_promedio:.4f} | "
              f"Precisión Train: {precision_train:.2f}% | Precisión Test: {precision_test:.2f}%")

        # Precisión por clase (detecta si sigue colapsando)
        print("  Precisión por clase en test:")
        for clase_idx in range(len(NOMBRES_CLASES)):
            if totales_por_clase[clase_idx] > 0:
                acc_clase = correctos_por_clase[clase_idx] / totales_por_clase[clase_idx] * 100
                alerta = " ⚠️ BAJO RENDIMIENTO" if acc_clase < 40 else ""
                print(f"    {NOMBRES_CLASES[clase_idx]}: {acc_clase:.1f}%{alerta}")

        # Distribución de predicciones (detecta colapso a una clase)
        print("  Distribución de predicciones en train:")
        for clase_idx in range(len(NOMBRES_CLASES)):
            pct = predicciones_por_clase[clase_idx] / total * 100
            alerta = " ⚠️ POSIBLE SESGO" if pct > 55 else ""
            print(f"    {NOMBRES_CLASES[clase_idx]}: {pct:.1f}%{alerta}")

    # ============================================================
    # 8. GUARDAR MODELO
    # ============================================================
    ruta_guardado = r"D:\Documentos\UNIVERSIDAD\INVESTIGACION PULMONES\Codigo\modelo_hibrido_pulmonar.pth"
    torch.save(model.state_dict(), ruta_guardado)
    print(f"\n[OK] Modelo guardado en: {ruta_guardado}")

if __name__ == '__main__':
    entrenar_pipeline_corregido()