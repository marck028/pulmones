# paso2_dataset.py
import os
import glob
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image

class RealChestXRayDataset(Dataset):
    def __init__(self, directorio_subconjunto, transform=None):
        self.clases = ['Normal', 'COVID', 'Lung_Opacity', 'Viral Pneumonia']
        self.transform = transform
        
        self.rutas_imagenes = []
        self.etiquetas = []
        
        for idx, clase in enumerate(self.clases):
            carpeta_img = os.path.join(directorio_subconjunto, clase, 'images')
            patron = os.path.join(carpeta_img, '*.png')
            lista_imgs = sorted(glob.glob(patron))
            
            for img_path in lista_imgs:
                self.rutas_imagenes.append(img_path)
                self.etiquetas.append(idx)

        print(f"[Dataset] {len(self.rutas_imagenes)} imágenes cargadas (sin máscaras).")

    def __len__(self):
        return len(self.rutas_imagenes)

    def __getitem__(self, idx):
        # Leer directamente en escala de grises — sin máscara
        img = cv2.imread(self.rutas_imagenes[idx], cv2.IMREAD_GRAYSCALE)
        
        # Convertir a RGB (requerido por ResNet18)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img_pil = Image.fromarray(img_rgb)
        
        label = self.etiquetas[idx]
        
        if self.transform:
            img_pil = self.transform(img_pil)
            
        return img_pil, label


# ============================================================
# Para pruebas directas del dataset
# ============================================================
if __name__ == '__main__':
    RUTA_DATASET_PROCESADO = r"D:\Documentos\UNIVERSIDAD\INVESTIGACION PULMONES\Codigo\Dataset_Procesado"

    transformaciones_base = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset_entrenamiento = RealChestXRayDataset(
        os.path.join(RUTA_DATASET_PROCESADO, 'train'),
        transform=transformaciones_base
    )
    dataset_evaluacion = RealChestXRayDataset(
        os.path.join(RUTA_DATASET_PROCESADO, 'test'),
        transform=transformaciones_base
    )

    train_loader = DataLoader(dataset_entrenamiento, batch_size=32, shuffle=True, num_workers=0)
    test_loader  = DataLoader(dataset_evaluacion,    batch_size=32, shuffle=False, num_workers=0)

    print(f"Lotes de entrenamiento: {len(train_loader)}")
    print(f"Lotes de prueba       : {len(test_loader)}")