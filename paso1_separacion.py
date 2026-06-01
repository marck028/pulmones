import os
import glob
import shutil
from sklearn.model_selection import train_test_split

RUTA_ORIGEN = r"D:\Documentos\UNIVERSIDAD\INVESTIGACION PULMONES\COVID-19_Radiography_Dataset"
RUTA_DESTINO = r"D:\Documentos\UNIVERSIDAD\INVESTIGACION PULMONES\Codigo\Dataset_Procesado"

CLASES = ['Normal', 'COVID', 'Lung_Opacity', 'Viral Pneumonia']
LIMITE_POR_ENFERMEDAD = 1000  # Solo 1000 imagenes por cada enfermedad

def organizar_dataset_express():
    print("Iniciando muestreo express de 1000 imágenes por clase...")
    
    img_train_master = []
    img_test_master = []
    mask_train_master = []
    mask_test_master = []
    y_train_master = []
    y_test_master = []
    
    for clase in CLASES:
        carpeta_clase_img = os.path.join(RUTA_ORIGEN, clase, 'images')
        patron = os.path.join(carpeta_clase_img, '*.png')
        lista_imgs = sorted(glob.glob(patron))
        
        rutas_imagenes_clase = []
        rutas_mascaras_clase = []
        etiquetas_clase = []
        
        # Recopilar solo hasta encontrar 1000 pares válidos de esta enfermedad
        contador = 0
        for img_path in lista_imgs:
            if contador >= LIMITE_POR_ENFERMEDAD:
                break
                
            nombre_archivo = os.path.basename(img_path)
            mask_path = os.path.join(RUTA_ORIGEN, clase, 'masks', nombre_archivo)
            
            if os.path.exists(mask_path):
                rutas_imagenes_clase.append(img_path)
                rutas_mascaras_clase.append(mask_path)
                etiquetas_clase.append(clase)
                contador += 1
                
        print(f"-> {clase}: Muestreadas {len(rutas_imagenes_clase)} imágenes y máscaras.")
        
        # Dividir esta clase específica: 80% (800) train, 20% (200) test
        img_tr, img_te, mask_tr, mask_te, y_tr, y_te = train_test_split(
            rutas_imagenes_clase, rutas_mascaras_clase, etiquetas_clase, 
            test_size=0.20, random_state=42
        )
        
        # Acumular en las listas maestras
        img_train_master.extend(img_tr)
        img_test_master.extend(img_te)
        mask_train_master.extend(mask_tr)
        mask_test_master.extend(mask_te)
        y_train_master.extend(y_tr)
        y_test_master.extend(y_te)

    # Limpiar cualquier residuo de la carpeta destino anterior para empezar de cero
    if os.path.exists(RUTA_DESTINO):
        shutil.rmtree(RUTA_DESTINO)
        
    def copiar_archivos(lista_imagenes, lista_mascaras, lista_etiquetas, subconjunto):
        for img_path, mask_path, clase in zip(lista_imagenes, lista_mascaras, lista_etiquetas):
            dir_dest_img = os.path.join(RUTA_DESTINO, subconjunto, clase, 'images')
            dir_dest_mask = os.path.join(RUTA_DESTINO, subconjunto, clase, 'masks')
            
            os.makedirs(dir_dest_img, exist_ok=True)
            os.makedirs(dir_dest_mask, exist_ok=True)
            
            nombre_archivo = os.path.basename(img_path)
            shutil.copy2(img_path, os.path.join(dir_dest_img, nombre_archivo))
            shutil.copy2(mask_path, os.path.join(dir_dest_mask, nombre_archivo))

    print("\nEscribiendo archivos estructurados en el disco duro...")
    copiar_archivos(img_train_master, mask_train_master, y_train_master, 'train')
    copiar_archivos(img_test_master, mask_test_master, y_test_master, 'test')
    
    print(f"\n¡Dataset Express listo en: {RUTA_DESTINO}!")
    print(f"Total Entrenamiento: {len(img_train_master)} (800 por clase)")
    print(f"Total Evaluación: {len(img_test_master)} (200 por clase)")

if __name__ == '__main__':
    organizar_dataset_express()