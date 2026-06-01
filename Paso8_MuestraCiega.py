import os
import glob
import shutil

# Rutas del ecosistema de tu proyecto en la computadora
RUTA_ORIGEN = r"D:\Documentos\UNIVERSIDAD\INVESTIGACION PULMONES\COVID-19_Radiography_Dataset"
RUTA_DESTINO_PRUEBAS = r"D:\Documentos\UNIVERSIDAD\INVESTIGACION PULMONES\Codigo\pruebas"

CLASES = ['Normal', 'COVID', 'Lung_Opacity', 'Viral Pneumonia']
SALTAR_IMAGENES = 1000  # Omitimos las que ya usamos en el entrenamiento/test anterior
LIMITE_MUESTRA_CIEGA = 500  # Extraemos 500 nuevas por enfermedad

def generar_muestra_ciega():
    print("=== Iniciando Extracción de Muestra Ciega de Validación ===")
    print(f"Saltando las primeras {SALTAR_IMAGENES} imágenes de Kaggle para evitar duplicados...\n")
    
    total_imagenes_copiadas = 0
    total_mascaras_copiadas = 0
    
    for clase in CLASES:
        carpeta_clase_img = os.path.join(RUTA_ORIGEN, clase, 'images')
        patron = os.path.join(carpeta_clase_img, '*.png')
        lista_imgs = sorted(glob.glob(patron))
        
        # Estructuramos las subcarpetas dentro de la nueva carpeta "pruebas"
        dir_dest_img = os.path.join(RUTA_DESTINO_PRUEBAS, clase, 'images')
        dir_dest_mask = os.path.join(RUTA_DESTINO_PRUEBAS, clase, 'masks')
        
        os.makedirs(dir_dest_img, exist_ok=True)
        os.makedirs(dir_dest_mask, exist_ok=True)
        
        contador_guardadas = 0
        
        # Iteramos empezando estrictamente DESPUÉS de las primeras 1000 imágenes
        for img_path in lista_imgs[SALTAR_IMAGENES:]:
            if contador_guardadas >= LIMITE_MUESTRA_CIEGA:
                break
                
            nombre_archivo = os.path.basename(img_path)
            mask_path = os.path.join(RUTA_ORIGEN, clase, 'masks', nombre_archivo)
            
            # Verificar que el par (Imagen/Máscara) exista físicamente
            if os.path.exists(mask_path):
                # Copiar Imagen
                shutil.copy2(img_path, os.path.join(dir_dest_img, nombre_archivo))
                # Copiar Máscara homóloga
                shutil.copy2(mask_path, os.path.join(dir_dest_mask, nombre_archivo))
                
                contador_guardadas += 1
                
        print(f"-> Clase [{clase}]: 500 imágenes nuevas extraídas con éxito.")
        total_imagenes_copiadas += contador_guardadas
        
    print("\n=======================================================")
    print(f"¡PROCESO COMPLETADO! Carpeta 'pruebas' creada exitosamente.")
    print(f"Ubicación: {RUTA_DESTINO_PRUEBAS}")
    print(f"Total de nuevas imágenes listas para inferencia: {total_imagenes_copiadas}")
    print("=======================================================")

if __name__ == '__main__':
    generar_muestra_ciega()