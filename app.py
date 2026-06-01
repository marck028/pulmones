import streamlit as st
import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image
import os
import sys

# Añadir el directorio actual al path para importar Paso3_Modelo
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from Paso3_Modelo import HybridCNNViT

# Configuración de la página
st.set_page_config(
    page_title="Diagnóstico Pulmonar IA",
    page_icon="🫁",
    layout="centered"
)

# Estilos personalizados básicos
st.markdown("""
<style>
    .reportview-container .main .block-container{
        padding-top: 2rem;
    }
    h1 {
        color: #4361ee;
    }
</style>
""", unsafe_allow_html=True)

# Título y descripción
st.title("🫁 Diagnóstico Pulmonar Asistido por IA")
st.markdown("Sube una radiografía de tórax para que el modelo híbrido CNN-ViT la analice.")

# Clases y transformaciones
CLASES_ORIGINALES = ['Normal', 'COVID', 'Lung_Opacity', 'Viral Pneumonia']
CLASES_TRADUCIDAS = {
    'Normal': 'Normal (Sano)',
    'COVID': 'COVID-19',
    'Lung_Opacity': 'Opacidad Pulmonar',
    'Viral Pneumonia': 'Neumonía Viral'
}

transformaciones = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Carga del modelo en caché para no recargarlo cada vez
@st.cache_resource
def cargar_modelo():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HybridCNNViT(num_classes=4).to(device)
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modelo_hibrido_pulmonar.pth")
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model, device
    except Exception as e:
        st.error(f"Error cargando el modelo: {e}")
        return None, device

modelo, dispositivo = cargar_modelo()

# UI de subida de archivo
archivo_subido = st.file_uploader("Arrastra y suelta tu radiografía (PNG, JPG)", type=["png", "jpg", "jpeg"])

if archivo_subido is not None:
    # Mostrar la imagen
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Radiografía Original")
        st.image(archivo_subido, use_container_width=True)
    
    with col2:
        st.subheader("Resultados del Análisis")
        if modelo is None:
            st.error("El modelo no está disponible.")
        else:
            with st.spinner('Analizando con CNN-ViT...'):
                try:
                    # Leer imagen
                    file_bytes = np.asarray(bytearray(archivo_subido.read()), dtype=np.uint8)
                    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
                    
                    # Preprocesamiento (Conversión a RGB sin máscara)
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                    img_pil = Image.fromarray(img_rgb)
                    
                    # Tensor y dispositivo
                    tensor_img = transformaciones(img_pil).unsqueeze(0).to(dispositivo)
                    
                    # Inferencia
                    with torch.no_grad():
                        outputs = modelo(tensor_img)
                        probabilities = torch.softmax(outputs, dim=1)
                        confidence, predicted = torch.max(probabilities, 1)
                        
                        confianza_pct = confidence.item() * 100
                        clase_original = CLASES_ORIGINALES[predicted.item()]
                        
                        # Umbral del 60%
                        if confianza_pct < 60.0:
                            diagnostico_final = "Otro (Baja confianza)"
                            color_alerta = "warning"
                        else:
                            diagnostico_final = CLASES_TRADUCIDAS[clase_original]
                            color_alerta = "success" if diagnostico_final == "Normal (Sano)" else "error"

                    # Mostrar resultado principal
                    if color_alerta == "success":
                        st.success(f"### {diagnostico_final}\n**Confianza: {confianza_pct:.2f}%**")
                    elif color_alerta == "warning":
                        st.warning(f"### {diagnostico_final}\n**Confianza del modelo: {confianza_pct:.2f}%**. La imagen podría no ser una radiografía válida o no pertenece a las clases conocidas.")
                    else:
                        st.error(f"### {diagnostico_final}\n**Confianza: {confianza_pct:.2f}%**")

                    st.markdown("---")
                    st.markdown("#### Desglose de Probabilidades")
                    
                    # Mostrar todas las probabilidades ordenadas
                    prob_dic = {CLASES_TRADUCIDAS[CLASES_ORIGINALES[i]]: probabilities[0][i].item() * 100 for i in range(4)}
                    prob_ordenadas = dict(sorted(prob_dic.items(), key=lambda item: item[1], reverse=True))
                    
                    for clase, prob in prob_ordenadas.items():
                        st.write(f"**{clase}**")
                        st.progress(int(prob) if prob > 1 else 1, text=f"{prob:.2f}%")

                except Exception as e:
                    st.error(f"Hubo un error al procesar la imagen: {e}")
