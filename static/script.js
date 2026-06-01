document.addEventListener('DOMContentLoaded', () => {
    const dropArea = document.getElementById('drop-area');
    const fileElem = document.getElementById('fileElem');
    const btnBrowse = document.getElementById('btn-browse');
    const uploadContent = document.querySelector('.upload-content');
    const previewContainer = document.getElementById('preview-container');
    const imagePreview = document.getElementById('image-preview');
    const btnAnalyze = document.getElementById('btn-analyze');
    const btnClear = document.getElementById('btn-clear');
    
    const loading = document.getElementById('loading');
    const resultsSection = document.getElementById('results-section');
    
    let currentFile = null;

    // Prevenir el comportamiento por defecto al arrastrar
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    // Efectos visuales al arrastrar
    ['dragenter', 'dragover'].forEach(eventName => {
        dropArea.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, unhighlight, false);
    });

    function highlight(e) {
        dropArea.classList.add('highlight');
    }

    function unhighlight(e) {
        dropArea.classList.remove('highlight');
    }

    // Manejar el archivo soltado
    dropArea.addEventListener('drop', handleDrop, false);

    function handleDrop(e) {
        let dt = e.dataTransfer;
        let files = dt.files;
        handleFiles(files);
    }

    // Manejar el click en el botón de seleccionar
    btnBrowse.addEventListener('click', () => {
        fileElem.click();
    });

    fileElem.addEventListener('change', function() {
        handleFiles(this.files);
    });

    function handleFiles(files) {
        if (files.length > 0) {
            const file = files[0];
            if (file.type.startsWith('image/')) {
                currentFile = file;
                showPreview(file);
            } else {
                alert('Por favor, selecciona un archivo de imagen válido.');
            }
        }
    }

    function showPreview(file) {
        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onloadend = function() {
            imagePreview.src = reader.result;
            uploadContent.classList.add('hidden');
            previewContainer.classList.remove('hidden');
            resultsSection.classList.add('hidden');
        }
    }

    // Limpiar selección
    btnClear.addEventListener('click', () => {
        currentFile = null;
        fileElem.value = '';
        imagePreview.src = '';
        previewContainer.classList.add('hidden');
        uploadContent.classList.remove('hidden');
        resultsSection.classList.add('hidden');
    });

    // Enviar a analizar
    btnAnalyze.addEventListener('click', async () => {
        if (!currentFile) return;

        // Mostrar loading
        loading.classList.remove('hidden');
        resultsSection.classList.add('hidden');
        btnAnalyze.disabled = true;

        const formData = new FormData();
        formData.append('file', currentFile);

        try {
            const response = await fetch('/predict/', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error('Error en el servidor al analizar la imagen.');
            }

            const data = await response.json();
            displayResults(data);
        } catch (error) {
            console.error('Error:', error);
            alert('Ocurrió un error al analizar la imagen: ' + error.message);
            loading.classList.add('hidden');
            btnAnalyze.disabled = false;
        }
    });

    function displayResults(data) {
        loading.classList.add('hidden');
        btnAnalyze.disabled = false;
        resultsSection.classList.remove('hidden');

        // Configurar título y confianza general
        const titleEl = document.getElementById('prediction-title');
        titleEl.textContent = traducirClase(data.prediction);
        
        // Colorear según resultado
        if (data.prediction === 'Normal') {
            titleEl.style.color = 'var(--success)';
            document.getElementById('confidence-bar-fill').style.backgroundColor = 'var(--success)';
        } else {
            titleEl.style.color = 'var(--danger)';
            document.getElementById('confidence-bar-fill').style.backgroundColor = 'var(--danger)';
        }

        document.getElementById('confidence-text').textContent = data.confidence;
        
        // Animar barra
        setTimeout(() => {
            document.getElementById('confidence-bar-fill').style.width = data.confidence;
        }, 100);

        // Llenar lista de probabilidades
        const listEl = document.getElementById('probabilities-list');
        listEl.innerHTML = ''; // Limpiar lista
        
        // Ordenar probabilidades de mayor a menor
        const probs = Object.entries(data.all_probabilities)
            .sort((a, b) => b[1] - a[1]);

        probs.forEach(([clase, prob]) => {
            const li = document.createElement('li');
            li.innerHTML = `
                <span class="prob-name">${traducirClase(clase)}</span>
                <span class="prob-value">${prob}%</span>
            `;
            listEl.appendChild(li);
        });
    }

    function traducirClase(claseOriginal) {
        const traducciones = {
            'Normal': 'Normal (Sano)',
            'COVID': 'COVID-19',
            'Lung_Opacity': 'Opacidad Pulmonar',
            'Viral Pneumonia': 'Neumonía Viral'
        };
        return traducciones[claseOriginal] || claseOriginal;
    }
});
