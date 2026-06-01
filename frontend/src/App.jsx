import { useState, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { UploadCloud, FileImage, Activity, CheckCircle2, AlertTriangle, AlertCircle, Loader2 } from 'lucide-react';
import './index.css';

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null);
  const fileInputRef = useRef(null);

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const processFile = (selectedFile) => {
    if (selectedFile && selectedFile.type.startsWith('image/')) {
      setFile(selectedFile);
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreview(reader.result);
        setResult(null); // Reset prev results
      };
      reader.readAsDataURL(selectedFile);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    const droppedFile = e.dataTransfer.files[0];
    processFile(droppedFile);
  };

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    processFile(selectedFile);
  };

  const analyzeImage = async () => {
    if (!file) return;
    
    setIsLoading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';
      const response = await fetch(`${apiUrl}/predict/`, {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) throw new Error('Error en el servidor');
      
      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error(error);
      alert('Error al analizar la imagen. Asegúrate de que el backend esté corriendo en localhost:8000.');
    } finally {
      setIsLoading(false);
    }
  };

  const resetState = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
  };

  const getStatusColor = (diagnosis, confidenceStr) => {
    const conf = parseFloat(confidenceStr);
    if (diagnosis.includes('Otro') || conf < 60) return 'var(--warning)';
    if (diagnosis === 'Normal' || diagnosis === 'Normal (Sano)') return 'var(--success)';
    return 'var(--danger)';
  };

  const getStatusIcon = (diagnosis, confidenceStr) => {
    const conf = parseFloat(confidenceStr);
    if (diagnosis.includes('Otro') || conf < 60) return <AlertTriangle size={32} color="var(--warning)" />;
    if (diagnosis === 'Normal' || diagnosis === 'Normal (Sano)') return <CheckCircle2 size={32} color="var(--success)" />;
    return <AlertCircle size={32} color="var(--danger)" />;
  };

  return (
    <div className="app-container">
      <motion.div 
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="header"
      >
        <h1>Diagnóstico Pulmonar IA</h1>
        <p>Análisis avanzado de radiografías de tórax potenciado por un modelo híbrido CNN-ViT. Detecta neumonía viral, COVID-19, opacidades pulmonares y pulmones sanos.</p>
      </motion.div>

      <AnimatePresence mode="wait">
        {!preview ? (
          <motion.div
            key="uploader"
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            className={`upload-area ${isDragging ? 'dragging' : ''}`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
          >
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileChange}
              accept="image/*"
              style={{ display: 'none' }}
            />
            <motion.div
              animate={{ y: isDragging ? -10 : 0 }}
              transition={{ type: "spring", stiffness: 300 }}
            >
              <UploadCloud size={64} className="upload-icon" />
            </motion.div>
            <h2 style={{ marginBottom: '0.5rem', fontWeight: 600 }}>Arrastra tu radiografía aquí</h2>
            <p style={{ color: 'var(--text-muted)' }}>o haz clic para explorar tus archivos (PNG, JPG)</p>
          </motion.div>
        ) : (
          <motion.div
            key="preview"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="glass-panel"
            style={{ padding: '2rem' }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '2rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                <FileImage size={24} color="var(--primary)" />
                <span style={{ fontWeight: 500 }}>{file.name}</span>
              </div>
              <button 
                onClick={resetState}
                style={{ background: 'transparent', border: '1px solid var(--surface-border)', color: 'white', padding: '0.5rem 1rem', borderRadius: '8px', cursor: 'pointer' }}
              >
                Cambiar Imagen
              </button>
            </div>

            <div className={result ? "results-grid" : ""} style={{ display: 'flex', flexDirection: result ? 'row' : 'column', alignItems: 'center', gap: '2rem' }}>
              <div style={{ flex: 1, width: '100%', display: 'flex', justifyContent: 'center' }}>
                <img src={preview} alt="Radiografía" className="image-preview" style={{ maxWidth: '400px', maxHeight: '400px', width: '100%', objectFit: 'contain', borderRadius: '16px' }} />
              </div>
              
              {result ? (
                <motion.div 
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  className="result-card glass-panel"
                  style={{ flex: 1, width: '100%', padding: '2rem', border: `1px solid ${getStatusColor(result.prediction, result.confidence)}33` }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginBottom: '1rem' }}>
                    {getStatusIcon(result.prediction, result.confidence)}
                    <div>
                      <div style={{ color: 'var(--text-muted)', fontSize: '0.9rem', textTransform: 'uppercase', letterSpacing: '1px' }}>Diagnóstico IA</div>
                      <div className="diagnosis-title" style={{ color: getStatusColor(result.prediction, result.confidence) }}>
                        {result.prediction}
                      </div>
                    </div>
                  </div>
                  
                  <div style={{ fontSize: '1.2rem', marginBottom: '2rem' }}>
                    Confianza: <strong>{result.confidence}</strong>
                  </div>

                  <div>
                    <h3 style={{ fontSize: '1rem', marginBottom: '1rem', color: 'var(--text-muted)' }}>Desglose de Probabilidades</h3>
                    {Object.entries(result.all_probabilities)
                      .sort(([,a], [,b]) => b - a)
                      .map(([clase, prob]) => (
                      <div key={clase} style={{ marginBottom: '1rem' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.9rem', marginBottom: '0.25rem' }}>
                          <span>{clase}</span>
                          <span>{prob}%</span>
                        </div>
                        <div className="progress-bar-bg">
                          <motion.div 
                            initial={{ width: 0 }}
                            animate={{ width: `${prob}%` }}
                            className="progress-bar-fill"
                            style={{ background: clase === result.prediction && prob >= 60 ? getStatusColor(result.prediction, result.confidence) : 'var(--primary)' }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </motion.div>
              ) : (
                <div style={{ marginTop: '2rem', textAlign: 'center', width: '100%' }}>
                  <button 
                    className="btn-primary" 
                    onClick={analyzeImage} 
                    disabled={isLoading}
                  >
                    {isLoading ? <Loader2 className="lucide-spin" size={24} /> : <Activity size={24} />}
                    {isLoading ? 'Analizando con IA...' : 'Analizar Radiografía'}
                  </button>
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

export default App;
