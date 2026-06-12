import { useState } from 'react';
import { Editor } from 'ketcher-react';
import { StandaloneStructServiceProvider } from 'ketcher-standalone';
import 'ketcher-react/dist/index.css';

const structServiceProvider = new StandaloneStructServiceProvider();

export default function App() {
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handlePredict = async () => {
    if (!window.ketcher) {
      setError("Ketcher editor is still loading...");
      return;
    }
    
    setLoading(true);
    setError(null);
    setResults(null);
    
    try {
      // 1. Extract SMILES from the Ketcher canvas
      const smiles = await window.ketcher.getSmiles();
      if (!smiles) {
        setError("Please draw a polymer repeating unit (monomer) first.");
        setLoading(false);
        return;
      }
      
      // 2. Call the FastAPI backend
      const response = await fetch('http://localhost:13789/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ smiles })
      });
      
      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || "Prediction failed on server");
      }
      
      const data = await response.json();
      setResults(data);
      
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', fontFamily: 'system-ui, sans-serif' }}>
      {/* Header / Toolbar */}
      <div style={{ padding: '15px 20px', display: 'flex', gap: '20px', alignItems: 'center', borderBottom: '1px solid #ddd', backgroundColor: '#fff' }}>
        <h2 style={{ margin: 0, color: '#2c3e50' }}>Polymer QSPR Predictor</h2>
        <button 
          onClick={handlePredict} 
          disabled={loading} 
          style={{ 
            padding: '10px 20px', 
            fontSize: '16px', 
            backgroundColor: loading ? '#95a5a6' : '#2980b9', 
            color: 'white', 
            border: 'none', 
            borderRadius: '6px', 
            cursor: loading ? 'not-allowed' : 'pointer' 
          }}
        >
          {loading ? 'Predicting...' : 'Predict Gas Permeability'}
        </button>
        {error && <span style={{ color: '#e74c3c', fontWeight: 'bold' }}>{error}</span>}
      </div>
      
      {/* Main Content Area */}
      <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
        {/* Ketcher Editor (Left Side) */}
        <div style={{ flex: 2, borderRight: '1px solid #ddd' }}>
          <Editor
            staticResourcesUrl=""
            structServiceProvider={structServiceProvider}
            onInit={(ketcher) => {
              window.ketcher = ketcher;
            
              ketcher.setMolecule('*C1C(C2)CCC2C1*'); 
            }}
          />
        </div>
        
        {/* Results Panel (Right Side) */}
        <div style={{ flex: 1, padding: '25px', overflowY: 'auto', backgroundColor: '#f8f9fa' }}>
          <h3 style={{ marginTop: 0, color: '#34495e', borderBottom: '2px solid #3498db', paddingBottom: '10px' }}>
            Prediction Results
          </h3>
          
          {!results && !loading && (
            <p style={{ color: '#7f8c8d', lineHeight: '1.6' }}>
              Draw a polymer repeating unit (monomer) on the left and click <strong>Predict</strong> to see gas permeability estimates.
            </p>
          )}
          
          {results && (
            <table style={{ width: '100%', borderCollapse: 'collapse', backgroundColor: 'white', boxShadow: '0 2px 5px rgba(0,0,0,0.05)' }}>
              <thead>
                <tr style={{ backgroundColor: '#ecf0f1' }}>
                  <th style={{ padding: '12px', textAlign: 'left', borderBottom: '2px solid #bdc3c7' }}>Gas</th>
                  <th style={{ padding: '12px', textAlign: 'right', borderBottom: '2px solid #bdc3c7' }}>Mean</th>
                  <th style={{ padding: '12px', textAlign: 'right', borderBottom: '2px solid #bdc3c7' }}>Std Dev</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(results).map(([gas, prediction]) => (
                  <tr key={gas} style={{ borderBottom: '1px solid #ecf0f1' }}>
                    <td style={{ padding: '12px', textTransform: 'uppercase', fontWeight: 'bold', color: '#2c3e50' }}>
                      {gas.replace('p_', '')}
                    </td>
                    <td style={{ padding: '12px', textAlign: 'right', fontFamily: 'monospace' }}>
                      {prediction.mean.toFixed(2)}
                    </td>
                    <td style={{ padding: '12px', textAlign: 'right', fontFamily: 'monospace', color: '#7f8c8d' }}>
                      ± {prediction.std.toFixed(2)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>
    </div>
  );
}
