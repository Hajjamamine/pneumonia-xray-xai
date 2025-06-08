import React, { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { FiUpload, FiActivity, FiCheckCircle, FiAlertCircle } from "react-icons/fi";

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [shapValues, setShapValues] = useState(null); // SHAP values state
  const [shapImage, setShapImage] = useState(null); // SHAP heatmap
  const [loading, setLoading] = useState(false);
  const [shapLoading, setShapLoading] = useState(false); // Separate loading state for SHAP
  const [dragActive, setDragActive] = useState(false);
  const inputRef = useRef(null);

  // Handle drag events
  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  // Handle dropped files
  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      handleFile(file);
    }
  };

  // Handle file selection
  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  };

  const handleFile = (file) => {
    if (file.size > 10 * 1024 * 1024) {
      alert("File size exceeds 10MB. Please upload a smaller file.");
      return;
    }
    setSelectedFile(file);
    setResult(null);
    setPreview(URL.createObjectURL(file));
  };

  // Cleanup preview URL
  useEffect(() => {
    return () => {
      if (preview) URL.revokeObjectURL(preview);
    };
  }, [preview]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!selectedFile) return;

    setLoading(true);
    const formData = new FormData();
    formData.append("image", selectedFile);

    try {
      const res = await fetch("http://localhost:8000/api/predict/", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      setResult(data);
    } catch (error) {
      setResult({ error: "Prediction failed. Please try again." });
    } finally {
      setLoading(false);
    }
  };

  const handleSHAP = async () => {
    if (!selectedFile) return;

    setShapLoading(true);
    const formData = new FormData();
    formData.append("image", selectedFile);

    try {
      const res = await fetch("http://localhost:8000/api/shap/", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const errorData = await res.json();
        throw new Error(errorData.error || "Failed to generate SHAP explanation.");
      }

      const data = await res.json();

      // Set SHAP values and heatmap for rendering
      setShapValues(data.shap_values); // SHAP values
      setShapImage(data.heatmap); // Base64-encoded heatmap
    } catch (error) {
      setShapValues(null);
      setShapImage(null);
      alert(error.message);
    } finally {
      setShapLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-purple-900 to-gray-900 flex flex-col items-center justify-center p-4 text-white">
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="text-center mb-8"
      >
        <h1 className="text-5xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-cyan-400 to-purple-400 mb-2">
          PneumoScan AI
        </h1>
        <p className="text-gray-300 text-lg">
          Advanced X-Ray Pneumonia Detection
        </p>
      </motion.div>

      <motion.div
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        className="w-full max-w-2xl"
      >
        <div
          className={`relative border-2 border-dashed rounded-2xl p-8 transition-all duration-300 ${
            dragActive
              ? "border-cyan-400 bg-gray-800"
              : "border-gray-700 bg-gray-800/50"
          }`}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
          onClick={() => inputRef.current.click()}
        >
          <input
            ref={inputRef}
            type="file"
            accept="image/*"
            onChange={handleFileChange}
            className="hidden"
          />
          <div className="text-center space-y-4">
            <div className="flex justify-center">
              <FiUpload className="text-4xl text-cyan-400" />
            </div>
            <motion.p
              animate={{ y: dragActive ? 2 : 0 }}
              className="text-xl font-medium"
            >
              {dragActive ? "Drop your X-Ray here" : "Drag & drop or click to upload"}
            </motion.p>
            <p className="text-gray-400">
              Supports JPEG, PNG (Max 10MB)
            </p>
          </div>
        </div>

        <AnimatePresence>
          {preview && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              className="mt-6"
            >
              <div className="relative group">
                <img
                  src={preview}
                  alt="Preview"
                  className="w-full rounded-xl shadow-lg border border-gray-700"
                />
                <div className="absolute inset-0 bg-gradient-to-t from-black/80 to-transparent opacity-0 group-hover:opacity-100 transition-opacity rounded-xl flex items-end p-4">
                  <button
                    onClick={() => setPreview(null)}
                    className="text-white bg-red-500/80 hover:bg-red-500 px-3 py-1 rounded-full text-sm"
                  >
                    Remove
                  </button>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: preview ? 1 : 0 }}
          className="mt-6 flex space-x-4"
        >
          {/* Button for Prediction */}
          <button
            onClick={handleSubmit}
            disabled={loading || !preview}
            className={`w-full py-3 px-6 rounded-xl font-bold text-lg flex items-center justify-center space-x-2 transition-all ${
              loading
                ? "bg-cyan-600/50 cursor-not-allowed"
                : "bg-gradient-to-r from-cyan-500 to-purple-600 hover:from-cyan-600 hover:to-purple-700 shadow-lg shadow-cyan-500/20"
            }`}
          >
            {loading ? (
              <>
                <svg
                  className="animate-spin -ml-1 mr-2 h-5 w-5 text-white"
                  xmlns="http://www.w3.org/2000/svg"
                  fill="none"
                  viewBox="0 0 24 24"
                >
                  <circle
                    className="opacity-25"
                    cx="12"
                    cy="12"
                    r="10"
                    stroke="currentColor"
                    strokeWidth="4"
                  ></circle>
                  <path
                    className="opacity-75"
                    fill="currentColor"
                    d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                  ></path>
                </svg>
                <span>Analyzing...</span>
              </>
            ) : (
              <>
                <FiActivity className="text-lg" />
                <span>Run AI Analysis</span>
              </>
            )}
          </button>

          {/* Button for SHAP */}
          <button
            onClick={handleSHAP}
            disabled={shapLoading || !preview}
            className={`w-full py-3 px-6 rounded-xl font-bold text-lg flex items-center justify-center space-x-2 transition-all ${
              shapLoading
                ? "bg-purple-600/50 cursor-not-allowed"
                : "bg-gradient-to-r from-purple-500 to-cyan-600 hover:from-purple-600 hover:to-cyan-700 shadow-lg shadow-purple-500/20"
            }`}
          >
            {shapLoading ? (
              <>
                <svg
                  className="animate-spin -ml-1 mr-2 h-5 w-5 text-white"
                  xmlns="http://www.w3.org/2000/svg"
                  fill="none"
                  viewBox="0 0 24 24"
                >
                  <circle
                    className="opacity-25"
                    cx="12"
                    cy="12"
                    r="10"
                    stroke="currentColor"
                    strokeWidth="4"
                  ></circle>
                  <path
                    className="opacity-75"
                    fill="currentColor"
                    d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                  ></path>
                </svg>
                <span>Generating...</span>
              </>
            ) : (
              <>
                <FiActivity className="text-lg" />
                <span>Explain with SHAP</span>
              </>
            )}
          </button>
        </motion.div>
      </motion.div>

      <AnimatePresence>
        {result && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 20 }}
            className={`mt-8 w-full max-w-2xl p-6 rounded-2xl backdrop-blur-sm ${
              result.error
                ? "bg-red-500/10 border border-red-500/30"
                : "bg-gray-800/70 border border-gray-700"
            }`}
          >
            {result.error ? (
              <div className="flex items-start space-x-3">
                <FiAlertCircle className="text-red-400 text-2xl mt-0.5 flex-shrink-0" />
                <div>
                  <h3 className="text-xl font-bold text-red-400">Error</h3>
                  <p className="text-gray-300">{result.error}</p>
                </div>
              </div>
            ) : (
              <div className="space-y-4">
                <div className="flex items-center space-x-3">
                  <FiCheckCircle
                    className={`text-2xl ${
                      result.prediction === "Normal"
                        ? "text-green-400"
                        : "text-red-400"
                    }`}
                  />
                  <h3 className="text-2xl font-bold">
                    Diagnosis:{" "}
                    <span
                      className={
                        result.prediction === "Normal"
                          ? "text-green-400"
                          : "text-red-400"
                      }
                    >
                      {result.prediction}
                    </span>
                  </h3>
                </div>

                <div className="space-y-2">
                  <div className="flex justify-between text-sm text-gray-400">
                    <span>Confidence Level</span>
                    <span>{(result.confidence * 100).toFixed(2)}%</span>
                  </div>
                  <div className="w-full bg-gray-700 rounded-full h-2.5">
                    <div
                      className={`h-2.5 rounded-full ${
                        result.prediction === "Normal"
                          ? "bg-green-500"
                          : "bg-red-500"
                      }`}
                      style={{
                        width: `${result.confidence * 100}%`,
                      }}
                    ></div>
                  </div>
                </div>

                {result.prediction === "Pneumonia" && (
                  <div className="mt-4 p-4 bg-red-500/10 rounded-lg border border-red-500/30">
                    <h4 className="font-medium text-red-400 flex items-center space-x-2">
                      <FiAlertCircle />
                      <span>Medical Attention Recommended</span>
                    </h4>
                    <p className="text-gray-300 mt-1 text-sm">
                      This result suggests possible pneumonia. Please consult a
                      healthcare professional for further evaluation.
                    </p>
                  </div>
                )}

                {/* Display SHAP Explanation */}
                {shapImage && (
                  <div className="mt-6">
                    <h4 className="text-lg font-bold text-cyan-400 mb-2">
                      SHAP Explanation
                    </h4>
                    <p className="text-gray-400 text-sm mb-4">
                      The SHAP heatmap highlights the regions of the X-ray that
                      contributed most to the model's prediction.
                    </p>
                    <div className="relative">
                      <img
                        src={`data:image/png;base64,${shapImage}`}
                        alt="SHAP Heatmap"
                        className="w-full rounded-xl shadow-lg border border-gray-700"
                      />
                    </div>
                  </div>
                )}
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>

      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.5 }}
        className="mt-12 text-center text-gray-400 text-sm"
      >
        <p>Powered by Deep Learning & Medical Imaging AI</p>
      </motion.div>
    </div>
  );
}

export default App;