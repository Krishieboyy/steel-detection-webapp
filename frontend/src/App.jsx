import React from "react";
import UploadBox from "./components/UploadBox";
import "./App.css";

export default function App() {
  return (
    <div className="page">
      <header className="hero-header">
        <h1>Industrial Steel Surface Defect Detection</h1>
        <p>AI-based detection of manufacturing defects from steel surface images</p>
      </header>

      <main className="main">
        <UploadBox />
      </main>
    </div>
  );
}
