class FileUploadHandler {
  constructor() {
    this.uploadArea = document.getElementById("upload-area");
    this.fileInput = document.getElementById("file-input");
    this.uploadContent = document.getElementById("upload-content");
    this.analyzeBtn = document.getElementById("analyze-btn");
    this.loadingDiv = document.getElementById("analysis-loading");
    this.resultDiv = document.getElementById("analysis-result");

    this.currentFile = null;
    this.setupEventListeners();
  }

  setupEventListeners() {
    this.uploadArea.addEventListener("dragenter", this.handleDragEnter.bind(this));
    this.uploadArea.addEventListener("dragover", this.handleDragOver.bind(this));
    this.uploadArea.addEventListener("dragleave", this.handleDragLeave.bind(this));
    this.uploadArea.addEventListener("drop", this.handleDrop.bind(this));
    this.uploadArea.addEventListener("click", () => this.fileInput.click());
    this.fileInput.addEventListener("change", this.handleFileSelect.bind(this));
    this.analyzeBtn.addEventListener("click", this.analyzeFile.bind(this));
  }

  handleDragEnter(e) {
    e.preventDefault();
    this.uploadArea.classList.add("drag-active");
  }

  handleDragOver(e) {
    e.preventDefault();
  }

  handleDragLeave(e) {
    e.preventDefault();
    if (!this.uploadArea.contains(e.relatedTarget)) {
      this.uploadArea.classList.remove("drag-active");
    }
  }

  handleDrop(e) {
    e.preventDefault();
    this.uploadArea.classList.remove("drag-active");

    const files = e.dataTransfer.files;
    if (files.length > 0) {
      this.handleFile(files[0]);
    }
  }

  handleFileSelect(e) {
    const files = e.target.files;
    if (files.length > 0) {
      this.handleFile(files[0]);
    }
  }

  handleFile(file) {
    this.currentFile = file;
    this.displayFileInfo(file);
    this.showAnalyzeButton();
    this.hideResult();
  }

  displayFileInfo(file) {
    const fileIcon = this.getFileIcon(file);
    this.uploadContent.innerHTML = `
      <i class="${fileIcon} upload-icon"></i>
      <p class="upload-title">${file.name}</p>
      <p class="upload-subtitle">${(file.size / 1024 / 1024).toFixed(2)} MB</p>
    `;
  }

  getFileIcon(file) {
    if (file.type.startsWith("video/")) return "fas fa-file-video text-blue";
    if (file.type.startsWith("image/")) return "fas fa-file-image text-green";
    return "fas fa-file";
  }

  showAnalyzeButton() {
    this.analyzeBtn.style.display = "block";
  }

  hideAnalyzeButton() {
    this.analyzeBtn.style.display = "none";
  }

  showLoading() {
    this.loadingDiv.style.display = "block";
  }

  hideLoading() {
    this.loadingDiv.style.display = "none";
  }

  showResult(result) {
    this.resultDiv.style.display = "block";

    const resultContent = this.resultDiv.querySelector(".result-content");
    const resultIcon = this.resultDiv.querySelector(".result-icon");
    const resultTitle = this.resultDiv.querySelector(".result-title");
    const resultConfidence = this.resultDiv.querySelector(".result-confidence");
    const resultExplanation = this.resultDiv.querySelector(".result-explanation");

    if (result.message === "No face detected") {
      resultContent.className = "result-content no-face";
      resultIcon.className = "result-icon no-face fas fa-exclamation-circle";
      resultTitle.textContent = "Aucun visage détecté";
      resultExplanation.textContent = "Aucun visage n'a été détecté dans cette image.";
      resultConfidence.innerHTML = "";

      const imagePreview = document.createElement("img");
      imagePreview.src = result.imagePath + `?t=${Date.now()}`;
      imagePreview.alt = "Image analysée";
      imagePreview.style.maxWidth = "100%";
      imagePreview.style.borderRadius = "10px";
      imagePreview.style.marginTop = "1rem";
      imagePreview.style.boxShadow = "0 0 10px rgba(0,0,0,0.2)";
      imagePreview.style.cursor = "pointer";

      resultConfidence.appendChild(imagePreview);
      return;
    }

    const hasDeepfake = result.faces.some(face => face.isDeepfake);
    if (hasDeepfake) {
      resultContent.className = "result-content deepfake";
      resultIcon.className = "result-icon deepfake fas fa-times-circle";
      resultTitle.textContent = "Deepfake détecté";
    } else {
      resultContent.className = "result-content authentic";
      resultIcon.className = "result-icon authentic fas fa-check-circle";
      resultTitle.textContent = "Contenu authentique";
    }

    resultExplanation.textContent = "Analyse visage par visage :";
    const faceDetails = document.createElement("div");
    faceDetails.style.marginTop = "1rem";
    faceDetails.style.fontSize = "14px";

    result.faces.forEach((face, index) => {
      const label = face.isDeepfake ? "Deepfake" : "Réel";
      const color = face.isDeepfake ? "red" : "green";
      const confidence = (face.confidence * 100).toFixed(1);
      const faceInfo = document.createElement("p");
      faceInfo.innerHTML = `👤 Visage ${index + 1} : <span style="color:${color}">${label}</span> – Confiance : ${confidence}%`;
      faceDetails.appendChild(faceInfo);
    });

    const imagePreview = document.createElement("img");
    imagePreview.src = result.imagePath + `?t=${Date.now()}`;
    imagePreview.alt = "Image analysée";
    imagePreview.style.maxWidth = "100%";
    imagePreview.style.borderRadius = "10px";
    imagePreview.style.marginTop = "1rem";
    imagePreview.style.boxShadow = "0 0 10px rgba(0,0,0,0.2)";
    imagePreview.style.cursor = "pointer";

    resultConfidence.textContent = "";
    resultConfidence.appendChild(faceDetails);
    resultConfidence.appendChild(imagePreview);
  }

  hideResult() {
    this.resultDiv.style.display = "none";
  }

  async analyzeFile() {
    if (!this.currentFile) return;

    this.hideAnalyzeButton();
    this.showLoading();

    const formData = new FormData();
    formData.append("file", this.currentFile);

    try {
      const response = await fetch("/predict", {
        method: "POST",
        body: formData
      });

      const result = await response.json();
      this.hideLoading();
      this.showResult(result);
    } catch (err) {
      console.error("Erreur de prédiction :", err);
      this.hideLoading();
      alert("Erreur lors de l’analyse.");
    }
  }
}

document.addEventListener("DOMContentLoaded", () => {
  new FileUploadHandler();

  // Modal logic
  const modal = document.getElementById("image-modal");
  const modalImg = document.getElementById("modal-img");
  const closeModal = document.querySelector(".close-modal");

  document.body.addEventListener("click", function (e) {
    if (e.target.tagName === "IMG" && e.target.alt === "Image analysée") {
      modal.style.display = "block";
      modalImg.src = e.target.src;
    }
  });

  closeModal.onclick = function () {
    modal.style.display = "none";
  };

  modal.onclick = function (e) {
    if (e.target === modal) {
      modal.style.display = "none";
    }
  };
});
