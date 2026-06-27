# 🔬 Mobile-First Malaria Cell Diagnostic System Using Fine-Tuned CNNs

A deep learning project focused on computer vision classification of blood smear images to automate malaria diagnosis. This system compares a custom Convolutional Neural Network built from scratch against advanced Transfer Learning architectures, ultimately delivering a live web application deployed via Streamlit.

🚀 **Live Application Link:** malaria-cell-classifier33.streamlit.app

---

## 📊 Project Performance Summary

| Architecture Model | Parameters/Size | Test Accuracy | Key Characteristic / Status |
| :--- | :---: | :---: | :--- |
| **Custom Scratch CNN** | Lightweight | 95.7% | Strong baseline, long CPU training times |
| **ResNet50 (Frozen)** | Heavy (~100MB) | 93.7% | Structurally heavy for fast deployment |
| **MobileNetV2 (Fine-Tuned)** | Tiny (~14MB) | **96.0%** | 🏆 **Production Champion Model** |

---

## 🧠 Core Engineering Workflow

### 1. High-Performance Data Streaming
To handle massive folder loads efficiently on standard CPU processing units, the dataset pipeline avoids raw RAM arrays. Instead, it leverages TensorFlow's native streaming tools:
* Automated categorization using `image_dataset_from_directory` with standard `(224, 224)` image reshaping.
* Optimized speed using memory caching via `.cache()`.
* Multi-threaded pipeline handling using `.prefetch(buffer_size=tf.data.AUTOTUNE)`.

### 2. Fine-Tuning Strategy
Initially, frozen transfer learning baselines struggled to beat the custom scratch network. To bridge this performance gap:
1. The entire main base of MobileNetV2 was activated for adjustment (`base_mobile.trainable = True`).
2. All initial structural layers were re-locked, leaving only the **deepest 20 layers open** for customization.
3. The network was re-compiled with a microscopic learning rate of **`1e-5`** using the Adam optimizer to avoid catastrophic forgetting.
4. This allowed deep layers to adapt specifically to malaria cell visual profiles, pushing test accuracy to **96.0%**.

---

## 🔍 Medical Error Analysis & Insights

Through clinical evaluation of misclassified instances using custom heatmaps and confusion matrices, a clear performance pattern was isolated:

1. **Faint/Microscopic Rings:** The network occasionally misses highly faint, tiny initialization spots inside infected cells, resulting in a false-negative classification.
2. **Obscured Cell Volumes:** Massive parasite clusters covering almost 90%+ of the cell body distort standard cell shapes, throwing off standard recognition profiles.

---

## 💻 Web App Deployment

The champion fine-tuned MobileNetV2 brain was exported as an optimized `.keras` file and deployed inside a production-ready **Streamlit** dashboard. 

### Features Built:
* **Drag-and-Drop Uploader:** Accepts `.png`, `.jpg`, and `.jpeg` micro-photography.
* **Side-by-Side Analysis Grid:** Renders user-submitted visuals alongside localized AI diagnostic metrics.
* **Clinical Export Tool:** Generates automated downloadable `.txt` report files documenting confidence percentages.

---

## 🛠️ Local Installation & Setup

1. Clone this repository to your system workspace.
2. Ensure you have dependencies ready via terminal:
   ```bash
   pip install -r requirements.txt
   ```
3. Boot up the local user dashboard interface:
   ```bash
   streamlit run app.py
   ```
