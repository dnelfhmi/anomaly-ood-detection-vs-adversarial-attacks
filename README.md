# 🤖 Anomaly, OOD Detection vs. Adversarial Attacks

This project investigates the interplay between adversarial attacks and anomaly/out-of-distribution (OOD) detection, combining the efforts of both a Red Team (attack/defense) and a Blue Team (detection/robustness analysis). The goal is to evaluate model robustness, develop effective defenses, and assess detection mechanisms for adversarial and OOD threats.

---

## 🏗️ Project Structure

- **🔴 Red Team:** Focuses on generating adversarial examples (L2-bounded PGD), evaluating their impact, and implementing defense mechanisms (JPEG compression, feature squeezing).
- **🔵 Blue Team:** Focuses on detecting anomalies and OOD samples, evaluating detection models, and analyzing robustness against both natural and adversarial distribution shifts.

---

## 🔵 Blue Team: Anomaly & OOD Detection

The Blue Team’s objective is to develop and evaluate methods for detecting anomalous and OOD samples, which are critical for real-world model reliability.

**Key Components:**
- **🧠 Detection Models:** Utilizes models such as autoencoders, variational autoencoders (VAE), and one-class SVMs to identify samples that deviate from the training distribution.
- **🔍 Feature Extraction:** Employs deep features from trained classifiers and unsupervised models for improved detection accuracy.
- **📊 Evaluation Metrics:** Assesses detection performance using metrics like ROC-AUC, F1-score, precision-recall, and confusion matrices.
- **🖼️ Visualization:** Provides t-SNE, PCA, and heatmap visualizations to illustrate the separation between in-distribution, OOD, and adversarial samples.
- **🛡️ Robustness Analysis:** Examines how well detection models can distinguish between clean, OOD, and adversarially perturbed data.

**Typical Workflow:**
1. **🗂️ Data Preparation:** Preprocesses CIFAR-10 and other datasets for OOD and anomaly detection.
2. **🏋️ Model Training:** Trains detection models on in-distribution data.
3. **🧪 Testing:** Evaluates on OOD datasets and adversarial examples generated by the Red Team.
4. **📈 Analysis:** Visualizes and reports detection results, highlighting strengths and weaknesses.

---

## 🔴 Red Team: Adversarial Attacks & Defenses

The Red Team’s objective is to challenge model robustness by generating adversarial examples and testing the effectiveness of various defense strategies.

**Key Components:**
- **⚔️ Adversarial Attacks:** Implements targeted and untargeted L2-bounded Projected Gradient Descent (PGD) attacks with varying epsilon values.
- **🛡️ Defense Mechanisms:** Applies JPEG compression and feature squeezing to mitigate adversarial perturbations.
- **📏 Performance Evaluation:** Compares model performance before and after defenses using metrics such as L2 norm, SSIM, PSNR, and attack success rate.
- **🖼️ Visualization:** Shows visual comparisons of clean, adversarial, and defended images, as well as perturbation maps.

**Typical Workflow:**
1. **🗂️ Data Preprocessing:** Loads and prepares datasets for adversarial testing.
2. **⚙️ Attack Generation:** Crafts adversarial examples using PGD.
3. **🛡️ Defense Application:** Applies defense mechanisms to adversarial samples.
4. **📈 Evaluation:** Measures the effectiveness of attacks and defenses using quantitative and qualitative metrics.

---

## 🧪 Results & Insights

- **🛡️ Adversarial Robustness:** The Red Team demonstrates that even strong models can be vulnerable to carefully crafted perturbations, but defenses like JPEG compression can partially restore accuracy.
- **🔎 Detection Effectiveness:** The Blue Team shows that anomaly/OOD detectors can often flag adversarial and OOD samples, but their performance varies depending on the attack strength and feature space.
- **🖼️ Visualization:** Both teams use dimensionality reduction and heatmaps to provide intuitive insights into model behavior under attack and detection scenarios.

---

## 🚀 Getting Started

### 💾 Installation

Install all required dependencies:
```bash
pip install -r requirements.txt
```

It is recommended to use a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Linux/macOS
# venv\Scripts\activate   # On Windows
pip install -r requirements.txt
```

### 🏃 Running the Project

1. **Google Colab:**  
   - Open Google Colab.
   - Upload the project files (including notebooks and scripts).
   - Enable GPU support:  
     `Runtime > Change runtime type > Set Hardware Accelerator to GPU (A100)`
   - Execute the cells or run the Python scripts step by step.

2. **Local Execution:**  
   - Run the main scripts or notebooks as described above.

---

## 📁 File Overview

- `blue_team_final.ipynb` – Blue Team’s detection and analysis notebook.
- `red_team_final.ipynb` – Red Team’s attack and defense notebook.
- `model.py` – Model architectures and utility functions.
- `load.py` – Data loading and evaluation scripts.
- `result/` – Contains result images, plots, and visualizations.
- `data/` – Contains datasets (e.g., CIFAR-10).

---

## 📚 References

- See `COMP90073_BlueTeam.pdf` and `COMP90073_RedTeam.pdf` for detailed methodology, experiments, and results.

---

**This README provides a high-level summary. For in-depth details, refer to the respective team reports and notebooks.**


