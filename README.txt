This project focuses on implementing adversarial attack techniques (L2-bounded PGD) and evaluating their impact on a model’s robustness, along with a defense mechanism combining JPEG compression and feature squeezing. Additionally, the project includes visualization and performance analysis of the adversarial attacks and defenses, examining metrics like L2 norm, SSIM, PSNR, and iterations. The goal is to explore the balance between adversarial success and perceptual quality, as well as demonstrate the defense’s effectiveness in mitigating adversarial threats.

The code is executed in Google Colab with an A100 GPU to ensure efficient computation of adversarial examples and defense evaluations. This setup ensures compatibility with Colab’s runtime environment, and the code has been thoroughly tested to perform optimally in this setting.

Project Overview

1. Data Preprocessing: Imports and preprocesses the dataset for adversarial testing.
2. Adversarial Attacks: Implements targeted and untargeted L2-bounded PGD attacks with 	varying epsilon values.
3. Defense Mechanism: Utilizes JPEG compression and feature squeezing to mitigate the impact of adversarial perturbations.
4. Performance Evaluation: Analyzes the model’s robustness by comparing metrics such as L2 norm, iterations, SSIM, and PSNR before and after applying the defense mechanism.
5. Visualization: Provides visual comparisons of adversarial images, perturbations, and the defense effect.

Install all required dependencies listed in the requirements.txt file using the following command:
```
pip install -r requirements.txt
```

To avoid conflicts with existing packages, it’s best to use a virtual environment:
```
# Create virtual environment
python3 -m venv venv

# Activate the virtual environment
# For Linux/macOS:
source venv/bin/activate

# For Windows:
venv\Scripts\activate

# Install dependencies within the environment
pip install -r requirements.txt
```

Follow these steps to run the project in Google Colab:
	1.	Open Google Colab.
	2.	Upload the project files (including the Jupyter notebook or Python script).
	3.	Enable GPU support:
	•	Go to Runtime > Change runtime type > Set Hardware Accelerator to GPU (A100).
	4.	Execute the cells or run the Python script step by step.


