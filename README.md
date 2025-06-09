# AetherFlow

AetherFlow is an embedded AI-powered system for early prediction of vaso-occlusive crises (VOC) in sickle cell disease patients. It uses symptom data and a machine learning model to provide real-time risk assessment, helping patients and caregivers take preventive action.

---

## 📦 Description

AetherFlow collects user symptom data (pain, fatigue, fever, hydration, prior crises, etc.), processes it through a trained machine learning model (deployed with TensorFlow Lite Micro), and predicts the likelihood of a crisis within 48 hours. The system is designed for microcontroller platforms and can be integrated into wearable or portable health devices.

---

## 🔗 GitHub Repository

[https://github.com/iPascal619/Aetherflow.git]

---

## ⚙️ Setup & Installation

### 1. **Clone the Repository**
```sh
git clone https://github.com/iPascal619/Aetherflow.git
cd aetherflow
```

### 2. **Python Environment (for data & model)**
```sh
python3 -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt
```

### 3. **Generate Data & Train Model**
```sh
python model/simulate_data.py
python model/train_model.py
```
- This will create a balanced dataset and export a TFLite model.

### 4. **Build Embedded Inference (C++)**
- Navigate to the TFLite Micro example directory:
  ```sh
  cd tflite-micro/tensorflow/lite/micro/examples/aetherflow
  ```
- Build using your platform’s toolchain (e.g., `make`, PlatformIO, Arduino IDE, etc.).
- Flash the binary to your microcontroller or run on your desktop for testing.


---

## Video Demo
 
 [https://www.loom.com/share/15ce6e4221454af1ac9ad67bcf78b592?sid=2b63b067-4add-43e7-acf4-232390f8e41f](Part 1)

 [https://www.loom.com/share/b44bde93d7f2421c8f8d4d87af3069f1?sid=18d9ff7d-0759-44ff-82fe-7e36d2779bcc](Part 2)

---

## 📊 Model Performance

![/home/pascal/Aetherflow/Screenshot (626).png](performance_metrics)
## 🚀 Deployment Plan

1. **Model Training:**  
   - Train and export the TFLite model using the provided Python scripts.

2. **Firmware Integration:**  
   - Embed the TFLite model as a C array in your microcontroller project.
   - Use the provided `main.cc` for real-time inference.

3. **Testing:**  
   - Test predictions using simulated or real symptom data.
   - Validate outputs against expected results.

5. **Production:**  
   - Integrate into wearable/portable device.
   - Provide user documentation and support.

---

## 📞 Support

For questions, issues, or contributions, please open an issue or pull request on the [https://github.com/iPascal619/Aetherflow.git]

---

**AetherFlow — Empowering proactive sickle cell care with embedded AI.**
