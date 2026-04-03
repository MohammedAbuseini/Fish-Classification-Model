# Fish Species Classifier

A deep learning project for **fish image recognition** using a **two-stage pipeline** and a simple **Flask web application** for inference.

The system first determines whether an uploaded image contains a fish. If the image is classified as a fish, it then predicts the fish species using an ensemble of three deep learning models.

---

## Project Overview

This project is organized into two main stages:

### Stage 1: Fish vs. Not Fish Detection
A binary classifier is used as a filtering step to decide whether an image contains a fish.

- **Model:** MobileNetV3-based binary classifier
- **Saved model:** `Binary.keras`
- **Goal:** Prevent non-fish images from being passed to the species classification stage

### Stage 2: Fish Species Classification
If an image is identified as a fish, the system runs an ensemble of three species classification models:

- **Swin Transformer** → `best_swin_fish_model1.pth`
- **EfficientNet-B4** → `best_efficientnet_b4.pth`
- **DenseNet121** → `densenet121_fish.pth`

The final prediction is produced using **majority voting**. If all three models disagree, the system selects the prediction with the **highest confidence**.

---

## Supported Fish Classes

The current application predicts the following 31 classes:

- Bangus
- Big Head Carp
- Black Spotted Barb
- Catfish
- Climbing Perch
- Fourfinger Threadfin
- Freshwater Eel
- Glass Perchlet
- Goby
- Gold Fish
- Gourami
- Grass Carp
- Green Spotted Puffer
- Indian Carp
- Indo-Pacific Tarpon
- Jaguar Gapote
- Janitor Fish
- Knifefish
- Long-Snouted Pipefish
- Mosquito Fish
- Mudfish
- Mullet
- Pangasius
- Perch
- Scat Fish
- Silver Barb
- Silver Carp
- Silver Perch
- Snakehead
- Tenpounder
- Tilapia

---

## How the Pipeline Works

1. The user uploads an image through the web interface.
2. The image is passed to the **binary model**.
3. If the output is **Not Fish**, the pipeline stops.
4. If the output is **Fish**, the image is passed to three classification models:
   - Swin Transformer
   - EfficientNet-B4
   - DenseNet121
5. The final species result is selected by:
   - **majority vote**, or
   - **highest-confidence fallback** if all predictions differ
6. The app returns:
   - whether the image is a fish,
   - the final predicted species,
   - confidence score,
   - each model’s individual prediction

---

## Project Structure

```bash
project/
├── App/
│   ├── app.py
│   └── index.html
├── Stage 1/
│   ├── Binary.keras
│   ├── MobileNetV3-Fish-classifier.ipynb
│   ├── shrinking_not-fish dataset.ipynb
│   └── Splitting_the_not-fish data.ipynb
├── Stage 2/
│   ├── best_efficientnet_b4.pth
│   ├── best_swin_fish_model1.pth
│   ├── densenet121_fish.pth
│   ├── DenseNet_121.ipynb
│   ├── EfficientNet_B4.ipynb
│   └── SWIN.ipynb
```

---

## Technologies Used

### Backend and App
- Python
- Flask
- HTML / CSS / JavaScript

### Deep Learning Frameworks
- TensorFlow / Keras
- PyTorch
- torchvision
- timm

### Image Processing
- Pillow
- NumPy

---

## Model Design

### 1. Binary Classification Stage
This stage acts as a gatekeeper for the pipeline.

- Resizes images to **224 × 224**
- Uses MobileNetV3 preprocessing
- Produces a binary decision:
  - **Fish**
  - **Not Fish**

This improves the reliability of the full system by reducing unnecessary species predictions on irrelevant images.

### 2. Species Classification Stage
Once a fish is detected, the image is processed by three separate classifiers.

#### Swin Transformer
A transformer-based vision model that captures both local and global visual relationships.

#### EfficientNet-B4
A convolutional neural network designed to scale depth, width, and resolution efficiently.

#### DenseNet121
A densely connected CNN that encourages feature reuse and stable gradient flow.

### 3. Ensemble Strategy
The project combines the strengths of all three models:

- If at least two models agree, that class is selected.
- If all predictions are different, the system picks the class with the highest confidence.

This makes the final decision more robust than relying on a single model.

---

## Running the Web App

### 1. Place Required Files
Make sure these files are available in the same directory where `app.py` expects them:

- `binary.keras`
- `best_swin_fish_model1.pth`
- `best_efficientnet_b4.pth`
- `densenet121_fish.pth`

### 2. Install Dependencies
You can install the main dependencies with:

```bash
pip install flask torch torchvision tensorflow pillow timm numpy
```

Depending on your environment, you may also need:

```bash
pip install keras
```

### 3. Start the Server
From the `App` folder, run:

```bash
python app.py
```

The server will start on:

```bash
http://localhost:5000
```

---

## Web Interface Features

The included frontend provides:

- image upload support,
- a visual preview of the uploaded image,
- final prediction display,
- binary fish / not-fish result,
- individual predictions from each model,
- confidence scores.

The interface is styled with an underwater theme to match the project domain.

---

## Training Files Included

### Stage 1 Notebooks
- `MobileNetV3-Fish-classifier.ipynb` — training and evaluation for the binary classifier
- `shrinking_not-fish dataset.ipynb` — reduces the not-fish dataset size
- `Splitting_the_not-fish data.ipynb` — splits the dataset into train / validation / test sets

### Stage 2 Notebooks
- `DenseNet_121.ipynb`
- `EfficientNet_B4.ipynb`
- `SWIN.ipynb`

These notebooks contain the training workflows for the species classification models.

---

## Why This Project Is Strong

This project demonstrates several valuable machine learning and software engineering ideas:

- **multi-stage pipeline design**,
- **binary filtering before detailed classification**,
- **ensemble learning for more robust predictions**,
- **integration of TensorFlow and PyTorch in one application**,
- **deployment through a Flask-based web app**,
- **clear separation between training and inference components**.

---

## Possible Future Improvements

Some useful extensions for the project would be:

- adding top-k predictions,
- showing class probabilities in charts,
- Dockerizing the app,
- adding model performance metrics to the interface,
- improving error handling for invalid image uploads,
- deploying the app online using Render, Hugging Face Spaces, or another cloud platform.

---

## Author

**Mohammed Abuseini**

A machine learning and software engineering project focused on practical computer vision, ensemble modeling, and deployment.

---

## License

This project is for educational and portfolio use unless a separate license is added.
