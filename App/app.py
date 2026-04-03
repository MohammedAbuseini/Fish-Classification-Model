from flask import Flask, render_template, request, jsonify, send_from_directory
import torch
from PIL import Image
import io
import base64
from torchvision import transforms
import timm
import torchvision.models as models
import torch.nn as nn
from collections import Counter
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os


app = Flask(__name__, template_folder='.')


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


classes = [
    "Bangus", "Big Head Carp", "Black Spotted Barb", "Catfish",
    "Climbing Perch", "Fourfinger Threadfin", "Freshwater Eel",
    "Glass Perchlet", "Goby", "Gold Fish", "Gourami", "Grass Carp",
    "Green Spotted Puffer", "Indian Carp", "Indo-Pacific Tarpon",
    "Jaguar Gapote", "Janitor Fish", "Knifefish", "Long-Snouted Pipefish",
    "Mosquito Fish", "Mudfish", "Mullet", "Pangasius", "Perch",
    "Scat Fish", "Silver Barb", "Silver Carp", "Silver Perch",
    "Snakehead", "Tenpounder", "Tilapia"
]


inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def load_binary_model():
            
            binary_model = keras.models.load_model(
            "binary.keras",
            
            )
            return binary_model


def load_classification_models():
    
    model_1 = timm.create_model(
        "swin_tiny_patch4_window7_224",
        pretrained=False,
        num_classes=len(classes)
    )
    model_1.load_state_dict(
        torch.load("best_swin_fish_model1.pth", map_location=DEVICE, weights_only=True)
    )
    model_1.to(DEVICE)
    model_1.eval()

    
    model_2 = timm.create_model(
        "efficientnet_b4",
        pretrained=False,
        num_classes=len(classes)
    )
    model_2.load_state_dict(
        torch.load("best_efficientnet_b4.pth", map_location=DEVICE, weights_only=True)
    )
    model_2.to(DEVICE)
    model_2.eval()

    
    try:
        model_3 = models.densenet121(weights=None)
    except TypeError:
        
        model_3 = models.densenet121(pretrained=False)
    
    model_3.classifier = nn.Linear(model_3.classifier.in_features, len(classes))
    model_3.load_state_dict(
        torch.load("densenet121_fish.pth", map_location=DEVICE, weights_only=True)
    )
    model_3.to(DEVICE)
    model_3.eval()

    return model_1, model_2, model_3

print("Loading binary model...")
binary_model = load_binary_model()

print("Loading classification models...")
swin_model, efficientnet_model, densenet_model = load_classification_models()
print("Classification models loaded successfully!")


    

def predict_binary_image(image_pil, model):
    img = image_pil.resize((224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    img_array = tf.keras.applications.mobilenet_v3.preprocess_input(img_array)

    prediction = model.predict(img_array, verbose=0)[0][0]
    return prediction
    
    
    return prediction


def predict_binary(image, threshold=0.3):
    
    prob = predict_binary_image(image,binary_model)
     
    
    if prob < threshold:
        return "Fish", float(1.0 - prob)
    else:
        return "Not Fish", float(prob)


def predict_single_model(image, model):
    input_tensor = inference_transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
    
    predicted_class = classes[pred_idx]
    confidence = probs[0, pred_idx].item()
    return predicted_class, confidence


def ensemble_predict_species(image):
    
    swin_class, swin_conf = predict_single_model(image, swin_model)
    eff_class, eff_conf = predict_single_model(image, efficientnet_model)
    dense_class, dense_conf = predict_single_model(image, densenet_model)

    predictions = [swin_class, eff_class, dense_class]
    confidences = [swin_conf, eff_conf, dense_conf]

   
    class_counts = Counter(predictions)
    most_common = class_counts.most_common()

    if most_common[0][1] >= 2:
        final_class = most_common[0][0]
    else:
        max_conf_idx = confidences.index(max(confidences))
        final_class = predictions[max_conf_idx]

    final_confidences = [conf for pred, conf in zip(predictions, confidences) if pred == final_class]
    avg_confidence = np.mean(final_confidences)

    return {
        "final_prediction": final_class,
        "confidence": float(avg_confidence),
        "model_predictions": {
            "Swin Transformer": {"class": swin_class, "confidence": float(swin_conf)},
            "EfficientNet B4": {"class": eff_class, "confidence": float(eff_conf)},
            "DenseNet121": {"class": dense_class, "confidence": float(dense_conf)}
        }
    }


def fish_pipeline(image):
    
   
    
    binary_label, binary_conf = predict_binary(image)
    
    print(f"Binary Model → {binary_label} ({binary_conf:.2f})")

   
    if binary_label == "Fish":
        print("✅ Fish detected → running ensemble models...\n")
        species_result = ensemble_predict_species(image)
        
        return {
            "is_fish": True,
            "binary_confidence": binary_conf,
            "final_prediction": species_result["final_prediction"],
            "confidence": species_result["confidence"],
            "model_predictions": species_result["model_predictions"]
        }
    else:
        print("⛔ Not a fish → stopping pipeline.")
        return {
            "is_fish": False,
            "binary_confidence": binary_conf,
            "final_prediction": "Not Fish",
            "confidence": binary_conf,
            "model_predictions": {}
        }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
       
        result = fish_pipeline(image)

        
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        result['image'] = f"data:image/jpeg;base64,{img_str}"

        return jsonify(result)

    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    
    import os
    required_files = [
        "binary.keras",
        "best_swin_fish_model1.pth",
        "best_efficientnet_b4.pth",
        "densenet121_fish.pth"
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print(f"\n⚠️  WARNING: Missing model files: {', '.join(missing_files)}")
        print("Please ensure all model files are in the same directory as app.py\n")
    
    print("\n" + "="*50)
    print("🐠 Fish Species Classifier Server Starting...")
    print("="*50)
    print(f"📍 Device: {DEVICE}")
    print(f"🌐 Server running at: http://localhost:5000")
    print(f"🌐 Network access at: http://0.0.0.0:5000")
    print("="*50 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
    