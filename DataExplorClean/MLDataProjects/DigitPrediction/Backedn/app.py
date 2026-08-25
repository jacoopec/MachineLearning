from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from PIL import Image
from torchvision import transforms
import io

# Initialize Flask app and enable CORS
app = Flask(__name__)
CORS(app)

# Load the TorchScript model
model = torch.jit.load("model_scripted.pt")
model.eval()

# Define class names
class_names = ["class1", "class2", "class3"]  # Replace with your actual class names

# Image transform
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((20, 20)),
    transforms.ToTensor()
])

@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    try:
        image = Image.open(io.BytesIO(file.read())).convert("L")
        tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            output = model(tensor)
            pred = output.argmax(dim=1).item()
            return jsonify({"prediction": class_names[pred]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
