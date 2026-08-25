import torch
from torchvision import transforms
from PIL import Image
import os
from DataExplorClean.MLDataProjects.DigitPrediction.MLPart.train import SimpleCNN

model_path = './model.pth'
num_classes = 3
image_size = (20, 20)
class_names = ['CROSS', 'LINE1', 'LINE2']  # Replace with your actual folder names

model = SimpleCNN(num_classes)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize(image_size),
    transforms.ToTensor()
])

def predict_image(image_path):
    image = Image.open(image_path)
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        predicted_class = output.argmax(dim=1).item()
        return class_names[predicted_class]

# Example usage
if __name__ == "__main__":
    test_img = 'predictThis.png'
    if os.path.exists(test_img):
        prediction = predict_image(test_img)
        print(f"Predicted class: {prediction}")
    else:
        print("⚠️ Please provide a valid image path.")
