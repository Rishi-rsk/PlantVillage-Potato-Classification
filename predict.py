import torch
from torchvision import transforms
from PIL import Image
from model import get_model

classes = ["Early Blight", "Late Blight", "Healthy"]

model = get_model(3)
model.load_state_dict(torch.load("saved_model/potato_model.pth", map_location="cpu"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

def predict(image_path):
    img = Image.open(image_path)
    img = transform(img).unsqueeze(0)
    with torch.no_grad():
        pred = model(img)
        return classes[pred.argmax().item()]

print(predict("test.jpg"))
