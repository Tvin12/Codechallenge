import torch
import torchvision.transforms as T
from torchvision.models.detection import deeplabv3_resnet50
from PIL import Image

# Load the pre-trained DeepLabV3 model
model = deeplabv3_resnet50(pretrained=True)
model.eval()

# Define a function to remove the background from an image
def remove_background(input_image_path, output_image_path):
    # Load the input image
    input_image = Image.open(input_image_path)

    # Preprocess the input image
    preprocess = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)

    # Make a prediction
    with torch.no_grad():
        output = model(input_batch)['out'][0]
    output_predictions = output.argmax(0)

    # Create a mask of the foreground object
    mask = output_predictions.byte().cpu().numpy()

    # Apply the mask to the input image
    input_image = input_image.convert("RGBA")
    input_data = input_image.getdata()
    new_image_data = []
    for i, item in enumerate(input_data):
        if mask[i] == 1:  # Foreground
            new_image_data.append(item[:-1] + (0,))  # Set alpha channel to 0 for the background
        else:  # Background
            new_image_data.append(item)
    input_image.putdata(new_image_data)

    # Save the resulting image with the background removed
    input_image.save(output_image_path, "PNG")

# Example usage
input_image_path = "input_image.jpg"
output_image_path = "output_image.png"
remove_background(input_image_path, output_image_path)
