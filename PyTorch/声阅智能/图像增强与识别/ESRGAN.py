import torch
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage


esrgan_model = torch.hub.load('esrgan', 'esrgan', pretrained=True)
low_res_image = Image.open(r'detection_results\cropped_objects\book_0_conf0.97.jpg')
low_res_tensor = ToTensor()(low_res_image).unsqueeze(0)

with torch.no_grad():
    high_res_tensor = esrgan_model(low_res_tensor)

high_res_image = ToPILImage()(high_res_tensor.squeeze(0))

high_res_image.show()