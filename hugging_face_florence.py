import torch
from PIL import Image

image1=r"D:\tessTrain\output_with_boxes.jpg"
# Load the model and processor
device = torch.device("cpu")

model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True).to(device)
processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)

# Prepare your image
image = Image.open(image1).convert("RGB")
prompt = "<OCR_with_detection_regions>"

# Generate the OCR output with detection regions
inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
generated_ids = model.generate(
    input_ids=inputs["input_ids"],
    pixel_values=inputs["pixel_values"],
    max_new_tokens=1024,
    num_beams=3
)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(generated_text)
