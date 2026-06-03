import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))




client = InferenceClient(
    api_key=os.environ["HF_HUB_TOKEN"],
)

# output is a PIL.Image object
image = client.text_to_image(
    "A cartoon doctor with a stethoscope, smiling and giving a thumbs up, in a bright and friendly style",
    model="black-forest-labs/FLUX.1-schnell",
)

if __name__ == '__main__':
    image.show()