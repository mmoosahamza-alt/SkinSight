# SkinSight — AI Skin Disease Screening Tool

A multimodal skin disease screening tool built in Python that combines 
image similarity matching with a clinical symptom engine.

## How it works
1. Patient uploads a skin image
2. System extracts visual embeddings using ResNet50
3. Compares against a labeled clinical image database
4. Asks targeted symptom questions to confirm the diagnosis
5. Returns the most likely condition, with early-exit logic that 
   pivots to the next most likely disease if symptoms don't match

## Tech Stack
- Python
- PyTorch + timm (ResNet50 embeddings)
- NumPy, Pillow

## Diseases Detected
- Melanoma
- Vitiligo
- Ringworm
- Acne
- Psoriasis

## Dataset
Images sourced from the ISIC Archive (isic-archive.com).
The image database is not included in this repo, as individual images 
carry their own licensing/attribution requirements that cannot be 
guaranteed at redistribution. See the demo video below for the tool in action.

## Demo
[Watch the demo video](https://youtu.be/hyrsuXu1K4s?si=fl_pKcGawjkJI-f6)

## Usage
1. Build a local `database/` folder with subfolders for each disease, 
   containing labeled reference images
2. Run `main.py`
3. Enter the path to a skin image when prompted
4. Answer the symptom questions
5. Receive a screening result

## Disclaimer
This is a screening tool only and is not a substitute 
for professional medical diagnosis. Always consult a dermatologist.

## License
All rights reserved. This code is shared for portfolio/demonstration 
purposes only. Please contact me if you'd like to use or reference it.
