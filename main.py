import numpy as np             
from PIL import Image     
import os
import torch
import timm
import time

# Load pretrained ResNet50 model for image embedding extraction
model = timm.create_model("resnet50", pretrained=True, num_classes=0)
model.eval()

transforms = timm.data.create_transform(**timm.data.resolve_model_data_config(model))

features = []
labels = []

# Build disease image database from local folders
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_PATH = os.path.join(BASE_DIR, "database")
folders = os.listdir(DATABASE_PATH)

def get_embedding(image_path):
    img = Image.open(image_path)
    img = img.convert("RGB")
    tensor = transforms(img)
    with torch.no_grad():
        embedding = model(tensor.unsqueeze(0))
    return embedding

for folder in folders:
    files = os.listdir(os.path.join(DATABASE_PATH, folder))
    for file in files:
        if not file.endswith('.jpg'):
            continue
        image_paths = os.path.join(DATABASE_PATH, folder, file)
        features.append(get_embedding(image_paths))
        labels.append(folder)

print("\n" + "="*50)
print("     SKINSIGHT — AI SKIN DISEASE SCREENER")
time.sleep(2)
print("="*50)
print("\nThis tool uses AI image analysis and clinical")
print("symptom questions to screen for skin conditions.\n")
time.sleep(2)

patients_image = input("Please paste an image of the issue: ")
patients_embedding = get_embedding(patients_image)

# Rank diseases by visual similarity to patient image
def find_match(patients_embedding):
    min_distances = {disease: float('inf') for disease in set(labels)}
    for index in range(len(features)):
        distance = np.linalg.norm(patients_embedding.detach().numpy() - features[index].detach().numpy())
        disease = labels[index]
        if distance < min_distances[disease]:
            min_distances[disease] = distance
    sorted_list = sorted(min_distances, key=lambda x: min_distances[x])
    return sorted_list

match_found = find_match(patients_embedding)

disease_qna = {
    "Vitiligo": ["Are the patches completely white or lighter than your normal skin tone? ", "Are the edges of the patches well defined and sharp? ", "Is there no itching or pain in the affected area? ", "Has a family member ever had similar white patches? ", "Have the patches been slowly growing over months? "],
    "Ringworm": ["Is there a clear raised red border forming a ring with healthy skin in the center? ", "Is there itching or burning in the area? ", "Is the border of the patch raised or scaly? ", "Did it appear and spread quickly over days or weeks? ", "Have you been in contact with animals or someone with a similar rash? "],
    "Acne": ["Do you have red or inflamed bumps on your skin? ", "Are the bumps filled with pus or white at the tip? ", "Is the affected area oily or greasy feeling? ", "Are the bumps mostly on your face, chest or back? ", "Have you been experiencing this during puberty or hormonal changes? "],
    "Psoriasis": ["Are the patches covered with thick silvery or white scales? ", "Are the edges of the patches very clearly defined and sharp? ", "Is the skin underneath the scales red and inflamed? ", "Do the patches appear on your elbows, knees or scalp? ", "Do joints near the affected area feel stiff or painful? "],
    "Melanoma": ["Is the mole or patch asymmetrical — different on each side? ", "Is the border irregular, ragged or uneven? ", "Does the patch have more than one color — brown, black, red or white? ", "Is the patch larger than 6mm — roughly the size of a pencil eraser? ", "Has the mole changed in size, shape or color recently? "]
}

# Ask symptom questions to confirm image diagnosis
def ask_symptoms(match_found):
    counter = 0
    for disease, questions in disease_qna.items():
        if disease == match_found:
            for i, question in enumerate(questions):
                answer = input(question)
                if answer.lower() == "yes":
                    counter += 1
                if i == 2 and counter < 2:
                    break

    if counter >= 4:
        return match_found
    else:
        diagnosis = "unclear"
        for disease, questions in disease_qna.items():
            if disease != match_found:
                counter2 = 0
                for i, question in enumerate(questions):
                    answer = input(question)
                    if answer.lower() == "yes":
                        counter2 += 1
                    if i == 2 and counter2 < 2:
                        break
                if counter2 >= 4:
                    diagnosis = disease
                    return diagnosis

        return diagnosis

symptoms = ask_symptoms(match_found[0])

time.sleep(1.75)

print("\n" + "="*50)
print("        SKINSIGHT — SCREENING RESULTS")
print("="*50)
time.sleep(1.75)
print(f"\n  Diagnosis: {symptoms.upper()}")
print(f"\n  ⚠️  This is a screening tool only.")
time.sleep(2)
print("  Please consult a dermatologist for")
print("  professional medical advice.")
print("\n" + "="*50)