
"""
Adversarial Makeup Transfer & Face Matching Pipeline
---------------------------------------------------
This script finds the most similar face in a reference dataset (by gender),
then applies adversarial makeup using DiffAM to test face recognition robustness.
"""

import os
import cv2
import argparse
import subprocess
import numpy as np
from insightface.app import FaceAnalysis
from scipy.spatial.distance import cosine

# --- Configuration ---
DATASETS = {'Male': './boys', 'Female': './girls'}
CHECKPOINT_PATH = 'checkpoint/test_MR_MT_t150_ninv20_ngen3_id1_l12_lr8e-06_face_without_makeup-0.pth'
MAIN_SCRIPT = 'main.py'
DEFAULT_CONFIG = 'MT.yml'
EXP_DIR = './runs/test'

def get_gender_label(gender_id):
    """Convert gender ID to string label."""
    return 'Male' if gender_id == 1 else 'Female'

def calculate_cosine_similarity(embedding1, embedding2):
    """Return cosine similarity between two embeddings."""
    return 1 - cosine(embedding1, embedding2)

def load_face_app():
    """Initialize the InsightFace analysis app."""
    app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app

def extract_faces_and_embeddings(app, image_path):
    """Detect faces and extract embeddings from an image."""
    img = cv2.imread(image_path)
    faces = app.get(img)
    return faces, img

def process_reference_image(app, reference_path):
    """Ensure at least one face is detected in the reference image."""
    faces, _ = extract_faces_and_embeddings(app, reference_path)
    if not faces:
        raise Exception("No face detected in reference image.")
    return faces

def get_best_match_face(app, gender, reference_embedding):
    """Find the most similar face in the dataset for the given gender."""
    best_score = -1
    best_match = None
    dataset_dir = DATASETS[gender]

    for root, _, files in os.walk(dataset_dir):
        for fname in files:
            if not fname.lower().endswith(('.jpg', '.png', '.jpeg')):
                continue
            path = os.path.join(root, fname)
            try:
                faces, _ = extract_faces_and_embeddings(app, path)
                if not faces:
                    continue
                similarity = calculate_cosine_similarity(reference_embedding, faces[0].embedding)
                if similarity > best_score:
                    best_score = similarity
                    best_match = path
            except Exception as e:
                print(f"Error processing {path}: {e}")
    return best_match, best_score

def compute_adversarial_intensity(similarity):
    """
    Adjust attack strength based on similarity.
    More similar faces require less noise to fool recognition.
    """
    base_noise = 40
    strength = int((1 - similarity) * base_noise)
    return max(10, strength)

def run_attack(selected_image, similarity_score):
    """Run the adversarial makeup transfer using DiffAM."""
    adv_strength = compute_adversarial_intensity(similarity_score)
    command = [
        'python', MAIN_SCRIPT,
        '--edit_one_image_MR',
        '--config', DEFAULT_CONFIG,
        '--exp', EXP_DIR,
        '--n_iter', '1',
        '--t_0', '300',
        '--n_inv_step', '40',
        '--n_train_step', '6',
        '--n_test_step', str(adv_strength),
        '--img_path', selected_image,
        '--model_path', CHECKPOINT_PATH
    ]
    print(f"\n[INFO] Running adversarial edit with noise level {adv_strength} on: {selected_image}")
    subprocess.run(command)

def main(reference_img_path):
    print("=== Adversarial Makeup Transfer & Face Matching ===")
    print(f"[INFO] Reference image: {reference_img_path}")

    app = load_face_app()
    ref_faces = process_reference_image(app, reference_img_path)

    for idx, face in enumerate(ref_faces):
        gender = get_gender_label(face.gender)
        print(f"\n[Face {idx}] Gender: {gender}")
        embedding = face.embedding
        best_match, similarity = get_best_match_face(app, gender, embedding)

        if best_match:
            print(f"[Match Found] {best_match} (Similarity: {similarity:.4f})")
            run_attack(best_match, similarity)
        else:
            print(f"[WARN] No match found for gender: {gender}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Find similar faces and launch adversarial attack')
    parser.add_argument('--ref_image', type=str, required=True, help='Path to reference face image')
    args = parser.parse_args()

    if not os.path.exists(args.ref_image):
        print(f"[ERROR] Image not found: {args.ref_image}")
        exit(1)

    main(args.ref_image)
