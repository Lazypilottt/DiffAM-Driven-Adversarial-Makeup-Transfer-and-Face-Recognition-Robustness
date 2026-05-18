# Face Similarity Matching and Adversarial Noise Strategy

This document describes the method used to calculate facial similarity and how this information is used to guide adversarial noise addition in the context of the DiffAM project.

---

## 1. Face Similarity Calculation

### Tools Used:

- **InsightFace (buffalo_l)** model for face detection and feature embedding.
- **Cosine similarity** to measure closeness between face embeddings.

### Process:

1. **Preprocessing**:
   - Convert input and gallery images to RGB.
   - Align faces using InsightFace detection.
   - Resize to 112x112 and normalize as required by the recognition model.

2. **Embedding Extraction**:
   - Pass each face through the InsightFace model to get a 512-dimensional embedding vector.

3. **Similarity Score**:
   - Use cosine similarity:  
     `similarity = 1 - (e1 ⋅ e2) / (||e1|| * ||e2||)`  
     Values closer to 1 indicate higher similarity.

4. **Gender Filtering**:
   - Detect gender using the `genderage.onnx` model from InsightFace.
   - Match only within the corresponding gender folder (`boys/`, `girls/`).

5. **Output**:
   - Best match is selected based on the highest similarity score.

---

## 2. Adversarial Noise Strategy

### Objective:

To guide the style transformation using makeup while retaining identity and maximizing adversarial effect (e.g., fooling a face recognizer).

### Strategy:

1. **Similarity-Guided Thresholding**:

   - Define three tiers of similarity:

     - **High (0.80–1.00)** → allow **more** adversarial noise  
     - **Medium (0.65–0.79)** → allow **moderate** noise  
     - **Low (<0.65)** → allow **minimal** perturbation  

2. **Integration with DiffAM**:

   - Use similarity tier to adjust loss weights:

     - High similarity → lower identity loss, higher makeup/adversarial loss  
     - Low similarity → balance identity strongly to avoid collapse  

3. **Parameter Tuning**:

   - Modify `n_train_step`, `n_inv_step`, `n_test_step`, and `t_0` parameters dynamically based on similarity.
   - Optionally scale `clip_loss`, `attack_loss` weights during training steps.

### Example:

For a similarity score of **0.92**:

```bash
python main.py --edit_one_image_MR --config MT.yml \
  --exp ./runs/test \
  --n_iter 1 --t_0 300 --n_inv_step 40 \
  --n_train_step 6 --n_test_step 40 \
  --img_path matched_image.jpg \
  --model_path checkpoint/identity_strong_adversarial.pth
```
