# Method

## 1. DiffAM in brief

[DiffAM](https://arxiv.org/abs/2405.09882) (Sun et al., CVPR 2024) protects face photos from unauthorised face
recognition (FR) by transferring natural-looking makeup that also carries an adversarial signal. The protected
photo still looks like the same person to a human, but FR models match it to a different target identity.

It works in two stages, both implemented as fine-tuning of a pretrained 256 x 256 DDPM:

**Makeup removal (MR).** A makeup diffusion model is fine-tuned with a CLIP directional loss from the prompt
`face with makeup` towards `face without makeup`, with an ArcFace identity loss, L1 and LPIPS keeping the face
unchanged otherwise. The result is a set of makeup / non-makeup pairs with masks, which define makeup styles.

**Adversarial makeup transfer (MT).** The CelebA-HQ model is fine-tuned per makeup style with:

| term | purpose |
|---|---|
| `loss_dir` | CLIP makeup direction: (source to output) should align with (non-makeup reference to makeup reference) in CLIP image space |
| `loss_dis` | histogram matching on eye, lip and skin regions against the reference makeup |
| L1 + LPIPS | keep the output close to the source image |
| `loss_adv` | ensemble attack pulling the output's embedding towards the target identity on surrogate FR models |

For `--target_model 2` the surrogates are FaceNet, IR152 and IRSE50, and the held-out MobileFace model is the
black-box target used for evaluation. The adversarial term is switched off for the first
`MT_iter_without_adv` iterations so the makeup settles first.

**Editing one image.** The input is inverted with deterministic DDIM to step `t0`, then regenerated with
`n_test_step` DDIM steps of the fine-tuned model.

## 2. Similarity-guided attack budget (this project)

DiffAM applies the same settings to every face. This project adds a per-face budget based on how close the face
already is to another identity.

### Reference gallery

The training split of the Kaggle
[Gender Classification Dataset](https://www.kaggle.com/datasets/cashutosh/gender-classification-dataset):
47,009 faces (23,766 male, 23,243 female), stored in gender folders.

### Embeddings and matching

* InsightFace `buffalo_l`: RetinaFace detector (`det_10g`), ArcFace recogniser (`w600k_r50`, 512-d) and a
  gender/age head (`genderage`).
* The largest detected face is used. Gallery images where detection fails (tight crops) are resized and passed
  straight to the recogniser.
* Similarity is the cosine similarity of the embeddings:

  $$s(q, g) = \frac{e_q \cdot e_g}{\lVert e_q \rVert \, \lVert e_g \rVert}$$

* The query's gender is predicted by the `genderage` head and only the matching gallery folder is searched.
  The score used downstream is the nearest neighbour, $s^* = \max_g s(q, g)$.

Gallery embeddings are computed once and cached (`python -m makeup_attack index`), so a query costs one
matrix-vector product. The first prototype re-ran detection on every gallery image for every query.

### From similarity to budget

Working hypothesis: a face whose nearest other identity is already close in ArcFace space needs a smaller push to
be misidentified.

* **Step budget** (passed to DiffAM as `--n_test_step`, clamped to `t0`):

  $$\text{steps} = \operatorname{round}\big(n_{\min} + (1 - s^*)(n_{\max} - n_{\min})\big), \quad n_{\min}=5,\ n_{\max}=40$$

* **Predicted fooling probability**, a logistic curve centred at $m = 0.6$:

  $$p_{\text{fool}} = \frac{1}{1 + e^{-k (s^* - m)}}, \quad k = 10$$

* **Tiers:** high ($s^* \ge 0.80$), medium ($0.65 \le s^* < 0.80$), low ($s^* < 0.65$).

```
$ python -m makeup_attack estimate
similarity  tier    steps  p(fool)
-----------------------------------
     0.950  high        7    97.1%
     0.850  high       10    92.4%
     0.750  medium     14    81.8%
     0.600  low        19    50.0%
     0.450  low        24    18.2%
     0.300  low        30     4.7%
```

Both curves are hand-set. They encode the hypothesis; they are not fitted to measured attack success.

## 3. Notes on the upstream code

* **`--model_ratio` has no effect.** In `utils/diffusion_utils.py::denoising_step`, both branches of the
  original/fine-tuned blend call `models[1]`, so the output always comes from the fine-tuned model. That is why
  the budget here is expressed as a step count rather than a blend ratio.
* **`--ref_img` must be a makeup pair.** DiffAM loads `assets/datasets/pairs/{ref}_m.png`, `_nm.png` and
  `_mask.png`. An early version of this project passed the best gallery match directly as `--ref_img`, which
  fails because gallery faces have no makeup pairs (see the Colab history). The gallery is therefore used to set
  the budget, not to pick the style.

## 4. Limitations and next steps

* **The Colab MT runs did not exercise the attack term.** They used `--n_iter 2`, while DiffAM only adds
  `loss_adv` to the objective once `it_out >= MT_iter_without_adv` (default 3). `loss_adv` was logged but never
  optimised, so the reported ASR (70% / 20% / 0% at FAR 0.1 / 0.01 / 0.001) is a baseline for the makeup edit
  alone. This also explains why all three styles scored the same. Re-running with `--n_iter 4` (upstream
  default) or `--MT_iter_without_adv 1` is the first experiment to repeat.
* ASR was measured on 10 test images per style, so each image moves the score by 10 points.
* The similarity-to-budget map should be fitted from data: bin faces by $s^*$, run the attack at several budgets
  and measure ASR per bin.
* Generate more makeup styles with the MR model and choose the style whose reference face is closest to the
  input, which would combine the gallery match with style selection.
* Gender labels come from the dataset's binary folders and a binary classifier, which is a coarse filter.
