# Region of Interest Detection in Melanocytic Skin Tumor Whole Slide Images — Nevus and Melanoma

**A detailed, code-grounded illustration of the paper and its accompanying implementation (PCLA-3C).**

---

## Bibliographic information

| Field | Value |
|---|---|
| Title | Region of Interest Detection in Melanocytic Skin Tumor Whole Slide Images—Nevus and Melanoma |
| Authors | Yi Cui, Yao Li, Jayson R. Miedema, Sharon N. Edmiston, Sherif W. Farag, James Stephen Marron, Nancy E. Thomas |
| Affiliation | University of North Carolina at Chapel Hill (Economics; Statistics & Operations Research; Pathology & Laboratory Medicine; Lineberger Comprehensive Cancer Center; Eshelman School of Pharmacy; Biostatistics; Dermatology) |
| Venue | *Cancers* **2024**, *16*, 2616 (MDPI) |
| DOI | [10.3390/cancers16152616](https://doi.org/10.3390/cancers16152616) |
| Dates | Received 29 June 2024; Revised 17 July 2024; Accepted 19 July 2024; Published 23 July 2024 |
| Model name | **PCLA-3C** (Patch CLAssifier, 3-Class) |
| Code | <https://github.com/cyMichael/ROI_Detection> (this repository is a modified/automated version) |
| License | MIT (code); CC BY 4.0 (paper) |

> **Precursor work (sourced from the code repository; not stated in the paper).** An extended abstract of this work was accepted at *Medical Imaging Meets NeurIPS (Med-NeurIPS) 2022* (Poster Session).

---

## Table of contents

1. [Executive summary](#1-executive-summary)
2. [Clinical background and motivation](#2-clinical-background-and-motivation)
3. [Problem definition](#3-problem-definition)
4. [Dataset: the UNC Melanocytic Tumor Dataset](#4-dataset-the-unc-melanocytic-tumor-dataset)
5. [Method: the PCLA-3C framework](#5-method-the-pcla-3c-framework)
6. [The code: repository walkthrough and paper-to-code mapping](#6-the-code-repository-walkthrough-and-paper-to-code-mapping)
7. [Results](#7-results)
8. [Misclassification analysis](#8-misclassification-analysis)
9. [Discussion, related work, and limitations](#9-discussion-related-work-and-limitations)
10. [Reproducibility and computational environment](#10-reproducibility-and-computational-environment)
11. [Reconciling the reported numbers (notes on internal inconsistencies)](#11-reconciling-the-reported-numbers)
12. [Glossary and abbreviations](#12-glossary-and-abbreviations)
13. [Key references](#13-key-references)
- [Appendix A: Peer-review clarifications](#appendix-a-peer-review-clarifications)

---

## 1. Executive summary

The paper addresses two coupled tasks on digitized **hematoxylin-and-eosin (H&E) whole slide images (WSIs)** of melanocytic skin tumors:

1. **Slide-level classification** — decide whether a slide shows **melanoma** (malignant) or **nevus** (benign).
2. **Region-of-interest (ROI) detection** — localize the diagnostically relevant tumor region within the slide, *without* requiring exhaustive pixel-level ground truth during training.

The core method, **PCLA-3C**, is deliberately simple and patch-based:

- Each gigapixel WSI is tiled into non-overlapping **256 × 256** patches at **20× magnification**.
- A **VGG16** convolutional neural network is fine-tuned as a **3-class patch classifier** with classes **{Melanoma, Nevus, Other}**. The third class ("Other") absorbs non-tumor background tissue, which lets the model learn a meaningful ROI signal even though annotations are incomplete.
- At inference, all patches of a slide are classified. The slide label is decided by **majority vote** between melanoma-patch and nevus-patch counts (ignoring "Other"). Patches are then **ranked** by the winning class's score to produce a predicted ROI.
- **ROI quality** is measured by **Intersection over Union (IoU)** against pathologist annotations on a held-out test set.

**Principal quantitative results** (test set of 26 WSIs, model trained on the full 134-WSI training set):

| Metric | PCLA-3C | CLAM (baseline) |
|---|---|---|
| Patch classification accuracy | **0.892** | — (CLAM does not classify patches) |
| Slide classification accuracy | **0.923** | 0.692 |
| ROI IoU | **0.382** | 0.112 |

The central methodological contribution is **exploiting partial annotations (which the authors describe as a semi-supervised setting) together with an explicit background class** to train a patch classifier that simultaneously yields accurate slide classification and a usable ROI map — substantially outperforming the state-of-the-art weakly-supervised baseline, **CLAM**, on this diagnostically challenging melanocytic dataset.

---

## 2. Clinical background and motivation

**Why melanocytic tumors matter.** The paper opens with epidemiological context: the American Cancer Society projected that in 2022 an estimated **99,780 invasive** and **97,920 in-situ** melanoma cases would be newly diagnosed in the US, with **~7,650 deaths**. The gold standard for diagnosis is a pathologist's visual assessment of H&E-stained tissue sections.

**The diagnostic problem.** The classification of melanocytic tumors is recognized as diagnostically challenging. Multiple studies report substantial **inter-observer discordance** among pathologists — the paper cites literature reporting **25–26% discordance** between individual pathologists for distinguishing a benign nevus from a malignant melanoma. The clinical stakes are asymmetric:

- **Under-diagnosis** of melanoma delays surgical excision and adjuvant therapy, risking metastasis.
- **Over-diagnosis** leads to unnecessary procedures and toxic adjuvant therapies.

**The opportunity.** Computational pathology — combining high-quality histopathology images with deep learning — offers the prospect of faster, cheaper, and more consistent assessment. Prior deep-learning work in pathology has scored tumor grade (e.g., breast cancer), performed histologic pattern classification (e.g., lung adenocarcinoma), and predicted treatment response. However, the authors note that **prior literature on skin cancer did not address ROI detection** for melanocytic tumors and achieved only limited classification accuracy in distinguishing among tumor types.

**The goal.** Build a deep-learning method that (a) **automatically detects the ROI** (the region a pathologist would circle) and (b) **classifies the slide** accurately — leveraging the partial annotations pathologists already produce, rather than requiring complete pixel-level labels.

---

## 3. Problem definition

Two outputs are required per WSI:

- **Slide classification** $\hat{y} \in \{\text{Melanoma}, \text{Nevus}\}$.
- **ROI mask** $B$ — the set of patches predicted to lie within the diagnostically relevant region, to be compared against the pathologist-annotated region $A$.

**The supervision challenge.** During training, only **slide-level labels** and **partial ROI annotations** are available:

- Every training slide has a slide-level label (melanoma vs. nevus).
- Pathologists annotated tumor boundaries on *some* slides/regions but **not all** ROIs were annotated. Consequently, the training annotations are incomplete rather than erroneous, and cannot serve as a complete ground truth for evaluating detection on the training data.
- Crucially, because annotation is incomplete, an unannotated region may still contain tumor — so naively labeling "everything outside an annotation" as non-tumor would inject label noise.

**The key idea.** Rather than treating this as pixel-level segmentation with exhaustive labels (which do not exist), PCLA-3C reframes it as **patch classification with three classes**, where a manually curated **"Other"** class provides clean negative (background/non-ROI) examples. Slide classification and ROI detection then both fall out of the same per-patch scores. This is described in the paper as leveraging *"partial information from annotations, also called semi-supervised learning."*

---

## 4. Dataset: the UNC Melanocytic Tumor Dataset

| Property | Value |
|---|---|
| Total WSIs | **160** H&E whole slide images of primary tumors |
| Class balance | **86 melanoma** (malignant) + **74 nevi** (benign) |
| Training set (80%) | **134 WSIs** — 71 melanoma + 63 nevus *(per main text)* |
| Test set (20%) | **26 WSIs** — 15 melanoma + 11 nevus |
| Scanner | **Aperio ScanScope Console**, 20× magnification |
| Annotation tool | **Aperio ImageScope Console**; boundaries exported as **XML** with region vertex coordinates |
| Test annotations | All 26 test slides manually annotated (melanoma/nevi circled on the glass slides) by an expert dermatopathologist (PAG — Pamela A. Groben) |
| Ethics | UNC Chapel Hill IRB #22-0611; waiver of informed consent [45 CFR 46.116(d)] and HIPAA authorization [45 CFR 164.512(i)(2)(ii)] |

**Effective data size.** Although only 160 slides exist, each gigapixel WSI yields **thousands of patches**, so the effective training set is **hundreds of thousands of patches** — this is what makes CNN training feasible and improves generalization across regions within a slide.

**Annotation asymmetry between train and test — a critical design point:**

- **Training slides** were used **without complete ground-truth annotations** (only partial region information). This mirrors the real-world situation: pathologists annotate for diagnosis, not for exhaustive ML labeling.
- **Test slides** were kept **unchanged with true, complete annotations**, so evaluation of both slide classification and IoU is trustworthy.

This asymmetry is what makes the reported results meaningful: the model never saw complete detection labels, yet it is evaluated against complete ones.

---

## 5. Method: the PCLA-3C framework

The framework (Figure 2 of the paper) has two stages — **Training** and **Testing** — each composed of well-defined steps. The mapping to code files is given in [Section 6](#6-the-code-repository-walkthrough-and-paper-to-code-mapping).

### 5.1 Preprocessing: color normalization

Scans performed in different labs — or even the same lab at different times — vary in color and quality. A CNN may exploit these spurious color differences rather than genuine tissue morphology. To prevent this, all WSIs are **color-normalized** into a common space using the two established stain-normalization methods the paper cites: Ruifrok & Johnston color deconvolution [28] and Macenko et al. [29]. This improves the robustness of both training and downstream quantitative analysis.

Both methods rest on the **Beer–Lambert law**. For incident light of intensity $I_0$ transmitted through stained tissue at measured intensity $I$ (per RGB channel), the **optical density (OD)** is

$$\text{OD} = -\log_{10}\!\left(\frac{I}{I_0}\right),$$

and OD is **linearly proportional to stain concentration**: each stain has a characteristic, constant absorption (OD) signature across the three RGB channels. This linearity is what makes the stains separable.

**Ruifrok & Johnston [28] — color deconvolution (fixed stain vectors).** The specimen is modeled as a mixture of up to three stains (for H&E: hematoxylin, eosin, and a residual/background channel).

1. Each stain $s$ is assigned a normalized OD vector — its RGB color signature — originally measured from control slides stained with a single stain.
2. These vectors are stacked into a $3\times3$ **stain matrix** $M$ (one row per stain).
3. For a pixel with measured optical density $\text{OD}$, the relationship is $\text{OD} = M^{\mathsf T} c$, where $c$ is the vector of per-stain concentrations. Because $M$ is (approximately) invertible, the concentrations are recovered by **deconvolution**: $c = (M^{\mathsf T})^{-1}\,\text{OD}$, which separates the RGB image into independent per-stain density channels.
4. **Normalization** re-projects those concentrations onto a set of *reference/standard* stain vectors, so that slides processed under different conditions share a common stain appearance.

**Macenko et al. [29] — automatic, data-driven stain estimation.** Rather than using pre-measured stain vectors, this method estimates them per slide directly from the image:

1. Convert RGB → OD and discard near-transparent background pixels (those with OD below a small threshold, e.g. $\beta = 0.15$).
2. Because H&E optical-density vectors lie approximately in a **2-D plane**, take the **singular value decomposition (SVD)** of the OD pixel matrix and keep the two directions with the largest singular values — the plane that best explains the stain variation.
3. Project the pixels onto that plane, compute each pixel's **angle**, and take the **robust extreme angles** (e.g. the 1st and 99th percentiles) as the two stain vectors (hematoxylin and eosin) — estimated automatically, with no control slides.
4. Solve the linear system (non-negative least squares) for the per-pixel stain **concentrations**, and take a robust maximum (e.g. the 99th percentile) as each stain's reference concentration.
5. **Normalize** by rescaling every slide's stain concentrations to a chosen *target* (reference-slide) stain matrix and maximum concentrations, then recompose OD → RGB. This maps all slides into the reference color space, removing inter-scanner and inter-batch color variation while preserving tissue morphology.

In this study, one of these normalizations is applied to every WSI before patch extraction, so the downstream VGG16 classifier learns morphology rather than staining artifacts.

### 5.2 Patch extraction and labeling

- **Tiling.** Slides are tiled into **non-overlapping 256 × 256** patches at 20× magnification. Tissue detection first discards background/empty patches.
- **Data augmentation.** Random crop, random horizontal flip, and normalization are applied so that edge features are preserved and the classifier generalizes.
- **Labeling rule** (uses the slide-level label together with the XML annotations):
  - If the slide is a **nevus**, patches **inside** the annotated regions are labeled **Nevus**.
  - If the slide is a **melanoma**, patches **inside** the annotated regions are labeled **Melanoma**.
  - Patches **outside** annotated regions are labeled **Other**.
- **Handling incomplete annotations.** Because not every ROI was annotated, some tumor patches lie *outside* the annotations and would be wrongly labeled "Other." To avoid this contamination, the authors **manually extracted "Other" patches from curated regions** rather than blindly labeling all extra-annotation patches as "Other."

The result is a labeled patch dataset with three classes: **Melanoma, Nevus, Other**.

### 5.3 Model and training

- **Architecture.** A **VGG16** CNN [10] is used as the base. The network's final fully connected classification layer is replaced with a three-unit output head, and the network is fine-tuned by backpropagation. (VGG is a classic deep CNN family; the paper also situates it among ZefNet, ResNet, and DenseNet.)
- **Output.** For each patch the model returns **three scores**, one per class (melanoma, nevus, other).
- **Optimization (from the code):** cross-entropy loss, **Adam** optimizer, learning rate **5 × 10⁻⁴**, weight decay **1 × 10⁻⁴**, with **early stopping** (patience 20) on validation loss.

#### 5.3.1 Detailed model architecture (VGG16)

The paper states only that VGG16 is the base and that "the last layer's parameters" were changed; the per-layer specification below follows the canonical **VGG16, configuration "D"** (Simonyan & Zisserman [10]) as instantiated by `torchvision`. The concrete `instantiate_model('vgg16', True, 3)` routine is not shown in the released scripts, so the specifics are reconstructed from the paper together with standard torchvision/VGG conventions.

- **Input.** An RGB patch of `224 × 224 × 3` during training (after `RandomCrop(224, padding=4)`) and `256 × 256 × 3` at inference. Channel-wise normalization uses the **dataset-specific H&E statistics** — mean `(0.6632, 0.4123, 0.5529)`, std `(0.1618, 0.1749, 0.1478)` — **not** ImageNet statistics.
- **Feature extractor.** Thirteen `3 × 3` convolutions (stride 1, padding 1), each followed by ReLU, organized into five blocks separated by `2 × 2` max-pooling (stride 2):

| Block | Layers (each `Conv 3×3` + ReLU) | Channels | Spatial output (224 input) |
|---|---|---|---|
| 1 | Conv, Conv, **MaxPool** | 64 | 112 × 112 |
| 2 | Conv, Conv, **MaxPool** | 128 | 56 × 56 |
| 3 | Conv, Conv, Conv, **MaxPool** | 256 | 28 × 28 |
| 4 | Conv, Conv, Conv, **MaxPool** | 512 | 14 × 14 |
| 5 | Conv, Conv, Conv, **MaxPool** | 512 | 7 × 7 |

- **Adaptive pooling.** `AdaptiveAvgPool2d((7, 7))` fixes the feature map to `7 × 7 × 512` regardless of input size. This is precisely why a `256 × 256` test patch (which pools to `8 × 8 × 512` before this layer) is accepted without error despite the `224 × 224` training crop.
- **Classifier head.** `Flatten (25 088) → Linear(25088, 4096) → ReLU → Dropout(0.5) → Linear(4096, 4096) → ReLU → Dropout(0.5) → Linear(4096, 3)`. The original ImageNet head's final `Linear(4096, 1000)` is **replaced by `Linear(4096, 3)`** — the "changed last layer" of the paper. The three outputs are the raw class logits {Melanoma (0), Nevus (1), Other (2)} — no softmax is applied (see §6.2).
- **Depth.** 13 convolutional + 3 fully connected = **16 weight layers**, hence "VGG-16." With the 3-way head the network has **≈ 134 million** trainable parameters (vs. ≈ 138 M for the 1000-class original).
- **Transfer learning.** The second argument of `instantiate_model('vgg16', True, 3)` denotes **ImageNet-pretrained initialization**; the backbone is fine-tuned end-to-end on the H&E patches rather than trained from scratch.
- **Training configuration.** CrossEntropyLoss on the logits; Adam (`lr = 5e-4`, `weight_decay = 1e-4`); batch size 100; up to 20 epochs. Note that early-stopping `patience = 20` **equals** `n_epochs = 20`, so under the default configuration early stopping can never actually fire — training runs the full 20 epochs and every per-epoch checkpoint is retained for later selection (§6.3).

### 5.4 Slide classification (testing stage)

At inference, **all patches** from a WSI are fed to the trained classifier, and each patch is assigned the class with the highest predicted score (the argmax over the three class scores). Then:

- **"Other" patches are ignored.**
- The slide label is decided by **majority vote** between the remaining two classes: if the number of patches predicted **melanoma** exceeds the number predicted **nevus**, the slide is classified **melanoma**; otherwise **nevus** (so that ties in patch counts are resolved in favor of nevus, consistent with the strict inequality `class_mel > class_nev` in the code).

### 5.5 ROI ranking and detection

Once a slide's class is decided, **all patches are ranked by the predicted score of the winning class**:

- For a **melanoma** slide, rank patches by their **melanoma** score.
- For a **nevus** slide, rank patches by their **nevus** score.

The predicted ROI is then the set of top-ranked patches. How many patches to include is governed by the **annotated ratio** $\beta$.

### 5.6 The two key metrics: β and IoU

**Annotated ratio $\beta$.** For a slide $C$ with annotated region $A$:

$$\beta = \frac{A_p}{C_p}$$

where $A_p$ is the number of patches inside the annotated region $A$, and $C_p$ is the total number of patches extracted from the slide $C$. Equivalently, $\beta$ is the proportion of a slide's extracted patches (by patch count, not area) that fall within the pathologist-annotated region.

**Top-$n\beta$ selection.** With $n$ = total number of patches in the slide, the model designates the **top $n\beta$ patches** (by predicted score) as the predicted ROI $B$. Example from the paper: if $\beta = 0.2$, then 20% of the slide is annotated ROI, so the model predicts its **top 20%** highest-scoring patches as the ROI.

**Intersection over Union (IoU).** Because the framework is patch-based, IoU is computed by counting patches:

$$\text{IoU} = \frac{\underline{AB}_p\;(|A \cap B|)}{\overline{AB}_p\;(|A \cup B|)}$$

where $\underline{AB}_p$ = number of patches in the intersection $A \cap B$, and $\overline{AB}_p$ = number of patches in the union $A \cup B$. Here $A$ is the annotated (ground-truth) region and $B$ is the predicted/highlighted region.

Equivalently (and as implemented in `analysis.py`):

$$\text{IoU} = \frac{\text{num\_inregion\_highlight}}{\text{num\_inregion} + \text{num\_highlight} - \text{num\_inregion\_highlight}}$$

- `num_inregion` = patches inside the annotation ($|A|$).
- `num_highlight` = patches predicted as ROI ($|B|$).
- `num_inregion_highlight` = patches both inside the annotation and predicted ($|A \cap B|$).

### 5.7 Visualization (three map types)

The predicted scores drive three complementary visualizations (Figure 3):

1. **Overlay map.** Highlights the top-ranked patches while masking the rest of the slide with a **transparent blue** tint. When ground-truth XML is available the threshold is set from $\beta$ (via `get_percent`); with `--no_xml` it falls back to `--percent` (default 0.2). The highlighted fraction is therefore *approximately* $\beta$ — not exactly, owing to the per-class-threshold and argmax gating described in §6.2 — so the highlighted region closely approximates the predicted ROI. This is the most direct visualization of "what the model considers ROI."
2. **Boundary map.** Draws the **boundary of the largest ROI cluster**. Highlighted patches are grouped with the **OPTICS** density-based clustering algorithm [30]; the largest cluster's outline is then rendered (via an alpha-shape hull). This shows how precisely the model captures tumor margins.
3. **Heatmap.** A continuous **color gradient** (matplotlib `coolwarm`) over predicted scores: **red = high** predicted score, **blue = low**. This conveys the model's confidence across the whole slide.

Figure 1 in the paper illustrates the end goal: the pathologist's ROI is annotated with black dots on the left; PCLA-3C reproduces it as a green boundary on the right.

---

## 6. The code: repository walkthrough and paper-to-code mapping

### 6.1 Repository layout

```
understand-deeplearningproject/
├── paper/cancers-16-02616.pdf     # the paper
└── codes/                         # all source; run commands from here
    ├── extract_patches_3class.py  # Stage 1: patch extraction + labeling
    ├── method_pcla_3class.py      # Stage 2: train VGG16 3-class classifier
    ├── score_pcla_3class.py       # Stage 3: score all patches, slide classification
    ├── visual.py                  # Stage 4: overlay / boundary / heatmap
    ├── analysis.py                # Stage 5: IoU computation
    ├── automated_pipeline.py      # config-driven orchestrator (subprocess wrapper)
    ├── batch_process.py           # run the pipeline over multiple configs
    ├── config.yaml / config_example.yaml
    ├── setup.py / requirements.txt
    └── utils/                     # CLAM-derived helpers (not referenced by the active pipeline)
```

Each stage is an independent argparse script; stages communicate only through files on disk (patch PNGs, HDF5 score files, text logs, and CSVs).

### 6.2 The five-stage pipeline

#### Stage 1 — `extract_patches_3class.py` (patch extraction and labeling)

Implements §5.2. Key functions:

- `makemask(w, h, xml_path)` — parses the Aperio XML, iterates over each `<Region>`'s `<Vertex>` coordinates, and rasterizes the polygons into a binary annotation mask via `cv2.fillPoly`. This is the code realization of "the annotated region $A$." Note that `makemask()` exists in **both** `extract_patches_3class.py` and `visual.py` with different internal conventions: the extraction copy allocates `np.zeros((h, w))` and is called `makemask(w, h, ...)`, whereas the visual copy allocates `np.zeros((w, h))` but is called `makemask(image.height, image.width, ...)`. Both ultimately yield a (height × width) mask, but the parameter order and names differ between files.
- `checkinout(patch_mask, annotation_ratio=1.00)` — decides whether a patch is *in-region*: a patch is kept if the fraction of annotated (nonzero) pixels within it meets the ratio threshold. During extraction the ratio is effectively hardcoded to `1.00` (fully inside): `compute_w_loader` calls `checkinout(img_mask)` without passing `annotation_ratio`, so the `config.yaml` value `annotation_ratio=0.5` affects only `visual.py`, not extraction.
- `compute_w_loader(...)` — iterates patch coordinates from the slide's HDF5 bag, checks each against the mask, and saves in-region patches as PNG files named `{output}_{x}_{y}.png` (the patch's coordinates are encoded in the filename).
- `make_dirs(...)` — creates the `feat_dir/{train,val,test}/{Melanoma,Nevi,Other}/` directory tree that the training stage consumes.

The two annotation sources map directly to the labeling rule: **`--xml_annotation_new`** determines the Melanoma/Nevus label for in-region patches, while **`--xml_annotation_other`** routes patches into the **Other** class (the manually curated background of §5.2).

#### Stage 2 — `method_pcla_3class.py` (train the PCLA-3C classifier)

Implements §5.3. Highlights:

- Loads the patch tree with `torchvision.datasets.ImageFolder` over `{train,val,test}`. Because `ImageFolder` assigns class indices alphabetically, the mapping is **`{Melanoma: 0, Nevi: 1, Other: 2}`** — consistent with the decision logic in later stages.
- **Training transform:** `RandomCrop(224, padding=4)` → `RandomHorizontalFlip` → `ToTensor` → `Normalize(mean=(0.6632, 0.4123, 0.5529), std=(0.1618, 0.1749, 0.1478))`. (Note the 224-pixel crop for training vs. full 256-pixel patches at test time; VGG16's adaptive pooling accepts both.)
- **Model:** `instantiate_model('vgg16', True, num_class=3)` — the VGG16 base with a 3-way head.
- **Training loop:** cross-entropy loss, Adam (`lr=5e-4`, `weight_decay=1e-4`), validation each epoch, **early stopping** with `patience=20`. Checkpoints are saved as `{exp_name}_epoch{N}_loss{L}_acc{A}.pt` — the accuracy token in the filename is later parsed to auto-select the best model.

#### Stage 3 — `score_pcla_3class.py` (score patches, slide classification)

Implements §5.4–§5.5. Highlights:

- For each slide, loads its patches from `{slide_id}.h5` and runs the trained model, accumulating `class_mel` (count of argmax==0) and `class_nev` (count of argmax==1).
- **Slide label = majority vote:** `label = 0` (melanoma) if `class_mel > class_nev`, else `1` (nevus) — the direct code form of §5.4. `check_correct` encodes the Melanoma↔0 / Nevi↔1 convention to tally slide accuracy.
- **Per-slide output HDF5** `results_dir/{exp_name}/score/{slide_id}.h5` with three datasets: `scores` (raw 3-class logits per patch), `coord` (patch coordinates), and `pred` (the slide-level predicted label). These files are the sole input to visualization and analysis. **No softmax is applied anywhere in the pipeline:** `ScoreToColor`, `get_thrd`, and `ScoreToRank` all threshold or rank these raw per-class logits (the `visual.py` variables named `probs`/`prob0`/`prob1` are logits, not probabilities).
- Writes `classification_{exp_name}.csv`, whose `classification_result` column is recomputed independently from the counts as `Melanoma` if `mel_con > nev_con` else `nevi` (note the lowercase `nevi`, inconsistent with the `Melanoma`/`Nevi` folder labels; ties fall to `nevi`). The slide-accuracy tally instead uses `check_correct`, comparing the argmax-derived label (0 = Melanoma, 1 = Nevi) against the ground-truth `label_name`. Overall slide-classification accuracy is printed.
- The normalization mean/std are passed in as `--mean1..3` / `--std1..3` flags (supplied by `config.yaml`), so they must be **kept in sync** with the hardcoded training values.

#### Stage 4 — `visual.py` (overlay, boundary, heatmap)

Implements §5.6–§5.7. Highlights:

- `get_percent(...)` computes the **annotated ratio $\beta$** from the ground-truth mask (fraction of patches in-region) — this is the code form of $\beta = A_p/C_p$.
- `get_thrd(probs, percent)` computes a per-class percentile threshold at the `100 − percent·100` percentile — separately, `thrd0` on the melanoma logit and `thrd1` on the nevus logit — thereby *approximately* selecting the **top $n\beta$** patches.
- `ScoreToColor(...)` (overlay/boundary) highlights a patch only when its winning-class logit exceeds that per-class threshold **and** its argmax agrees with the slide label; because of the extra argmax gate, the highlighted set is *at most* the top $n\beta$ patches, not exactly $n\beta$. It returns a discrete decision (0 = highlight-melanoma, 1 = highlight-nevus, −1 = not highlighted), whereas `ScoreToRank(...)` (heatmap) returns a continuous min–max-normalized percentile rank mapped to a `coolwarm` color.
- **Overlay:** masks the whole slide with a transparent blue (`plt.cm.coolwarm(0.05, alpha=0.5)`) and reveals only highlighted patches; simultaneously logs per-slide `num_inregion_highlight`, `num_inregion`, and `num_highlight` to `{exp_name}.txt` — the exact counts IoU needs.
- **Boundary:** clusters highlighted patches with `sklearn.cluster.OPTICS`, keeps the largest cluster, fits an `alphashape` hull, and draws the contour with `cv2.drawContours` (green line, as in Figure 1).
- **Heatmap:** blends the `coolwarm`-colored score map with the original slide (`original_image * 0.5 + heatmap_image * 0.5`).
- Outputs are pyramidal `.tiff` files written with **pyvips**, enabling WSI-scale rendering.

#### Stage 5 — `analysis.py` (IoU)

Implements the IoU metric of §5.6. It parses the `{exp_name}.txt` log written by `visual.py` and computes, per slide,

```
union = num_inregion + num_highlight − num_inregion_highlight
IoU   = num_inregion_highlight / union
```

skipping slides with zero in-region patches, and writes `summary_iou_final.csv`. (Note: `analysis.py` hardcodes reading a label file named `melanomal.csv` from the CSV directory and a single experiment name `pcla_3class`.)

### 6.3 Orchestration layer

- **`automated_pipeline.py`** — the `ROIDetectionPipeline` class reads `config.yaml` and runs all five stages via `subprocess`, mapping config keys to CLI flags. It **skips** stages whose outputs already exist (`skip_existing`) and **auto-selects a checkpoint** by parsing the `_acc` token from checkpoint filenames. Because that token records per-epoch *training* accuracy, this selects the highest-training-accuracy checkpoint (ties broken arbitrarily by `max()`), which is not necessarily the best-validation model that early stopping would favor. `--validate-only` checks the config without running.
- **`batch_process.py`** — runs the pipeline once per `config_*.yaml` in a directory and prints a success/failure summary.
- **`config.yaml`** — single source of truth for paths and hyperparameters (architecture, `batch_size=100`, `n_epochs=20`, `learning_rate=5e-4`, `patch_size=256`, `annotation_ratio=0.5`, `percent=0.2`, and the normalization mean/std). `config_example.yaml` is the template; the validator rejects any path still set to a `PATH_TO_*` placeholder.

### 6.4 Paper → code mapping (quick reference)

| Paper concept | Code location |
|---|---|
| Color normalization (§5.1) | Not included in repo (external "Step 0" preprocessing) |
| Tiling into 256×256 patches | HDF5 patch bags consumed by `extract_patches_3class.py` (`h5_to_patch_new`) |
| Annotation mask $A$ from XML | `makemask()` + `cv2.fillPoly` in `extract_patches_3class.py` / `visual.py` |
| In-region test (annotation ratio) | `checkinout()` |
| Patch labels {Melanoma, Nevi, Other} | `make_dirs()` tree + `--xml_annotation_new` / `--xml_annotation_other` |
| VGG16 3-class classifier (§5.3) | `method_pcla_3class.py` (`instantiate_model('vgg16', ...)`) |
| Data augmentation | `transforms.RandomCrop/RandomHorizontalFlip/Normalize` |
| Slide classification by majority vote (§5.4) | `class_mel > class_nev` in `score_pcla_3class.py` |
| Patch ranking by winning-class score (§5.5) | `ScoreToColor` / `ScoreToRank` in `visual.py` |
| Annotated ratio $\beta = A_p/C_p$ | `get_percent()` in `visual.py` |
| Top-$n\beta$ selection | `get_thrd()` percentile threshold |
| IoU (§5.6) | `analysis.py` (`num_inregion_highlight / union`) |
| Overlay / Boundary / Heatmap (§5.7) | `visual.py` (`--`/`--boundary`/`--heatmap`) |
| OPTICS clustering for boundary | `sklearn.cluster.OPTICS` in `visual.py` |

---

## 7. Results

### 7.1 Method comparison — full training set (Table 1)

Both methods trained on the 134-WSI training set; evaluated on the 26-WSI test set.

| Evaluation metric | PCLA-3C | CLAM |
|---|---|---|
| Patch classification accuracy | **0.892** | — (CLAM does not classify patches) |
| Slide classification accuracy | **0.923** | 0.692 |
| IoU | **0.382** | 0.112 |

PCLA-3C raises slide classification accuracy from 69.2% to 92.3% and IoU from 0.112 to 0.382, a more than threefold increase in region-of-interest overlap relative to CLAM.

**Comparison protocol.** Both methods were trained on the identical 134-WSI split and evaluated on the identical 26-WSI test set. CLAM is *label-free* (slide labels only), whereas PCLA-3C additionally consumes the pathologists' *partial* ROI annotations; the comparison is therefore best read as quantifying the value that those partial annotations add, rather than as an architecture-only contest (see Reviewer 1 in [`reply_to_reviewer.md`](reply_to_reviewer.md)).

### 7.2 Confusion matrix (Table 2)

The paper designates **nevi as the positive class and melanoma as the negative class**, reflecting the clinical priority of correctly identifying benign cases to avoid unnecessary intervention.

| | True: Nevi | True: Melanoma |
|---|---|---|
| **Predicted: Nevi** | 20 | 0 |
| **Predicted: Melanoma** | 2 | 9 |

Reported summary metrics: **accuracy 93.5%**, **sensitivity 81.8%**, **specificity 100%**. The reported specificity of 100% indicates that, on this test set, no malignant (melanoma) case was misclassified as benign (nevus) — a property of particular clinical importance given the asymmetric cost of under-diagnosis. (See [Section 11](#11-reconciling-the-reported-numbers) for a note reconciling these counts with the 26-slide test set and the 92.3% figure.)

### 7.3 Robustness across training-set subsampling (Table 3)

To probe robustness, the model was retrained on random **subsets** of the original training set (20%, 40%, 60%, 80%), while the **test set was held fixed**. CLAM has no patch-classification row because it does not classify patches. Values are **mean [95% CI]**.

| Split | Method | Patch acc. | Slide acc. | IoU |
|---|---|---|---|---|
| **20%** | PCLA-3C | 0.6397 [0.5193, 0.7601] | 0.7406 [0.6627, 0.8185] | 0.3026 [0.2394, 0.3327] |
| | CLAM | — | 0.6710 [0.6386, 0.7033] | 0.0427 [0.0342, 0.0512] |
| **40%** | PCLA-3C | 0.7887 [0.7536, 0.8238] | 0.8430 [0.8043, 0.8817] | 0.3402 [0.3057, 0.3784] |
| | CLAM | — | 0.6976 [0.6619, 0.7333] | 0.0524 [0.0297, 0.0751] |
| **60%** | PCLA-3C | 0.8191 [0.7766, 0.8616] | 0.8721 [0.8458, 0.8985] | 0.3652 [0.3369, 0.3934] |
| | CLAM | — | 0.7097 [0.6830, 0.7364] | 0.0621 [0.0428, 0.0814] |
| **80%** | PCLA-3C | 0.8210 [0.7949, 0.8471] | 0.8885 [0.8607, 0.9163] | 0.3710 [0.3335, 0.4084] |
| | CLAM | — | 0.7258 [0.7117, 0.7399] | 0.1103 [0.0529, 0.1677] |

**Interpretation.** PCLA-3C exhibits a gradual, monotonic decline in performance as the training set is reduced: even at a **20%** training subset its IoU (0.30) already **exceeds CLAM's best IoU** (0.11, at the 80% subset), and its slide classification accuracy increases monotonically with training-set size (0.74 → 0.89). Accuracy and IoU both increase with more data, consistent with the value of the annotations. The paper additionally reports, at the 80% subset (107 WSIs), a **patch-level accuracy of 0.7866 [0.761, 0.813]** and **slide-level accuracy of 0.885 [0.857, 0.914]**.

**Takeaway.** The consistent margin over CLAM across all subsampling levels indicates that the combination of **partial annotations** with a **patch classifier that includes an explicit background class** is the principal source of the observed improvement, and that PCLA-3C is comparatively data-efficient and robust.

---

## 8. Misclassification analysis

PCLA-3C misclassified **only two slides** in the test set. Both were **true nevi predicted as melanoma** (consistent with the specificity being higher for the malignant class):

- **Figure 4 case** — *not a typical nevus*: it exhibits features of a **pigmented spindle cell nevus**, one of the recognized diagnostic challenges of melanocytic tumors. Even expert pathologists find such cases hard.
- **Figure 5 case** — a **routine nevus**, but the model was misled by **color**. Melanoma ROIs are generally **dark**, whereas nevus regions are typically **light**; this slide had **dark areas outside the annotated ROI** that pushed the model toward a melanoma prediction and an incorrect ROI.

This analysis transparently identifies a genuine limitation: the model can be confounded by staining and darkness cues, and by rare morphological variants — precisely the cases that are also difficult for human readers.

---

## 9. Discussion, related work, and limitations

### 9.1 Positioning against related work

- **CLAM** [31] (clustering-constrained attention multiple-instance learning; Lu et al., *Nat. Biomed. Eng.* 2021) is the primary baseline. CLAM performs slide classification and ROI detection **without pixel/patch labels**, and works well on renal-cell and lung cancers — but its **ROI detection is unsatisfactory on the melanocytic dataset**, which motivates PCLA-3C.
- **CNN-based classifiers** for breast/skin cancer [13–16, 37, 38] mostly target the classification task; PCLA-3C additionally does ROI detection and reports higher accuracy on melanocytic tumors.
- **Transformer methods** for medical image enhancement — Feng et al.'s **T2Net** (joint MRI reconstruction/super-resolution) [34] and Wang et al.'s **TED-Net** (low-dose CT denoising) [35] — and transfer-learning skin-cancer classifiers (Khalid et al. [36]) are cited as complementary directions.
- **Weakly-supervised segmentation** — Lerousseau et al.'s **WMIL** [39] generates patch pseudo-labels from slide labels during training and is highlighted as a promising avenue for future WSI study.

### 9.2 Strengths (in detail)

1. **Detection without complete labels — the central novelty.** PCLA-3C learns ROI localization from *partial* annotations plus a manually curated **"Other" background class**, rather than from exhaustive pixel/patch masks (which do not exist for this cohort). This directly addresses the defining obstacle in melanocytic pathology: exhaustive annotation is infeasible, and the ground truth is itself uncertain given the documented **25–26% inter-pathologist discordance** [40].
2. **One model, two tasks.** Slide classification (majority vote over patch predictions) and ROI localization (ranking patches by the winning-class score) both emerge from the *same* patch classifier — no separate detector or segmentation head is trained, which keeps the method simple and internally consistent.
3. **Large margin over the state-of-the-art baseline.** On the fixed 26-WSI test set PCLA-3C improves slide accuracy from 0.692 → **0.923** and IoU from 0.112 → **0.382** (a more than threefold gain in ROI overlap) relative to CLAM.
4. **Data efficiency and robustness.** In the subsampling study (Table 3), PCLA-3C trained on only **20%** of the data already surpasses CLAM's *best* IoU (obtained at 80%), and its accuracy/IoU improve monotonically with more data, with comparatively tight 95% confidence intervals.
5. **Clinically aligned evaluation.** By treating **nevi as the positive class**, the reported **100% specificity** means no malignant case in the test set was labeled benign — the error direction that matters most clinically.
6. **Interpretability for pathologists.** The three visualization modes (overlay, boundary, heatmap) provide spatially grounded, human-readable explanations of *where* the model sees tumor, supporting review and trust rather than a black-box slide label.
7. **Accessibility.** A VGG16 patch classifier runs on a single RTX 3090 and is compatible with open tooling such as FastPathology — a low barrier to reproduction and deployment.

### 9.3 Weaknesses (in detail)

1. **Not autonomous.** Slide-classification accuracy is **92.3%**, so the model cannot replace a pathologist; two true nevi were classified as melanoma. The authors explicitly frame it as decision support.
2. **Residual color/stain confounding.** Even after color normalization, dark tissue *outside* the annotated ROI can bias the model toward "melanoma" (Figure 5), because melanoma regions are typically darker than nevi — the model may key on staining darkness rather than morphology.
3. **Small, single-institution cohort.** The study uses **160 WSIs** from one institution (UNC), with only **26 test slides** and no external, multi-scanner validation in the published work; generalization to other labs, scanners, and populations is unproven.
4. **Annotation and curation dependence.** The gains rely on the partial annotations, and the **"Other"-class curation is manual** and not fully specified, which limits exact reproducibility; performance degrades as training data shrinks.
5. **Dated backbone with no domain pretraining.** VGG16 (2014) predates modern vision Transformers and pathology **foundation models**; it is initialized from ImageNet rather than self-supervised pathology pretraining, bounding its representational power (see §9.5).
6. **No slide-level context.** Patches are classified **independently**; the method models no inter-patch or global-slide context (unlike attention-MIL or dedicated slide encoders), which can fragment the predicted ROI.
7. **Evaluation coupling and implementation subtleties.** IoU is computed at the ground-truth annotated ratio $\beta$ (which sizes the prediction), and ranking operates on raw logits with a per-class-threshold-plus-argmax gate; these are defensible but should be kept in mind when comparing to other protocols.

### 9.4 Conclusions and future work

The authors conclude that the framework produces an **accurate and robust** approach to detect skin tumors and predict tumor type, which they suggest could reduce clinician workload and improve diagnostic efficiency. The authors anticipate that the approach may **generalize to other cancers** (not just skin) and to vision-based treatment-outcome prediction. **Future work stated in the paper:** extract richer information from high-quality WSIs and incorporate **extra modalities such as gene expression and clinical data** to further improve detection and prediction.

### 9.5 Beyond the paper: post-2024 medical-imaging models and how they could extend PCLA-3C

The most impactful development since this paper is the arrival of **pathology foundation models** and **promptable segmentation models**. The VGG16 patch classifier is the natural component to modernize; the models below (all 2024) are the leading candidates, with their architectures summarized.

- **UNI** (Chen et al., *Nature Medicine* 2024) — a general-purpose **tile encoder** for pathology. Architecture: a **Vision Transformer, ViT-Large/16** (~307 M parameters) pretrained with the **DINOv2** self-supervised objective on *Mass-100K* (~100 million H&E tiles from ~100,000 diagnostic WSIs across 20 tissue types). It emits a 1024-dimensional embedding per 256×256 tile. **How it would extend PCLA-3C:** replace the VGG16 backbone with a frozen or lightly fine-tuned UNI encoder and train only a small 3-class head — typically far higher accuracy and data efficiency than an ImageNet-pretrained CNN.
- **Virchow / Virchow2** (Paige, 2024) — a large pathology tile encoder. Architecture: a **ViT-Huge/14** (~632 M parameters) trained with **DINOv2** on ~1.5 million WSIs (billions of tiles); Virchow2 adds pathology-specific augmentations and multi-magnification training. Tile embeddings concatenate the class token with pooled patch tokens. **Extension:** a stronger drop-in replacement for the patch encoder where compute permits.
- **Prov-GigaPath** (Xu et al., *Nature* 2024) — a **whole-slide** foundation model that adds the slide-level context PCLA-3C lacks. Architecture: **(i)** a ViT **tile encoder** (DINOv2) over 256×256 tiles, followed by **(ii)** a **slide encoder built on LongNet**, a Transformer with *dilated attention* that scales to tens of thousands of tiles per gigapixel slide. Pretrained on *Prov-Path* (~1.3 billion tiles from ~171,000 WSIs). **Extension:** a LongNet-style aggregator over patch embeddings would let the model reason about global slide layout instead of voting over independent patches.
- **MedSAM** (Ma et al., *Nature Communications* 2024) — the medical adaptation of the **Segment Anything Model**, directly relevant to the ROI-*detection* goal. Architecture: a **ViT-Base image encoder** producing a dense image embedding, a **prompt encoder** (here, bounding-box prompts), and a **lightweight two-layer transformer mask decoder** with transposed-convolution upsampling to a full-resolution mask; trained on ~1.5 million image–mask pairs spanning ~10 modalities. **Extension:** PCLA-3C's coarse patch-level ROI could seed box prompts for MedSAM to produce a **pixel-precise** ROI boundary, improving IoU beyond the patch grid.
- **CONCH** (Lu et al., *Nature Medicine* 2024) — a **vision–language** pathology model. Architecture: a **CoCa**-style dual encoder — a ViT image encoder and a text encoder trained jointly with contrastive and image-captioning objectives on ~1.17 million pathology image–caption pairs. **Extension:** enables **zero-shot / text-promptable** classification ("melanoma vs. nevus") and report-style outputs, reducing dependence on curated patch labels.

**Synthesis for future work.** A modern successor to PCLA-3C would (a) swap VGG16 for a **DINOv2-pretrained pathology tile encoder** (UNI or Virchow2), (b) add a **slide-level LongNet aggregator** (Prov-GigaPath) for global context, and (c) refine the coarse patch ROI into a precise boundary with a **promptable segmenter** (MedSAM) — while retaining this paper's key insight that *partial annotations plus an explicit background class* are enough to supervise ROI detection.

---

## 10. Reproducibility and computational environment

| Aspect | Detail |
|---|---|
| Language | Python |
| Image I/O | **OpenSlide** (reading `.svs` WSIs); **pyvips/libvips** (writing pyramidal `.tiff`) |
| DL framework | **PyTorch ≥ 1.7.1**, torchvision |
| Hardware | UNC **Longleaf Cluster** (Linux, tested on Ubuntu 18.04); **NVIDIA GeForce RTX 3090** on local workstations |
| CUDA | Tested on **CUDA 11.3** |
| Complementary tool | **FastPathology** [32] acknowledged as a user-friendly platform compatible alongside the Python framework |
| Data availability | Available from the corresponding author on reasonable request; a robust methylation classifier reference dataset is cited [41] |

**Typical run order** (from `codes/`, once the `models/` and `datasets/` helper packages are on the `PYTHONPATH` and a color-normalized dataset is available):

1. `python extract_patches_3class.py --data_dir ... --csv_path ... --xml_annotation_new ... --xml_annotation_other ... --feat_dir ...`
2. `python method_pcla_3class.py --exp_name pcla_3class --data_folder <feat_dir> --batch_size 100 --n_epochs 20`
3. `python score_pcla_3class.py --exp_name pcla_3class --model_load <ckpt> --csv_path ... --patch_path ... --results_dir ... --classification_save_dir ...`
4. `python visual.py --exp_name pcla_3class --csv_path ... --wsi_dir ... --results_dir ... --xml_dir ... [--heatmap|--boundary] --percent 0.2`
5. `python analysis.py --results_dir <results_dir>/pcla_3class --csv_dir <dir with melanomal.csv>`

Or run everything via the orchestrator: `python automated_pipeline.py --config config.yaml` (use `--validate-only` first to check paths).

**Metadata CSV contract:** columns `slide_id`, `data_split` (train/val/test), and `label_name` (Melanoma/Nevi/Other). WSIs are `.svs`; per-slide patch bags are `.h5` keyed by `slide_id`.

---

## 11. Reconciling the reported numbers

For a fully faithful reading, a few figures in the paper do not perfectly reconcile with one another. These are noted here for the careful reader; none change the paper's conclusions.

- **160 vs. 165 WSIs.** The paper consistently states **160 WSIs** (86 melanoma + 74 nevi). The repository README's overview text mentions **165** — treat **160** as authoritative (it is used throughout the paper and its tables).
- **Training-set composition.** The main text gives the training split as **71 melanoma + 63 nevus = 134**, and the test split as **15 + 11 = 26** (these sum correctly and are self-consistent with the 86/74 totals). The Figure 2a *panel* labels the training set as "Melanoma 67, Nevus 57" (which sums to 124, not 134); the **main-text numbers (71/63)** are the consistent ones.
- **Confusion-matrix totals vs. 26-slide test set.** Table 2's cells (20 / 0 / 2 / 9) sum to **31** and yield **93.5%** accuracy, whereas the headline slide-classification accuracy is **92.3%** (i.e., **24 of 26** correct, with exactly **2 nevi misclassified as melanoma** per §3.3 of the paper). The **2 false-melanoma** count is consistent across the confusion matrix and the misclassification discussion; the absolute cell counts and the 93.5% figure do not fully reconcile with a 26-slide test set. When in doubt, the **92.3% / 24-of-26 / 2-misclassified** framing is the internally consistent one. (The companion [`reply_to_reviewer.md`](reply_to_reviewer.md), Reviewer 2, presents a corrected confusion matrix that sums to 26 and matches 92.3%.)

---

## 12. Glossary and abbreviations

| Term | Meaning |
|---|---|
| **H&E** | Hematoxylin and eosin — the standard histology stain |
| **WSI** | Whole slide image — a gigapixel digitized microscope slide |
| **ROI** | Region of interest — the diagnostically relevant tumor region |
| **CNN** | Convolutional neural network |
| **Patch** | A small fixed-size tile (256×256 px here) cut from a WSI |
| **Nevus (pl. nevi)** | Benign melanocytic tumor (mole) |
| **Melanoma** | Malignant melanocytic tumor (skin cancer) |
| **PCLA-3C** | The proposed 3-class patch classifier |
| **CLAM** | Clustering-constrained attention multiple-instance learning (baseline) |
| **IoU** | Intersection over Union — overlap metric for detection |
| **β (annotated ratio)** | Fraction of a slide's patches inside the annotation |
| **OPTICS** | Ordering Points To Identify the Clustering Structure — density-based clustering |
| **MIL / WMIL** | (Weakly-supervised) multiple-instance learning |

---

## 13. Key references

Selected references cited in this document (numbering follows the paper):

- **[10]** Simonyan, K.; Zisserman, A. *Very Deep Convolutional Networks for Large-Scale Image Recognition* (VGG). ICLR 2015.
- **[28]** Ruifrok, A.C.; Johnston, D.A. *Quantification of Histochemical Staining by Color Deconvolution.* Anal. Quant. Cytol. Histol. 2001.
- **[29]** Macenko, M. et al. *A Method for Normalizing Histology Slides for Quantitative Analysis.* IEEE ISBI 2009.
- **[30]** Ankerst, M.; Breunig, M.M.; Kriegel, H.P.; Sander, J. *OPTICS: Ordering Points to Identify the Clustering Structure.* SIGMOD 1999.
- **[31]** Lu, M.Y.; Williamson, D.F.; Chen, T.Y.; Chen, R.J.; Barbieri, M.; Mahmood, F. *Data-efficient and weakly supervised computational pathology on whole-slide images* (**CLAM**). Nat. Biomed. Eng. 2021.
- **[32]** Pedersen, A. et al. *FastPathology: An Open-Source Platform for Deep Learning-Based Research and Decision Support in Digital Pathology.* IEEE Access 2021.
- **[39]** Lerousseau, M. et al. *Weakly Supervised Multiple Instance Learning Histopathological Tumor Segmentation* (**WMIL**). MICCAI 2020.
- **[40]** Hekler, A. et al. *Pathologist-level classification of histopathological melanoma images with deep neural networks.* Eur. J. Cancer 2019.

Full reference list (41 entries) is in the paper, pages 11–12.

---

## Appendix A: Peer-review clarifications

A companion document, [`reply_to_reviewer.md`](reply_to_reviewer.md), gives point-by-point responses to three *Cancers* reviewers. The substantive points relevant to this explainer are summarized below:

- **Reviewer 1 (ML methodology).** Requested a full architecture specification (addressed in [§5.3.1](#531-detailed-model-architecture-vgg16)), clarification of the CLAM comparison protocol ([§7.1](#71-method-comparison--full-training-set-table-1)), confirmation that ranking uses raw logits ([§6.2](#62-the-five-stage-pipeline)), and documentation of the "Other"-class curation ([§5.2](#52-patch-extraction-and-labeling)).
- **Reviewer 2 (clinical).** Raised single-institution / small-cohort external validity, the Table 2 confusion-matrix inconsistency ([§11](#11-reconciling-the-reported-numbers)), color/stain confounding ([§8](#8-misclassification-analysis)), and the intended decision-support role.
- **Reviewer 3 (statistics).** Asked for the confidence-interval resampling protocol ([§7.3](#73-robustness-across-training-set-subsampling-table-3)), correction of the Figure 2a training-split counts ([§11](#11-reconciling-the-reported-numbers)), justification of the β-sized top-$n\beta$ IoU protocol ([§5.6](#56-the-two-key-metrics-β-and-iou)), and broader baselines ([§9.1](#91-positioning-against-related-work)).
