# Image Captioning Using KNN

## Overview
This project implements an algorithm for image captioning using K-Nearest Neighbors (KNN). The approach is inspired by the paper [A Distributed Representation Based Query Expansion Approach for Image Captioning](https://aclanthology.org/P15-2018.pdf). The task involves leveraging image and caption embeddings to predict captions for unseen images by finding their nearest neighbors in the embedding space.

### Key Features
- Utilizes the MS COCO 2014 validation dataset for images and captions.
- Employs [Faiss](https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-library-for-efficient-similarity-search/) for efficient nearest neighbor search.
- Extracts image and text embeddings using the [CLIP](https://openai.com/research/clip) model.
- Measures performance using BLEU scores.
- Visualizes images, ground truth captions, and predicted captions for qualitative analysis.

---

## Dataset
### MS COCO 2014 Validation Set
- **Images**: A collection of real-world images from diverse categories.
- **Captions**: Five human-annotated captions per image.
- **Annotations**: Captions and other metadata are stored in JSON format.

---

## Algorithm
The image captioning process is implemented as follows:

1. **Input**:
    - Precomputed image and caption embeddings.
    - K value for nearest neighbor search.

2. **Steps**:
    - For each image embedding, find the k-nearest image embeddings using Faiss.
    - Compute a query vector as the mean of the embeddings of captions from the k-nearest neighbors.
    - Identify the caption closest to the query vector in the embedding space.

3. **Output**:
    - Predicted caption for each image.

---

## Implementation Details
### Libraries Used
- `torch` and `torchvision` for image preprocessing and handling.
- `faiss` for efficient nearest neighbor search.
- `nltk` for BLEU score computation.
- `matplotlib` for visualization.

### Preprocessing
- Images are resized to 224x224 and converted to tensors.
- Captions are tokenized for BLEU score evaluation.

### Faiss Index
- **IndexFlatL2**: Used for computing nearest neighbors based on L2 distance.
- **GPU Acceleration**: Faiss leverages GPU when available to speed up computations.

### BLEU Score
- Evaluates the similarity between predicted captions and ground truth captions.
- Uses tokenized captions for more accurate comparisons.

---

## Usage
### Dependencies
Install the required libraries:
```bash
pip install torch torchvision faiss-cpu nltk matplotlib
```

### Steps to Run
1. **Download the Dataset**:
   ```bash
   wget http://images.cocodataset.org/zips/val2014.zip
   unzip val2014.zip
   wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
   unzip annotations_trainval2014.zip
   ```

2. **Run the Script**:
   - Ensure precomputed image and caption embeddings are available.
   - Execute the notebook or Python script.

3. **Visualize Results**:
   - The script displays images, their ground truth captions, and the predicted captions.

---

## Tasks
### 1. Experiment with Different k Values
- Test the algorithm with varying k values (e.g., 1, 3, 5).
- Record BLEU scores and analyze performance.

### 2. Optimize Faiss Index
- Experiment with different Faiss index factories to speed up computation.
- Document the trade-off between speed and accuracy.

### 3. Qualitative Analysis
- Visualize five images with:
  - Their ground truth captions.
  - Predicted captions.

---

## Results
### Quantitative Metrics
- BLEU scores for various k values.

### Qualitative Observations
- Visual inspection of predictions vs. ground truth captions.

---

## References
1. [A Distributed Representation Based Query Expansion Approach for Image Captioning](https://aclanthology.org/P15-2018.pdf)
2. [Faiss: A Library for Efficient Similarity Search](https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-library-for-efficient-similarity-search/)
3. [CLIP: Connecting Text and Images](https://openai.com/research/clip)
4. [MS COCO Dataset](https://cocodataset.org/#home)

---

## Future Work
- Extend the approach to other datasets or tasks.
- Explore the impact of weighted averages for query vector computation.
- Incorporate advanced metrics like CIDEr or METEOR for caption evaluation.

