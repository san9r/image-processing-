# RAG for image captioning: preprocessing (phase 1)

Shared preprocessing for a comparative study of RAG techniques on MS COCO.

## Setup
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

Download the COCO 2017 annotations (annotations_trainval2017.zip) from
cocodataset.org and set `ANN_DIR` at the top of the scripts.

## Pipeline (run in this order)
1. `explore.py`            quick look at the annotation files
2. `split.py`              knowledge-base split (10,000 train2017 images, seed 42)
3. `validation.py`         validation (100) and test (100, stratified by object count)
4. `images.py`             downloads the images into images/kb, images/val, images/test
5. `stats.py`              dataset statistics and figures
6. `preprocess.py`         raw + clean captions (processed/)
7. `embeddings.py`         CLIP ViT-B/32 image and caption embeddings (embeddings/)
8. `build-index.py`        FAISS flat inner-product indexes (indexes/)
9. `extract_metadata.py`   object metadata from COCO annotations (metadata/)
10. `audit.py`             quality control and leakage audit (reports/qc_report.json)

## Splits
| Split | Source | Size |
|---|---|---|
| Knowledge base | COCO train2017 | 10,000 images, 5 captions each |
| Validation | train2017, not in the knowledge base | 100 |
| Test | COCO val2017, stratified by object count | 100 |

Evaluation captions are never in the knowledge base (checked in `audit.py`).

## Results of phase 1
- Embeddings: CLIP ViT-B/32, 512-d, L2-normalised
- Index: IndexFlatIP (exact), cosine similarity
- Image-to-caption sanity check: Recall@1 = 0.443, Recall@5 = 0.650
- Quality control: all checks passed