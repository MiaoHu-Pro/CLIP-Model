# Vision Transformer and CLIP implementation

This directory is an educational implementation of Vision Transformer (ViT)
and CLIP (**Contrastive Language-Image Pre-training**). It progresses from a
single image encoder to a complete dual-encoder model that learns a shared
embedding space for images and text.

## Model overview

ViT divides an image into non-overlapping patches, projects each patch into a
token embedding, adds position information, and processes the sequence with
Transformer self-attention. In the MNIST classifier, the final class-token
representation is used to predict one of ten digits.

CLIP combines two independent encoders:

- a ViT image encoder produces an image embedding;
- a Transformer text encoder produces a caption embedding.

Both embeddings are L2-normalized. For a batch of paired images and captions,
their cosine similarities form a square matrix:

$$
s_{ij} =
\exp(\alpha)
\frac{f_{\mathrm{image}}(x_i)}{\lVert f_{\mathrm{image}}(x_i)\rVert}
\cdot
\frac{f_{\mathrm{text}}(t_j)}{\lVert f_{\mathrm{text}}(t_j)\rVert}.
$$

The symmetric contrastive loss trains image $i$ to retrieve its matching text
and text $i$ to retrieve its matching image. Other examples in the batch act
as negatives. After training, the embeddings support zero-shot-style
classification and image-text retrieval without adding a task-specific
classifier.

## Project scripts

- `code-1-vit-fixed.py` implements and trains a small ViT classifier on MNIST.
- `code-2-clip-fixed.py` trains CLIP on MNIST image/prompt pairs. Because many
  examples share the same digit caption, it uses a multi-positive contrastive
  loss rather than treating matching same-digit captions as negatives.
- `code-2-clip-flickr8k-demo.py` trains CLIP from scratch on local Flickr8k
  image-caption data. It randomly selects from five captions during training
  and reports image-to-text and text-to-image Recall@1, Recall@5, and Recall@10.
- `code-3-test-pretrained-clip.py` demonstrates similarity scoring with a
  separately downloaded pretrained ChineseCLIP model.
- `code-1-vit.py` and `code-2-clip.py` are the original learning versions; the
  `*-fixed.py` files are recommended for execution.

The Flickr8k model uses a small word/punctuation vocabulary built only from
training captions. The MNIST CLIP demo uses a simpler UTF-8 byte tokenizer.
These are intentionally compact teaching implementations rather than the BPE
tokenizer and web-scale training data used by production CLIP systems.

## Server commands

Run the ViT demonstration:

```bash
cd ~/scratch/dips_project/reinforcement_learning/multip_modal/vit_and_clip
sbatch submit-code-1-vit-fixed.sh
```

Run the MNIST CLIP demonstration:

```bash
sbatch submit-code-2-clip-fixed.sh
```

Run the more realistic Flickr8k CLIP experiment:

```bash
sbatch submit-code-2-clip-flickr8k-demo.sh
```

For a quick server check, append `--epochs 1` to any submission command. The
scripts save their best validation checkpoints under `checkpoints/` and write
Slurm output under `result_out/`.

This project demonstrates CLIP's architecture, contrastive objective, learned
temperature, in-batch negatives, shared multimodal embeddings, and retrieval
evaluation. Flickr8k and MNIST are far too small to reproduce the broad
open-vocabulary capability of CLIP trained on hundreds of millions of pairs.
