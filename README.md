# artsygest
[Project introduction](https://prezi.com/view/5GRIWdsUUeVozPxdd8ZO/) 26.10.2023

[Figma design](https://www.figma.com/proto/S1Tj3vlQRewG8MVdCTKAu6/artsygest?type=design&node-id=2-11&t=QmXnyRIsfRxfw3UM-1&scaling=contain&page-id=0%3A1&mode=design)

## Project description

Recommendation system for finding similar art.

Based on [wikiart](https://www.wikiart.org/) data available on kaggle [here](https://www.kaggle.com/ipythonx/wikiart-gangogh-creating-art-gan).

Procedure similar to [this](https://towardsdatascience.com/developing-art-style-embeddings-for-visual-similarity-comparison-of-artworks-7a9d4ade2045) article.

### Style embedding

It is based on the PyTorch implementation of [Neural Style Transfer](https://github.com/ProGamerGov/neural-style-pt) ([DOI](10.5281/zenodo.6967432)). The main script is changed so that it outputs the style embedding vector instead of the stylized image. 

To run the script you need to first set up the venv and install the requirements:
```bash
python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

Then you need to download VGG-19 architecture with the `models/download_models.py` book

```bash
python models/download_models.py
```

Then you can run the script:
```bash
cd style_embedding
python3 style_embedding.py -gpu 0 -style_path examples/inputs
```

It should create a `style_embedding.pt` file in the current directory.
