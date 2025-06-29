# Week-2
Garbage Classification with CNN


This project uses the [Trash Type Image Dataset](https://www.kaggle.com/datasets/farzadnekouei/trash-type-image-dataset) from Kaggle.

To download it automatically, the code uses `kagglehub`:

```python
import kagglehub
path = kagglehub.dataset_download("farzadnekouei/trash-type-image-dataset")

