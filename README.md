# AttireCV

**AttireCV** is an application that attempts to classify blouses and trousers based on their picture
by using pre-trained CV models.

## Dataset

Dataset was downloaded via `opendatasets` library
from [this](https://www.kaggle.com/competitions/lamoda-images-classification/) Kaggle competition:

```python
import opendatasets
url = "https://www.kaggle.com/competitions/lamoda-images-classification/"
opendatasets.download(url)
```

The dataset is quite good, and so it does not have any corrupt files, outliers or any distortions.
