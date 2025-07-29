# AttireCV

**AttireCV** is an application that attempts to classify blouses and trousers based on their picture
by using pre-trained CV models.

## Dataset

Dataset was downloaded via `opendatasets` library:

```python
import opendatasets
url = 'https://www.kaggle.com/competitions/lamoda-images-classification/'
opendatasets.download(url)
```
