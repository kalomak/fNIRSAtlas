# Running a Custom Model

Put your sklearn-compatible model in a [pipeline](pipelines.md) and pass it to a MOABB evaluation with a [paradigm](paradigms.md):

```python
from moabb.evaluations import WithinSessionEvaluation
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fnirsatlas import Bnci2014FnirsMA, FNIRSClassification
from fnirsatlas.pipelines import WindowedFeatureExtractor
from my_package import MyModel

model = make_pipeline(
    WindowedFeatureExtractor(),
    StandardScaler(),
    MyModel(),
)
evaluation = WithinSessionEvaluation(
    paradigm=FNIRSClassification(),
    datasets=[Bnci2014FnirsMA()],
)
results = evaluation.process({"My model": model})
print(results)
```

For models implemented with PyTorch, see [Running a Custom PyTorch Model](custom_pytorch_models.md).
