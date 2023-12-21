---
title: Pythonのscikit-learnによる分類まとめ4
tags:
  - Python
  - 機械学習
  - scikit-learn
  - 教師あり学習
private: false
updated_at: '2023-10-11T22:17:54+09:00'
id: 31cafb7a89bd50e8bcd1
organization_url_name: null
slide: false
ignorePublish: false
---
前回の続きです。

https://qiita.com/wawana12/items/202a37382ff61b4d27a9

Pythonのscikit-learnによる分類をまとめました。
今回は、マルチクラスとマルチ出力アルゴリズム、半教師あり学習、確率キャリブレーション、ニューラルネットワークモデルを用いた分類を行います。

# マルチクラスとマルチ出力アルゴリズム
OneVsRestClassifier
https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html
```Python
from sklearn.datasets import load_iris
from sklearn.multiclass import OneVsRestClassifier # OneVsRestClassifierのインポート
from sklearn.svm import SVC
X, y = load_iris(return_X_y=True)
model = OneVsRestClassifier(SVC()).fit(X, y)
model.score(X, y)
```
スコアは0.9533333333333334でした。

OneVsOneClassifier
https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsOneClassifier.html
```Python
from sklearn.datasets import load_iris
from sklearn.multiclass import OneVsOneClassifier # OneVsOneClassifierのインポート
from sklearn.svm import SVC
X, y = load_iris(return_X_y=True)
model = OneVsOneClassifier(SVC()).fit(X, y)
model.score(X, y)
```
スコアは0.9733333333333334でした。

OutputCodeClassifier
https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OutputCodeClassifier.html
```Python
from sklearn.datasets import load_iris
from sklearn.multiclass import OutputCodeClassifier # OutputCodeClassifierのインポート
from sklearn.svm import SVC
X, y = load_iris(return_X_y=True)
model = OutputCodeClassifier(SVC()).fit(X, y)
model.score(X, y)
```
スコアは0.9466666666666667でした。

MultiOutputClassifier
https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html
出力が1次元であるirisデータセットには適していないようです。

ClassifierChain
https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.ClassifierChain.html
出力が1次元であるirisデータセットには適していないようです。

# 半教師あり学習
SelfTrainingClassifier
https://scikit-learn.org/stable/modules/generated/sklearn.semi_supervised.SelfTrainingClassifier.html
```Python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.semi_supervised import SelfTrainingClassifier # SelfTrainingClassifierのインポート
from sklearn.svm import SVC
rng = np.random.RandomState(42)
X, y = load_iris(return_X_y=True)
random_unlabeled_points = rng.rand(y.shape[0]) < 0.3
y[random_unlabeled_points] = -1
model = SelfTrainingClassifier(SVC(probability=True, gamma="auto")).fit(X, y)
model.score(X, y)
```
スコアは0.6533333333333333でした。

# 確率キャリブレーション
CalibratedClassifierCV
https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html
```Python
from sklearn.datasets import load_iris
from sklearn.calibration import CalibratedClassifierCV # CalibratedClassifierCVのインポート
from sklearn.svm import SVC
X, y = load_iris(return_X_y=True)
model = CalibratedClassifierCV(SVC()).fit(X, y)
model.score(X, y)
```
スコアは0.6533333333333333でした。

# ニューラルネットワークモデル
MLPClassifier
https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
```Python
from sklearn.datasets import load_iris
from sklearn.neural_network import MLPClassifier # MLPClassifierのインポート
from sklearn.svm import SVC
X, y = load_iris(return_X_y=True)
model = MLPClassifier(max_iter=1000).fit(X, y)
model.score(X, y)
```
エラーが出たため、max_iter=1000を追加しました。
スコアは0.98でした。

記事を書き、scikit-learnには多くの分類モデルがあることを学びました。これらの分類モデルを上手に使いこなせるようになりたいです。
