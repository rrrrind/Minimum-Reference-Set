# Minimum Reference Set
## Contents
Minimum Reference Setのモジュール  
2次元・2クラスに対応  

## Algorithm
1. クラス毎にプロットを分割し，ユークリッドでクラス毎のプロット間の距離を測る  
1. 最も近い2プロットを選択する  
1. 選択した2プロットで全データを分類する(最近傍法を使用)  
1. 分類誤差が0の場合は終了し、0でない場合は更に2プロット選択し全データを分類する  
1. 分類する際に選択したプロット数で特徴量空間の評価を行う(選択されたプロットが少ない空間を良いとする)  

※アルゴリズムが間違えてる等ございましたら，ご指摘いただけると幸いです．

## How to use
1. from mrs import MinimumReferenceSet
1. mrs = MinimumReferenceSet()
1. 評価値 = mrs.run(1つの特徴量空間, クラスラベル, 探索上限)

## Paper
[Minimum reference set based feature selection for small sample classifications](https://dl.acm.org/doi/abs/10.1145/1273496.1273516)
