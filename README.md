# Multilingual-Deep-Neural-Math-Word-Problem-Solver

In this project, we develop a language agnostic math word problem solver using deep learning-based methods. Subsets of two multilingual  large-scale datasets, Math23K (Chinese language) and Dolphin-S (English language) are used to train and test the language-agnostic model.  Detailed descriptions can be found [here](https://github.com/shrija14/Multilingual-Deep-Neural-Math-Word-Problem-Solver/tree/master/Reports).

Steps to reproduce results:  
- Preprocessing 
  - For Math23K, run Preprocess/Preprocess_Math23K.ipynb
  - For DolphinS, run Preprocess/Preprocess_Dolphin.ipynb  
  This will prepare data in a format that the model needs.
- First Stage
  - Run FirstStage/src/main.py  
  This will give the equation templates.
