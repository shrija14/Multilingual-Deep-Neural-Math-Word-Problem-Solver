# Multilingual-Deep-Neural-Math-Word-Problem-Solver

This project, develops a language agnostic math word problem solver through the implementation of deep learning-based methods. It  utilizes  subsets  of  two  multilingual  large-scale datasets, Math23K (Chinese language) and Dolphin-S (English language), to train and test a language-agnostic model.  Detailed descriptions have been provided in Report.pdf.

Steps to reproduce results:  
- Preprocessing 
  - For Math23K, run Preprocess/Preprocess_Math23K.ipynb
  - For DolphinS, run Preprocess/Preprocess_Dolphin.ipynb  
  This will prepare data in a format that the model needs.
- First Stage
  - Run FirstStage/src/main.py  
  This will give the equation templates.
