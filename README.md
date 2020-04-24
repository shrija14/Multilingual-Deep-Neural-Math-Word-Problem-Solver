# Multilingual-Deep-Neural-Math-Word-Problem-Solver

In this project, we develop a language agnostic math word problem solver using deep learning-based methods. Subsets of two multilingual  large-scale datasets, Math23K (Chinese language) and Dolphin-S (English language) are used to train and test the language-agnostic model.  Detailed descriptions can be found [here](https://github.com/shrija14/Multilingual-Deep-Neural-Math-Word-Problem-Solver/tree/master/Reports).

Data is prepared in the following way:  
The notebooks mentioned here are in Miscellaneous folder.  
- Dolphin subset is prepared using Initial_Data_Cleaning.ipynb
- Dolphin is preprocessed in Preprocess_Dolphin.ipynb
- Math23K is preprocessed in Preprocess_Math23K.ipynb
- Dolphin is then replicated using SecondStage/replicate.py.
- LAMP32K is prepared in LAMP32K.ipynb
- LAMP32K is split in train,validation,test and postfix template preparation is in Split_Postfix.ipynb

All generated files including the dataset and splits are provided in the data/ folder.

Generator:  
The code in FirstStage/src contains the model that will generate equations. It can be run using the command  
> python main.py --cuda-use --checkpoint-dir-name params_12 --mode 0 --teacher-forcing-ratio 0.5 --input-dropout 0.4 --encoder-hidden-size 512 --decoder-hidden-size 1024  --generator 1

This will create 3 files for train, validation and test with generated equations. Sample has been provided in Results/GeneratorModel.

Predictor:  
The code in SecondStage takes template equations and predicts operators between them.
In order to train a model predict flag to True. By default it is False.
To predict we need the model that can be downloaded from [here](https://drive.google.com/file/d/1EZ8-55lvaa__VlAhm-NZhqETZ-hGqpTP/view?usp=sharing) to SecondStage/data/ 
> python main.py true #predict  
> python main.py false #train

This will output sample correct/matched equations with ground truth.

GenPred:  
In this we generate equation templates as opposed to entire equation in Generator. The command to train this models is  
> python main.py --cuda-use --checkpoint-dir-name params_12 --mode 0 --teacher-forcing-ratio 0.5 --input-dropout 0.4 --encoder-hidden-size 512 --decoder-hidden-size 1024  --generator 0

This will create train, validation and test files with equation templates. These files can later be fed into second stage with predictor flag = true

#### References
[Template-Based Math Word Problem Solvers with Recursive Neural Networks](https://github.com/uestc-db/T-RNN)
