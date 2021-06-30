Main Program:
1. Run UI program to recognize the formula (need trained model file: mix2.h5):nr_split_ui.py
2. Train Model(build model file: mix2.h5): nr_mix_train.py
Other Program:
1. Expression calculation methods: calculate_expression.py
2. Generate left and right parentheses util: datagen/datagen.py
3. Formula recognition class: sliding/formula_recognizer.py

Tips about how to run the program:
1. Use files in the zip package directly or download dataset from https://www.kaggle.com/clarencezhao/handwritten-math-symbol-dataset
2. Put the dataset under "dataset" folder in the project root dir
3. Copy parentheses dataset from "datagen/gen" to "dataset" folder (or run datagen.py to generate again)
4. Run "nr_mix_train.py" to build the model file "mix2.h5" (as this file is too large, we didn't upload it)
5. Run "nr_split_ui.py" to run the ui program, happy testing.

Tips about how to install tensorflow and other libs on Windows:
1. Install python 3.8 64 bit
2. python -m pip install –upgrade pip
3. python -m pip install --upgrade https://storage.googleapis.com/tensorflow/windows/cpu/tensorflow_cpu-2.4.0-cp38-cp38-win_amd64.whl
4. python -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
5. pip install pywin32
6. pip install Pillow

