## FixMatch for Audio-based COVID-19 Detection

Code for the paper: "[[Exploring Semi-supervised Learning for Audio-based COVID-19 Detection using FixMatch](https://mobile-systems.cl.cam.ac.uk/papers/interspeech22.pdf)]()" by Ting Dang, Thomas Quinnell, and Cecilia Mascolo.



## Previous work

For the code used as the foundation for the project and details on accessing the dataset, please see the [COVID-19 Sound Team's repository](https://github.com/cam-mobsys/covid19-sounds-neurips) by Tong Xia, Dimitris Spathis, and Andreas Grammenos.



## Code

The models are implemented in Python3 (tested using `3.6` and `3.7`) using Tensorflow `1.15`.
Before running the code, the relevant data files must be pre-processed and inserted into the correct
directories, which will be explained in later sections.
The libraries must be manually installed onto the system. For example, this could be done through a Virtual
Environment:

```bash
# create a virtual environment for this project (assuming you are in the root directory of the repository)
python3 -m venv ./venv
# now activate it
source ./venv/bin/activate
# and finally, install the dependencies
pip install -r ./requirements.txt
```

## Algorithms

There are 4 sections of the code, namely:
- Pre-processing
- Supervised code
- Psuedo-label
- FixMatch

## Pre-processing

1. Navigate to the `pre-process` directory
1. Convert the .m4a files to .wav files by extracting the data to `unlabelled_data` directory
    and run 
    ```
     python convert_wav.py
    ```
1. to fairly split positive and negative samples, run
    ```
     python data_merge.py
    ```
1. Rename the data to `0426_EN_used_task2` and run
    ```
     python data_flatten.py
    ```
    Then follow the instructions below and in the README of the preprocess directory to remove noisy samples.
    In order to decide if a provided audio sample is of sufficient quality to be used for inference we provide a tool that
    automatically detects whether a sample is of high-quality. This tool employs another network, namely
    Yamnet. The sample should contain either:

- breathing (will be tagged with `'b'`),
- cough (will be tagged with `'c'`),
- or voice (will be tagged with `'v'`).

Silent and noisy samples will be filtered accordingly and labelled as `'n'`. This labelling will exclude such files
from further experiments. Please note that this tool requires a **different** environment
to be used, as Yamnet _requires_ Tensorflow 2.0.


## Supervised model

- Prepare input

  ```shell
   cd ./COVID19_prediction/data
   python pickle_data.py
  ```

- Go to model's path `cd ./COVID19_prediction/COVID_model`
- Train the model `sh run_train.sh`
- Test the model `sh run_test.sh`

edit the .sh files to include extra arguments such as:
- shuffle\_vad - shuffle validation data at end of 15 epochs
- train\_data\_portion - portion of labelled data to train on

## Pseudo-label

- Prepare input

  ```shell
   cd ./COVID19_prediction/data
   python pickle_data.py
   python pickle_unlabelled_data.py
  ```

- Go to model's path `cd ./Pseudo_label/COVID_model`
- Train the model `sh run_pseudo.sh`
- Visualise the histogram of confidences
  ```shell
   sh run_confidences.sh
   python vis_predictions.py
  ```

## FixMatch

- Prepare input (just copy the data files if done already in Pseudo-label stage)

  ```shell
   cd ./COVID19_prediction/data
   python pickle_data.py
   python pickle_unlabelled_data.py
  ```

- Go to model's path `cd ./FixMatch/COVID_model`
- Train the model `sh run_fm.sh`
- Train the model with FixMatch with MixUp `sh run_fixup.sh`
- Visualise the predictions
  ```shell
   sh run_pred.sh
   python vis_predictions.py
  ```

## Citing this work

```shell
@article{dang2022exploring,
  title={Exploring Semi-supervised Learning for Audio-based COVID-19 Detection using FixMatch},
  author={Dang, Ting and Quinnell, Thomas and Mascolo, Cecilia},
  journal={Proc. Interspeech 2022},
  pages={2468--2472},
  year={2022}
}
```

## 
