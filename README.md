## Recurrent Filter Learning for Visual Tracking
This is the implementation of our RFL tracker published in ICCV2017 workshop on VOT. 
Our code is written in **python3(3.5)** using **Tensorflow(>=1.2)** toolbox

**For easy comparison, we upload our OTB100 results files to the main directory `./otb100_results.zip`**

### Tracking
You use our pretrained model to test our tracker first. 
1. Download the model from the link: [model](https://drive.google.com/drive/folders/0BzxOz7xyra_-dzJaY2d0Y1RiZFk?resourcekey=0-aOh9gKNpJ-UHkR0LVJcZNA&usp=sharing)
2. Put the model into directory `./output/models`
3. Run `python3 tracking_demo.py` in directory `./tracking`

### Training
1. Download the ILSRVC data from the official website and set proper paths for ISLVRC and their tfrecords in `config.py`
2. Then run the `process_data.sh` in `./data_preprocssing` directory to convert ILSVRC data to tfrecords.
3. Run `python3 train.py` to train the model.

If you find the code is helpful, please cite
```
@inproceedings{Yang2017,
    author = {Yang, Tianyu and Chan, Antoni B.},
    booktitle = {ICCV Workshop on VOT},
    title = {Recurrent Filter Learning for Visual Tracking},
    url = {http://arxiv.org/abs/1708.03874},
    year = {2017}
}
```
