# Seq-NMS + YOLO
This is an anofficial implementation of Seq-nms. The original paper is [here](https://arxiv.org/abs/1602.08465). The original code is [here](https://github.com/tmoopenn/seq-nms).

The reason of re-implementing this algorithm is to combine it with YOLO. We like to see the difference between the original detection results and the updated detection results. 

## Setup Details 
```
python setup.py build_ext --inplace
```
More details can be found in the original repo.

## Usage
The main program is ```find_seq.py```, which is based on the original function ```seq_nms.py```. To run the main program, use the following command:
```
python find_seq.py --img [IMAGE FOLDER] --yolo [YOLO DETECTION FOLDER]
```

We provide 4 frames for better visualization. These 4 frames are come from our koala detection dataset. Without using seq-nms, our detector is struggle for detecting the class we are interested in. Using seq-nms, we can use the frame with higher confidence to boost the ones with lower confidence. 

To run the main program and generate the new detection results, use the following command:
```
python find_seq.py --img dataset/detected_images --yolo dataset/yolo_labels/
```
It will create a new folder called ```updated_labels``` which contains the updated detection results. We show the comparison between the original detection results and the updated detection results in the following figure.

## Visualization of the results
![Effect of Seq-NMS](./result.jpg)
Each row is a frame. We start the frame from index 0. The maximum number of bbox for each frame is 4 here. We name each bbox in the first row like``` 0_0, 0_1,..0_4```. We separate classes by different colors. The last 2 white boxes in frame 3 means bbox with 0 coordinates, confidence and label.

# Reference
1. https://arxiv.org/abs/1602.08465
2. https://github.com/tmoopenn/seq-nms