import os
import numpy as np
import argparse
import glob
import shutil
from pathlib import Path

from seq_nms import build_box_sequences, find_best_sequence, rescore_sequence, delete_sequence
from compute_overlap import compute_overlap_areas_given, compute_area
from utils.sort_name import natural_keys
from utils.yolo_to_coco import yolo_to_coco


def main(args):
    # ==========create boxes, scores, labels==========
    num_frames = len(os.listdir(args.img))
    num_boxes = []
    for label in os.listdir(args.yolo):
        num_boxes.append(len(open(args.yolo+'/'+label).readlines()))
    max_num_boxes = max(num_boxes)
    boxes = np.zeros((num_frames, max_num_boxes, 4))
    scores = np.zeros((num_frames, max_num_boxes))
    labels = np.zeros((num_frames, max_num_boxes))
    print(f"Total {num_frames} frames; maximum {max_num_boxes} boxes for each frame")

    img_files = sorted(glob.glob(args.img + "/*.jpg"), key=natural_keys)
    for img_idx in range(len(img_files)):
        image_name = img_files[img_idx]
        label_name = os.path.join(args.yolo, os.path.basename(img_files[img_idx])[:-4]+'.txt' ) 
        # print(f"Image file: {image_name}, label file: {label_name}")
        if os.path.exists(label_name):
            num_det = len(open(label_name).readlines())
            # print(f"The {img_idx}-th image {img_files[img_idx]} has label file {os.path.join(args.yolo, img_files[img_idx][:-4]+'.txt')} with {num_det} detected objects. Modify the {img_idx}-th row of boxes, scores and labels")
            for det in range(num_det):
                line = open(label_name).readlines()[det].split()
                # convert to x1, y1, x2, y2 format
                voc_label = yolo_to_coco(line[1:5])
                # save each coordiates with 4 decimal places
                boxes[img_idx][det] = np.array([float(i) for i in voc_label])
                scores[img_idx][det] = float(line[5])
                labels[img_idx][det] = float(line[0])

        else:
            # print(f"No label for {img_files[img_idx]}. Paddding with zeros")
            boxes[img_idx] = np.zeros((max_num_boxes, 4))
    assert boxes.shape == (num_frames, max_num_boxes, 4), "boxes shape is not correct"
    assert scores.shape == (num_frames, max_num_boxes), "scores shape is not correct"
    assert labels.shape == (num_frames, max_num_boxes), "labels shape is not correct"

    # ==========build box sequences==========
    linkage_threshold=0.2
    score_metric='avg'
    nms_threshold=0.3
    box_graph = build_box_sequences(boxes, scores, labels, linkage_threshold=linkage_threshold)
    assert box_graph.shape == (num_frames - 1, max_num_boxes), "box_graph shape is not correct"
    
    # ==========recursively find the best sequence and delete other boxes based on the IoU==========
    sequence_to_use = [] # list of dict. Each key-value is: [frame index]-[box index on that frame]
    accu_conf = []
    while True: 
            frame_index_sequence = {} 
            sequence_frame_index, best_sequence, best_score = find_best_sequence(box_graph, scores)
            # print(f"\nsequence_frame_index: {sequence_frame_index}, best_sequence: {best_sequence}, best_score: {best_score}")
            frame_index_sequence[sequence_frame_index] = best_sequence
            accu_conf.append(best_score)
            sequence_to_use.append(frame_index_sequence)
            if len(best_sequence) <= 1:
                break 
            rescore_sequence(best_sequence, scores, sequence_frame_index, best_score, score_metric=score_metric)
            delete_sequence(best_sequence, sequence_frame_index, scores, boxes, box_graph, suppress_threshold=nms_threshold)
    assert len(sequence_to_use) == len(accu_conf), "The number of sequences and the number of accumulated confidence are not the same"
    print(f"Build {len(sequence_to_use)} best sequence(s)")
    # print(f"sequence_to_use: {sequence_to_use}. The accumulated confidence for each sequence is {accu_conf}")

    # ==========generate updated label files for each sequence==========
    updated_label = os.path.join(str(Path(args.img).parent), "updated_labels")
    if os.path.exists(updated_label):
        shutil.rmtree(updated_label)
    os.mkdir(updated_label)
    print(f"Create {updated_label} to save the updated label files")

    for seq_idx in range(len(sequence_to_use)):
        frame_box_list = []
        for start_frame, box_idxs in sequence_to_use[seq_idx].items():
            print(f"The {seq_idx}-th sequence is for class {int(labels[start_frame][box_idxs[0]])}", end=' ')
            # print(f"    Starts from the {start_frame}-th frame. Box indices: {box_idxs}")
            start_frame = int(start_frame)
            for box_in_frame in range(len(box_idxs)):
                frame_box_list.append(f"{start_frame}_{box_idxs[box_in_frame]}")
                start_frame += 1
        # print(f"    The trace is: {' -> '.join(frame_box_list)}. ")
        # print(f"    {frame_box_list}")
        print(f"with {len(frame_box_list)} nodes and rescore to {accu_conf[seq_idx] / len(frame_box_list)}")
        
        # ==========update the label files==========
        for tem in frame_box_list:
            for img_idx in range(len(img_files)):
                if int(tem.split('_')[0]) == img_idx:
                    # print(f"    Update {tem}, the {img_idx}-th imaage with name: {os.path.basename(img_files[img_idx])}, and {tem.split('_')[-1]}-th label with name: { os.path.basename(os.path.join(args.yolo, os.path.basename(img_files[img_idx]).replace('.jpg', '.txt'))) }")
                    file_old = os.path.join(args.yolo, os.path.basename(img_files[img_idx]).replace('.jpg', '.txt'))
                    with open(file_old, 'r') as f:
                        for line_idx, line in enumerate(f.readlines()):
                            file_new = os.path.join(updated_label, os.path.basename(file_old))
                            with open(file_new, 'a') as f_new:
                                if line_idx == int(tem.split('_')[-1]):
                                    new_label = " ".join(line.split()[:-1]) + " " +str( accu_conf[seq_idx] / len(frame_box_list) )
                                    # print(f"    update label {line} to new label {new_label}")
                                    f_new.write(new_label + "\n")


    print(f"Finish updating the label files.")
                                    

    

if __name__ == '__main__':
    # set arguments
    parser = argparse.ArgumentParser(description='Use seq-nms to find the best sequence(s) after yolo detection')
    parser.add_argument('--img', type=str, required=True, help='path to the folder containing the original images')
    parser.add_argument('--yolo', type=str, required=True, help='path to the folder containing the yolo labels')
    
    args = parser.parse_args()

    main(args)