from ultralytics import YOLO
import cv2 
import numpy as np
import os  
import glob  
from tqdm import tqdm 

'''
{0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}
'''

if __name__ == "__main__":
    # Load a model
    model = YOLO('yolov8l-seg.pt')  # load an official model
    model = YOLO('yolov8s-seg.pt')
    img_dir = "/home/spurs/dataset/kitti_raw/2011_10_03/2011_10_03_drive_0047_sync/image_02/data"

    img_paths = glob.glob(os.path.join(img_dir, "*png"))
    img_paths.sort()

    des_dir = os.path.dirname(img_dir)
    des_dir = os.path.join(des_dir, "seg")
    os.makedirs(des_dir, exist_ok=True)
    #img_paths = ['/home/spurs/dataset/kitti_raw/2011_10_03/2011_10_03_drive_0047_sync/image_02/data/0000000791.png']
    
    # for kitti
    dynamic_object = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat'] 
    for img_path in tqdm(img_paths):
        img_name = os.path.basename(img_path)
        img = cv2.imread(img_path)
        H, W, _ = img.shape

        # Predict with the model
        result = model(img)[0]  # predict on an image
        #annotated_frame = result.plot()
        #cv2.imwrite(f'./output_9.png', annotated_frame)
        #continue

        color = np.random.randint(low=0, high=255, size=(1, 3))[0]
        color = [0, 0, 255]
        alpha = 0.5
        img_seg = np.zeros((H, W), dtype=np.uint8)
        for j, mask in enumerate(result.masks):
            #print(mask.xy[0][:, 0].min(), mask.xy[0][:, 0].max(), mask.xy[0][:, 1].min(), mask.xy[0][:, 1].max())
            seg = mask.xy
            id_to_name = result.names
            class_id = int(result.boxes.cls.cpu()[j])
            object_name = id_to_name[class_id]
            bbox = result.boxes.xyxy.cpu().numpy()[j].astype(np.int32).tolist() #start_point, end_point
            prob = result.boxes.conf.cpu().numpy()[j]
            #print(f"class_id={class_id}, object={object_name}, bbox={bbox}, prob={prob}")
            #print(id_to_name)
            if False:
                seg = seg[0].astype(np.int32) # n * 2
                seg = seg[:, ::-1]
                seg = np.asarray(seg, dtype=np.int32)

                img_c = img.copy()
                img_c[seg[:, 0], seg[:, 1]] = color 
                img = img * alpha + img_c * (1 - alpha)
                cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color[::-1], 3)

            else:
                try:
                    overlay = img.copy()
                    cv2.drawContours(img, [seg[0].astype(np.int32)], -1, color, -1) # area
                    # cv2.drawContours(img, [seg[0].astype(np.int32)], -1, color, 2) # contours
                    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
                    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color[::-1], 1) # cv2.rectangle(image, start_point, end_point, color, thickness)
                    #cv2.imshow("yolov8 seg", img)
                    #cv2.waitKey(1)
                except:
                    print(f"seg={seg}, class_id={class_id}, object={object_name}")
                    continue 
                    assert 0, f"img_path={img_path}"
                    
            mask = result.masks.data[j].numpy() #mask.data
            #print(mask[mask>0].tolist())
            mask = mask * 255
            
            mask_1 = cv2.resize(mask, (W, H))
            if object_name in dynamic_object:
                # background is 0
                img_seg[mask_1 > 0] = (class_id + 1)
            #print(f"img.shape={img.shape}, img_seg.shape={img_seg.shape}, mask_1.shape={mask_1.shape}")

            if j == 0:
                mask_2 = mask_1 
            else:
                mask_2 += mask_1
            
            #cv2.imwrite(f'./output_{j}.png', mask_1)
        
        cv2.imwrite("output_seg.png", img)
        #break
        #cv2.imwrite(os.path.join(des_dir, img_name), img_seg)

