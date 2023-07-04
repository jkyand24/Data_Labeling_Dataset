import json
import os
import cv2
import glob
import numpy as np

json_dir = "./data/anno"
json_paths = glob.glob(os.path.join(json_dir, "*.json"))

label_dict = {"수각류": 0}

new_width = 1024
new_height = 768

for json_path in json_paths: # json_path: ~~~.json
    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f) # json_data의 key: 'info', 'images', 'annotations', 'licenses'
    
    images_info = json_data['images'] # {}
    annotations_info = json_data['annotations'] # [{}]
    
    image_filename = images_info['filename']
    image_filename_temp = image_filename.replace(".jpg", "")
    image_filepath = os.path.join("./data/images", image_filename)
    
    image_id = images_info['id']
    
    for annotation_info in annotations_info:
        if image_id == annotation_info['image_id']: # 이 예제에서 사용하는 data에서는, 모두 같음
            image = cv2.imread(image_filepath)
            resized_image =  cv2.resize(image, (new_width, new_height))
            
            # points 만들기
            
            points = []
            
            polygons_info = annotation_info['polygon']
            
            scale_x = new_width / image.shape[1]
            scale_y = new_height / image.shape[0]
            
            for polygon_info in polygons_info:
                x = polygon_info['x']
                y = polygon_info['y']
                resized_x = int(x * scale_x)
                resized_y = int(y * scale_y)
                
                points.append((resized_x, resized_y))
                
            # polylines 그리기
            
            cv2.polylines(resized_image, [np.array(points, np.int32).reshape((-1, 1, 2))], isClosed=True, color=(0, 255, 0), thickness=2)
                # np.array(points, np.int32)의 shape: (point 개수, 2)
            
            # 좌표들 찾기
            
            xs = [point[0] for point in points]
            ys = [point[1] for point in points]
            x_min = min(xs)
            y_min = min(ys)
            x_max = max(xs)
            y_max = max(ys)
            
            # rectangle 그리기
            
            cv2.rectangle(resized_image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
            
            # label과 yolo 형식의 좌표들을 담은 txt 파일 생성
            
            yolo_xcenter = (x_max + x_min) / (2 * new_width)
            yolo_ycenter = (y_max + y_min) / (2 * new_height)
            yolo_width = (x_max - x_min) / new_width
            yolo_height = (y_max - y_min) / new_height
            
            category_name = annotation_info['category_name']
            label_number = label_dict[category_name]
            
            os.makedirs("./data/yolo_label_data", exist_ok=True)
            with open(f"./data/yolo_label_data/{image_filename_temp}.txt", "a") as f:
                f.write(f"{label_number} {yolo_xcenter} {yolo_ycenter} {yolo_width} {yolo_height}\n")   
                
            # 시각화 
            
            cv2.imshow("Polygon", resized_image)
            
            if cv2.waitKey(0) & 0xFF == ord('q'):
                exit()