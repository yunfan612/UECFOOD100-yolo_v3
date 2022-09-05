環境:
tensorflow-gpu 2.4.0
Keras          2.4.3

描述:使用yolo v3對uecfood100做object detection,共有100種日本食物類別

dataset下載於: https://www.kaggle.com/datasets/rkuo2000/uecfood100

下載後資料夾內bb_info為食物在照片中對應的位置

box_label_generator.py:
讀1~100資料夾之bb_info並產生label

merge_multiple_food.py:
讀label,將重複檔案但不同box的label merge起來,產生label_multiple_merge

train_test_list_generator.py:
將label_multiple_merge檔分成train,valid,test

kmeans.py:
重新產生適合training data的anchors(寬.高)

train.py
訓練模型
pretrain model為yolo_weights.h5(可上網抓)
參考:https://zhuanlan.zhihu.com/p/137533458
至作者網址:https://pjreddie.com/darknet/yolo/下載weight(可先試YOLOv3-320)
打開anaconda並進入環境,cd到資料夾下,執行 python convert.py yolov3.cfg yolov3.weights model_data/yolo_weights.h5

yolo.py
測試模型
輸入單張照片並測試結果
score為信心分數超過才顯現
iou為相同類別重疊超過多少則只選score大的

yolo_test.py
測試模型
輸入多張照片並測試結果
跑出資料夾內所有圖形結果(32行path),結果放於35行result_path,並會產生測試結果的label於249行txt_path(跑分數用)


輸出中英文轉換
於測試模型時
classes_path: food_classes_chinese.txt為中文
              food_classes.txt為英文

font = ImageFont.truetype的font為mingliu.ttc 為中文
                                 FiraMono-Medium.otf 為英文
