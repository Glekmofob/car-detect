import cv2 
from ultralytics import YOLO


from configRead import LoadConfig
from video import GetVideo

config = LoadConfig()
videoSource = config["video"]["source"]
modelWeights = config["model"]["weights"] #Using nano model bacuse my laptop has no GPU and wants to kill itself
confThresh = config["model"]["confThreshold"]

model = YOLO(modelWeights)
cv2.namedWindow("Result", cv2.WINDOW_NORMAL)

capture = GetVideo(videoSource)


while True:
    success, frame = capture.read()

    if not success:
        print("Видева Кончилось")
        break

    results = model.predict(frame, conf = confThresh, classes = [0, 2], verbose = False)
    coords = []
    boxes = results[0].boxes
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist()) 
        #тут двумерные массивы поэтому формат который выдает for это [[...]] для себя + tensor
        x, y = int(box.xywh[0].tolist()[0]), int(box.xywh[0].tolist()[1])
        conf = int(box.conf[0].item()*100)
        objId = int(box.cls[0].item())
        objName = results[0].names[objId]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        text = f"{objName}, conf:{conf}% coords: {x}, {y}"

        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.5
        thickness = 2

        (textWidth, textHieght), baseline = cv2.getTextSize(text,fontFace= font,fontScale= fontScale, thickness= thickness)
        text_x = x2 - textWidth - 5
        text_y = y2 + textHieght + 5

        text_x = max(text_x, x1 + 5)
        cv2.putText(frame, text, (text_x, text_y),fontFace=font, fontScale=fontScale, color=(0,255,0), thickness= thickness)



    
    cv2.imshow("Result", frame) # можно по идее просто пихнуть results[0].plot(), но практика кастомизации ╰(▔∀▔)╯ 
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
capture.release()
cv2.destroyAllWindows()