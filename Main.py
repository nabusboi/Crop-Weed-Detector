from ultralytics import YOLO
import os

model_path=r"runs\detect\train\weights\best.pt"
predicted_folder="predicted"

def main():
    model=YOLO(model_path)

    image_path=input("Enter image path: ")
    image_path=image_path.strip().strip('"')

    result=model.predict(
        source=image_path,
        conf=0.4,
        iou=0.35,
        agnostic_nms=True,
        save=True,
        show=True
    )

    for results in result:
        for box in results.boxes:
            class_id=int(box.cls[0])
            confidence=float(box.conf[0])
            class_name=results.names[class_id]

            print(class_name,round(confidence*100,2),"%")

        output_path=os.path.join(predicted_folder,os.path.basename(image_path))

        results.save(filename=output_path)
        results.show()

if __name__ =="__main__":
    main()