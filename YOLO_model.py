import os
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from ultralytics import YOLO

seed = 42
image_size = 960
epochs=40
batch_size=16

dataset_path='weed_crop_yolo_dataset'
yaml_file='weeds_crops.yaml'

def cleanNaming(name):
    clean_name=""

    for char in name:
        if(char.isalnum()):
            clean_name+=char.lower()
    return clean_name

def Preprocess():

    dataset = load_dataset(
        "Project-AgML/weed_crop_detection", 
        split="train")
    
    crop_names=[]

    for names in dataset.unique("crop"):
        crop_names.append(cleanNaming(names))

    X=list(range(len(dataset)))
    y=dataset["crop"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=seed,
        stratify=y)
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_test, y_test,
        test_size=0.5,
        random_state=seed,
        stratify=y_test)

    return dataset,X_train, X_val, X_test, crop_names

def createSplit(dataset,crop_names,X,split):
    img_folder=os.path.join(dataset_path,'images',split)
    label_folder=os.path.join(dataset_path,'labels',split)

    os.makedirs(img_folder, exist_ok=True)
    os.makedirs(label_folder, exist_ok=True)

    category_names=(
        dataset.features["objects"]["categories"].feature.names
    )

    for dataset_pos in X:
        sample=dataset[dataset_pos]
        img=sample["image"]
        boxes=sample["objects"]["bbox"]
        categories=sample["objects"]["categories"]

        img_w,img_h=img.size

        img_destination=os.path.join(
            img_folder,f"{dataset_pos}.jpg"
        )

        label_destination=os.path.join(
            label_folder,f"{dataset_pos}.txt"
        )

        img.convert("RGB").save(img_destination)

        with open(label_destination,"w") as file:
            for pos in range(len(boxes)):
                box=boxes[pos]
                category_id=categories[pos]
                category_name=cleanNaming(category_names[category_id])
                if category_name in crop_names:
                    class_id=0
                else:
                    class_id=1

                x=box[0]
                y=box[1]
                width=box[2]
                height=box[3]

                x_center=(x+width/2)/img_w
                y_center=(y+height/2)/img_h
                norm_width=width/img_w
                norm_height=height/img_h

                file.write(
                    f"{class_id} "
                    f"{x_center} "
                    f"{y_center} "
                    f"{norm_width} "
                    f"{norm_height}\n"
                )

def main():
    dataset,X_train,X_val,X_test,crop_names=Preprocess()

    createSplit(dataset,crop_names,X_train,"train")
    createSplit(dataset,crop_names,X_val,"valid")
    createSplit(dataset,crop_names,X_test,"test")

    with open(yaml_file,"w") as file:
        file.write(f"path: {os.path.abspath(dataset_path)}\n")
        file.write("train: images/train\n")
        file.write("val: images/valid\n")
        file.write("test: images/test\n")
        file.write("names:\n")
        file.write("  0: crop\n")
        file.write("  1: weed\n")
    model=YOLO("yolo26n.pt")

    model.train(
        data=yaml_file,
        epochs=epochs,
        batch=batch_size,
        imgsz=image_size
    )
    best_model=YOLO(model.trainer.best)

    metrics=best_model.val(
        data=yaml_file,
        split="test"
    )

    print("Precision: ",metrics.box.mp)
    print("Recall: ",metrics.box.mr)
    print("Log folder",model.trainer.save_dir)
if __name__ =="__main__":
    main()