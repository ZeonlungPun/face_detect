import os,cv2,json
import tensorflow as tf
import numpy as np
import albumentations as alb

import matplotlib.pyplot as plt
image_path="E:\\face_detect\\images"

#gpu
gpus=tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu,True)
print('gpu avaliable:',tf.test.is_gpu_available())


#image augmentation
augmentor = alb.Compose([alb.RandomCrop(width=450, height=450),
                         alb.HorizontalFlip(p=0.5),
                         alb.RandomBrightnessContrast(p=0.2),
                         alb.RandomGamma(p=0.2),
                         alb.RGBShift(p=0.2),
                         alb.VerticalFlip(p=0.5)],
                       bbox_params=alb.BboxParams(format='albumentations',
                                                  label_fields=['class_labels']))
#Load a Test Image and Annotation with OpenCV and JSON
# img = cv2.imread(os.path.join('data','train', 'images','chantel41.jpg'))
#img (h,w)
# with open(os.path.join('data', 'train', 'labels', 'chantel41.json'), 'r') as f:
#     label = json.load(f)
# coords = [0,0,0,0]
# coords[0] = label['shapes'][0]['points'][0][0]
# coords[1] = label['shapes'][0]['points'][0][1]
# coords[2] = label['shapes'][0]['points'][1][0]
# coords[3] = label['shapes'][0]['points'][1][1]
# coords = list(np.divide(coords, [img.shape[1],img.shape[0],img.shape[1],img.shape[0]]))
#
# print(coords)
# augmented = augmentor(image=img, bboxes=[coords], class_labels=['Yiu'])
# cv2.rectangle(augmented['image'],
#               tuple(np.multiply(augmented['bboxes'][0][:2], [450,450]).astype(int)),
#               tuple(np.multiply(augmented['bboxes'][0][2:], [450,450]).astype(int)),
#                     (255,0,0), 2)
# plt.imshow(augmented['image'])
# plt.show()

#Build and Run Augmentation Pipeline

for partition in ['train','test','val']:
    for image in os.listdir(os.path.join('data', partition, 'images')):
        #img shape (h,w)
        img = cv2.imread(os.path.join('data', partition, 'images', image))

        coords = [0,0,0.00001,0.00001]
        label_path = os.path.join('data', partition, 'labels', f'{image.split(".")[0]}.json')
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                label = json.load(f)

            coords[0] = label['shapes'][0]['points'][0][0]
            coords[1] = label['shapes'][0]['points'][0][1]
            coords[2] = label['shapes'][0]['points'][1][0]
            coords[3] = label['shapes'][0]['points'][1][1]
            coords = list(np.divide(coords, [img.shape[1],img.shape[0],img.shape[1],img.shape[0]]))

        try:
            #augmen image by 60 for each image
            for x in range(60):
                augmented = augmentor(image=img, bboxes=[coords], class_labels=['Yiu'])
                cv2.imwrite(os.path.join('aug_data', partition, 'images', f'{image.split(".")[0]}.{x}.jpg'), augmented['image'])

                annotation = {}
                annotation['image'] = image

                if os.path.exists(label_path):
                    if len(augmented['bboxes']) == 0:
                        annotation['bbox'] = [0,0,0,0]
                        annotation['class'] = 0
                    else:
                        annotation['bbox'] = augmented['bboxes'][0]
                        annotation['class'] = 1
                else:
                    annotation['bbox'] = [0,0,0,0]
                    annotation['class'] = 0


                with open(os.path.join('aug_data', partition, 'labels', f'{image.split(".")[0]}.{x}.json'), 'w') as f:
                    json.dump(annotation, f)

        except Exception as e:
            print(e)




