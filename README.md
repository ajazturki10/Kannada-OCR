# Kannada-Object Character Recognition

The project aims at Optical Character Recognition of handwritten documents in Kannada, a official State Language of Karnataka.

This project has been trained on ResNet-50 pretrained network.


## Dataset
The dataset used is the Chars74K[http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/] dataset. It consists of a collection of images that belong to 657+ classes. Each class consists of 25 handwritten characters. Since a Deep Learning approach has also been used in this paper, the dataset needed to be expanded. This was done by using various augmentation techniques.

## NetworK

Pretrained Network : `ResNet -50`

Setting `layer.trainable =  False` (Freezing layers)
   
   - SGD as an optimizer with a learning rate of 0.2
   
   - Epochs : 5
   
   - Batch Size : 16

After unfreezing, set `layer.trainable = True`
- Optimizer : SGD with a learning rate of 0.01
- Epochs : 50
- Batch Size : 16

# Performance


![image](https://user-images.githubusercontent.com/81867085/123999766-7ac45e00-d9f0-11eb-9f93-95e31b190dcb.png)


![image](https://user-images.githubusercontent.com/81867085/123999837-90d21e80-d9f0-11eb-83fc-a346adf395f6.png)

Our model got the testing accuracy of about 97.82%

## Sample Predictions

![image](https://github.com/ajazturki10/Kannada-OCR/blob/70751bcb2fcc93bd508c4d26f9115d5a8fcbfc0c/predict_2.PNG)


### Author

- [Ajazahmed Turki](https://www.linkedin.com/in/ajazahmed-turki-7a632120b/)
