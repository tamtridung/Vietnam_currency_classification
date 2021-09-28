# VN_currency_classification

## Description:
I want to apply the knowledge I have learned about deep learning to help blind people easily recognize money.
This project is also suitable for banking application

<p align="center">
  <img src="https://www.itourvn.com/images/easyblog_articles/585/vietnamese-banknotes.jpg" />
</p>

## Technologies and techniques
- Tensorflow 2.6.0
- Data augmentation: to make dataset more various
- Pre-trained model: MobileNetV2
- Evaluation: Confusion matrix, Classification report, GramCad

## Result:
### Accuracy:
After fine tunning, model can reach `98.2%` of accuracy. Not a bad result!

<p align="center">
  <img src="https://user-images.githubusercontent.com/87942072/135127562-4580393c-0c01-4939-93c3-bbdfe04fc9e6.png" />
</p>

### Right prediction with real images
<p align="center">
  <img src="https://user-images.githubusercontent.com/87942072/135125969-53043431-e918-430a-8128-ad520da05314.png" />
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/87942072/135126074-7df81354-4bf0-42be-8cd6-6c08db37c69e.png" />
</p>


<p align="center">
  <img src="https://user-images.githubusercontent.com/87942072/135126216-9e0c2b36-074b-4ee2-a03d-4c2ecd435f7f.png" />
</p>

### Wrong prediction with real images
<p align="center">
  <img src="https://user-images.githubusercontent.com/87942072/135126321-e1235ad8-48a3-4c9d-b0dd-da9e863d2f5f.png" />
</p>

<p align="center">
  <img src="[image](https://user-images.githubusercontent.com/87942072/135126378-96e777c4-c15d-4a98-9332-9f1483cfc00f.png" />
</p>
	
## Next Improvement:
- Model is not good to recognize class 50000, because images of this class is not good
- Lack of diversity in data
- Streamlit deploy
- Model still a little bit confusing between 2 classes: 20000 and 500000

<p align="center">
  <img src="[image](https://user-images.githubusercontent.com/87942072/135128401-5b616d3a-0aa4-4f6a-b423-a6b40bc57191.png" />
</p>











