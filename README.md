# VN_currency_classification

## Why is Vietnamese Currency Classification?
- In order to highlight what i have leant about Deep Learning models
- Tương lai có thể tạo ra một ứng dụng có thể giúp cho người mù nhận biết được tiền Việt Nam
- Ứng dụng trong ngân hàng
## What I have done?
### Balancing data:
	- Initial data set with 8 classes, detail are:
		|class name | files|
		|-----------|------|
		|1000 | 90 |
		|10000 | 170 |
		|100000 | 66 |
		|2000 | 211 |
		|20000 | 218 |
		|200000 | 242 |
		|5000 | 276 |
		|50000 |221 |
		|500000 | 219 |

	We can see that our data is imbalancing, the class 1000 and 100000 is quite less file than others classes
	- I used data augmentation to make dataset more balance. After done it, we have nicer dataset looked like:
	
		|class name | files|
		|1000 | 628 |
		|10000 | 548 |
		|100000 | 517 |
		|2000 | 420 |
		|20000 | 434 |
		|200000 | 481 |
		|5000 | 551 |
		|50000 |442 |
		|500000 | 437 |

### Using MobileNetV2 to train on data set

	- first training:
	[In cái hình loss, acc]
	
	Yep!!!! Very nice result with above 97.52% accuracy in validation set
	
	- Fine tunning:
	[In cái hình sau khi acc]
	After fine-tuning the accuracy of our model has increase to `98.2%` at epoch 21st (has improved from 97.2% at first-run).

### Confusion Matrix:

	[Hình confusion matrix]
	
	+ Model is confusing between 2 classes 20000 and 500000 
	
	[Hình 20k, vs 50k]

## Prediction on real money
	
	
## GramCad
	 















