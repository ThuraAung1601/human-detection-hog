from skimage.feature import hog
import joblib,glob,os,cv2

from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn import svm
import numpy as np 
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt

X = []
Y = []

pos_im_path = './DATAIMAGE/positive'
neg_im_path = './DATAIMAGE/negative'

# Load the positive features
for filename in glob.glob(os.path.join(pos_im_path,"*.png")):
    fd = cv2.imread(filename,0)
    fd = cv2.resize(fd,(64,128))
    fd = hog(fd,orientations=9,pixels_per_cell=(8,8),visualize=False,cells_per_block=(3,3))
    X.append(fd)
    Y.append(1)

# Load the negative features
for filename in glob.glob(os.path.join(neg_im_path,"*.jpg")):
    fd = cv2.imread(filename,0)
    fd = cv2.resize(fd,(64,128))
    fd = hog(fd,orientations=9,pixels_per_cell=(8,8),visualize=False,cells_per_block=(3,3))
    X.append(fd)
    Y.append(0)

X = np.float32(X)
Y = np.array(Y)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

print('Train Data:',len(X_train))
print('Train Labels (1,0)',len(y_train))

model = LinearSVC()
model.fit(X_train,y_train)

# predict
y_pred = model.predict(X_test)

# confusion matrix and accuracy

from sklearn import metrics
from sklearn.metrics import classification_report 

print(f"Classification report for classifier {model}:\n"
      f"{metrics.classification_report(y_test, y_pred)}\n")

joblib.dump(model, 'models.dat')
print('Model saved : {}'.format('models.dat'))
