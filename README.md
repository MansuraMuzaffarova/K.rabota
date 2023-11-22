# K.rabota
import cv2
import tkinter as tk
from PIL import Image, ImageTk

class CarDetector:
    def _init_(self):
        self.root = tk.Tk()
        self.root.title("Car Detector")
        self.cap = cv2.VideoCapture(0)
        self.video_label = tk.Label(self.root)
        self.video_label.pack()
        self.alert_window = tk.Toplevel(self.root)
        self.alert_window.title("Alert Window")
        self.canvas = tk.Canvas(self.alert_window, width=200, height=200, bg="red")
        self.car_rect = self.canvas.create_rectangle(50, 50, 150, 150, fill="black")
        self.canvas.pack()
        self.car_detected = False
        self.update()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    def update(self):
        ret, frame = self.cap.read()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_frame)
        img = ImageTk.PhotoImage(image=img)
        self.video_label.img = img
        self.video_label.config(image=img)
        cars = self.detect_cars(frame)
        if len(cars) > 0:
            if not self.car_detected:
                self.alert_window.deiconify()
                self.car_detected = True
        else:
            self.alert_window.withdraw()
            self.car_detected = False
        self.root.after(10, self.update)
    def detect_cars(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cars = car_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
        for (x, y, w, h) in cars:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        return cars
    def on_closing(self):
        self.cap.release()
        self.root.destroy()
if _name_ == "_main_":
    car_cascade = cv2.CascadeClassifier('haarcascade_car.xml')
    car_detector = CarDetector()
    car_detector.root.mainloop()


import os
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import to_categorical
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

data_directory = "train_resized100"
class_names = sorted(os.listdir(data_directory))
num_classes = len(class_names)

images = []
labels = []
for class_index, class_name in enumerate(class_names):
    class_directory = os.path.join(data_directory, class_name)
    for filename in os.listdir(class_directory):
        img = plt.imread(os.path.join(class_directory, filename))
        images.append(img)
        labels.append(class_index)

X = np.array(images)
y = np.array(labels)

X = X / 255.0

y = to_categorical(y, num_classes)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(100, 100, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")


import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
data_directory = "train_resized100"
class_names = sorted(os.listdir(data_directory))
num_classes = len(class_names)
images = []
labels = []
for class_index, class_name in enumerate(class_names):
    class_directory = os.path.join(data_directory, class_name)
    for filename in os.listdir(class_directory):
        img = cv2.imread(os.path.join(class_directory, filename))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        images.append(img.flatten())
        labels.append(class_index)

X = np.array(images)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
svm_model = SVC(kernel='linear', C=1.0)
svm_model.fit(X_train, y_train)
joblib.dump(svm_model, 'svm_model.pkl')
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

