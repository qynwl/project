import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D
from keras.utils import Sequence
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 划分训练集、验证集和测试集，并保存到csv文件中
data = pd.read_csv('D:\\CSV5\\processed_features.csv', encoding='ANSI')
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

train_data, test_data, train_label, test_label = train_test_split(data.iloc[:, 1:], data.label, test_size=0.2,
                                                                  random_state=42)
train_data, val_data, train_label, val_label = train_test_split(train_data, train_label, test_size=0.2, random_state=42)

train_data.to_csv('train_data.csv', index=False)
val_data.to_csv('val_data.csv', index=False)
test_data.to_csv('test_data.csv', index=False)
train_label.to_csv('train_label.csv', index=False)
val_label.to_csv('val_label.csv', index=False)
test_label.to_csv('test_label.csv', index=False)


# 将csv文件转换成深度学习平台所需要的输入格式
class DataGenerator(Sequence):
    def __init__(self, data_file, label_file, batch_size=32, shuffle=True):
        self.data = pd.read_csv(data_file, encoding='ANSI')
        self.label = pd.read_csv(label_file, encoding='ANSI')
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.data) / float(self.batch_size)))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__data_generation(indexes)
        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.data))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        X = np.zeros((len(indexes), 1000))
        y = np.zeros((len(indexes), 2))

        for i in range(len(indexes)):
            filename = self.data.iloc[indexes[i], 0]
            features = np.array(self.data.iloc[indexes[i], 1:].tolist())
            label = int(self.label.iloc[indexes[i], 0])

            X[i] = features
            y[i][label] = 1

        return X, y


# 构建CNN模型，并进行训练
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(1000, 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.25))
model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

train_generator = DataGenerator('train_data.csv', 'train_label.csv')
val_generator = DataGenerator('val_data.csv', 'val_label.csv')

history = model.fit(train_generator,
                    epochs=50,
                    validation_data=val_generator)

# 使用测试集评估模型性能
test_data = pd.read_csv('test_data.csv', encoding='ANSI')
test_label = pd.read_csv('test_label.csv', encoding='ANSI')

X_test = np.zeros((len(test_data), 1000))
y_test = np.zeros((len(test_data), 2))

for i in range(len(test_data)):
    features = np.array(test_data.iloc[i, :].tolist())
    label = int(test_label.iloc[i, 0])

    X_test[i] = features
    y_test[i][label] = 1

X_test = np.expand_dims(X_test, axis=2)

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

accuracy = accuracy_score(y_true_classes, y_pred_classes)
precision = precision_score(y_true_classes, y_pred_classes)
recall = recall_score(y_true_classes, y_pred_classes)
f1score = f1_score(y_true_classes, y_pred_classes)

print('Accuracy: {:.3f}'.format(accuracy))
print('Precision: {:.3f}'.format(precision))
print('Recall: {:.3f}'.format(recall))
print('F1-score: {:.3f}'.format(f1score))
