import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as metrics

if __name__ == '__main__':
  # Load data
  (train_data, train_label), (test_data, test_label) = mnist.load_data()

  # Reshape data
  train_data = np.reshape(train_data, (60000, 28 * 28))
  test_data = np.reshape(test_data, (10000, 28 * 28))
  train_data, test_data = train_data / 255.0, test_data / 255.0

  # Build logistic regression model
  log_reg = LogisticRegression(
    solver='saga',
    multi_class='multinomial',
    max_iter=100,
    verbose=2
  )

  # Train the model
  log_reg.fit(train_data, train_label)

  # Predict the result of test data
  predition = log_reg.predict(test_data)

  # Show the accuracy and confusion matrix
  print(metrics.accuracy_score(test_label, predition))
  print(metrics.confusion_matrix(test_label, predition))