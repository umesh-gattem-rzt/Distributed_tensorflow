import tensorflow as tf
import rztutil
import Utils

# class Sample(object):
#     def __init__(self):
#         self.train_data, self.train_label, self.test_data, self.test_label = rztutils.read_csv('mnist.csv',
#                                                                                                split_ratio=80,
#                                                                                                delimiter=";",
#                                                                                                output_label=True,
#                                                                                                label_vector=True)
#
#         print(len(self.train_data))
#         print()
#         self.train_data = Utils.split(self, 3, [50, 20, 30], self.train_data)
#         data = Utils.get_train_data(self, i=0)
#         print(len(data))
#
#
# if __name__ == '__main__':
#     model = Sample()


train_data, train_label, test_data, test_label = rztutil.read_csv('mnist.csv',
                                                                  split_ratio=80,
                                                                  delimiter=";",
                                                                  output_label=True,
                                                                  label_vector=True)

print(len(train_data))
model = Utils()
model.split(3, [50, 20, 30], train_data)
data = model.get_train_data(i=0)
print(len(data))
