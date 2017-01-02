class Utils(object):
    """
    Utils Class
    """

    def split(self, no_of_workers, split_ratio, csv_data):
        if no_of_workers != len(split_ratio):
            raise Exception("No of workers and split ratio should be equal")
        if sum(split_ratio) != 100:
            raise Exception("Split ratio should be total 100 percent")
        self.data = [0 for i in range(no_of_workers)]
        start = 0
        for i in range(no_of_workers):
            start = start
            end = start + int(split_ratio[i] / 100 * len(csv_data))
            self.data[i] = csv_data[start:end]
            start = end
        pass

    def get_train_data(self, index):
        train_data = self.data[index]
        return train_data

    def get_train_label(self, index):
        train_label = self.data[index]
        return train_label
