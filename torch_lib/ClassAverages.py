import numpy as np
import os
import json
"""
Enables writing json with numpy arrays to file
"""


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


"""
Class will hold the average dimension for a class, regressed value is the residual
"""


class ClassAverages:
    def __init__(self, classes=[]):
        self.dimension_map = {}
        self.filename = os.path.abspath(os.path.dirname(__file__)) + os.path.sep + 'class_averages.txt'

        if len(classes) == 0:  # eval mode
            self.load_items_from_file()

        for detection_class in classes:
            class_ = detection_class.lower()
            if class_ in self.dimension_map.keys():
                continue
            self.dimension_map[class_] = {}
            self.dimension_map[class_]['count'] = 0
            self.dimension_map[class_]['total'] = np.zeros(3, dtype=np.double)

    def add_item(self, class_, dimension):
        class_ = class_.lower()
        # 不会出现类别不存在的错误。因为已经枚举了KITTI数据集中
        # 出现的所有的类别
        self.dimension_map[class_]['count'] += 1
        self.dimension_map[class_]['total'] += dimension
        # self.dimension_map[class_]['total'] /= self.dimension_map[class_]['count']

    # f返回特定类别的平均三维值，单位为米
    def get_item(self, class_):
        class_ = class_.lower()
        res = self.dimension_map[class_]['total'] / self.dimension_map[class_]['count']

        # array([1.73780633, 0.65711191, 0.84038649])
        # 即高为1.73m 宽为0.65 长为0.84
        return res

    def dump_to_file(self):
        f = open(self.filename, "w")
        f.write(json.dumps(self.dimension_map, cls=NumpyEncoder))
        f.close()

    def load_items_from_file(self):
        f = open(self.filename, 'r')
        dimension_map = json.load(f)

        for class_ in dimension_map:
            dimension_map[class_]['total'] = np.asarray(dimension_map[class_]['total'])

        self.dimension_map = dimension_map

    def recognized_class(self, class_):
        return class_.lower() in self.dimension_map
