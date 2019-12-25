from easydict import EasyDict as edict
d = edict()  # 这个是输出{}
d.foo = 3  # 我们可以直接赋值语句对字典元素进行创建
d.bar = {'prob':'value'}  # 另外我们也可以创建字典中的字典
d.bar.prob = 'newer'  # 另外我们也可以很方便的修改字典中元素的值
print(d)