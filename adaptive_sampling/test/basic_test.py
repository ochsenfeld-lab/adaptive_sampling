import numpy as np

list =[]
for i in range(0, len(list)+1,1):
    list = list + [1.0]
    print(list)
    print(len(list))

a = np.array([2,3,4])
b = np.array([2,3,4])
print(np.sqrt(np.sum(np.square((a - b)/a))))

class Test():
    def __init__(self,dimension):
        self.dimension = dimension

Test1 = Test(dimension=5)
print(Test1.dimension)

print(6/2/3)