#%%

class test():
    def __init__(self):
        self.a = 1
        self.b = 2
        
    def __call__(self, inputs):
        self.call(inputs)
        
        

A = test()
#%%
A()

#%%
class testb(test):
    def __init__(self):
        super().__init__()
        
    def call(self, inputs):
        print(inputs)


B = testb()

#%%
B("1")



#%%
from itertools import combinations, product

wid = [1,2,3]
dep = [5,6,7]
a = [wid,dep]

perm = list(product(*a)) 
perm