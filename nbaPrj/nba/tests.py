#%%
from django.test import TestCase

# Create your tests here.

#%%
a = ''
len(a)
# %%
a = "['abcde', 'fghij']"
table = str.maketrans("[' ]", '    ')
a = a.translate(table).replace(' ','')
# %%
a.split(",")
# %%
a = [1,2]
print(type(a))
# %%
