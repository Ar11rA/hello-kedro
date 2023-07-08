# list
arr = [1, 2, 3, 4]
print(arr)
print(arr[2])
print(len(arr))
arr.append(5)
arr.reverse()
print(arr)

# tuple
tup = (1, 2)
print(tup)
print(tup[1])
tup = ("S", "Excellent", "Swati")
print(tup[2])
arr = [("S", "Excellent", "Swati"), ("S", "Excellent", "Sonali")]

# dictionary
d = {
    'name': 'Swati',
    'grade': 'S',
    'remarks': 'Excellent',
    'marks': 90,
    'age': 1
}

print(d)

# set
s = {1, 2, 3, 4, 3, 2, 2, 1, 1, 1, 1}
print(s)

# list comprehension
# [expression for item in iterable]
d = [i for i in range(10)]
print(d)
# [expression for item in iterable if condition]
d = [i for i in range(10) if i % 2 == 0]
print(d)
# [expression if condition else expression for item in iterable]
d = [i if i % 2 == 1 else i * 2 for i in range(10)]
print(d)


# custom classes
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def get_details(self):
        print(f'Name: {self.name}, Age: {self.age}')


Person('Swati', 10).get_details()
