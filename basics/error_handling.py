# error handling divide by zero
def divide(x, y):
    try:
        result = x / y
        print("Result is", result)
    except ZeroDivisionError:
        print("Division by zero not allowed")


divide(10, 2)
divide(10, 0)

# error handling file not present
def file_read(file):
    try:
        with open(file, 'r') as f:
            print(f.read())
    except FileNotFoundError:
        print("File not found")


# error handling key not present
def key_not_present():
    d = {'name': 'Swati', 'age': 1}
    try:
        print(d['address'])
    except KeyError:
        print("Key not found")
