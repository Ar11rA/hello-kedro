# function
def double_number(n) -> int:
    result = n * 2
    return result


def add_numbers(a, b) -> int:
    result = a + b
    return result


sum = add_numbers(1, 2)

print("double of sum is", double_number(sum))
print(double_number(2))
print(add_numbers(2, 4))

# function to convert celsius to fahrenheit and kelvin
# Kelvin = Celsius + 273.15
# Fahrenheit = Celsius * 1.80 + 32.00