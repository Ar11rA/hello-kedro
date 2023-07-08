list = [162, 181, 192, 185, 188, 186, 190]
sum = 0
for item in list:
    sum = sum + item
print(sum)
avarage_height = sum / len(list)
print(avarage_height)


def get_average_height(list):
    sum = 0
    for item in list:
        sum = sum + item
    return sum / len(list)


print(get_average_height(list))

students_marks = [20, 22, 46, 89, 56, 91, 89, 96, 21]
compare_value = 0
for individual_marks in students_marks:
    if individual_marks > compare_value:
        compare_value = individual_marks
print(compare_value)

# calculate sum of all even number in 1...100
even_number_sum = 0
for number in range(1, 10):
    if (number % 2) == 0:
        even_number_sum = even_number_sum + number

print(even_number_sum)
