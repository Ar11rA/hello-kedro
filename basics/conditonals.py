def conditionals(income):
    if income <= 10000:
        print('poor')
    elif 10000 < income <= 500000:
        print('lower middle class')
    elif 500000 < income <= 1000000:
        print('upper middle class')
    else:
        print('rich')


conditionals(5000)
conditionals(12321312312)
conditionals(400000)
conditionals(800000)


# function to explain match case
def match_case(character):
    match character:
        case 'A|B':
            print('poor')
        case 'C|D|E':
            print('lower middle class')
        case 'F|G|H|I|J|K|L|M|N|O|P|Q|R|S|T|U|V|W':
            print('upper middle class')
        case _:
            print('rich')


match_case('A')
match_case('Z')
match_case('D')
match_case('G')

# function to calulate tax in india
# args: income, investment
def calculate_tax(income, investment):
    taxable_income = income - investment
    if taxable_income <= 250000:
        cumulated_tax = 0
    elif 250000 < taxable_income <= 500000:
        cumulated_tax = 0.05 * (taxable_income - 250000)
    elif 500000 < taxable_income <= 1000000:
        cumulated_tax = 0.2 * (taxable_income - 500000) + 12500
    else:
        cumulated_tax = 0.3 * (taxable_income - 1000000) + 112500
    return cumulated_tax


tax = calculate_tax(1000000, 100000)
print(tax)
tax = calculate_tax(300000, 100000)
print(tax)
tax = calculate_tax(100000, 100000)
print(tax)
tax = calculate_tax(800000, 100000)
print(tax)
tax = calculate_tax(2500000, 100000)
print(tax)

# abccba
