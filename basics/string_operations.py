# function to take 2 strings, reverses the first one, uppercases the second one
# and returns (first half of  string A + last half string B)
# e.g. abcd, efgh -> dcGH
# dcba
# EFGH
# dcGH

def form_final_string(str1, str2):
    str1 = str1[::-1]
    str2 = str2.upper()
    return str1[:len(str1) // 2] + str2[len(str2) // 2:]


print(form_final_string("swati", "aritra"))
print(form_final_string("abcd", "efgh"))