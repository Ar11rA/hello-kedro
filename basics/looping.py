# loop through list various ways
def loop_through_list(arr: list):
    for item in arr:
        print("normal loop", item)
    for i in range(len(arr)):
        print("indexed loop", arr[i])
    ctr = 0
    while ctr < len(arr):
        print("while loop", arr[ctr])
        ctr += 1
    for i, item in enumerate(arr):
        print("enumerate loop", i, item)
    [print("single line loop", item) for item in arr]


loop_through_list([1, 2, 3, 4, 5])


# loop through dictionary various ways
def loop_through_dict(d: dict):
    for key in d:
        print("normal loop", key, d[key])
    for key, value in d.items():
        print("items loop", key, value)
    for key in d.keys():
        print("keys loop", key)
    for value in d.values():
        print("values loop", value)


loop_through_dict({'name': 'Swati', 'age': 10})