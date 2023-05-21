import json

x = '''
{
  "subnetIds": {
    "uat": [
      "abc",
      "def",
      "ghij"
    ],
    "prod": [
      "ijks",
      "lmn"
    ],
    "dev": [
      "opq",
      "rstd",
      "uvw"
    ]
  },
  "threshold": 256
}
'''

event = json.loads(x)

print(event['subnetIds'])
print(event['subnetIds'].keys())

result = {}
should_send_mail = False


def calc_free_subnets(subnets, threshold=4):
    arr = []
    for subnet in subnets:
        if len(subnet) == threshold:
            arr.append(subnet)
    return arr


for key in event['subnetIds'].keys():
    result[key] = calc_free_subnets(event['subnetIds'][key])
    if len(result[key]) > 0:
        should_send_mail = True

print(result)

print(json.dumps(result))
