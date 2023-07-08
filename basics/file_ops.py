import pandas_series_examples as pd
import xml.etree.ElementTree as ET


# read txt file
def read(file):
    with open(file, 'r') as f:
        print(f.read())


read('basics/resources/sample.txt')


# read csv file (using pandas)
def read_csv(file):
    df = pd.read_csv(file)
    print(df)


read_csv('basics/resources/sample.csv')


# read json file (using pandas)
def read_json(file):
    df = pd.read_json(file, typ='dictionary')
    print(df)


read_json('basics/resources/sample.json')


# read xml file
def read_xml(file):
    tree = ET.parse(file)
    root = tree.getroot()

    # Access elements and attributes
    for child in root:
        print(f"Element: {child.tag}, Attribute: {child.attrib}, value: {child.text}")
        for subchild in child:
            print(f"  Subelement: {subchild.tag}, Text: {subchild.text}")


read_xml('basics/resources/sample.xml')


# write file
def write_file(file, text, mode):
    with open(file, mode) as f:
        f.write(text)


write_file('basics/resources/sample.txt', 'Hello World 1\n', 'w')
write_file('basics/resources/sample.txt', 'Hello World 2\n', 'a')
write_file('basics/resources/sample.txt', 'Hello World 3\n', 'a')
