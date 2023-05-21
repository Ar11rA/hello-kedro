from api.models.sample import Sample;

def get_hello():
    return {"message": "Hello World"}


def create_hello(sample: Sample):
    return {"message": sample.name}