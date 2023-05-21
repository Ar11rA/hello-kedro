from kedro.io import DataCatalog, MemoryDataSet
from kedro.pipeline import node, Pipeline
from kedro.runner import SequentialRunner

from api.models.sample import Sample


# Prepare first node
def return_greeting(msg):
    return "Hello " + msg

# Prepare second node
def join_statements(greeting):
    return f"{greeting} Kedro!"

def run(sample: Sample):
    return_greeting_node = node(return_greeting,
                                inputs="my_base",
                                outputs="my_salutation")
    join_statements_node = node(join_statements,
                                inputs="my_salutation",
                                outputs="my_message")
    pipeline = Pipeline([return_greeting_node, join_statements_node])
    data_catalog = DataCatalog({
        "my_salutation": MemoryDataSet(), "my_base": MemoryDataSet(sample.name)
    })

    # Create a runner to run the pipeline
    runner = SequentialRunner()
    # Run the pipeline
    return runner.run(pipeline, data_catalog)