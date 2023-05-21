"""
This is a boilerplate pipeline 'games'
generated using Kedro 0.18.4
"""

from kedro.pipeline import Pipeline, node, pipeline

from . import nodes as fn

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=fn.clean_data,
            inputs=["example_games_data"],
            outputs="games",
            name="clean",
        ),
        node(
            func=fn.linear_regression,
            inputs=[],
            outputs="lin_reg_accuracy_result",
            name="linear_regression",
        ),
        node(
            func=fn.logistic_regression,
            inputs=[],
            outputs="log_reg_accuracy_result",
            name="logistic_regression",
        ),
        node(
            func=fn.knn,
            inputs=[],
            outputs="knn_accuracy_result",
            name="knn",
        ),
        node(
            func=fn.decision_tree,
            inputs=[],
            outputs="decision_tree_result",
            name="dt",
        ),
        node(
            func=fn.random_forest,
            inputs=[],
            outputs="random_forest_result",
            name="rf",
        )
    ])
