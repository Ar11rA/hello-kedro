from pydantic import BaseModel

class IrisParams(BaseModel):
    train_fraction: float
    random_state: int
    target_column: str


