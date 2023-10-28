from pydantic import BaseModel


class Layer(BaseModel):
    size: int
    activation: str = "sigmoid"
    label: str | None = None
