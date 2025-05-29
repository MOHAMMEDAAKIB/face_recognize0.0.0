from pydantic import BaseModel

class AddUserRequest(BaseModel):
    name: str

class RecognizeResponse(BaseModel):
    identity: str
    distance: float
