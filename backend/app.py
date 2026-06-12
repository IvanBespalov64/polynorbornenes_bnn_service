import json
import typing
import pathlib
import pydantic

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from fastapi import FastAPI

from model import MTLBNNRegressor

class PredictionRequest(pydantic.BaseModel):
    smiles : str = pydantic.Field(...)

class SinglePrediction(pydantic.BaseModel):
    mean : float = pydantic.Field(...)
    std : float = pydantic.Field(...)

class PredictionRespone(pydantic.BaseModel):
    p_n2 : SinglePrediction
    p_he : SinglePrediction
    p_h2 : SinglePrediction
    p_o2 : SinglePrediction
    p_ch4 : SinglePrediction
    p_c2h6 : SinglePrediction
    p_c3h8 : SinglePrediction
    p_c4h10 : SinglePrediction

import os
print(os.system('ls -lah'))
print(os.system('pwd'))

app = FastAPI()

prediction_metadata_path = (pathlib.Path(__file__) / '..' / 'data' / 'prediction_metadata.json').resolve()
model_checkpoint_path = (pathlib.Path(__file__) / '..' / 'data' / 'bnn_mtl_on_full_data.pt').resolve()

prediction_metadata : typing.Dict[str, typing.Dict[str, float]] = None
with open(prediction_metadata_path, 'r') as fp:
    prediction_metadata = json.load(fp)

generator = rdFingerprintGenerator.GetMorganGenerator(
    radius=2, 
    fpSize=1024, 
)

model = MTLBNNRegressor(
    task_names=prediction_metadata.keys(),
    input_dim=1024,
    hidden_dims=(64, 32),
    head_hidden_dim=32,
    activation='relu',
    norm_type='layernorm',
    use_residual=False,
)
model.load_state_dict(torch.load(model_checkpoint_path))
model.eval()

@app.get('/status')
def status() -> int:
    return 0
