import json
import typing
import pathlib
import pydantic

import numpy as np

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from fastapi import FastAPI
from fastapi.exceptions import HTTPException
from fastapi.middleware.cors import CORSMiddleware

import torch
from backend.model import MTLBNNRegressor

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

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace "*" with your actual frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

prediction_metadata_path = (pathlib.Path(__file__) / '..' / '..' / 'data' / 'prediction_metadata.json').resolve()
model_checkpoint_path = (pathlib.Path(__file__) / '..' / '..' / 'data' / 'bnn_mtl_on_full_data.pt').resolve()

NUM_PRED_SAMPLES : int = 100

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

@app.post('/predict')
def predict(
    request : PredictionRequest
) -> PredictionRespone:

    if Chem.MolFromSmiles(request.smiles) is None:
        raise HTTPException(status_code=400, detail="Invalid SMILES!")

    fps = generator.GetCountFingerprintAsNumPy(Chem.MolFromSmiles(request.smiles))
    cur_x = torch.tensor(fps, dtype=torch.float32).view(1, -1)
    
    preds = {task : [] for task in prediction_metadata.keys()}
    preds_mean, preds_std = {}, {}
    with torch.inference_mode():
        for task in prediction_metadata.keys():
            for _ in range(NUM_PRED_SAMPLES):
                preds[task].append(
                    model(cur_x, task, sample=True).item()
                )
            preds_mean[task] = np.mean(preds[task]) * prediction_metadata[task]['target_std'] + prediction_metadata[task]['target_mean']
            preds_std[task] = np.std(preds[task]) * prediction_metadata[task]['target_std'] + prediction_metadata[task]['target_mean']

            preds_std[task] = max(0, np.sqrt(preds_std[task]**2 - prediction_metadata[task]['target_min_pred_std']**2))

            # anti box-cox
            preds_std[task] = np.exp(preds_std[task]) - 1
            preds_mean[task] = np.exp(preds_mean[task]) - 1

    return PredictionRespone(
        p_n2=SinglePrediction(
            mean=preds_mean['inhs_p(n2)'],
            std=preds_std['inhs_p(n2)'],
        ),
        p_he=SinglePrediction(
            mean=preds_mean['inhs_p(he)'],
            std=preds_std['inhs_p(he)'],
        ),
        p_h2=SinglePrediction(
            mean=preds_mean['inhs_p(h2)'],
            std=preds_std['inhs_p(h2)'],
        ),
        p_o2=SinglePrediction(
            mean=preds_mean['inhs_p(o2)'],
            std=preds_std['inhs_p(o2)'],
        ),
        p_ch4=SinglePrediction(
            mean=preds_mean['inhs_p(ch4)'],
            std=preds_std['inhs_p(ch4)'],
        ),
        p_c2h6=SinglePrediction(
            mean=preds_mean['inhs_p(c2h6)'],
            std=preds_std['inhs_p(c2h6)'],
        ),
        p_c3h8=SinglePrediction(
            mean=preds_mean['inhs_p(c3h8)'],
            std=preds_std['inhs_p(c3h8)'],
        ),
        p_c4h10=SinglePrediction(
            mean=preds_mean['inhs_p(c4h10)'],
            std=preds_std['inhs_p(c4h10)'],
        ),
    )

@app.get('/status')
def status() -> int:
    return 0
