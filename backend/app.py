import json
import typing
import pathlib
import pydantic
import random

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
    confidence : float = pydantic.Field(...)

class PredictionRespone(pydantic.BaseModel):
    p_n2 : SinglePrediction
    p_he : SinglePrediction
    p_h2 : SinglePrediction
    p_o2 : SinglePrediction
    p_ch4 : SinglePrediction
    p_co2 : SinglePrediction

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["digimatter.ru"],  # In production, replace "*" with your actual frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

prediction_metadata_path = (pathlib.Path(__file__) / '..' / '..' / 'data' / 'prediction_metadata.json').resolve()
model_checkpoint_path = (pathlib.Path(__file__) / '..' / '..' / 'data' / 'bnn_on_full_data_revised.pt').resolve()

NUM_PRED_SAMPLES : int = 100

prediction_metadata : typing.Dict[str, typing.Dict[str, float]] = None
with open(prediction_metadata_path, 'r') as fp:
    prediction_metadata = json.load(fp)

generator = rdFingerprintGenerator.GetMorganGenerator(
    radius=2, 
    fpSize=1024, 
)

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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

    seed_everything(42)

    print(f"Got request f{request.model_dump()}")

    smiles = request.smiles

    if "|" in smiles:
        smiles = smiles[:smiles.find('|')].strip()
        print(f"Find pipe in SMILES, fixed to {smiles}")

    if Chem.MolFromSmiles(smiles) is None:
        raise HTTPException(status_code=400, detail="Invalid SMILES!")

    fps = generator.GetCountFingerprintAsNumPy(Chem.MolFromSmiles(smiles))
    cur_x = torch.tensor(fps, dtype=torch.float32).view(1, -1)
    
    preds = {task : [] for task in prediction_metadata.keys()}
    preds_mean, preds_confidence = {}, {}
    with torch.inference_mode():
        for task in prediction_metadata.keys():
            for _ in range(NUM_PRED_SAMPLES):
                preds[task].append(
                    model(cur_x, task, sample=True).item()
                )
            preds_mean[task] = np.mean(preds[task]) * prediction_metadata[task]['target_std'] + prediction_metadata[task]['target_mean']
            preds_confidence[task] = np.std(preds[task]) * prediction_metadata[task]['target_std'] + prediction_metadata[task]['target_mean']

            preds_confidence[task] = np.clip(1 - np.log(preds_confidence[task] / prediction_metadata[task]['target_median_pred_std']) / np.log(2), 0, 1) 

    return PredictionRespone(
        p_n2=SinglePrediction(
            mean=preds_mean['inhs_p(n2)'],
            std=preds_confidence['inhs_p(n2)'],
        ),
        p_he=SinglePrediction(
            mean=preds_mean['inhs_p(he)'],
            std=preds_confidence['inhs_p(he)'],
        ),
        p_h2=SinglePrediction(
            mean=preds_mean['inhs_p(h2)'],
            std=preds_confidence['inhs_p(h2)'],
        ),
        p_o2=SinglePrediction(
            mean=preds_mean['inhs_p(o2)'],
            std=preds_confidence['inhs_p(o2)'],
        ),
        p_ch4=SinglePrediction(
            mean=preds_mean['inhs_p(ch4)'],
            std=preds_confidence['inhs_p(ch4)'],
        ),
        p_co2=SinglePrediction(
            mean=preds_mean['inhs_p(co2)'],
            std=preds_confidence['inhs_p(co2)'],
        ),
    )

@app.get('/status')
def status() -> int:
    return 0
