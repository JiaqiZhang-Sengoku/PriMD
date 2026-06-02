import os


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATASET_ROOT = os.environ.get("PRIMD_DATASET_ROOT", os.path.join(PROJECT_ROOT, "dataset"))
SAVED_ROOT = os.environ.get("PRIMD_SAVED_ROOT", os.path.join(PROJECT_ROOT, "saved"))


DATA_DIR = {
    "CMUMOSI": os.path.join(DATASET_ROOT, "CMUMOSI"),
    "CMUMOSEI": os.path.join(DATASET_ROOT, "CMUMOSEI"),
    "IEMOCAPSix": os.path.join(DATASET_ROOT, "IEMOCAP"),
    "IEMOCAPFour": os.path.join(DATASET_ROOT, "IEMOCAP"),
}


PATH_TO_FEATURES = {
    name: os.path.join(path, "features")
    for name, path in DATA_DIR.items()
}


PATH_TO_LABEL = {
    "CMUMOSI": os.path.join(DATA_DIR["CMUMOSI"], "CMUMOSI_features_raw_2way.pkl"),
    "CMUMOSEI": os.path.join(DATA_DIR["CMUMOSEI"], "CMUMOSEI_features_raw_2way.pkl"),
    "IEMOCAPSix": os.path.join(DATA_DIR["IEMOCAPSix"], "IEMOCAP_features_raw_6way.pkl"),
    "IEMOCAPFour": os.path.join(DATA_DIR["IEMOCAPFour"], "IEMOCAP_features_raw_4way.pkl"),
}


MODEL_DIR = os.path.join(SAVED_ROOT, "model")
LOG_DIR = os.path.join(SAVED_ROOT, "log")
NPZ_DIR = os.path.join(SAVED_ROOT, "npz")


for path in [MODEL_DIR, LOG_DIR, NPZ_DIR]:
    os.makedirs(path, exist_ok=True)
