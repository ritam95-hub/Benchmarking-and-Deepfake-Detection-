from src.models.xception_model import build_xception
from src.training.trainer import Trainer

model = build_xception()

trainer = Trainer(model)

trainer.compile()

trainer.fit(
    X_train,
    y_train
)

model.save(
    "checkpoints/xception/model.h5"
)