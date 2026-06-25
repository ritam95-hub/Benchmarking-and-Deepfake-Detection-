MODELS = [
    "Xception",
    "PatchResNet",
    "EfficientNetB0",
    "MesoNet"
]

for model_name in MODELS:

    train_model(model_name)

    evaluate_model(model_name)

    measure_latency(model_name)

    save_results(model_name)