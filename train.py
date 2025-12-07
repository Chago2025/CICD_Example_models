import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from mlflow import MlflowClient

# MLflow experiment
mlflow.set_tracking_uri("http://127.0.0.1:5000")
experiment_name = "IrisCIExperiment"
mlflow.set_experiment(experiment_name)

with mlflow.start_run() as run:
    # Datos
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    
    # Modelo
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    
    # Métricas
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    mlflow.log_metric("accuracy", acc)
    
    # Loggear modelo
    mlflow.sklearn.log_model(model, artifact_path="model")
    
    print(f"🏁 Accuracy: {acc}")

# Registrar en Model Registry
client = MlflowClient()
model_name = "IrisCICDModel"
run_id = run.info.run_id
model_uri = f"runs:/{run_id}/model"

try:
    client.create_registered_model(model_name)
except mlflow.exceptions.RestException:
    pass  # Ya existe

mv = client.create_model_version(
    name=model_name,
    source=model_uri,
    run_id=run_id,
    description="Modelo subido por CI/CD"
)

# Ahora decidir si promovemos basado en accuracy
if acc >= 0.90:
    client.transition_model_version_stage(
        name=model_name,
        version=mv.version,
        stage="Production",
        archive_existing_versions=True
    )
    print(f"🎯 Modelo versión {mv.version} promovido a Production (accuracy={acc:.2f}).")
else:
    client.transition_model_version_stage(
        name=model_name,
        version=mv.version,
        stage="Staging",
        archive_existing_versions=False
    )
    print(f"⚠️ Modelo versión {mv.version} enviado a Staging (accuracy={acc:.2f}).")
