from azureml.core import Workspace, Dataset, Environment

# Load the stored workspace
ws = Workspace.from_config()

# Get the registered dataset from azure
dataset = Dataset.get_by_name(ws, name='input_images')

## Try with our saved image
env = Environment.get(workspace=ws, name="seg-env-gpu")

# get our compute target
compute_target = ws.compute_targets["gpu-cluster-NC-16"]


from azureml.core import Experiment

# define the expiriment
#exp = Experiment(workspace=ws, name='image_segmentation_re-run_and_prediction_try')
exp = Experiment(workspace=ws, name='image_segmentation_prediction_previous_BS128')

from azureml.core import ScriptRunConfig

# setup the run details
src = ScriptRunConfig(source_directory='C:/Users/DavidS10/PycharmProjects/pythonProject/image_classification/code',
                      script='prediction.py',
                      arguments=['--data-path', dataset.as_mount()],
                      compute_target=compute_target,
                      environment=env)

# Submit the model to azure!
run = exp.submit(config=src)