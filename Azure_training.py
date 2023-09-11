# Run this code to create a new resource group with a new workspace and upload data to the new WS.
from azureml.core import Workspace

subscription_id = '0761ae77-61d2-4602-a658-66b9812c0323'
resource_group = 'seg_resource_re-train_previous_new_6'
workspace_name = 'seg_ws_RE-TRAINING_6'


location='eastus' # set to the location you used in your quota request

# create the Workspace
ws = Workspace.create(workspace_name,
                             subscription_id=subscription_id,
                             resource_group=resource_group,
                             location=location
)

ws.write_config(path='.azureml')

from azureml.core.compute import AmlCompute, ComputeTarget

# the name we are going to use to reference our cluster
# compute_name = "image-segmentation-gpu"
compute_name = "gpu-cluster-NC-16"

# the azure machine type
# vm_size = 'Standard_NC6_Promo'
# vm_size = "Standard_NC6_v3"
#vm_size = "STANDARD_NC6S_V3"
vm_size = "Standard_NC16as_T4_v3"
#vm_size="Standard_NCS_v3"
# define the cluster and the max and min number of nodes
provisioning_config = AmlCompute.provisioning_configuration(vm_size = vm_size,
                                                            min_nodes = 0,
                                                            max_nodes = 10)
# create the cluster
compute_target = ComputeTarget.create(ws, compute_name, provisioning_config)

from azureml.core.environment import Environment
from azureml.core.conda_dependencies import CondaDependencies

# create an enviornment
env = Environment(name='seg-env-gpu')

# define packages for image
cd = CondaDependencies.create(pip_packages=['azureml-dataset-runtime[pandas,fuse]',
                                            'azureml-defaults',
                                            'Pillow','tensorflow',
                                            'tifffile','numpy','matplotlib','scikit-image'],
                             conda_packages=['SciPy'])

env.python.conda_dependencies = cd

# Specify a docker image to use.
# env.docker.base_image = (
#     "mcr.microsoft.com/azureml/minimal-ubuntu18.04-py37-cuda11.0.3-gpu-inference:latest"
# )

# Register environment to re-use later
env = env.register(workspace = ws)


datastore = ws.get_default_datastore()

# upload the data to the datastore
datastore.upload(src_dir='C:/Users/DavidS10/PycharmProjects/pythonProject/image_classification',
                 target_path='/data/',
                 overwrite=False,
                 show_progress=True)

from azureml.core import Dataset

# create the dataset object
dataset = Dataset.File.from_files(path=(datastore, '/data'))

# register the dataset for future use
dataset = dataset.register(workspace=ws,
                           name='input_images',
                           description='input_images_for_training')