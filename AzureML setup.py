def initial_setup(subscription_id,resource_group,workspace_name,location,compute_name,vm_size,env_name):
    from azureml.core import Workspace

# create the Workspace
    ws = Workspace.create(workspace_name,
                             subscription_id=subscription_id,
                             resource_group=resource_group,
                             location=location
)

    ws.write_config(path='.azureml')

    from azureml.core.compute import AmlCompute, ComputeTarget



# define the cluster and the max and min number of nodes
    provisioning_config = AmlCompute.provisioning_configuration(vm_size = vm_size,
                                                            min_nodes = 0,
                                                            max_nodes = 10)
# create the cluster
    compute_target = ComputeTarget.create(ws, compute_name, provisioning_config)

    from azureml.core.environment import Environment
    from azureml.core.conda_dependencies import CondaDependencies

# create an enviornment
    env = Environment(name=env_name)

# define packages for image
    cd = CondaDependencies.create(pip_packages=['azureml-dataset-runtime[pandas,fuse]',
                                                'azureml-defaults',
                                                'Pillow','tensorflow',
                                                'tifffile','numpy','matplotlib'],
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

subscription_id = '0761ae77-61d2-4602-a658-66b9812c0323'
resource_group = 'image_seg_training_resource_10'
workspace_name = 'image_seg_training_workspace_10'
# set to the location you used in your quota request:
location = 'eastus'
compute_name = "gpu-cluster-NC6"
vm_size = 'Standard_NC6_Promo'
env_name='waste-env-gpu'