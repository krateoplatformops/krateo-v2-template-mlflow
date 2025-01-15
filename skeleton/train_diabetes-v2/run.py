import argparse
import azureml.core
from azureml.core import Workspace, Experiment
import logging, os
from azureml.core.authentication import ServicePrincipalAuthentication

print("SDK version:", azureml.core.VERSION)

logging.getLogger("azure").setLevel(logging.DEBUG)

# load .env file without dotenv package
from dotenv import load_dotenv
load_dotenv()

#Enter details of your AzureML workspace
subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
resource_group = os.getenv("AZURE_RESOURCE_GROUP")
workspace_name = os.getenv("AZURE_WORKSPACE_NAME")

if subscription_id is None:
    raise ValueError("Set AZURE_SUBSCRIPTION_ID environment variable")
if resource_group is None:
    raise ValueError("Set AZURE_RESOURCE_GROUP environment variable")
if workspace_name is None:
    raise ValueError("Set AZURE_WORKSPACE_NAME environment variable")

cli_auth = ServicePrincipalAuthentication(
    tenant_id=os.getenv("AZURE_TENANT_ID"),
    service_principal_id=os.getenv("AZURE_CLIENT_ID"),
    service_principal_password=os.getenv("AZURE_CLIENT_SECRET"),
)

ws = Workspace.get(name=workspace_name,
                   subscription_id=subscription_id, 
                   resource_group=resource_group,
                   auth=cli_auth)

from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--alpha', type=float, default=0.03, help='Alpha value for Ridge regression')
parser.add_argument('--experiment_name', type=str, help='Experiment name', required=True)
parser.add_argument('--run_id', type=str, help='Run ID', required=True)
args = parser.parse_args()

# Choose a name for your CPU cluster
cluster_name = "cpu-cluster"

# Verify that cluster does not exist already
try:
    cpu_cluster = ComputeTarget(workspace=ws, name=cluster_name)
    print("Found existing cpu-cluster")
except ComputeTargetException:
    print("Creating new cpu-cluster")
    
    # Specify the configuration for the new cluster
    compute_config = AmlCompute.provisioning_configuration(vm_size="STANDARD_D2_V2",
                                                           min_nodes=0,
                                                           max_nodes=2)

    # Create the cluster with the specified name and configuration
    cpu_cluster = ComputeTarget.create(ws, cluster_name, compute_config)
    
    # Wait for the cluster to complete, show the output log
    cpu_cluster.wait_for_completion(show_output=True)


from azureml.core import Experiment
from azureml.core import Run

exp = Experiment(workspace=ws, name=args.experiment_name)
run = Run(exp, run_id=args.run_id)

training_script = 'train_diabetes.py'
with open(training_script, 'r') as f:
    print(f.read())


from azureml.core import Environment
from azureml.core.conda_dependencies import CondaDependencies

env = Environment(name="mlflow-env")

# Specify conda dependencies with scikit-learn and temporary pointers to mlflow extensions
cd = CondaDependencies.create(
    pip_packages=["azureml-mlflow", "scikit-learn", "matplotlib", "pandas", "numpy", "protobuf==5.28.3"]
    )

env.python.conda_dependencies = cd

from azureml.core import ScriptRunConfig

src = ScriptRunConfig(source_directory=".",
                      arguments=["--alpha", args.alpha, "--run_id", args.run_id],
                      script=training_script,
                      compute_target=cpu_cluster,
                      environment=env)

import mlflow 
#   # Get the current tracking uri
# tracking_uri = mlflow.get_tracking_uri()
print(f"Current tracking uri: {ws.get_mlflow_tracking_uri()}")

run = exp.submit(src)
run.wait_for_completion(show_output=True)