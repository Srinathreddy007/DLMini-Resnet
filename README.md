# DL-MiniProject
<h2>Running the Code on the HPC</h2> 
Follow these steps to run the code on a High-Performance Computing (HPC) cluster.

### 1. Navigate to the Scratch Folder
Create a directory in the `/scratch/your_net_id/` folder. Replace `your_net_id` with your actual net ID.

```bash
cd /scratch/your_net_id/

```
### 2. Clone the Git Repo
Clone the repository using the following command: 
``` bash
git clone https://github.com/Srinathreddy007/DLMini-Resnet.git
```

### 3. Create a Conda Environment
Create a conda environment in the scratch folder and initialize it:
```bash
conda create --name /scratch/your_net_id/ENV_NAME python=3.9
conda init
```

Navigate to the project folder:
```bash
cd /scratch/your_net_id/DLMini-Resnet/
```
### 4. Modify the .sbatch File
There is an `.sbatch` file in the repository that needs to be modified to run the code on the HPC cluster. Open it using a text editor like vim or a command-line editor.

Change the line `--mail-user ` to receive updates about the model status:
```bash
#SBATCH --mail-user=netid@nyu.edu
```
In the `.sbatch` file change the path to navigate to the folder:
```bash
cd /scratch/your_net_id/DLMini-Resnet
```
Replace the line that activates the environment:
```bash
source activate /scratch/your_net_id/ENV_NAME
```
If you wish to change the name of the model, you can do so, else leave it.

Save the changes to the  `.sbatch` file.

### 5. Submit the Job
Submit the `.sbatch` file for running:
```bash
sbatch submit.sbatch
```
Ensure that you have appropriate permissions and resources allocated on the HPC cluster before submitting the job.

The required libraries will be downloaded when you submit the `.sbatch` file. It contains a line `pip insall -r requirements.txt` that takes care of the necessary downloads. 

`Note: Replace your_net_id and ENV_NAME with your actual Net ID and environment name respectively.`

## Running the Code on the Local System
### 1. Install the anaconda 2020.07 version to run without compatibility issues. 
### 2. Clone the Git Repo
Clone the repository using the following command: 
``` bash
git clone https://github.com/Srinathreddy007/DLMini-Resnet.git
```
### 3. Create a Conda Environment
Navigate to destination folder. Create the conda environment using the `.yml` file provided in the repo and activate the environment.
```bash
conda create --name /path/to/your/folder/ENV_NAME --file gpu_env.yml
conda activate /path/to/your/folder/ENV_NAME 
```

### 4. Install Necessary Libraries
Run the command. The `requirements.txt` file is also provided in the repo. 
```bash
pip install -r requirements.txt
```

### 5. Run the Code:
Run the `main.py` file using the command:
```bash
python main.py --batch_size_train=64 --num_blocks='[3, 4, 6, 3]' --channel_size='[64, 96, 128, 188]' --he_init=True --epochs=250 --learning_rate=0.01 --weight_decay=5e-04 --model_name=18_he_BN_188
```

 `Don't change the command-line arguments, as they are the parameters set for replicating the best model.`






