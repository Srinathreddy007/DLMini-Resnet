# DL-MiniProject
<h2>Running the Code on the HPC</h2> 
Follow these steps to run the code on a High-Performance Computing (HPC) cluster.

### 1. Create a Directory in the Scratch Folder
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
conda create --name /scratch/your_net_id/ENV_NAME
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







