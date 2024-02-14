# Setup without Visual Studio Code
When using VS Code, simply press Ctrl+Shift+P and follow the [setup instructions](https://code.visualstudio.com/docs/python/environments).

Follow the steps below to set up the environment for the MLflow on your local machine.

## Initial Setup

1. **Open a Terminal** under your desired folder where you want to clone the repository.

2. **Clone the Repository:** Use the following command to clone the repository to your local machine. 
    ```
    git clone https://github.com/georgeh12/mlflow.git
    cd mlflow
    ```

## Setting Up a Virtual Environment
Requires git and python prior to running the code below.

1. **Create a Virtual Environment:** First, create a new virtual environment using Python's built-in `venv` module. Replace `myenv` with your preferred environment name.
    ```
    python -m venv myenv
    ```

2. **Activate the Virtual Environment:** The activation command depends on your operating system.

    - **Windows:**
        ```
        myenv\Scripts\activate
        ```

    - **MacOS & Linux:**
        ```
        source myenv/bin/activate
        ```

3. **Install Packages:** With the virtual environment activated, use `pip` to install the packages.
    ```
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    ```

4. **Verify Installation:** After installation, you can verify the installed packages and their versions using:
    ```
    pip list
    ```

## Installing Jupyter Notebook

1. **Install Jupyter:** With your virtual environment activated, install Jupyter Notebook using `pip`. This will install Jupyter Notebook locally within your virtual environment.
    ```
    pip install notebook
    ```

2. **Install Kernel for Virtual Environment:** To ensure that Jupyter uses your virtual environment as the kernel, you need to install `ipykernel` and create a kernel for your environment.
    ```
    pip install ipykernel
    python -m ipykernel install --user --name=myenv
    ```

3. **Launch Jupyter Notebook:** Once Jupyter is installed, you can start the Jupyter Notebook server by running:
    ```
    jupyter notebook
    ```

4. **Select the Kernel in Jupyter Notebook:** When you create or open a notebook in Jupyter, you can select the kernel corresponding to your virtual environment. Click on "Kernel" > "Change kernel" and select the kernel you named in the previous step.

## Deactivating the Virtual Environment

- **Deactivate the Virtual Environment:** Once you are done working in the virtual environment, you can deactivate it by simply running:
    ```
    deactivate
    ```

