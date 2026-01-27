# Environment Setup

There are several ways to install Python and set up your computing environment. Here, I share my personal preferences.

The uv package has gained significant traction as it provides a faster and more efficient way to install packages and resolve dependencies

&nbsp;
## Python Setup

This section guides you through the Python setup and package installation procedure using `uv` via its `uv pip` interface. 

### 1. Install Python (if not already installed)

Make sure you have Python installed on your system. You can download it from the official website: [https://www.python.org/downloads/](https://www.python.org/downloads/)

To check if Python is installed, open your terminal or command prompt and run:

```bash
python --version
```
or
```bash
python3 --version
```
If it returns 3.10 or newer, no further action is required.

&nbsp;
### 2. Create a Virtual Environment

I highly recommend installing Python packages in a seperate virtual environment to avoid modifying the system-wide packages that your OS may depend on. 

To create a virtual environment, navigate to your project directoy and follow these stesps

**Install uv**

```bash
pip install uv
```

**Create a virtual environment**

```bash
uv venv --python=python3.12
```
**Activate the virtual environment**
```bash
.\venv\Scripts\activate
```

&nbsp;
> **Note**
> You need to activate the virtual environment each time you start a new terminal session. For example, if you restart your terminal or computer and want to continue working on project the next day, simply run `.\venv\Scripts\activate` in project directory to reactivate your virtual environment.

&nbsp;
### 3. Install Required Packages

To install all required packages from a requirements.txt file) run the following command, assuming the file is in the same directory as your terminal session

```bash
uv pip install -r requirements.txt
```

&nbsp;
### Finalizing the setup

That's it! Your enivornment should now be ready for running the code in the repository.

**Start working with the code**

Once everything is set up, you can start working with the code files. For instance, lunch JupyerLab by running

```bash
jupyter lab
```
