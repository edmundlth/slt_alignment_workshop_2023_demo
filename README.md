[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/edmundlth/slt_alignment_workshop_2023_demo/main)

# slt_alignment_workshop_2023_demo
Code and notebooks for demonstrating Singular Learning Theory related computational experiments. 

# Binder image
A docker image of the main branch is built with [binder](https://mybinder.org/) and can be launched using the link above. That will allow you to run the included notebooks on a server hosted by [JupyterHub](https://jupyterhub.readthedocs.io/en/latest/)

# Local installation Guide
 1. Clone repository
 ```
 git clone https://github.com/edmundlth/slt_alignment_workshop_2023_demo.git
 ```
 2. Create Python virtual environment and install required packages specified in `Pipfile.lock` using [pipenv](https://pipenv.pypa.io/en/latest/)
 ```
 cd slt_alignment_workshop_2023_demo
 pipenv install 
 ```
or if you don't have `pipenv`, just create your own virtual environment and install the required packages from `requirement.txt`
```
# ... create you own virtualenv and activates it. 
pip install -r requirement.txt
```

 3. Run `jupyter` within virtual environment
 ```
 pipenv run jupyter notebook
 ```


