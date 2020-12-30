# Weblingual Server
This local daemon leverages recent progresses in machine learning and natural language processing to support the frontend chrome extension on various NLP tasks.

# Deployment
Because we're running full transformers in the server, if you're running it locally, you'll need a GPU with a good amount of VRAM.

# Notes

## For Windows users:
This daemon is not designed for the purpose of running on Windows environment. Instead, you'll probably need to run CUDA in WSL (IDK how AMD GPUs work with pytorch because I don't have one, please send a pull request/issue if you have any thoughts). Nvidia's developer website <https://docs.nvidia.com/cuda/wsl-user-guide/index.html> has more information about how this works.

Generally, you'll need to enroll in the Dev channel, install the experimental CUDA WSL driver, and install the CUDA tooklit package from NVIDIA.
