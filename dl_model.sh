mkdir checkpoints
cd checkpoints
wget "https://civitai.com/api/download/models/245598?type=Model&format=SafeTensor&size=pruned&fp=fp16" --content-disposition
cd ..
mkdir onnx
