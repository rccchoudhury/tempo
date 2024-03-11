conda create -n tempo_v0 python=3.10 -y
conda activate tempo_v0

# Change this line to match system requirements.
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch -y

mkdir third_party
cd third_party
git clone https://github.com/open-mmlab/mmpose.git
cd mmpose
git checkout dd3c49418730e2d3651ff62fe04fb4319168c7c1
pip install -r requirements/build.txt
python setup.py develop

cd ..

