n_samples=30
n_models=5
model_name=andromeda
noise_type=g
toymodel=singlequbit
p1=0
p2=.05

python --version
cd ../simulations
python regular_data.py $toymodel $n_samples $noise_type $p1 $p2
cd ../nn-models
python model.py $toymodel $n_samples $n_models $model_name $noise_type $p1 $p2
