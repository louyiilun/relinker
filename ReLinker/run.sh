export reinforce='rclogp'

for i in $(seq 1 100)
do 

python -W ignore sample_test.py  \
                 --checkpoint "$reinforce"/model \
                 --linker_size_model models/zinc_size_gnn.ckpt \
                 --samples "$reinforce"/samples$i \
                 --data datasets \
                 --prefix zinc_final_test \
                 --n_samples 1 \
                 --device cuda:0
echo "=============================="
echo "sample$i"
echo "=============================="
python -W ignore preprocess_test.py \
                 --samples "$reinforce"/samples$i \
                 --scoring_function "$reinforce" \
                 --out_dir "$reinforce"/out_dir$i \
                 --true_smiles_path datasets/zinc_final_test_smiles.smi \
                 --template template \
                 --n_samples 1 

echo "=============================="
echo "preprocess$i"
echo "=============================="
python -W ignore train_test.py --config configs/difflinker_test.yml \
                 --data "$reinforce"/out_dir$i \
                 --samples_path "$reinforce"/samples$i \
                 --scoring_function "$reinforce" \
                 --checkpoints "$reinforce"/model
echo "=============================="     
echo "train$i"
echo "=============================="

done






