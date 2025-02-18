i=100
reinforce=qed

python -W ignore preprocess_test.py \
                 --samples $reinforce/samples$i \
                 --scoring_function $reinforce \
                 --out_dir $reinforce/out_dir$i \
                 --true_smiles_path datasets/zinc_final_test_smiles.smi \
                 --template template \
                 --n_samples 1 

python -W ignore compute_metrics.py \
                 ZINC \
                 $reinforce/samples$i/metrics.smi \
                 datasets/zinc_final_train_linkers.smi \
                 1 1 None \
                 resources/wehi_pains.csv \
                 diffusion