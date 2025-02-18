启动环境:
```shell
conda activate difflinker
```

打开run.sh文件，把run.sh文件中reinforce改成需要的强化名，定义需要的训练轮数：

例如：
export reinforce='rclogp'
（可选的强化名可以在preprocess中选择强化函数的函数“get_scoring_function”中找到）
for i in $(seq 1 100)

准备数据文件：
```shell
mkdir -p datasets
wget https://zenodo.org/record/7121271/files/zinc_final_test.pt?download=1 -O datasets/zinc_final_test.pt
```

准备初始模型：
```shell
mkdir $reinforce$/model/
wget https://zenodo.org/record/7121300/files/zinc_difflinker.ckpt?download=1 -O $reinforce$/model/zinc_difflinker.ckpt
```
训练过程中wandb会读取出这个模型epoch=299，需要将格式改为适当的格式并设置epoch=299，
代码会自动读取epoch数并添加训练轮数再更新配置文件，如读取epoch=299，设置参数n_epochs=309(299+10)

运行shell文件：
```shell
bash run.sh
```

