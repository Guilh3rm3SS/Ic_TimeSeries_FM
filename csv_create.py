import pandas as pd

# Dados como dicionário
dados = {
    'Nome': ['Alice', 'Bob', 'Carol'],
    'Idade': [30, 25, 27],
    'Email': ['alice@example.com', 'bob@example.com', 'carol@example.com']
}

# Cria um DataFrame
df = pd.DataFrame(dados)

# Salva em CSV
df.to_csv('dados.csv', index=False, encoding='utf-8')

print('Arquivo "dados.csv" criado com sucesso usando pandas!')


pip install ml-dtypes==0.5.0 protobuf==3.20.3 tensorboard==2.18.0 torch==2.6.0
pip install "numpy>=1.21.6,<1.28.0" --force-reinstall
pip install gluonts
pip install --upgrade numpy==1.26.4
pip install --upgrade ml-dtypes==0.5.0
pip install --upgrade torch==2.6.0
pip install uni2ts
export MPLCONFIGDIR=/tmp/matplotlib

pip install torch==2.6.0 torchvision==0.21.0
pip install ml_dtypes>=0.5.0


pip install --upgrade --force-reinstall \
    
    pandas==2.2.2 \
    torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    fsspec==2025.3.2 \
    numpy==1.16.4 \
    scipy==1.14.0
    gluonts==0.14.3
pip install uni2ts


export HF_HOME=./.cache

nohup python ./moirai/forecast_mercedes_level.py > output.log 2>&1 &


pip install --upgrade --force-reinstall \
    pandas==1.2.0 \
    torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    fsspec==2025.3.2 \
    numpy==1.16.5 \
    scipy==1.14.0
    gluonts==0.14.3
pip install uni2ts
pip install matplotlib
export MPLCONFIGDIR=/tmp/matplotlib
export HF_HOME=./.cache


pip3 install timesfm[torch]