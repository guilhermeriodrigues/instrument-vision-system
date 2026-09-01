import os

# Caminho da pasta
pasta = r"C:\Users\guilh\instrument-vision-system\images\Digital_Instruments"

# Número inicial
contador = 1

# Lista arquivos JPG
arquivos = [f for f in os.listdir(pasta) if f.lower().endswith(".jpg")]
arquivos.sort()

# =========================
# ETAPA 1 - nomes temporários
# =========================
temporarios = []

for i, arquivo in enumerate(arquivos):
    caminho_antigo = os.path.join(pasta, arquivo)

    nome_temp = f"temp_{i}.jpg"
    caminho_temp = os.path.join(pasta, nome_temp)

    os.rename(caminho_antigo, caminho_temp)

    temporarios.append(nome_temp)

# =========================
# ETAPA 2 - nomes finais
# =========================
for arquivo_temp in temporarios:
    caminho_temp = os.path.join(pasta, arquivo_temp)

    novo_nome = f"digital_instrument{contador}.jpg"
    caminho_novo = os.path.join(pasta, novo_nome)

    os.rename(caminho_temp, caminho_novo)

    print(f"{arquivo_temp} -> {novo_nome}")

    contador += 1

print("Renomeação concluída.")