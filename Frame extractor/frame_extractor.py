import cv2
import os
import zipfile
import numpy as np

# === CONFIGURAÇÕES ===
video_path = "video_5.mp4"          # nome do vídeo
output_folder = "dataset_placas"  # pasta principal
max_frames = 1000                  # máximo de frames salvos
threshold = 23.0                  # sensibilidade para detectar mudança entre quadros

# === PREPARAR PASTAS ===
images_folder = os.path.join(output_folder, "images")
labels_folder = os.path.join(output_folder, "labels")
os.makedirs(images_folder, exist_ok=True)
os.makedirs(labels_folder, exist_ok=True)

# === FUNÇÃO PARA ESCOLHER LABEL ===
def escolher_label():
    """Pergunta ao usuário qual label deseja usar"""
    print("\n🏷️  SELECIONE O LABEL PARA AS PLACAS")
    print("=" * 40)
    print("1. placa_azul  (ID: 0)")
    print("2. placa_cinza (ID: 0)")
    print("=" * 40)
    
    while True:
        try:
            opcao = input("Digite o número da opção desejada (1 ou 2): ").strip()
            if opcao == "1":
                return "placa_azul", 0
            elif opcao == "2":
                return "placa_cinza", 0
            else:
                print("❌ Opção inválida! Digite 1 ou 2.")
        except KeyboardInterrupt:
            print("\n\n❌ Operação cancelada pelo usuário.")
            exit(1)
        except Exception as e:
            print(f"❌ Erro: {e}")

# === ESCOLHER LABEL ===
placa_tipo, classe_id = escolher_label()
print(f"\n✅ Label selecionado: {placa_tipo} (ID: {classe_id})")

# === CARREGAR MODELO DE DETECÇÃO DE PLACAS ===
cascade_path = cv2.data.haarcascades + "haarcascade_russian_plate_number.xml"
plate_cascade = cv2.CascadeClassifier(cascade_path)
if plate_cascade.empty():
    raise Exception("Erro ao carregar o classificador Haar Cascade.")

# === ABRIR VÍDEO ===
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise Exception("Erro ao abrir o vídeo.")

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"🎥 Vídeo: {total_frames} frames a {fps:.2f} fps.\n")

# === FUNÇÕES ===
def frame_diff_score(f1, f2):
    f1_gray = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
    f2_gray = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(f1_gray, f2_gray)
    return np.mean(diff)

def detectar_placas(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(60, 20))
    return plates

def salvar_label_yolo(filename, img_w, img_h, boxes, classe_id):
    """Salva anotação no formato YOLO compatível com Make Sense.ai"""
    with open(filename, "w", encoding="utf-8") as f:
        for i, (x, y, w, h) in enumerate(boxes):
            x_center = (x + w / 2) / img_w
            y_center = (y + h / 2) / img_h
            width = w / img_w
            height = h / img_h
            
            # Garantir que as coordenadas estão dentro dos limites [0, 1]
            x_center = max(0.000001, min(0.999999, x_center))
            y_center = max(0.000001, min(0.999999, y_center))
            width = max(0.000001, min(0.999999, width))
            height = max(0.000001, min(0.999999, height))
            
            # Formato YOLO: <class> <x_center> <y_center> <width> <height>
            linha = f"{classe_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            
            # Se for a última bounding box, não adiciona quebra de linha
            if i == len(boxes) - 1:
                f.write(linha)
            else:
                f.write(linha + "\n")

def criar_arquivo_labels_txt():
    """Cria arquivo labels.txt dentro da pasta labels com o ID da classe"""
    labels_txt_path = os.path.join(labels_folder, "labels.txt")
    
    # Criar arquivo com o ID da classe, sem espaços extras
    with open(labels_txt_path, "w", encoding="utf-8") as f:
        f.write(f"{classe_id}")
    
    print(f"📝 Arquivo labels.txt criado em: {labels_txt_path}")
    print(f"🏷️  Label configurado: '{placa_tipo}' (ID: {classe_id})")

# === LOOP PRINCIPAL ===
saved = 0
last_frame = None
frame_index = 0

print(f"🎥 Iniciando extração de frames...")
print(f"📁 Saída: {output_folder}")
print(f"🎯 Alvo: {max_frames} frames máximo")
print(f"🏷️  Label selecionado: {placa_tipo} (ID: {classe_id})")

while cap.isOpened() and saved < max_frames:
    ret, frame = cap.read()
    if not ret:
        break

    plates = detectar_placas(frame)
    if len(plates) > 0:
        diff = threshold + 1 if last_frame is None else frame_diff_score(frame, last_frame)

        if diff > threshold:
            # Usar nomes "frameXXX" começando em 000
            img_name = f"frame{saved:03d}.jpg"
            label_name = f"frame{saved:03d}.txt"
            img_path = os.path.join(images_folder, img_name)
            label_path = os.path.join(labels_folder, label_name)

            cv2.imwrite(img_path, frame)
            h, w = frame.shape[:2]
            salvar_label_yolo(label_path, w, h, plates, classe_id)

            saved += 1
            last_frame = frame
            progresso = (saved / max_frames) * 100
            print(f"✅ Frame {frame_index} salvo ({saved:03d}/{max_frames}) [Label: {placa_tipo} (ID: {classe_id})] [{progresso:5.1f}%]")

    frame_index += 1

cap.release()
print(f"\n📸 Total de frames salvos: {saved}")

# === CRIAR labels.txt NA PASTA LABELS ===
criar_arquivo_labels_txt()

# === CRIA ARQUIVO ZIP FINAL ===
zip_name = f"{output_folder}.zip"
with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, _, files in os.walk(output_folder):
        for file in files:
            file_path = os.path.join(root, file)
            arcname = os.path.relpath(file_path, output_folder)
            zipf.write(file_path, arcname=arcname)

# === RELATÓRIO FINAL ===
print(f"\n{'='*50}")
print("📊 RELATÓRIO FINAL")
print(f"{'='*50}")
print(f"✅ Frames salvos: {saved}")
print(f"🏷️  Label utilizado: {placa_tipo} (ID: {classe_id})")
print(f"📁 Dataset salvo em: {output_folder}")
print(f"📦 Arquivo compactado: {zip_name}")
print(f"🖼️  Imagens em: {images_folder}")
print(f"📝 Labels em: {labels_folder}")
print(f"📋 Arquivo labels.txt criado na pasta labels")
print(f"\n💡 Estrutura de arquivos gerada:")
print(f"   📁 {output_folder}/")
print(f"   ├── 📁 images/")
print(f"   │   ├── frame001.jpg")
print(f"   │   ├── frame002.jpg")
print(f"   │   └── ...")
print(f"   ├── 📁 labels/")
print(f"   │   ├── labels.txt  (contém: '{classe_id}')")
print(f"   │   ├── frame001.txt")
print(f"   │   ├── frame002.txt")
print(f"   │   └── ...")
print(f"\n🎯 Para usar no Make Sense.ai:")
print(f"   - Ao definir labels, use: {placa_tipo}")
print(f"{'='*50}")