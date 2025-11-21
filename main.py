import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os

# comparação das imagens usando ORB + RANSAC
def processar_comparacao(img_a_path, img_b_path):
    LIMIAR_RATIO = 0.85
    MIN_INLIERS = 10

    try:
        a_gray = cv2.imread(img_a_path, cv2.IMREAD_GRAYSCALE)
        b_gray = cv2.imread(img_b_path, cv2.IMREAD_GRAYSCALE)
        a_color = cv2.imread(img_a_path)
        b_color = cv2.imread(img_b_path)

        if a_gray is None or b_gray is None:
            raise Exception("Erro ao ler as imagens.")

    except Exception as erro:
        messagebox.showerror("Falha ao carregar", str(erro))
        return None

    detector = cv2.ORB_create(nfeatures=10000)
    kp_a, desc_a = detector.detectAndCompute(a_gray, None)
    kp_b, desc_b = detector.detectAndCompute(b_gray, None)

    if desc_a is None or desc_b is None or len(desc_a) < 2 or len(desc_b) < 2:
        return cv2.drawMatches(a_color, kp_a, b_color, kp_b, [], None)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    pares = matcher.knnMatch(desc_a, desc_b, k=2)

    filtrados = []
    for par in pares:
        if len(par) == 2:
            p1, p2 = par
            if p1.distance < LIMIAR_RATIO * p2.distance:
                filtrados.append(p1)

    if len(filtrados) > MIN_INLIERS:
        origem = np.float32([kp_a[m.queryIdx].pt for m in filtrados]).reshape(-1, 1, 2)
        destino = np.float32([kp_b[m.trainIdx].pt for m in filtrados]).reshape(-1, 1, 2)

        H, mascara = cv2.findHomography(origem, destino, cv2.RANSAC, 5.0)

        if mascara is not None:
            mask_list = mascara.ravel().tolist()
            qtd_inliers = sum(mask_list)

            if qtd_inliers > MIN_INLIERS:

                params1 = dict(matchColor=(0, 255, 0),
                               singlePointColor=None,
                               matchesMask=mask_list,
                               flags=2)

                img_f2 = cv2.drawMatches(a_color, kp_a, b_color, kp_b, filtrados, None, **params1)

                params2 = dict(matchColor=(0, 255, 0),
                               singlePointColor=None,
                               matchesMask=mask_list,
                               flags=6)

                img_f6 = cv2.drawMatches(a_color, kp_a, b_color, kp_b, filtrados, None, **params2)

                inl_kpA = [kp_a[m.queryIdx] for i, m in enumerate(filtrados) if mask_list[i] == 1]
                inl_kpB = [kp_b[m.trainIdx] for i, m in enumerate(filtrados) if mask_list[i] == 1]

                imgA_kp = cv2.drawKeypoints(a_color, inl_kpA, None, color=(0, 0, 255), flags=4)
                imgB_kp = cv2.drawKeypoints(b_color, inl_kpB, None, color=(0, 0, 255), flags=4)

                hA, wA = imgA_kp.shape[:2]
                hB, wB = imgB_kp.shape[:2]
                canvas = np.zeros((max(hA, hB), wA + wB, 3), dtype=np.uint8)
                canvas[:hA, :wA] = imgA_kp
                canvas[:hB, wA:wA + wB] = imgB_kp

                # salvamento dos resultados
                try:
                    pasta = "resultados"
                    os.makedirs(pasta, exist_ok=True)
                    cv2.imwrite(os.path.join(pasta, "resultado_linhas_simples.png"), img_f2)
                    cv2.imwrite(os.path.join(pasta, "resultado_linhas_pontos.png"), img_f6)
                    cv2.imwrite(os.path.join(pasta, "resultado_pontos.png"), canvas)
                except Exception as erro:
                    print("Erro ao salvar:", erro)

                return img_f2

        return cv2.drawMatches(a_color, kp_a, b_color, kp_b, [], None)

    return cv2.drawMatches(a_color, kp_a, b_color, kp_b, [], None)

img_a = ""
img_b = ""

def escolher_img_a():
    global img_a
    caminho = filedialog.askopenfilename(
        title="Escolha a primeira imagem",
        filetypes=[("Imagens", "*.jpg *.png *.jpeg *.bmp *.tiff")]
    )
    if caminho:
        img_a = caminho
        lbl_a.config(text=os.path.basename(caminho), fg="white")

def escolher_img_b():
    global img_b
    caminho = filedialog.askopenfilename(
        title="Escolha a segunda imagem",
        filetypes=[("Imagens", "*.jpg *.png *.jpeg *.bmp *.tiff")]
    )
    if caminho:
        img_b = caminho
        lbl_b.config(text=os.path.basename(caminho), fg="white")

def executar():
    if not img_a or not img_b:
        messagebox.showinfo("Atenção", "Selecione as duas imagens.")
        return

    lbl_status.config(text="Processando...", fg="#3ba55d")
    root.update()

    resultado = processar_comparacao(img_a, img_b)

    if resultado is None:
        lbl_status.config(text="Erro durante o processo.", fg="red")
        return

    lbl_status.config(text="Concluído! Resultados salvos.", fg="#3ba55d")

    largura_max = 1200
    altura_max = 700

    h, w = resultado.shape[:2]

    esc_larg = largura_max / w
    esc_alt = altura_max / h
    esc_final = min(esc_larg, esc_alt, 1)

    nova_w = int(w * esc_final)
    nova_h = int(h * esc_final)

    resultado = cv2.resize(resultado, (nova_w, nova_h), interpolation=cv2.INTER_AREA)

    rgb = cv2.cvtColor(resultado, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    tkimg = ImageTk.PhotoImage(pil)

    lbl_out.config(image=tkimg)
    lbl_out.image = tkimg

# interface
root = tk.Tk()
root.title("Comparador de Imagens - ORB + RANSAC")
root.geometry("1280x800")
root.configure(bg="#1e1e1e")

root.option_add("*Font", "SegoeUI 10")

BTN_COR = "#3ba55d"
BTN_HOVER = "#49c46d"
BG = "#1e1e1e"
CARD = "#2b2b2b"

def on_enter(e):
    e.widget['background'] = BTN_HOVER

def on_leave(e):
    e.widget['background'] = BTN_COR

frame_top = tk.Frame(root, bg=BG, pady=10)
frame_top.pack()

btn1 = tk.Button(frame_top, text="Selecionar Imagem 1", width=20,
                 command=escolher_img_a, bg=BTN_COR, fg="white",
                 activebackground=BTN_HOVER)
btn1.pack(side=tk.LEFT, padx=10)
btn1.bind("<Enter>", on_enter)
btn1.bind("<Leave>", on_leave)

lbl_a = tk.Label(frame_top, text="Nenhuma", fg="gray", bg=BG, width=30)
lbl_a.pack(side=tk.LEFT)

btn2 = tk.Button(frame_top, text="Selecionar Imagem 2", width=20,
                 command=escolher_img_b, bg=BTN_COR, fg="white",
                 activebackground=BTN_HOVER)
btn2.pack(side=tk.LEFT, padx=10)
btn2.bind("<Enter>", on_enter)
btn2.bind("<Leave>", on_leave)

lbl_b = tk.Label(frame_top, text="Nenhuma", fg="gray", bg=BG, width=30)
lbl_b.pack(side=tk.LEFT)

frame_mid = tk.Frame(root, bg=BG, pady=10)
frame_mid.pack()

btn = tk.Button(frame_mid,
                text="COMPARAR IMAGENS",
                width=30,
                height=2,
                bg=BTN_COR,
                fg="white",
                font=("Segoe UI", 12, "bold"),
                command=executar,
                activebackground=BTN_HOVER)
btn.pack()
btn.bind("<Enter>", on_enter)
btn.bind("<Leave>", on_leave)

lbl_status = tk.Label(root, text="Selecione duas imagens e compare",
                      font=("Segoe UI", 10, "italic"),
                      fg="white",
                      bg=BG)
lbl_status.pack(pady=5)

frame_img = tk.Frame(root, bg=CARD, borderwidth=1, relief=tk.SUNKEN)
frame_img.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

lbl_out = tk.Label(frame_img, bg=CARD)
lbl_out.pack(fill=tk.BOTH, expand=True)

root.mainloop()
