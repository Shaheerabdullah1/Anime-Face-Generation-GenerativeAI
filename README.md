
# AnimeFaces: Exploring Generative Models with VAEs, GANs, and DCGANs

## 📌 Project Overview

This project presents a **comparative analysis** of three popular generative deep learning models:
- **Variational Autoencoders (VAEs)**
- **Generative Adversarial Networks (GANs)**
- **Deep Convolutional GANs (DCGANs)**

Each model was trained to **generate anime-style faces**, using a Kaggle anime face dataset.
The practical application of this project is implemented in `app.py`, where users can generate
anime faces with a single click.

---

## 🧠 Author

**Shaheer Abdullah**  
Department of Computer Science  
National University of Computer & Emerging Sciences (NUCES), Islamabad  

---

## 🖼️ Model Comparison Result

This project compares the quality of generated images using VAE, GAN, and DCGAN models.

Only this image is added as requested:  
![Model Comparison](assets/model_comparison.png)

---

## 📁 Repository Structure

```
├── Anime-Face-Generation-DCGANs.ipynb
├── Anime-Face-Generation-Using-Gans.ipynb
├── Anime-Face-Generation-VAEs.ipynb
├── Anime-Faces-Style-GANs.ipynb
├── app.py                    # Face generator app
├── discriminator_model.h5   # Saved discriminator model
├── generator_model.h5       # Saved generator model
├── Dockerfile               # For containerized deployment
├── requirements.txt         # Dependencies
└── assets/
    └── model_comparison.png # Image used in README
```

---

## ⚙️ Installation & Setup

### ✅ Clone the Repository
```bash
git clone https://github.com/<your-username>/anime-face-generation.git
cd anime-face-generation
```

### ✅ Create Virtual Environment
```bash
python -m venv env
source env/bin/activate      # On Mac/Linux
env\Scripts\activate       # On Windows
```

### ✅ Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Run the Anime Face Generator (`app.py`)

```bash
python app.py
```

If using Streamlit:
```bash
streamlit run app.py
```

This will launch a local web app where users can generate anime faces with a button click.

---

## 📊 Results & Observations

| Model   | Visual Quality | Sharpness | Diversity | Notes                         |
|---------|----------------|-----------|-----------|-------------------------------|
| **VAE** | Low            | Blurry    | Medium    | Good latent space, blurry     |
| **GAN** | Medium         | Better    | Medium    | Slight distortions in features|
| **DCGAN** | High        | Sharp     | High      | Best quality & realism        |

---

## 🧪 Dataset

- **Source:** Kaggle Anime Faces Dataset  
- **Images:** 21,551 cropped anime face images  
- **Size:** 64×64 pixels  
- **Preprocessing:** Normalized between `[0, 1]`

---

## 📚 References

Please refer to the detailed research citations in your provided PDF/LaTeX file. They include works like:
- StyleGAN2 (Karras et al., 2020)
- AnimeGAN (Jiang et al., 2020)
- PGAN, MaskGAN, CapsuleGAN, etc.

---

## 📜 License & Citation

If you use this repository for academic or research purposes, please cite:

```
@article{abdullah2024animefaces,
  title={AnimeFaces: Exploring Generative Models with VAEs, GANs, and DCGANs},
  author={Abdullah, Shaheer},
  year={2024},
  institution={FAST NUCES}
}
```

---

## ⭐ Support

If you found this repository useful:
✅ Star the repo  
✅ Share it with others  
✅ Connect with me for collaborations

---

✨ *Happy Generating Anime Faces!*  
