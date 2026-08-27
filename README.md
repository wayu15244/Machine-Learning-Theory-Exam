# Theory part - Machine Learning (สอบ Midterm) งานกลุ่ม

**รายวิชา:** Machine Learning (การเรียนรู้ของเครื่อง)  
**อาจารย์ผู้สอน:** อาจารย์ ดร. โอฬาริก สุรินต๊ะ (Olarik Surinta)  
**คะแนนเต็ม:** 15 คะแนน (กำหนดส่ง 7 กันยายน เวลา 17:00 น.)  
**เกณฑ์คะแนน:**
- Visualization ของ KNN = 5 คะแนน
- Visualization ของ KNN Regression = 5 คะแนน
- Visualization ของ K-Means = 5 คะแนน

---

## 🌐 ลิงก์ผลงานออนไลน์ (Live Demos & Artifacts)

- **Interactive Visualization Dashboard (GitHub Pages):**  
  `https://[username].github.io/Machine-Learning-Theory-Exam/` *(เปิดใช้งานผ่านไฟล์ `index.html`)*
- **รายงานฉบับสมบูรณ์ (PDF File พร้อมส่ง):**  
  [`Machine_Learning_Theory_Exam.pdf`](Machine_Learning_Theory_Exam.pdf)

---

## 👥 รายชื่อสมาชิกในกลุ่ม (5 คน)

> **หมายเหตุ:** กำหนดให้หัวหน้ากลุ่มเป็นผู้ส่งงานเพียง 1 คนผ่านระบบ Google Classroom

| ลำดับ | บทบาท | ชื่อ - นามสกุล | รหัสนิสิต | หน้าที่รับผิดชอบ |
| :---: | :---: | :--- | :---: | :--- |
| **1** | **หัวหน้ากลุ่ม** | [ชื่อ-สกุล นิสิตคนที่ 1] | [รหัสนิสิต 1] | รวบรวมรายงาน, ส่งไฟล์ PDF, ตรวจสอบภาพรวม |
| 2 | สมาชิก | [ชื่อ-สกุล นิสิตคนที่ 2] | [รหัสนิสิต 2] | สรุปทฤษฎี & ตัวอย่างคำนวณ KNN Classification |
| 3 | สมาชิก | [ชื่อ-สกุล นิสิตคนที่ 3] | [รหัสนิสิต 3] | สรุปทฤษฎี & ตัวอย่างคำนวณ KNN Regression |
| 4 | สมาชิก | [ชื่อ-สกุล นิสิตคนที่ 4] | [รหัสนิสิต 4] | สรุปทฤษฎี & ตัวอย่างคำนวณ K-Means Clustering |
| 5 | สมาชิก | [ชื่อ-สกุล นิสิตคนที่ 5] | [รหัสนิสิต 5] | พัฒนา Interactive Visualizations & จัดทำ GitHub Repo |

---

## 🧠 สรุปสาระสำคัญทางทฤษฎี (Theoretical Summary)

### 1. K-Nearest Neighbor (KNN) Classification (5 คะแนน)
- **ประเภทการเรียนรู้:** Supervised Learning (Instance-Based / Lazy Learning)
- **หลักการสำคัญ:** *"สิ่งที่คล้ายกันมักจะอยู่ใกล้กัน" (Things that are similar are close to each other)* โดยไม่มีขั้นตอนเทรนโมเดลล่วงหน้า แต่จะเก็บตัวอย่างข้อมูลฝึกฝนทั้งหมดไว้ แล้วคำนวณระยะห่างเมื่อมี Query Point
- **การวัดระยะทาง:** Euclidean, Manhattan, Minkowski Distance
- **กระบวนการทำงาน 6 ขั้นตอน:**
  1. Input New Sample
  2. Calculate Distance
  3. Sort Distance (น้อย $\rightarrow$ มาก)
  4. Select K Nearest
  5. Majority Vote (หรือ Distance-Weighted Vote)
  6. Prediction
- **ผลของค่า K:** 
  - $K$ เล็ก $\implies$ Flexible, ขอบเขตซับซ้อน แต่ Noise Sensitive (เสี่ยง Overfitting)
  - $K$ ใหญ่ $\implies$ Stable, ลด Noise แต่เสี่ยง Underfitting (ขอบเขตเรียบเกินไป)
- **ตัวอย่างการคำนวณ:** จุด $A(2,3), B(3,5), C(6,8), D(7,6), E(5,4)$ ทดสอบกับ $P_1(4,5)$ ที่ $K=3$ ได้ผลทำนายเป็น **Class A**

### 2. K-Nearest Neighbor (KNN) Regression (5 คะแนน)
- **ประเภทการเรียนรู้:** Supervised Learning สำหรับพยากรณ์ **ค่าตัวเลขต่อเนื่อง (Continuous Target)** เช่น ราคาบ้าน, ยอดขาย, หรือดัชนีหุ้น
- **สูตรการพยากรณ์:**
  - **Simple Average:** $\hat{y} = \frac{1}{K}\sum_{i=1}^K y_i$
  - **Distance-Weighted Average:** $\hat{y} = \frac{\sum w_i y_i}{\sum w_i}$ โดยที่ $w_i = \frac{1}{d_i^2 + 10^{-5}}$
- **ตัวอย่างการคำนวณตามสไลด์:**
  - ตัวอย่าง 5 Features: จุดทดสอบ $[3, 3, 4, 3, 3], K=3 \implies \hat{y} = 51.00$
  - ตัวอย่างทำนายราคาบ้าน: จุดทดสอบ $[72 \text{ ตร.ม.}, 2 \text{ ห้องนอน}], K=5 \implies \hat{y} = 1,217.59$ พันบาท
- **การประยุกต์ใช้จริง (Colab):** การทำนายราคาปิดหุ้น Apple (AAPL) ด้วย Sliding Window ขนาด 5 วัน ประเมินผลลัพธ์ด้วย MAE (1.83 USD)

### 3. K-Means Clustering (5 คะแนน)
- **ประเภทการเรียนรู้:** Instance-based Unsupervised Learning (ไม่มี Label)
- **หลักการสำคัญ:** Partitioning Algorithm ที่แบ่งข้อมูลออกเป็น $K$ กลุ่ม โดยลดผลรวมความแปรปรวนภายในกลุ่ม (WCSS / Inertia)
- **กระบวนการทำงาน 5 ขั้นตอน:**
  1. กำหนดค่า $K$
  2. กำหนด Initial Centroids ($C_1, \dots, C_K$)
  3. Cluster Assignment (จัดจุดข้อมูลสังกัด Centroid ที่ใกล้ที่สุด)
  4. Update Centroid (คำนวณพิกัดเฉลี่ยใหม่ของสมาชิกในกลุ่ม $C_j = \frac{1}{|S_j|}\sum x$)
  5. Iteration & Convergence (ทำซ้ำจนตำแหน่ง Centroid นิ่ง)
- **ตัวอย่างการคำนวณตามสไลด์:** จุด $A(2,3), B(3,4), C(4,5), D(8,8), E(9,9), F(8,10)$ เริ่มต้นที่ $C_1(2,3), C_2(9,9)$ อัปเดตในรอบที่ 1 เป็น $C_1(3.00, 4.00)$ และ $C_2(8.33, 9.00)$ และลู่เข้าในรอบที่ 2
- **การประยุกต์ใช้จริง (Colab):** การแบ่งกลุ่มลูกค้าห้างสรรพสินค้า (Mall Customer Segmentation) เป็น 5 กลุ่มพฤติกรรม

---

## 📊 ตารางเปรียบเทียบคุณสมบัติ (Comparison Matrix)

| คุณสมบัติ | 1) KNN Classification | 2) KNN Regression | 3) K-Means Clustering |
| :--- | :--- | :--- | :--- |
| **Learning Paradigm** | Supervised Learning | Supervised Learning | Unsupervised Learning |
| **Target Variable** | Discrete Class (กลุ่มคุณภาพ) | Continuous Value (ตัวเลขต่อเนื่อง) | ไม่มี Label |
| **Model Type** | Lazy / Instance-Based | Lazy / Instance-Based | Iterative Partitioning |
| **Decision Rule** | Majority Vote / Weighted Vote | $\frac{1}{K}\sum y_i$ หรือ $\frac{\sum w_i y_i}{\sum w_i}$ | $\arg\min_j \|x - C_j\|^2$ |
| **จุดเด่น** | เข้าใจง่าย ปรับตัวกับข้อมูลใหม่ทันที | จับความสัมพันธ์ไม่เป็นเส้นตรงได้ดี | รวดเร็วกับข้อมูลขนาดใหญ่ เหมาะทำ EDA |
| **ข้อพึงระวัง** | Curse of Dimensionality, คำนวณช้า | ไวต่อ Outlier, ไม่สามารถทำ Extrapolate | ไวต่อ Initial Centroids (Local Minima) |

---

## 📁 โครงสร้างไฟล์ในโปรเจกต์ (Project Structure)

```text
Theory part/
├── index.html                      # Interactive Visualization Web Dashboard (สำหรับ GitHub Pages)
├── report.html                     # เอกสารรายงานฉบับสมบูรณ์จัดหน้า A4 สำหรับแปลงเป็น PDF
├── Machine_Learning_Theory_Exam.pdf # ไฟล์ PDF รายงานฉบับสมบูรณ์ (8 หน้า A4 พิมพ์ส่งอาจารย์)
├── README.md                       # เอกสารสรุปโปรเจกต์และรายชื่อสมาชิก
├── figures/                        # แผนภาพกราฟความละเอียดสูงสำหรับรายงาน
│   ├── knn_classification_slide.png
│   ├── knn_regression_slide.png
│   ├── stock_knn_regression.png
│   ├── kmeans_clustering_slide.png
│   ├── kmeans_good_vs_poor.png
│   ├── mall_customers_kmeans.png
│   └── pdf_preview/                # ภาพพรีวิวแต่ละหน้าของ PDF
└── src/                            # ซอร์สโค้ด Python สำหรับคำนวณและสร้าง Visualization
    ├── knn_classification.py
    ├── knn_regression.py
    ├── stock_knn_regression.py
    ├── kmeans_clustering.py
    └── mall_customers_kmeans.py
```

---

## 🚀 ขั้นตอนการนำขึ้น GitHub และเปิดใช้งาน GitHub Pages

### 1. เริ่มต้น Git และ Commit โค้ด
```bash
git init
git add .
git commit -m "Complete ML Theory Midterm Project with Interactive Visualizations and PDF Report"
```

### 2. ลิงก์กับ GitHub Repository
```bash
# เปลี่ยน [username] เป็นชื่อบัญชี GitHub ของหัวหน้ากลุ่ม
git branch -M main
git remote add origin https://github.com/[username]/Machine-Learning-Theory-Exam.git
git push -u origin main
```

### 3. เปิดใช้งาน GitHub Pages เพื่อสร้างลิงก์ออนไลน์
1. ไปที่หน้า GitHub Repository ของคุณ
2. คลิกที่แถบ **Settings** ด้านบน
3. ที่เมนูด้านซ้าย เลือก **Pages**
4. ในส่วน **Build and deployment** $\rightarrow$ **Branch**:
   - เลือก Branch: `main`
   - เลือก Folder: `/ (root)`
5. กดปุ่ม **Save**
6. รอประมาณ 1-2 นาที คุณจะได้รับ URL ออนไลน์ เช่น:  
   `https://[username].github.io/Machine-Learning-Theory-Exam/`

---

## 📚 เอกสารอ้างอิง (References)
1. สุรินต๊ะ, โอฬาริก. (2567). เอกสารประกอบการบรรยายรายวิชา Machine Learning: เรื่อง K-Nearest Neighbor (KNN), KNN Regression และ K-Means Clustering.
2. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.
3. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning: Data Mining, Inference, and Prediction*. Springer.
