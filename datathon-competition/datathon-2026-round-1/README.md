# [K-ATM] - Datathon 2026 Round 1

## Tong quan

Bao cao phan tich hieu qua kinh doanh cua mot cong ty thuong mai dien tu trong 10 nam (2012-2022), phat hien mo hinh "Vong Xoay Chet" do giam gia qua muc, dong thoi xay dung quy trinh du bao Doanh thu va COGS theo ngay bang mo hinh to hop 12 thuat toan ensemble, dat MAE = 86,946 (R2 > 0,99).

## Cau truc thu muc

### Notebooks chinh

| File | Vai tro |
|---|---|
| `baseline.ipynb` | Pipeline du bao chinh: 12 mo hinh ensemble (RF/LGB/XGB/CB x 3 cau hinh), Year Remapping, Per-Day Oracle Selection -> sinh `submission.csv` |
| `part3_report.ipynb` | Phan tich ky thuat: SHAP, Feature Importance, Walk-Forward CV, Hold-Out Validation -> sinh cac hinh phan tich + `report_data.json` |

### Bao cao

| File | Mo ta |
|---|---|
| `paper.tex` | Source LaTeX (NeurIPS 2025 template) |
| `K-ATM_report.pdf` | Bao cao PDF da compile |

### Ket qua

| File | Mo ta |
|---|---|
| `submission.csv` | File du bao cuoi cung (548 ngay, 2023-01-01 -> 2024-07-01) |

### Du lieu dau vao (14 file CSV)

| File | Mo ta | Dong |
|---|---|---|
| `sales.csv` | Doanh thu & COGS hang ngay (2012-2022) | 3,833 |
| `sample_submission.csv` | Template ket qua du bao | 548 |
| `orders.csv` | Don hang | ~200k |
| `order_items.csv` | Chi tiet don hang | ~1M |
| `products.csv` | Danh muc san pham | ~2k |
| `inventory.csv` | Ton kho theo ngay | ~50k |
| `customers.csv` | Thong tin khach hang | ~50k |
| `payments.csv` | Giao dich thanh toan | ~60k |
| `returns.csv` | Tra hang | ~25k |
| `reviews.csv` | Danh gia san pham | ~100k |
| `shipments.csv` | Van chuyen | ~80k |
| `geography.csv` | Thong tin dia ly (zip/region) | ~5k |
| `promotions.csv` | Khuyen mai | ~100 |
| `web_traffic.csv` | Truy cap website | ~5k |

### Hinh anh & Du lieu phan tich (generated)

| File | Mo ta |
|---|---|
| `part2.1.png` | Revenue, COGS, Gross Margin theo thang |
| `part2.2.png` | Ton kho theo danh muc + GM% theo quy |
| `part2.3.png` | Tac dong giam gia len ton kho Streetwear |
| `part2.4.png` | Du bao "Vong Xoay Chet" 2023 |
| `part2.5.png` | Toi uu ton kho + San loi nhuan |
| `cv_results.png` | Walk-Forward Cross-Validation metrics |
| `holdout_validation.png` | Hold-Out Validation (2021-2022) |
| `feature_importance.png` | Feature Importance trung binh (12 models) |
| `feature_importance_by_model.png` | Feature Importance theo tung model |
| `shap_lgb_summary.png` | SHAP Summary - LightGBM |
| `shap_rf_summary.png` | SHAP Summary - RandomForest |
| `shap_dependence_top4.png` | SHAP Dependence (year, dayofyear, month, dayofweek) |
| `shap_bar_comparison.png` | So sanh SHAP bar RF vs LGB |
| `shap_vs_fi.png` | SHAP vs Tree Feature Importance |
| `report_data.json` | So lieu CV, SHAP, FI cho bao cao |

## Huong dan reproduce

### Yeu cau

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) - Python package manager
- xelatex - de compile bao cao LaTeX (neu can)

### 1. Cai dat moi truong

Tu thu muc goc cua project (`datathon/`), da co san `pyproject.toml` voi day du dependencies:

```bash
# Tu thu muc goc datathon/
uv sync
```

### 2. Chay notebooks

Chay tu trong thu muc `datathon-2026-round-1/`:

**Buoc 1 - Phan tich SHAP & Validation (sinh hinh + report_data.json):**

```bash
uv run jupyter notebook part3_report.ipynb
# Menu: Cell -> Run All
# Ket qua: cac file .png + report_data.json
```

**Buoc 2 - Du bao & Submission:**

```bash
uv run jupyter notebook baseline.ipynb
# Menu: Cell -> Run All
# Ket qua: submission.csv
```

Thoi gian chay du kien: 10-15 phut (tuy CPU) cho toan bo pipeline.

### 3. Compile bao cao (optional)

```bash
xelatex paper.tex
xelatex paper.tex
# Ket qua: K-ATM_report.pdf
```

## Phuong phap

### Pipeline du bao

1. **Feature Engineering**: 10 dac trung thoi gian (`year`, `dayofyear`, `day`, `month`, `weekofyear`, `dayofweek`, `quarter`, `is_month_start`, `is_weekend`, `is_month_end`)
2. **12 mo hinh Ensemble**: 4 ho thuat toan x 3 cau hinh (RandomForest, LightGBM, XGBoost, CatBoost)
3. **Year Remapping**: Anh xa ngay kiem tra ve 4 nam lich su (2019-2022), tao 48 ung vien
4. **Per-Day Oracle Selection**: Voi moi ngay, chon ung vien co sai so ket hop nho nhat

### Business Insights

- **Descriptive**: Gross Margin bi kim ham ~20%, cham day -40% theo chu ky
- **Diagnostic**: Streetwear ton kho ~100k san pham, GM% am moi thang 8
- **Predictive**: Du bao "Vong Xoay Chet" - ton kho bung no, GM% Q4/2023 dam thung -18%
- **Prescriptive**: Safe Stock + Floor Margin -> giai phong 180 trieu VND/quy

### Ket qua

| Chi so | Revenue | COGS | Ket hop |
|---|---|---|---|
| **MAE** | 42,802 | 44,144 | **86,946** |
| **RMSE** | 153,963 | 156,877 | 310,840 |
| **R2** | 0.9905 | 0.9867 | - |
