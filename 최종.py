import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def estimate_Mc(mags, bin_width=0.1, correction=0.2):
   
    mags = mags[np.isfinite(mags)]
    bins = np.arange(np.floor(mags.min()), np.ceil(mags.max()) + bin_width, bin_width)
    hist, edges = np.histogram(mags, bins=bins)
    
    
    peak_idx = np.argmax(hist)
    Mc_raw = 0.5 * (edges[peak_idx] + edges[peak_idx+1])
    
  
    return Mc_raw + correction


def b_value_mle(mags, Mc, M_bin=0.1):
    mags = mags[mags >= Mc]
    if len(mags) < 20:
        return np.nan
    mean_M = np.mean(mags)
    denom = mean_M - (Mc - 0.5 * M_bin) #간단하게 모델 작성했는데 오류가 나와서 해결책을 물어보고 참고하여 작성(0으로 나누는 경우의 오류)
    if denom <= 0:
        return np.nan
    return np.log10(np.e) / denom

# 중간에 1차 함수인 부분만 잘라내어 피팅하고싶을때 어떻게 해? 라고 질문후 참고하여 작성 
def get_gr_points(mags, bin_width=0.1):

    bins = np.arange(np.floor(mags.min()), np.ceil(mags.max()) + bin_width, bin_width)
    hist, edges = np.histogram(mags, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    
  
    cum = np.cumsum(hist[::-1])[::-1]
    
  
    valid = cum > 0
    x = centers[valid]
    y = np.log10(cum[valid])
    
    return x, y

def analyze_region(csv_path, label):
    try:
        df = pd.read_csv(csv_path)
    except:
        return None

    mags = df["mag"].dropna().values
    

    Mc = estimate_Mc(mags, correction=0.2)
    

    all_x, all_y = get_gr_points(mags)
    

    mask = all_x >= Mc
    fit_x = all_x[mask]
    fit_y = all_y[mask]
    

    if len(fit_x) > 2:
        coeffs = np.polyfit(fit_x, fit_y, 1) 
        b_val = -coeffs[0] 
    else:
        coeffs = [np.nan, np.nan]
        b_val = np.nan

    return {
        "label": label,
        "Mc": Mc,
        "mags": mags,
        "all_x": all_x,     
        "all_y": all_y,
        "fit_x": fit_x,     
        "fit_y": fit_y,
        "coeffs": coeffs,   
        "b_value": b_val
    }


def region_feature_points(csv_path, label, Mc,
                          n_samples=40, sample_size=200):

    try:
        df = pd.read_csv(csv_path)
    except:
        return pd.DataFrame()
        
    mags_all = df["mag"].dropna().values
    mags_all = mags_all[mags_all >= Mc]

    points = []
    if len(mags_all) < 10:
        return pd.DataFrame()

    for _ in range(n_samples):

        sel = np.random.choice(
            mags_all,
            size=min(sample_size, len(mags_all)),
            replace=True 
        )
        mean_mag = np.mean(sel)
        b_val = b_value_mle(sel, Mc)
        
        if np.isfinite(b_val):
            points.append([mean_mag, b_val, label])

    return pd.DataFrame(points, columns=["mean_mag", "b_value", "label"])


if __name__ == "__main__":


    japan = analyze_region("japan.csv", "Japan")
    california = analyze_region("cal.csv", "California")
    mid_atlantic = analyze_region("mid_atlantic.csv", "MidAtlantic")


    if not all([japan, california, mid_atlantic]):
        print("CSV 파일 로드에 실패했습니다. 파일 경로를 확인하세요.")
        exit()

    print("Japan  : Mc =", round(japan["Mc"], 2), " b =", round(japan["b_value"], 2))
    print("Calif. : Mc =", round(california["Mc"], 2), " b =", round(california["b_value"], 2))
    print("MidAtl : Mc =", round(mid_atlantic["Mc"], 2), " b =", round(mid_atlantic["b_value"], 2))


    plt.figure(figsize=(14, 4))

    plt.subplot(1, 3, 1)
    plt.hist(japan["mags"], bins=30, alpha=0.5, label="Japan")
    plt.hist(california["mags"], bins=30, alpha=0.5, label="California")
    plt.hist(mid_atlantic["mags"], bins=30, alpha=0.5, label="MidAtlantic")

    plt.xlabel("Magnitude")
    plt.ylabel("Frequency")
    plt.title("Magnitude histogram")
    plt.legend()
    plt.grid(alpha=0.3)


    plt.subplot(1, 3, 2)
    
    

    regions = [japan, california, mid_atlantic]
    colors = ['tab:blue', 'tab:green', 'tab:purple']
    
    for reg, color in zip(regions, colors):
        label = reg['label']
        Mc = reg['Mc']
        
    
        plt.plot(reg['all_x'], reg['all_y'], '.', color=color, alpha=0.3, markersize=8)
        plt.plot(reg['fit_x'], reg['fit_y'], 'o', color=color, label=f"{label} Data")
        
       
        slope, intercept = reg['coeffs']
        if np.isfinite(slope):
            
            x_line = np.linspace(Mc, reg['fit_x'].max(), 10)
            y_line = slope * x_line + intercept
            
     
            plt.plot(x_line, y_line, '-', color=color, linewidth=2, 
                     label=f"{label} Fit (b={-slope:.2f})")
            

            plt.vlines(Mc, ymin=0, ymax=y_line[0], colors=color, linestyles=':', alpha=0.5)

    plt.xlabel("Magnitude (M)")
    plt.ylabel("log10 N(>M)")
    plt.title("Gutenberg-Richter Law (Fit above Mc)")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()


# 3가지 데이터를 학습시켜서 classify 하고싶을때 어떻게 해? 라고 질문후 참고하여 작성 
    jp_pts = region_feature_points("japan.csv", "Japan", japan["Mc"])
    ca_pts = region_feature_points("cal.csv", "California", california["Mc"])
    ma_pts = region_feature_points("mid_atlantic.csv", "MidAtlantic", mid_atlantic["Mc"])

    df_cls = pd.concat([jp_pts, ca_pts, ma_pts], ignore_index=True)

  
    X = df_cls[["mean_mag", "b_value"]].values
    label_map = {"Japan": 0, "California": 1, "MidAtlantic": 2}
    inv_label_map = {v: k for k, v in label_map.items()}
    y = df_cls["label"].map(label_map).values

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    clf = LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=500)
    clf.fit(Xs, y)

    print("\n--- 3-class Earthquake Classifier ---")
    try:
        mean_in = float(input("Enter mean magnitude (e.g. 4.5): "))
        b_in = float(input("Enter b-value (e.g. 1.0): "))
        
       
        X_user = scaler.transform([[mean_in, b_in]])
        probs = clf.predict_proba(X_user)[0]
        pred_class = np.argmax(probs)
        pred_label = inv_label_map[pred_class]

        print("\nPrediction Result:")
        for i, p in enumerate(probs):
            print(f"  Probability of {inv_label_map[i]:<12}: {p*100:.1f}%")
        print(f"→ Final Decision: {pred_label}")
        
    except ValueError:
        print("잘못된 입력입니다. 그래프만 출력합니다.")
        mean_in, b_in = None, None
    plt.subplot(1, 3, 3)

    plt.scatter(jp_pts["mean_mag"], jp_pts["b_value"], c="red", alpha=0.3, label="Japan")
    plt.scatter(ca_pts["mean_mag"], ca_pts["b_value"], c="green", alpha=0.3, label="California")
    plt.scatter(ma_pts["mean_mag"], ma_pts["b_value"], c="blue", alpha=0.3, label="Mid-Atlantic")

    if mean_in is not None and b_in is not None:
        plt.scatter(mean_in, b_in, c="black", marker="*", s=300, label="Your Input", zorder=10)

    plt.xlabel("Mean Magnitude")
    plt.ylabel("b-value")
    plt.title("Tectonic Classification")
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()