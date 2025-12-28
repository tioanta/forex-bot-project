import yfinance as yf
import pandas as pd
from prophet import Prophet
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

# --- KONFIGURASI VISUALISASI ---
# Mengatur gaya plot agar terlihat modern (mirip Seaborn darkgrid)
sns.set_theme(style="darkgrid")
plt.rcParams['figure.figsize'] = (12, 12) # Ukuran gambar persegi (12x12 inci)
plt.rcParams['font.family'] = 'sans-serif' # Menggunakan font sans-serif agar bersih

# Palet Warna untuk Rekomendasi
COLORS = {
    'BELI': '#2ecc71', # Hijau Neon
    'JUAL': '#e74c3c', # Merah
    'HOLD': '#95a5a6'  # Abu-abu
}

TICKERS = {
    'USD': 'USDIDR=X',
    'JPY': 'JPYIDR=X',
    'KRW': 'KRWIDR=X',
    'CNY': 'CNYIDR=X'
}

def get_recommendation(current_price, predicted_price, threshold=0.002):
    diff_percent = (predicted_price - current_price) / current_price
    if diff_percent > threshold:
        return "BELI", f"+{diff_percent*100:.2f}%"
    elif diff_percent < -threshold:
        return "JUAL", f"{diff_percent*100:.2f}%"
    else:
        return "HOLD", f"{diff_percent*100:.2f}%"

# --- FUNGSI BARU UNTUK MENGGAMBAR ---
def plot_currency(ax, currency, df_recent, current_price, predicted_price, signal, change_txt):
    """Menggambar satu kotak subplot untuk satu mata uang."""
    
    # 1. Plot Data Historis (Garis Biru)
    ax.plot(df_recent['ds'], df_recent['y'], label='Historis (90 Hari)', color='#3498db', linewidth=2)
    
    # 2. Plot Titik Prediksi (Titik Oranye)
    prediction_date = df_recent['ds'].iloc[-1] + timedelta(days=1)
    ax.scatter(prediction_date, predicted_price, color='#e67e22', s=150, zorder=5, label='Prediksi Besok')
    
    # 3. Anotasi Harga
    # Harga Sekarang
    last_date = df_recent['ds'].iloc[-1]
    ax.annotate(f"Now: {current_price:,.0f}", 
                (last_date, current_price),
                xytext=(10, -20), textcoords='offset points',
                arrowprops=dict(arrowstyle="->", color='white'),
                fontsize=10, color='white', fontweight='bold')

    # Harga Prediksi
    ax.annotate(f"Pred: {predicted_price:,.0f}", 
                (prediction_date, predicted_price),
                xytext=(10, 20), textcoords='offset points',
                arrowprops=dict(arrowstyle="->", color='#e67e22'),
                fontsize=10, color='#e67e22', fontweight='bold')

    # 4. Kotak Rekomendasi (Summary Box) di pojok kiri atas
    rec_color = COLORS.get(signal, COLORS['HOLD'])
    summary_text = f"{currency}/IDR\n{signal}\n({change_txt})"
    
    # Membuat kotak teks dengan background warna sesuai sinyal
    props = dict(boxstyle='round,pad=0.5', facecolor=rec_color, alpha=0.9, edgecolor='none')
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=14,
            fontweight='bold', color='white', verticalalignment='top', bbox=props)

    # 5. Formatting Sumbu
    ax.set_title(f"Analisa {currency} terhadap Rupiah", fontsize=12, color='white', pad=20)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b')) # Format tanggal: 28-Dec
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9) # Miringkan tanggal
    ax.grid(True, which='major', linestyle='--', linewidth=0.5, color='gray')
    
    # Set background subplot menjadi agak gelap
    ax.set_facecolor('#2c3e50') 


def run_analysis_and_plot():
    today_str = datetime.now().strftime('%Y-%m-%d')
    print(f"--- MEMULAI ANALISA & VISUALISASI: {today_str} ---")
    
    # Siapkan Canvas Gambar (Grid 2x2)
    # Menggunakan style gelap untuk keseluruhan gambar
    plt.style.use('dark_background') 
    fig, axs = plt.subplots(2, 2)
    fig.suptitle(f"FOREX DAILY FORECAST (IDR)\nGenerated on: {today_str}", fontsize=18, fontweight='bold', color='white', y=0.98)
    
    # Flatten axs agar mudah di-loop (dari matriks 2x2 jadi list 1 baris)
    axs_flat = axs.flatten()
    
    currency_data_storage = []

    # Loop through currencies and corresponding axes
    for i, (currency, ticker) in enumerate(TICKERS.items()):
        ax = axs_flat[i] # Ambil axis yang sesuai
        print(f"Memproses {currency}...")
        
        try:
            df = yf.download(ticker, period="2y", interval="1d", progress=False)
            if df.empty or len(df) < 30: continue

            df.reset_index(inplace=True)
            # Standardisasi kolom (Kode lama Anda)
            if 'Date' in df.columns: df['ds'] = df['Date']
            else: df['ds'] = df.index
            if isinstance(df.columns, pd.MultiIndex):
                 try: df['y'] = df[('Close', ticker)]
                 except KeyError: df['y'] = df['Close']
            else: df['y'] = df['Close']
            df = df[['ds', 'y']].dropna()
            
            # --- DATA UNTUK PLOTTING ---
            # Kita hanya mengambil 90 hari terakhir agar grafik terlihat jelas
            df_recent = df.tail(90).copy() 
            current_price = float(df.iloc[-1]['y'])

            # --- MODELING ---
            m = Prophet(daily_seasonality=True, yearly_seasonality=True)
            m.fit(df)
            future = m.make_future_dataframe(periods=1)
            forecast = m.predict(future)
            predicted_price = forecast.iloc[-1]['yhat']
            
            signal, change_txt = get_recommendation(current_price, predicted_price)
            
            # --- PANGGIL FUNGSI PLOTTING ---
            plot_currency(ax, currency, df_recent, current_price, predicted_price, signal, change_txt)
            print(f"-> Plot {currency} berhasil.")

        except Exception as e:
            print(f"Error pada {currency}: {e}")
            ax.text(0.5, 0.5, f"Error Data {currency}", ha='center', color='red') # Tulis error di gambar jika gagal
            continue

    # Merapikan layout agar tidak bertabrakan
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Memberi ruang untuk judul di atas
    
    # Footer Credit
    fig.text(0.5, 0.01, "Automated Analysis by ForexBot | Not Financial Advice", ha='center', fontsize=10, color='gray')

    # --- SIMPAN GAMBAR ---
    filename = "forex_forecast.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight') # DPI 300 agar tajam
    print(f"\n=== GAMBAR BERHASIL DIBUAT: {filename} ===")
    print("Silakan cek di bagian 'Artifacts' pada hasil workflow GitHub Actions.")

if __name__ == "__main__":
    run_analysis_and_plot()
