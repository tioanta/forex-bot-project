import os
import random
import yfinance as yf
import pandas as pd
from prophet import Prophet
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from instagrapi import Client


# --- KONFIGURASI ---
sns.set_theme(style="darkgrid")
plt.rcParams['figure.figsize'] = (12, 12)
plt.rcParams['font.family'] = 'sans-serif'

COLORS = {'BELI': '#2ecc71', 'JUAL': '#e74c3c', 'HOLD': '#95a5a6'}
TICKERS = {
    'USD': 'USDIDR=X', 'JPY': 'JPYIDR=X', 
    'KRW': 'KRWIDR=X', 'CNY': 'CNYIDR=X'
}
pertanyaan = [
    "Menurutmu besok bakal naik atau turun lagi? 🤔",
    "Tim Jual atau Tim Tahan nih hari ini? ☝️",
    "Ada yang lagi nabung buat ke Jepang? 🇯🇵",
    "Rate segini worth it gak buat tukar sekarang? 💸"
]
def get_recommendation(current_price, predicted_price, threshold=0.002):
    diff_percent = (predicted_price - current_price) / current_price
    if diff_percent > threshold: return "BELI", f"+{diff_percent*100:.2f}%"
    elif diff_percent < -threshold: return "JUAL", f"{diff_percent*100:.2f}%"
    else: return "HOLD", f"{diff_percent*100:.2f}%"

def plot_currency(ax, currency, df_recent, current_price, predicted_price, signal, change_txt):
    ax.plot(df_recent['ds'], df_recent['y'], label='Historis', color='#3498db', linewidth=2)
    prediction_date = df_recent['ds'].iloc[-1] + timedelta(days=1)
    ax.scatter(prediction_date, predicted_price, color='#e67e22', s=150, zorder=5)
    
    ax.annotate(f"{current_price:,.0f}", (df_recent['ds'].iloc[-1], current_price), 
                xytext=(10, -20), textcoords='offset points', color='white', fontsize=9)
    ax.annotate(f"{predicted_price:,.0f}", (prediction_date, predicted_price), 
                xytext=(10, 20), textcoords='offset points', color='#e67e22', fontweight='bold', fontsize=9)

    rec_color = COLORS.get(signal, COLORS['HOLD'])
    props = dict(boxstyle='round,pad=0.5', facecolor=rec_color, alpha=0.9, edgecolor='none')
    ax.text(0.05, 0.95, f"{currency}\n{signal}", transform=ax.transAxes, fontsize=12,
            fontweight='bold', color='white', verticalalignment='top', bbox=props)

    ax.set_title(f"{currency} to IDR", fontsize=12, color='white')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
    ax.set_facecolor('#2c3e50')

def upload_to_instagram(image_path, caption_text):
    print("--- MENCOBA UPLOAD KE INSTAGRAM ---")
    
    username = os.environ.get("IG_USERNAME")
    password = os.environ.get("IG_PASSWORD")
    session_id = os.environ.get("IG_SESSION_ID")

    cl = Client()
    cl.delay_range = [1, 3]

    try:
        # 1. PROSES LOGIN
        if session_id:
            print("Mencoba Login menggunakan Session ID...")
            cl.login_by_sessionid(session_id)
        else:
            print("Menggunakan Username & Password...")
            cl.login(username, password)

        # 2. VERIFIKASI LOGIN (Langkah Baru)
        # Kita coba ambil info akun sendiri untuk memastikan tiket valid
        try:
            cl.account_info()
            print("✅ Login Terverifikasi! Sesi Valid.")
        except Exception as e:
            print(f"❌ Login Gagal/Sesi Kadaluwarsa. Pesan: {e}")
            raise Exception("Session Invalid")

        # 3. PROSES UPLOAD
        print("Sedang mengupload foto...")
        media = cl.photo_upload(
            path=image_path,
            caption=caption_text
        )
        print(f"🎉 SUKSES MUTLAK! Foto berhasil diupload. Media PK: {media.pk}")
        
    except Exception as e:
        print(f"!! Gagal Upload Instagram: {e}")
        print("Saran: Ambil ulang Session ID baru dari browser (Incognito) dan JANGAN Log Out.")

def run_bot():
    today_str = datetime.now().strftime('%Y-%m-%d')
    print(f"--- MULAI PROSES: {today_str} ---")
    
    plt.style.use('dark_background')
    fig, axs = plt.subplots(2, 2)
    fig.suptitle(f"FOREX PREDICTION (IDR)\n{today_str}", fontsize=18, fontweight='bold', color='white')
    axs_flat = axs.flatten()
    
    caption_summary = f"🤖 Prediksi Kurs Rupiah (IDR) - {today_str}\n\n"
    has_data = False

    for i, (currency, ticker) in enumerate(TICKERS.items()):
        ax = axs_flat[i]
        try:
            df = yf.download(ticker, period="1y", interval="1d", progress=False)
            if df.empty: continue
            
            df.reset_index(inplace=True)
            if 'Date' in df.columns: df['ds'] = df['Date']
            else: df['ds'] = df.index
            
            if isinstance(df.columns, pd.MultiIndex):
                try: df['y'] = df[('Close', ticker)]
                except KeyError: df['y'] = df['Close']
            else: df['y'] = df['Close']
            
            df = df[['ds', 'y']].dropna()
            if len(df) < 30: continue

            m = Prophet(daily_seasonality=True)
            m.fit(df)
            future = m.make_future_dataframe(periods=1)
            forecast = m.predict(future)
            
            current = f
            loat(df.iloc[-1]['y'])
            pred = forecast.iloc[-1]['yhat']
            
            signal, change = get_recommendation(current, pred)
            plot_currency(ax, currency, df.tail(60), current, pred, signal, change)
            caption_summary += f"💵 {currency}: {signal} ({change})\n"
            has_data = True
            
        except Exception as e:
            print(f"Skip {currency}: {e}")
            continue

    if not has_data:
        print("Tidak ada data.")
        return

    plt.tight_layout(rect=[0, 0.03, 1, 0.90])
    filename = "forex_forecast.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print("Gambar berhasil dibuat.")
    selected_question = random.choice(pertanyaan)
    caption_summary += f"\n\n{selected_question}\n\nDisclaimer: Not Financial Advice.\n#forex #investasi..."

    
    upload_to_instagram(filename, caption_summary)

if __name__ == "__main__":
    run_bot()
