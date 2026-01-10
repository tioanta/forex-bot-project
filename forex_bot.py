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
# Ukuran diperpanjang sedikit ke bawah agar muat 3 baris
plt.rcParams['figure.figsize'] = (12, 18) 
plt.rcParams['font.family'] = 'sans-serif'

COLORS = {'BELI': '#2ecc71', 'JUAL': '#e74c3c', 'HOLD': '#95a5a6'}

# 1. Ticker Forex (DITAMBAHKAN SAR/ARAB SAUDI)
FOREX_TICKERS = {
    'USD': 'USDIDR=X', 
    'JPY': 'JPYIDR=X', 
    'KRW': 'KRWIDR=X', 
    'CNY': 'CNYIDR=X',
    'SAR': 'SARIDR=X'  # <--- Riyal Arab Saudi
}

# 2. Ticker Saham LQ45 Pilihan (Untuk Caption)
STOCK_TICKERS = [
    'BBCA.JK', 'BBRI.JK', 'BMRI.JK', 'BBNI.JK', 
    'TLKM.JK', 'ASII.JK', 'UNTR.JK', 'ICBP.JK',
    'ADRO.JK', 'PGAS.JK', 'MDKA.JK', 'GOTO.JK',
    'UNVR.JK', 'BRIS.JK'
]

# --- BANK PERTANYAAN INTERAKTIF ---
QUESTIONS = [
    "Menurutmu besok Rupiah bakal menguat atau melemah lagi? 🤔",
    "Tim 'Serok Bawah' atau Tim 'Tunggu Gajian' nih? ☝️",
    "Ada yang lagi nabung buat liburan ke Jepang/Korea/Umroh tahun ini? 🕋✈️", # Update Umroh
    "Jujur, rate segini udah worth it buat tukar atau tunggu dulu? 💸",
    "Buat yang suka jastip, momen kayak gini bikin cuan atau boncos? 📦",
    "Kalau punya uang dingin, mending beli Dollar atau beli Seblak? 🤣",
    "Prediksi kamu: Minggu depan Dollar tembus berapa? 📉📈",
    "Tag temanmu yang lagi butuh banget info kurs ini! 👇",
    "Siapa yang dompetnya ikut menangis lihat grafik hari ini? 😭",
    "Mending nyesel beli sekarang atau nyesel gak beli kemarin? 🧐",
    "Lagi pantau mata uang apa nih buat traveling? ✈️",
    "Gimana strategi cuan kamu hari ini? Share dong di komen! 🧠"
]

def get_recommendation(current_price, predicted_price, threshold=0.002):
    diff_percent = (predicted_price - current_price) / current_price
    if diff_percent > threshold: return "BELI", diff_percent
    elif diff_percent < -threshold: return "JUAL", diff_percent
    else: return "HOLD", diff_percent

def plot_currency(ax, currency, df_recent, current_price, predicted_price, signal, change_pct):
    ax.plot(df_recent['ds'], df_recent['y'], label='Historis', color='#3498db', linewidth=2)
    prediction_date = df_recent['ds'].iloc[-1] + timedelta(days=1)
    ax.scatter(prediction_date, predicted_price, color='#e67e22', s=150, zorder=5)
    
    ax.annotate(f"{current_price:,.0f}", (df_recent['ds'].iloc[-1], current_price), 
                xytext=(10, -20), textcoords='offset points', color='white', fontsize=9)
    ax.annotate(f"{predicted_price:,.0f}", (prediction_date, predicted_price), 
                xytext=(10, 20), textcoords='offset points', color='#e67e22', fontweight='bold', fontsize=9)

    rec_color = COLORS.get(signal, COLORS['HOLD'])
    props = dict(boxstyle='round,pad=0.5', facecolor=rec_color, alpha=0.9, edgecolor='none')
    
    change_txt = f"+{change_pct*100:.2f}%" if change_pct > 0 else f"{change_pct*100:.2f}%"
    
    ax.text(0.05, 0.95, f"{currency}\n{signal} ({change_txt})", transform=ax.transAxes, fontsize=12,
            fontweight='bold', color='white', verticalalignment='top', bbox=props)

    ax.set_title(f"{currency} to IDR", fontsize=12, color='white')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
    ax.set_facecolor('#2c3e50')

def analyze_top_stocks(tickers, top_n=3):
    print(f"--- MENGANALISA {len(tickers)} SAHAM LQ45 ---")
    recommendations = []

    for ticker in tickers:
        try:
            df = yf.download(ticker, period="1y", interval="1d", progress=False)
            if df.empty or len(df) < 60: continue
            
            df = df.reset_index()
            if 'Date' in df.columns: df['ds'] = df['Date']
            else: df['ds'] = df.index

            if isinstance(df.columns, pd.MultiIndex):
                try: df['y'] = df[('Close', ticker)]
                except KeyError: df['y'] = df['Close']
            else: df['y'] = df['Close']
            
            df = df[['ds', 'y']].dropna()
            current_price = float(df.iloc[-1]['y'])

            m = Prophet(daily_seasonality=True)
            m.fit(df)
            future = m.make_future_dataframe(periods=7)
            forecast = m.predict(future)
            
            future_price = forecast.iloc[-1]['yhat']
            diff_percent = (future_price - current_price) / current_price
            
            if diff_percent > 0.005:
                recommendations.append({
                    'code': ticker.replace('.JK', ''),
                    'buy': current_price,
                    'sell': future_price,
                    'potential': diff_percent * 100
                })
        except Exception as e:
            continue
    
    recommendations.sort(key=lambda x: x['potential'], reverse=True)
    return recommendations[:top_n]

def upload_to_instagram(image_path, caption_text):
    print("--- MENCOBA UPLOAD KE INSTAGRAM ---")
    
    username = os.environ.get("IG_USERNAME")
    password = os.environ.get("IG_PASSWORD")
    session_id = os.environ.get("IG_SESSION_ID")

    cl = Client()
    cl.delay_range = [1, 3]

    try:
        if session_id:
            print("Mencoba Login menggunakan Session ID...")
            cl.login_by_sessionid(session_id)
        else:
            print("Menggunakan Username & Password...")
            cl.login(username, password)

        try:
            cl.account_info()
            print("✅ Login Terverifikasi! Sesi Valid.")
        except Exception as e:
            print(f"❌ Login Gagal/Sesi Kadaluwarsa. Pesan: {e}")
            raise Exception("Session Invalid")

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
    
    # 1. BUAT GAMBAR FOREX
    plt.style.use('dark_background')
    
    # --- PERUBAHAN DISINI (LAYOUT) ---
    # Ubah grid menjadi 3 Baris x 2 Kolom untuk menampung 5 mata uang
    fig, axs = plt.subplots(3, 2) 
    fig.suptitle(f"MARKET UPDATE (IDR)\n{today_str}", fontsize=18, fontweight='bold', color='white')
    axs_flat = axs.flatten()
    
    caption_summary = f"🤖 Prediksi Pasar & Valas - {today_str}\n\n"
    caption_summary += "🌏 **UPDATE KURS VALAS**:\n"
    
    has_forex_data = False

    for i, (currency, ticker) in enumerate(FOREX_TICKERS.items()):
        # Pastikan index tidak melebihi jumlah grid yang tersedia
        if i >= len(axs_flat): break
            
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
            
            current = float(df.iloc[-1]['y'])
            pred = forecast.iloc[-1]['yhat']
            
            signal, change = get_recommendation(current, pred)
            plot_currency(ax, currency, df.tail(60), current, pred, signal, change)
            
            icon = "🟢" if signal == "BELI" else "🔴" if signal == "JUAL" else "⚪"
            caption_summary += f"{icon} {currency}: {signal} (IDR {current:,.0f})\n"
            has_forex_data = True
            
        except Exception as e:
            print(f"Skip {currency}: {e}")
            continue

    # Matikan (hide) slot grafik yang kosong sisa grid
    for j in range(len(FOREX_TICKERS), len(axs_flat)):
        axs_flat[j].axis('off')

    if not has_forex_data:
        print("Tidak ada data forex.")
        return

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    filename = "forex_forecast.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print("Gambar Forex berhasil dibuat.")

    # 2. ANALISA SAHAM
    top_stocks = analyze_top_stocks(STOCK_TICKERS, top_n=3)
    
    if top_stocks:
        caption_summary += "\n🚀 **REKOMENDASI SAHAM LQ45** (Prediksi 7 Hari):\n"
        for s in top_stocks:
            caption_summary += (
                f"💎 **{s['code']}** (Potensi +{s['potential']:.2f}%)\n"
                f"   🛒 Area Beli: {s['buy']:,.0f}\n"
                f"   🎯 Target Jual: {s['sell']:,.0f}\n"
            )
    else:
        caption_summary += "\n⚠️ Pasar saham sedang volatile/sideways, tidak ada sinyal kuat hari ini.\n"

    # 3. PENUTUP CAPTION
    selected_question = random.choice(QUESTIONS)
    caption_summary += f"\n❓ QOTD: {selected_question}\n"
    caption_summary += (
        "\nDisclaimer: Analisis berbasis AI (Prophet). DYOR.\n"
        "#saham #investasi #IHSG #forex #SAR #Riyal #Umroh #cuan"
    )
    
    # 4. UPLOAD
    upload_to_instagram(filename, caption_summary)

if __name__ == "__main__":
    run_bot()
