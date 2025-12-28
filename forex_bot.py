import yfinance as yf
import pandas as pd
from prophet import Prophet
from datetime import datetime, timedelta

# --- KONFIGURASI TICKER ---
# Menggunakan format USDIDR=X yang lebih stabil
TICKERS = {
    'USD': 'USDIDR=X', 
    'JPY': 'JPYIDR=X',
    'KRW': 'KRWIDR=X',
    'CNY': 'CNYIDR=X'
}

def get_recommendation(current_price, predicted_price, threshold=0.002):
    """
    Menentukan sinyal Beli/Jual berdasarkan selisih persentase.
    Threshold 0.002 artinya 0.2%
    """
    diff_percent = (predicted_price - current_price) / current_price
    if diff_percent > threshold:
        return "BELI (Naik)"
    elif diff_percent < -threshold:
        return "JUAL (Turun)"
    else:
        return "HOLD (Stabil)"

def run_analysis():
    print(f"--- ANALISA FOREX: {datetime.now().strftime('%Y-%m-%d')} ---\n")
    results = []

    for currency, ticker in TICKERS.items():
        print(f"Sedang memproses {currency} ({ticker})...")
        
        try:
            # 1. Download Data (2 Tahun terakhir)
            # progress=False agar log tidak penuh loading bar
            df = yf.download(ticker, period="2y", interval="1d", progress=False)
            
            # Cek apakah data kosong
            if df.empty:
                print(f"!! Gagal mengambil data untuk {currency}. Lewati.")
                continue

            df.reset_index(inplace=True)
            
            # 2. Standardisasi kolom Date
            if 'Date' in df.columns:
                df['ds'] = df['Date']
            else:
                df['ds'] = df.index
            
            # 3. Standardisasi kolom Close (Handling MultiIndex yfinance baru)
            if isinstance(df.columns, pd.MultiIndex):
                try:
                    # Coba ambil kolom ('Close', 'USDIDR=X')
                    df['y'] = df[('Close', ticker)]
                except KeyError:
                    # Fallback jika struktur berbeda
                    df['y'] = df['Close']
            else:
                df['y'] = df['Close']
            
            # Bersihkan data
            df = df[['ds', 'y']].dropna()

            # Pastikan data cukup untuk prediksi
            if len(df) < 10:
                print(f"!! Data terlalu sedikit untuk {currency}. Lewati.")
                continue

            # Ambil harga terakhir (Real)
            current_price = float(df.iloc[-1]['y'])
            
            # 4. Training Model Prophet
            m = Prophet(daily_seasonality=True, yearly_seasonality=True)
            m.fit(df)
            
            # 5. Prediksi Besok (H+1)
            future = m.make_future_dataframe(periods=1)
            forecast = m.predict(future)
            predicted_price = forecast.iloc[-1]['yhat']
            
            # 6. Buat Rekomendasi
            signal = get_recommendation(current_price, predicted_price)
            
            results.append({
                'Mata Uang': currency,
                'Harga Skrg': f"Rp {current_price:,.0f}",
                'Prediksi': f"Rp {predicted_price:,.0f}",
                'Sinyal': signal
            })
            print(f"OK. {currency} Selesai.")

        except Exception as e:
            print(f"Error pada {currency}: {e}")
            continue

    # --- BAGIAN REPORTING ---
    print("\n" + "="*65)
    print("HASIL AKHIR PREDIKSI")
    print("="*65)
    
    if not results:
        print("Tidak ada data yang berhasil diolah.")
    else:
        # Header Tabel
        print(f"{'Mata Uang':<10} | {'Harga Skrg':<15} | {'Prediksi':<15} | {'Sinyal'}")
        print("-" * 65)
        # Isi Tabel
        for row in results:
            print(f"{row['Mata Uang']:<10} | {row['Harga Skrg']:<15} | {row['Prediksi']:<15} | {row['Sinyal']}")

if __name__ == "__main__":
    run_analysis()
