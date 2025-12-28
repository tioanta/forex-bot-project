import yfinance as yf
import pandas as pd
from prophet import Prophet
from datetime import datetime, timedelta

# --- UPDATE 1: Perbaikan Kode Ticker (USDIDR=X lebih stabil) ---
TICKERS = {
    'USD': 'USDIDR=X', 
    'JPY': 'JPYIDR=X',
    'KRW': 'KRWIDR=X',
    'CNY': 'CNYIDR=X'
}

def get_recommendation(current_price, predicted_price, threshold=0.002):
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
            # Download Data
            df = yf.download(ticker, period="2y", interval="1d", progress=False)
            
            # --- UPDATE 2: Cek apakah data kosong agar tidak error ---
            if df.empty:
                print(f"!! Gagal mengambil data untuk {currency}. Lewati.")
                continue

            df.reset_index(inplace=True)
            
            # Standardisasi kolom Date
            if 'Date' in df.columns:
                df['ds'] = df['Date']
            else:
                df['ds'] = df.index
            
            # Standardisasi kolom Close (Flatten multi-index jika ada)
            if isinstance(df.columns, pd.MultiIndex):
                # Cara aman ambil kolom Close jika multi-level
                try:
                    df['y'] = df[('Close', ticker)]
                except KeyError:
                    df['y'] = df['Close']
            else:
                df['y'] = df['Close']
            
            df = df[['ds', 'y']].dropna()

            if len(df) < 10:
                print(f"!! Data terlalu sedikit untuk {currency}. Lewati.")
                continue

            # Ambil harga terakhir
            current_price = float(df.iloc[-1]['y'])
            
            # Training Prophet
            m = Prophet(daily_seasonality=True, yearly_seasonality=True)
            m.fit(df)
            
            # Prediksi Besok
            future = m.make_future_dataframe(periods=1)
            forecast = m.predict(future)
            predicted_price = forecast.iloc[-1]['yhat']
            
            # Buat Rekomendasi
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

    print("\n" + "="*50)
    print("HASIL AKHIR")
    print("="*50)
    
print("\n" + "="*50)
    print("HASIL AKHIR")
    print("="*50)
    
    if not results:
        print("Tidak ada data yang berhasil diolah.")
    else:
        # Print manual agar rapi tanpa library tambahan
        # Perhatikan baris ini harus satu baris panjang (jangan diputus)
        print(f"{'Mata Uang':<10} | {'Harga Skrg':<15} | {'Prediksi':<15} | {'Sinyal'}")
        print("-" * 60)
        for row in results:
            print(f"{row['Mata Uang']:<10} | {row['Harga Skrg']:<15} | {row['Prediksi']:<15} | {row['Sinyal']}")

if __name__ == "__main__":
    run_analysis()
