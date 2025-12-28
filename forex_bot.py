import yfinance as yf
import pandas as pd
from prophet import Prophet
from datetime import datetime, timedelta

# Konfigurasi Mata Uang (Yahoo Finance Tickers)
# Won, Yen, Yuan, Dollar ke IDR
TICKERS = {
    'USD': 'IDR=X',
    'JPY': 'JPYIDR=X',
    'KRW': 'KRWIDR=X',
    'CNY': 'CNYIDR=X'
}

def get_recommendation(current_price, predicted_price, threshold=0.002):
    """
    Logika sederhana: 
    Jika prediksi naik > 0.2% dari harga sekarang -> BELI
    Jika prediksi turun > 0.2% dari harga sekarang -> JUAL
    Sisanya -> HOLD
    """
    diff_percent = (predicted_price - current_price) / current_price
    
    if diff_percent > threshold:
        return "BELI (Harga diprediksi NAIK)"
    elif diff_percent < -threshold:
        return "JUAL (Harga diprediksi TURUN)"
    else:
        return "HOLD (Pergerakan stabil)"

def run_analysis():
    print(f"--- ANALISA FOREX: {datetime.now().strftime('%Y-%m-%d')} ---\n")
    
    results = []

    for currency, ticker in TICKERS.items():
        print(f"Sedang memproses {currency}...")
        
        # 1. Download Data (2 Tahun terakhir)
        df = yf.download(ticker, period="2y", interval="1d", progress=False)
        df.reset_index(inplace=True)
        
        # Handling format data yfinance terbaru
        if 'Date' in df.columns:
            df['ds'] = df['Date']
        else:
            df['ds'] = df.index
            
        # Ambil kolom Close price saja
        # Note: yfinance kadang return MultiIndex, kita flatten dulu
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
            
        df['y'] = df['Close']
        df = df[['ds', 'y']].dropna()

        # Ambil harga penutupan terakhir (Today)
        current_price = df.iloc[-1]['y']
        
        # 2. Training Model Prophet
        m = Prophet(daily_seasonality=True, yearly_seasonality=True)
        m.fit(df)
        
        # 3. Prediksi Besok (H+1)
        future = m.make_future_dataframe(periods=1)
        forecast = m.predict(future)
        
        predicted_price = forecast.iloc[-1]['yhat']
        
        # 4. Buat Rekomendasi
        signal = get_recommendation(current_price, predicted_price)
        
        # Simpan hasil untuk report
        results.append({
            'Mata Uang': currency,
            'Harga Skrg': f"Rp {current_price:,.2f}",
            'Prediksi Besok': f"Rp {predicted_price:,.2f}",
            'Rekomendasi': signal
        })

    print("\n" + "="*50)
    print("HASIL PREDIKSI HARIAN")
    print("="*50)
    
    # Print Tabel Sederhana
    df_res = pd.DataFrame(results)
    print(df_res.to_markdown(index=False))

if __name__ == "__main__":
    run_analysis()
