act:
	source venv/bin/activate

pengu1:
	analyze -p 1 -s PENGU/USDT -t 1d -m sma,ema
	analyze -m sma,ema,ema_cross,macd,rsi -s PENGU/USDT -p 2 -a 7 -t 1h