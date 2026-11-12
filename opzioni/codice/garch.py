"""GJR-GARCH(1,1) con innovazioni t di Student: stima a finestra espandente, ristima ogni 21 sedute,
previsione della varianza giornaliera sigma2_{t+1} con sole informazioni fino a t."""
import sys, numpy as np, pandas as pd, warnings
from arch import arch_model
warnings.filterwarnings('ignore')
out = sys.argv[1]
u = pd.read_parquet(out + '/sottostante.parquet').close
r = (100 * np.log(u).diff()).dropna()
x = r.values; n = len(x)
MIN, STEP = 504, 21
res = np.full((n, 4), np.nan)     # sigma2_next, sigma2_lr, persistenza, nu
pars = []
for s in range(MIN, n, STEP):
    fit = arch_model(x[:s], mean='Constant', vol='GARCH', p=1, o=1, q=1, dist='t').fit(disp='off')
    mu, om, al, ga, be, nu = fit.params
    p = min(al + ga / 2 + be, 0.999)
    lr = om / (1 - p)
    s2 = fit.conditional_volatility[-1] ** 2      # sigma2 della seduta s-1
    e = x[s - 1] - mu
    for t in range(s - 1, min(s - 1 + STEP, n)):  # t = ultima seduta osservata
        if t > s - 1:
            e = x[t] - mu
        s2n = om + (al + ga * (e < 0)) * e * e + be * s2
        res[t] = (s2n, lr, p, nu)
        s2 = s2n
    pars.append((r.index[s - 1], mu, om, al, ga, be, nu, p, np.sqrt(lr * 252)))
g = pd.DataFrame(res, index=r.index, columns=['s2next', 's2lr', 'pers', 'nu'])
g['hv21'] = np.sqrt((r ** 2).rolling(21).sum() * 252 / 21) / 100
g['hv10'] = np.sqrt((r ** 2).rolling(10).sum() * 252 / 10) / 100
g.to_parquet(out + '/garch.parquet')
pp = pd.DataFrame(pars, columns=['data', 'mu', 'omega', 'alpha', 'gamma', 'beta', 'nu', 'pers', 'vol_lr'])
pp.to_csv(out + '/garch_parametri.csv', index=False)
print(pp.describe().T[['mean', 'min', 'max']].round(4))
print(g.dropna().tail(3))
