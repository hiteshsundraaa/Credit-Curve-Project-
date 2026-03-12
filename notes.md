NOTES:
The Credit Curve (Term Structure of Credit)
The Credit Curve is a graphical representation of the relationship between the credit spread (risk premium) and the time to maturity for a specific issuer or credit rating.

Normal (Upward Sloping): Investors demand higher compensation for the uncertainty of lending over longer periods.

Inverted: A rare signal suggesting that the issuer is at high risk of default in the short term, but may be safer if they survive the immediate crisis.

Humped: Often indicates a period of mid-term economic transition or specific "refinancing" risks for a company.

2. Credit Spreads
The Credit Spread is the additional yield an investor requires to hold a risky bond instead of a "risk-free" government bond (like a US Treasury) of the same maturity.

Spread=Yield 
Risky
​	
 −Yield 
Risk−Free
​	
 
Z-Spread (Zero-Volatility Spread): A more advanced version of the simple spread. It is the constant spread that must be added to the risk-free spot rate curve to make the present value of the bond's cash flows equal to its market price.

3. The Bootstrapping Method
Market data usually only gives us "Par Yields" for specific maturities (e.g., 2yr, 5yr, 10yr). Bootstrapping is the recursive process used to derive Spot Rates (zero-coupon rates) from these yields.

We start with the shortest maturity (where yield = spot rate).

We use that rate to discount the first coupon of a longer bond.

We solve for the unknown spot rate of the longer maturity.

4. Interpolation Techniques
Since we don't have a bond for every single day of the year, we must "fill the gaps" between known data points:

Linear Interpolation: Connects dots with straight lines. Simple, but can create "kinks" in the curve that aren't realistic for trading.

Cubic Splines: Creates a smooth, continuous curve. This is the industry standard as it better reflects the fluid nature of interest rate movements.

5. Probability of Default (PD) & Recovery Rate
The credit curve is a market-implied estimate of risk:

Hazard Rate: The instantaneous probability of default.

Recovery Rate: The percentage of the bond's face value that investors expect to get back if the company fails (standard industry assumption is often 40% for senior unsecured debt).
