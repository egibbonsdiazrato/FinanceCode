# Bond Course:

The most basic equation for TVM is:

$$ PV = \frac{FV}{\left( 1 + r \right)^T}$$

where PV is present value, FV is future/face value, r is the interest paid over one period and T is the number of 
periods to maturity.

This equation can be applied to zero coupon bonds (ZCB). ZCB are a type of bonds which pay no coupon. This means that
only one payment of the future or face value is made at maturity.

Bonds which pay coupons can be thought of as a sum of zero coupon bonds. As bonds usually pay fixed coupons, they may
be expressed using the equation below:

$$ TFV = \sum_{t=1}^{T}\frac{c*FV}{(1 + r)^t} + \frac{FV}{(1 + r)^T}$$

where TFV stands for the theoretical fair value, T is the number of periods to maturity, c is the coupon as a fraction
of the FV, r is the interest rate paid over one period. The two terms are the sum of the discounting of the coupon 
payments and the discounted final payment at maturity. This equation can be further simplified to:

$$ PV = C\frac{1 - 1(1+r)^{-n}}{r} + \frac{FV}{(1 + r)^T}$$

where C is the absolute value of the coupon payment.