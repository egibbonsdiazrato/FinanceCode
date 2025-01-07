# Bond Course:

The most basic equation for TVM is:

$$ PV = \frac{FV}{\left( 1 + r \right)^T}$$

where PV is present value, FV is future/face value, r is the interest paid over one period and T is the number of 
periods to maturity.

This equation can be applied to zero coupon bonds (ZCB). ZCB are a type of bonds which pay no coupon. This means that
only one payment of the future or face value is made at maturity.

Bonds which pay coupons can be thought of as a sum of zero coupon bonds. As bonds usually pay fixed coupons, they may
be expressed using the equation below:

$$ PV = \sum_{t=1}^{T} \frac{c*FV}{(1 + r)^t} + \frac{FV}{(1 + r)^T}$$

where TFV stands for the theoretical fair value, T is the number of periods to maturity, c is the coupon as a fraction
of the FV, r is the interest rate paid over one period. The two terms are the sum of the discounted coupon 
payments and the discounted final payment at maturity. This equation can be further simplified to:

$$ V = C \sum_{t=1}^{T} \frac{1}{(1 + r)^t} + \frac{FV}{(1 + r)^T} = C \left( \frac{1 - 1(1+r)^{-n}}{r} \right) + \frac{FV}{(1 + r)^T}$$

where $V$ is value and $C=cFV$ is the absolute value of the coupon payment.

The Duration (or Macaulay duration) is defined as the sum of the weighted cashflows over the lifetime of a bond. This
returns 

The modified duration is a measure of sensitivity of bond price to yield change. This is given by:

$$ ModD(y) = \frac{-1}{V}\frac{\partial V}{\partial y} = -\frac{\partial ( \ln ( V ) )}{\partial y} $$

This modified duration may be approximated 