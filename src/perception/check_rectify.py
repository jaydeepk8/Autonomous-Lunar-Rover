import numpy as np
import cv2

K1 = np.array([[2068.44, 0, 964.405],
               [0, 2064.29, 593.512],
               [0, 0, 1]])
d1 = np.array([-0.108441, 0.158389, 0.000537106, -0.00127904])

K2 = np.array([[2065.6, 0, 952.098],
               [0, 2061.9, 606.559],
               [0, 0, 1]])
d2 = np.array([-0.114041, 0.178219, -0.000146877, -0.00112736])

R = np.array([[0.999997, 0.00172602, 0.0019164],
              [-0.00172807, 0.999998, 0.00106908],
              [-0.00191455, -0.00107239, 0.999998]])
T = np.array([[-0.301556], [0.000599], [0.001489]], dtype=np.float64)

SIZE = (1936, 1216)

for alpha in [0.0, 0.5, 1.0, -1.0]:
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K1, d1, K2, d2, SIZE, R, T, alpha=alpha)
    print(f"alpha={alpha:5}  fx={P1[0,0]:9.2f}  cx={P1[0,2]:8.2f}  cy={P1[1,2]:8.2f}")

print("\ntarget from rectifiedMaps.calibration:")
print("           fx=   1991.49  cx=  966.26  cy=  600.35")