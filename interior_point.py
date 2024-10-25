import numpy as np
import matplotlib.pyplot as plt
# khoi tao
c = np.array([1, 2, 0])
A = np.array([1, 1, 1])
x = np.array([2, 2, 4])
I = np.eye(x.size)
xx = x[0:2]
n = 0
# mang chua cac diem da di qua 
vectors = np.array([[0, 0], xx])
while np.linalg.norm(vectors[n + 1] - vectors[n]) > 0.0000001:
  D = np.diag(x)
  n = n + 1
  A1 = np.dot(A, D)
  A1 = np.array([A1])
  c1 = np.dot(D, c)
  A1_T = A1.reshape(-1, 1)
  P = I - np.dot(np.dot(A1_T, np.linalg.inv(np.dot(A1, A1_T))), A1)
  cp = np.dot(P, c1)
  v = 100000000.0
  for i in cp:
    if v > i:
      v = i
  if v > 0: 
    print("khong co gia tri v nao be hon 0")
  v = -v 
  alpha = 0.5
  x1 = np.array([1,1,1]) + cp * (alpha/v)
  x = np.dot(D, x1)
  # lay hai phan tu dau cua x (x1, x2) va them vao tap chua cac diem da di qua
  xxx = x[0:2]
  xxx = [xxx]
  vectors = np.append(vectors, xxx, axis = 0)

print("so lan lap can thiet: ", n)
print("dap an cua bai toan la: x1 = ", vectors[n, 0], ", x2 = ", vectors[n, 1])
print("Gia tri toi uu cua ham muc tieu la: ", c[0] * vectors[n, 0] + c[1] * vectors[n, 1])

x = vectors[1:, 0]
y = vectors[1:, 1]

# ve cac diem lan luot di chuyen qua
plt.scatter(x, y, color='red', marker='o', s=12, label='Data points')
plt.plot(x,y, color = 'black')

plt.title('Scatter Plot of Data Points')
plt.xlabel('x1-axis')
plt.ylabel('x2-axis')
plt.xlim(-1, 10)
plt.ylim(-1, 10)

# khu vuc xac dinh
x1 = np.linspace(0, 10, 400)
x2 = 8 - x1
# duong x1 + x2 <= 8
x11 = np.linspace(-1, 10, 400)
x22 = 8 - x11
plt.fill_between(x1, 0 , x2, where=(x2 > 0), color='skyblue', alpha=0.5, label = '$x1 + x2 \\leq 8$')  

plt.plot(x11, x22, color = 'red', label = '$x1 + x2 = 8$')
plt.plot(x, y, color = 'blue', label = 'increase direction')
# x1 = 0
plt.plot([0, 0], [-1, 10], color = 'black', label = '$x1 = 0$')
# x2 = 0
plt.plot([-1, 10], [0, 0], color = 'black', label = '$x2 = 0$')

plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Feasible Region for $x1 > 0$, $x2 > 0$, and $x1 + x2 \\leq 8$')

plt.legend()
plt.grid(True)
plt.show() 