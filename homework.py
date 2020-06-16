'''
author = 'caoguolin.曹国琳'
date = '2020.6.15'
student ID = '119116011340'
email = 'kafeicgl@sina.com'
'''

#####Read ME:
#####本脚本旨在模拟二维温度场随时间的变化，使用的解法是有限差分方法，求解有限差分方程的方法是直接迭代法
#####最终的结果主要是一些图像，包括对给定温度下不同区域节点的冷却过程的模拟，不同浇筑温度下各个区域节点冷却过程对比
#####同时，最后还分别画出了在开始时刻，中间时刻，结束时刻的物件的温度场分布三维线框图
#####程序中共包含17处注释，可放心食用(*^_^*)


#####程序逻辑：
#####首先先从整体上看看程序的主题逻辑：
#####1.根据实际的工艺过程和工艺材料来设置一些对应的工艺参数
#####2.设置有限差分的节点大小和时间步长等迭代参数
#####3.将根据实际情况推导出来的具体数学形式的有限差分方程写入到代码中的封包函数，方便调用
#####4.由于后面的实验对比需要，先设置一个初始温度取不同值的小循环，分别赋值初始温度给初始温度矩阵
#####5.根据时间步长和时间迭代上限设置迭代大循环，在每个循环里用定义好的有限差分方程和上一次的温度场矩阵求解下一次的温度场矩阵
#####6.大循环结束，根据循环中提取出来的相关信息来画相关图像



#####以下为程序代码，会在相应的地方用注释对代码进行解释
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D       ###note1：导入需要的包，这里主要是数学和画图相关的包


###设置物化性参数和
# T0 = 860            ###note2:由于后面需要改变初始温度进行对照实验，所以这里暂且将T0隐藏
TA = 20
C = 500
P = 7800
K = 30
C = C/4186
P = P/1000
K = K/418.6
BT = 0.5              ###note3：T0为初始温度，TA为环境温度,C为比热,P为密度,K为热传导率,BT为换热系数

###设置节点数和节点大小
DX = 1
DY = 1
DS = 0.01
N1 = 30
N2 = 30
S2 = 50               ###note4:DX为X方向步长，DY为y方向步长，DS为时间步长，N1为X方向节点数，N2为y方向节点数,S2为时间上限

####设置节点数索引，range是从0开始数的
n1 = N1 - 1
n2 = N2 - 1           ###note5:设置n1和n2是为了方便range函数的遍历，其实就是节点数

###定义公式
FX = K*DS/(C*P*DX*DX)
FY = K*DS/(C*P*DY*DY)
F1 = 1-2*FX-2*FY      ###note6:定义这两个公式是在数学推导中为了简化内部节点差分方程的形式


###定义差分方程
def fd(T1):
    for i in range(1,n1):
        for j in range(1,n2):
            T2[i][j] = F1*T1[i][j] + FX*(T1[i+1][j]+T1[i-1][j]) + FY*(T1[i][j+1]+T1[i][j-1])      
    ###note7:以上为内部节点的差分方程

    for j in range(1,n2):
        T2[0][j] = F1*T1[0][j] + 2*FX*T1[1][j] + FY*(T1[0][j+1]+T1[0][j-1]) + 2*BT*DS*(TA-T1[0][j])/(P*C*DX)
        T2[n1][j] = F1*T1[n1][j] + 2*FX*T1[n1-1][j] + FY*(T1[n1][j+1]+T1[n1][j-1]) + 2*BT*DS*(TA-T1[n1][j])/(P*C*DX)

    for i in range(1,n1):
        T2[i][0] = F1*T1[i][0] + 2*FY*T1[i][1] + FX*(T1[i+1][0]+T1[i-1][0]) + 2*BT*DS*(TA-T1[i][0])/(P*C*DX)
        T2[i][n2] = F1*T1[i][n2] + 2*FY*T1[i][n2-1] + FX*(T1[i+1][n2]+T1[i-1][n2]) + 2*BT*DS*(TA-T1[i][n2])/(P*C*DX)
    ###note8:以上为四条边界线的差分方程

    
    T2[0][0] = F1*T1[0][0] + 2*FX*T1[1][0] + 2*FY*T1[0][1] + (2*BT*DS/(P*C))*(1/DX+1/DY)*(TA-T1[0][0])
    T2[0][n2] = F1*T1[0][n2] + 2*FX*T1[1][n2] + 2*FY*T1[0][n2-1] + (2*BT*DS/(P*C))*(1/DX+1/DY)*(TA-T1[0][n2])
    T2[n1][0] = F1*T1[n1][0] + 2*FX*T1[n1-1][0] + 2*FY*T1[n1][1] + (2*BT*DS/(P*C))*(1/DX+1/DY)*(TA-T1[n1][0])
    T2[n1][n2] = F1*T1[n1][n2] + 2*FX*T1[n1-1][n2] + 2*FY*T1[n1][n2-1] + (2*BT*DS/(P*C))*(1/DX+1/DY)*(TA-T1[n1][n2])
    ###note9:以上为四个角的差分方程

    return(T2)

###note10：以上函数是通过上一次的温度场矩阵来求解下一次的温度矩阵，整个温度场由内部，边界，角组成


T11 = np.zeros((N1,N2),dtype=float)
T12 = np.zeros((N1,N2),dtype=float)
T13 = np.zeros((N1,N2),dtype=float)
T2 = np.zeros((N1,N2),dtype=float)    ###note11:初始化温度场矩阵，并定义其格式为数组

T0 = [500,860,1000]
T01 = T0[0]
T02 = T0[1]
T03 = T0[2]          

for i in range(n1+1):
    for j in range(n2+1):
        T11[i][j] = T01
        T12[i][j] = T02
        T13[i][j] = T03      ###note12:设置三种初始温度500，860，1000进行对照实验


S = 0
Tm1 = T11
Tm2 = T12
Tm3 = T13
node1 = [] 
node2 = []
node3 = []
tem1 = []
tem3 = []
gett1 = []
gett2 = []
gett3 = []
TT = []
while S < S2 - 0.000005:
    node1.append(Tm2[0][0])
    node2.append(Tm2[0][15])
    node3.append(Tm2[15][15])
    TT.append(Tm2.tolist())
    tem1.append(Tm1[15][15])
    tem3.append(Tm3[15][15])
    T22 = fd(Tm2).copy()
    T21 = fd(Tm1).copy()
    T23 = fd(Tm3).copy()
    Tm2 = T22
    Tm1 = T21
    Tm3 = T23
    S = S + DS                 ###note13:设置大循环，按时间步长叠加循环，跳出条件为到达时间上限



#####对比画冷却曲线
y1 = []
y2 = []
y3 = []
y4 = []
y5 = []
for i in range(0,500,5):
    y1.append(node1[i])
    y2.append(node2[i])
    y3.append(node3[i])
    y4.append(tem1[i])
    y5.append(tem3[i])


x = range(1,np.array(y1).shape[0]+1)

plt.plot(x,y1,label='Corner',color='r')
plt.plot(x,y2,label='Boundary',color='b')
plt.plot(x,y3,label='Center',color='g')
plt.xlabel('time step')
plt.ylabel('temperature')
plt.title('change of temperature')
plt.legend()


plt.plot()
plt.savefig('temperature.png', dpi=200)       
plt.show()                                ###note14：此图像为860℃下，温度场内角点，边界点，中心点的冷却曲线对比
                                          ########## 同时需要说明这里画出了整体5000步的整体趋势，等步长取出了50个时间节点



plt.plot(x,y3,label='860℃',color='r')
plt.plot(x,y4,label='500℃',color='b')
plt.plot(x,y5,label='1000℃',color='g')
plt.xlabel('time step')
plt.ylabel('temperature')
plt.title('change of center temperature')
plt.legend()


plt.plot()
plt.savefig('center temperature.png', dpi=200)
plt.show()                               ###note15：这里包含三张图像，分别是角点，边界点，中心点在不同初始温度下的冷却曲线对比
                                         #########  这里由于上面画的总体趋势图中温度变化主要集中在前端
                                         #########  所以这里和后面都只取了前500步，等步长取出了100个时间节点

####缩小区间画860℃下的温度变化图
y11 = []
y22 = []
y33 = []
for i in range(0,500,5):
    y11.append(node1[i])
    y22.append(node2[i])
    y33.append(node3[i])



x = range(1,np.array(y11).shape[0]+1)

plt.plot(x,y11,label='Corner',color='r')
plt.plot(x,y22,label='Boundary',color='b')
plt.plot(x,y33,label='Center',color='g')
plt.xlabel('time step')
plt.ylabel('temperature')
plt.title('change of temperature')
plt.legend()


plt.plot()
plt.savefig('temperature_new.png', dpi=200)
plt.show()                                  ###note16：此图像放大温度变化趋势而缩小时间区间的新的860℃下不同区域节点的冷却曲线
       


###三维热图
gett1 = TT[1]
gett2 = TT[500]
gett3 = TT[4950]


fig=plt.figure()
ax2 = Axes3D(fig)

xx = np.arange(1,31,1)
yy = np.arange(1,31,1)
X, Y = np.meshgrid(xx, yy)

Z1 = np.array(gett1)
Z2 = np.array(gett2)
Z3 = np.array(gett3)


#作图
ax2.plot_wireframe(X,Y,Z2,rstride = 1, cstride = 1,cmap='rainbow')
# plt.savefig('initial.png',dpi=200)
plt.savefig('intermediate',dpi=200)
# plt.savefig('end',dpi=200)
plt.show()                                 ###note17：这里包含三张三维图像，对应860℃下初始，中间，结束时刻的温度分布线框图







