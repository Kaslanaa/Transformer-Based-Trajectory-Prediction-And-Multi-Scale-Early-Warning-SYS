import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
'''
初始画布
'''
fig = plt.figure()
plt.grid(ls='--')

x = np.linspace(0,2*np.pi,100)#100个横坐标
y = np.sin(x)
'''
对象初始化
'''
crave_ani = plt.plot(x,y,'red',alpha=0.5)[0]#返回线条对象
point_ani = plt.plot(0,0,'r',alpha=0.4,marker='o')[0]#返回第一个点的对象
xtext_ani = plt.text(5,0.8,'',fontsize=12)#x的文本对象
ytext_ani = plt.text(5,0.7,'',fontsize=12)#y的文本对象
ktext_ani = plt.text(5,0.6,'',fontsize=12)#k的文本对象

def tangent_line(x0,y0,k):
	xs = np.linspace(x0 - 0.5,x0 + 0.5,100)
	ys = y0 + k * (xs - x0)
	return xs,ys

#计算斜率的函数
def slope(x0):
	num_min = np.sin(x0 - 0.05)
	num_max = np.sin(x0 + 0.05)
	k = (num_max - num_min) / 0.1
	return k

#绘制切线
k = slope(x[0])
xs,ys = tangent_line(x[0],y[0],k)
tangent_ani = plt.plot(xs,ys,c='blue',alpha=0.8)[0]#第一条切线的对象

def updata(num):
	k=slope(x[num])#更新斜率
	xs,ys = tangent_line(x[num],y[num],k)#更新切线
	tangent_ani.set_data(xs,ys)#用set_data方法通过帧数快速地更新切线
	point_ani.set_data(x[num],y[num])#更新点，用set_data方法
	xtext_ani.set_text('x=%.3f'%x[num])#更新横坐标，用set_text方法
	ytext_ani.set_text('y=%.3f'%y[num])#更新纵坐标，用set_text方法
	ktext_ani.set_text('k=%.3f'%k)#更新斜率，用set_text方法
	return [point_ani,xtext_ani,ytext_ani,tangent_ani,k]

ani = animation.FuncAnimation(fig=fig,func=updata,frames=np.arange(0,100),interval=100)
plt.show()


