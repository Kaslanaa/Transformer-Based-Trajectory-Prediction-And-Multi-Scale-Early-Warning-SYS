import Warning_system
import matplotlib.pylab as plt

if __name__=='__main__':
    Object=[0.23,0.45]#当前真实轨迹点
    plt.scatter(0.23,0.45,marker='o', facecolors='red')
    Object_Prediction=[0.25,0.5]#预测轨迹点
    plt.scatter(0.25,0.5,marker='*',facecolors='green')#
    #所有的机密点
    Confidential_Point_Set={"wallet":[0.14,0.32],"meal card":[0.5,0.2],"My glasses":[0.4,0.3],"My slippers":[0.8,0.7],"My garbage":[0.9,1.0],"Can't make it up anymore = =":[0.7,0.75]}
    system=Warning_system.warning_system(Object,Object_Prediction,Confidential_Point_Set)
    res=system.Calculate_Threat_level()
    for key in Confidential_Point_Set:
        if key in res:
            plt.scatter(Confidential_Point_Set.get(key)[0],Confidential_Point_Set.get(key)[1],marker='x',facecolors='blue',label=key)
        else:
            plt.scatter(Confidential_Point_Set.get(key)[0], Confidential_Point_Set.get(key)[1], marker='s',
                        facecolors='yellow',label=key)
    # 设置坐标轴标签和标题
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('warning system')
    plt.legend()
    # 显示图形
    plt.show()