import numpy as np
class warning_system:
    """
    @w
    Source="基于历史轨迹集的轨迹预测与目标预警系统_王娜"
    This class is designed for early warning, which is based on the object and its prediction, as well as the confidential points set
    The inputs are : Current object, Its prediction, The full Confidential_Point_Set
    The output is: Threat level of WarningSet
    =======================================================================================
    Step1: find the nearest Confidential Point, append it to WarningSet
    Step2: computing the One-step transition probability matrix and two-step transition probability matrix to get sitej and siteI, append them
    Step3: find the nearest Confidential Point(Object_Prediction), append it to WarningSet
    Step4: Find a point with a cosine similarity greater than 0 to the actual direction of motion
    ======================================================================================
    Object=[x,y](list)
    Object_Prediction=[x,y](list)
    Confidential_Point_Set={"name1":[x1,y1],"name2":[x2,y2],"name3":[x3,y3]}
    """
    def __init__(self,Object,Object_Prediction,Confidential_Point_Set):
        self.Confidential_Point_Set=Confidential_Point_Set
        self.Object=Object
        self.Object_Prediction=Object_Prediction
        self.WarningSet=[]
        self.results={}
    def find_nearst_point(self,point):
        distance_dic={}
        for key in self.Confidential_Point_Set:
            dx=point[0]-self.Confidential_Point_Set[key][0]
            dy=point[1]-self.Confidential_Point_Set[key][1]
            distance=np.sqrt(dx**2+dy**2)
            distance_dic[key]=distance
        sorted_names = sorted(distance_dic, key=distance_dic.get, reverse=False)#将距离进行排序，找出小的键
        return next(iter(sorted_names))
    def One_step_transition_probability_matrix(self):#计算所有site的一阶概率转移矩阵，注意，这是基于欧式距离的，如果要改的话，学长自行定义判别函数
        n=len(self.Confidential_Point_Set)
        Sites=list(self.Confidential_Point_Set.keys())
        matrix=np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                dx=self.Confidential_Point_Set.get(Sites[i])[0]-self.Confidential_Point_Set.get(Sites[j])[0]
                dy=self.Confidential_Point_Set.get(Sites[i])[1]-self.Confidential_Point_Set.get(Sites[j])[1]
                matrix[i][j]=np.sqrt(dx**2+dy**2)
        return matrix
    def Two_step_transition_probability_matrix(self):
        One_step_matrix=self.One_step_transition_probability_matrix()
        return np.dot(One_step_matrix,One_step_matrix)
    def CreatingWarningSet(self):
        Sites = list(self.Confidential_Point_Set.keys())
        Site_i=self.find_nearst_point(self.Object)#找到距离Object最近的机密点Site_i
        self.WarningSet.append(Site_i)
        Site_m=self.find_nearst_point(self.Object_Prediction)#找到距离预测预测点最近的机密点Site_m
        self.WarningSet.append(Site_m)
        One_step_matrix=self.One_step_transition_probability_matrix()#一步概率转移矩阵
        Two_step_matrix=self.Two_step_transition_probability_matrix()#二步概率转移矩阵
        Site_i_index=[i for i in range(len(self.Confidential_Point_Set)) if Sites[i]==Site_i]#找出Site_i的索引
        One_step_matrix=One_step_matrix[Site_i_index,:]
        Two_step_matrix=Two_step_matrix[Site_i_index,:]
        self.WarningSet.append(Sites[np.argmin(One_step_matrix)])
        self.WarningSet.append(Sites[np.argmin(Two_step_matrix)])
        return self.WarningSet
    def Calculate_Threat_level(self):
        warning_set=self.CreatingWarningSet()
        Threat_table={}
        for i in warning_set:
            dx=self.Object[0]-self.Confidential_Point_Set.get(i)[0]
            dy=self.Object[1]-self.Confidential_Point_Set.get(i)[1]
            dis=np.sqrt(dx**2+dy**2)
            Threat=1/(1+np.exp(-dis))
            Threat_table[i]=Threat
        return Threat_table
