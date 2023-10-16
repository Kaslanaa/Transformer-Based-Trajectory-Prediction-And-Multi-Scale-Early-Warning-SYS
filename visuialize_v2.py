import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import ast
import torch.nn as nn
from matplotlib.font_manager import FontProperties
import Warning_system
from matplotlib.legend_handler import HandlerTuple
import matplotlib.patches as patches
import time
import matplotlib.patches as patches
import matplotlib.lines as lines
font = FontProperties(family='Times New Roman', size=12)
'''
#========================================应用(double alarm-double confidential)========================================#
'''
input_size = 2  # 输入特征维度
output_size = 2  # 输出特征维度
num_layers = 2  # Transformer编码器和解码器的层数
hidden_size = 64  # Transformer内部隐藏层维度
num_heads = 4  # 多头注意力机制中的头数
class TransformerModel(nn.Module):
    def __init__(self, input_size, output_size, num_layers, hidden_size, num_heads):
        super(TransformerModel, self).__init__()

        self.embedding = nn.Linear(input_size, hidden_size)
        self.transformer = nn.Transformer(
            d_model=hidden_size,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=hidden_size
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.embedding(x)  # 嵌入层
        x = x.permute(1, 0, 2)  # 调整维度顺序
        x = self.transformer(x, x)  # Transformer模型的前向传播
        x = x.permute(1, 0, 2)  # 调整维度顺序
        x = self.fc(x)  # 全连接层
        return x
model = TransformerModel(input_size, output_size, num_layers, hidden_size, num_heads)
model.load_state_dict(torch.load('best_model.pt'))
"""
导入训练后的模型
"""
hunman_trajority = pd.read_excel('testset_for_application.xlsx')
hunman_trajority = hunman_trajority.iloc[1:]
hunman_trajorities = np.array(hunman_trajority.iloc[:,8].values)
processed_human_trajorites = []
for i in range(hunman_trajorities.shape[0]):
    curr_person = ast.literal_eval(hunman_trajorities[i])[:7]
    processed_human_trajorites.append(curr_person)
processed_human_trajorites = torch.tensor(processed_human_trajorites).permute(1, 0, 2)
human_trajorites_prediction = model(processed_human_trajorites)
processed_human_trajorites = processed_human_trajorites.squeeze(1)
human_trajorites_prediction = human_trajorites_prediction.squeeze(1)

original_traj = processed_human_trajorites.detach().numpy()#原始轨迹点
predicted_traj = human_trajorites_prediction.detach().numpy()#得到预测轨迹点

"""
第一个人的原始轨迹以及预测轨迹
"""

hunman_trajority_plus = pd.read_excel('testset_for_application_plus.xlsx')
hunman_trajority_plus = hunman_trajority_plus.iloc[1:]
hunman_trajorities_plus = np.array(hunman_trajority_plus.iloc[:,8].values)
processed_human_trajorites_plus = []
for i in range(hunman_trajorities_plus.shape[0]):
    curr_person_plus = ast.literal_eval(hunman_trajorities_plus[i])[:7]
    processed_human_trajorites_plus.append(curr_person_plus)
processed_human_trajorites_plus = torch.tensor(processed_human_trajorites_plus).permute(1, 0, 2)
human_trajorites_prediction_plus = model(processed_human_trajorites_plus)
processed_human_trajorites_plus = processed_human_trajorites_plus.squeeze(1)
human_trajorites_prediction_plus = human_trajorites_prediction_plus.squeeze(1)

original_traj_plus = processed_human_trajorites_plus.detach().numpy()#另一个原始轨迹点
predicted_traj_plus = human_trajorites_prediction_plus.detach().numpy()#得到另一个点的预测轨迹点

Confidential_Point_Set = {"Confidential Point I":[-8.775,41.4],"Confidential Point II":[-8.825,41.25]}




"""
可视化
"""
import matplotlib.patches as patches

fig, axs = plt.subplots(2,1,figsize=(10, 10))
ax1 = axs[0]
ax2 = axs[1]

"""
初始化预警点，以及威胁程度
"""
ax2.set_xlabel('Time Serials',fontweight='bold')
ax2.set_ylabel('Threat Level Of All Confidential Point',fontweight='bold')
ax2.text(0,1.03,'SOTA Warning SYS (Multiview)',fontweight='bold')
ax2.text(5.8,1.145,'TRESHOLD',fontweight='bold',color='red')
ax2.grid(ls='--')
ax2.set_xlim(-0.1,6.5)
ax2.set_ylim(1.025,1.2)
ax2.axhline(1.14, color='red', linestyle='--',alpha=1)

system1 = Warning_system.warning_system(list(original_traj[6]),list(predicted_traj[0]),Confidential_Point_Set)
system2 = Warning_system.warning_system(list(original_traj_plus[6]),list(predicted_traj_plus[0]),Confidential_Point_Set)
res1_dic = system1.Calculate_Threat_level()
res2_dic = system2.Calculate_Threat_level()
res = []
for key in res1_dic:
    res.append(res1_dic[key]+res2_dic[key])
res1_cnt = res[0]
res2_cnt = res[1]
ax2.plot(0,res1_cnt,'g',alpha=0.5,marker='o',label='Confidential Point I')
ax2.plot(0,res2_cnt,'r',alpha=0.5,marker='o',label='Confidential Point II')

alarm1_cnt_sum = []
alarm2_cnt_sum = []

alarm1_cnt_sum.append(res1_cnt)
alarm2_cnt_sum.append(res2_cnt)
#================================================================================
ax1.set_xlim((-8.9, -8.5))
ax1.set_ylim((41.0, 41.5))
ax1.grid(ls='--')
ax1.set_xlabel('Detected X value',fontweight='bold')
ax1.set_ylabel('Detected Y value',fontweight='bold')
ax1.set_title('SOTA-TP-SYS For Multi-Scale Prediction AND Multi-Alerts', fontweight='bold')

x_cnt = []
y_cnt = []
x_cnt_plus = []
y_cnt_plus = []

x_current, y_current = original_traj[6, 0], original_traj[6, 1]
x_current_plus, y_current_plus = original_traj_plus[6, 0],original_traj_plus[6,1]

x_cnt.append(predicted_traj[0, 0])#先加入各自的第一个预测轨迹点
y_cnt.append(predicted_traj[0, 1])
x_cnt_plus.append(predicted_traj_plus[0,0])
y_cnt_plus.append(predicted_traj_plus[0,1])



Source_point = ax1.plot(x_current, y_current, 'g', alpha=0.8, marker='o',label='Source Point (X1,Y1)') # 当前真实轨迹的最后一个点
Source_point_plus = ax1.plot(x_current_plus, y_current_plus, 'r' , alpha=0.8, marker='o', label='Source Point (X2,Y2)')
# plt.legend(handles=[Source_point[0], Source_point_plus[0]], labels=['Source Point'], handler_map={tuple: HandlerTuple(ndivide=None)})

First_line = ax1.plot([x_current,predicted_traj[0,0]], [y_current,predicted_traj[0,1]], linestyle='--', color='green',alpha=0.2,label='Predicted Trajectory Line')
First_line_plus = ax1.plot([x_current_plus,predicted_traj_plus[0,0]],[y_current_plus,predicted_traj_plus[0,1]],linestyle='--',color='black',alpha=0.2,label='Predicted Trajectory Line')
# plt.legend(handles=[(First_line[0],First_line_plus[0])],labels=['Predicted Trajectory Line'],handler_map={tuple: HandlerTuple(ndivide=None)})

ax1.plot(-8.775,41.4,color = 'purple',alpha=1,marker='8',label='Confidential point I')#机密点1
ax1.plot(-8.825,41.25,color='purple',alpha=1,marker='^',label='Confidential point II')#机密点2
ax1.text(-8.89,41.01,'Demo.',fontsize=12, fontweight='bold')
ax2.text(0.1,0.1,'SOTA Warning SYS',fontsize=12, fontweight='bold')
ax1.text(-8.75,41.3,'ALARM AREA',fontsize=10, fontweight='bold',color='red')

predict_point_ani = ax1.scatter(x_cnt, y_cnt, c='b', alpha=0.4, marker='o',label='Predicted Trajectory point 1')  # 预测轨迹的第一个点
predict_point_ani_plus = ax1.scatter(x_cnt_plus,y_cnt_plus,c='y',alpha=0.4,marker='o', label='Predicted Trajectory point 2')

x_text_ani_prediction = ax1.text(-8.62, 41.05, '', fontsize=9,fontweight='bold')  # 预测轨迹的x标签
x_text_ani_prediction.set_text('Predicted X1:%.3f' % predicted_traj[0, 0])
y_text_ani_prediction = ax1.text(-8.62, 41.02, '', fontsize=9,fontweight='bold')  # 预测轨迹的y标签
y_text_ani_prediction.set_text('Predicted Y1:%.3f' % predicted_traj[0, 1])
x_text_ani_prediction_plus = ax1.text(-8.85,41.09,'',fontsize=9,fontweight='bold')
x_text_ani_prediction_plus.set_text('Predicted X2:%.3f'%predicted_traj_plus[0,0])
y_text_ani_prediction_plus = ax1.text(-8.85,41.06,'',fontsize=9,fontweight='bold')
y_text_ani_prediction_plus.set_text('Predicted Y2:%.3f'%predicted_traj_plus[0,1])

Thre_level_point1_ani_text = ax2.text(5,1.04,'',fontsize=9,fontweight='bold')
Thre_level_point2_ani_text = ax2.text(5,1.03,'',fontsize=9,fontweight='bold')


arrow_annotation = patches.FancyArrowPatch(
    posA=(x_current, y_current),
    posB=(predicted_traj[0,0], predicted_traj[0,1]),
    arrowstyle='->',
    alpha=0.4,
    color='green',
    mutation_scale=10
)

arrow_annotation_plus = patches.FancyArrowPatch(
    posA=(x_current_plus,y_current_plus),
    posB=(predicted_traj_plus[0,0],predicted_traj_plus[0,1]),
    arrowstyle='->',
    alpha=0.4,
    color='black',
    mutation_scale=10
)

ax1.add_patch(arrow_annotation)
ax1.add_patch(arrow_annotation_plus)

circle1 = patches.Circle((-8.775,41.4), 0.07, linestyle='dashed',edgecolor='red',alpha=0.4, facecolor='none',label='Warning area(Base)')
circle2 = patches.Circle((-8.775,41.4), 0.025, edgecolor='red',alpha=0.4, facecolor='none')
circle3 = patches.Circle((-8.775,41.4), 0.02, linestyle='dashed',edgecolor='red',alpha=0.4, facecolor='none')

circle4 = patches.Circle((-8.825,41.25), 0.07, linestyle='dashed',edgecolor='red',alpha=0.4, facecolor='none')
circle5 = patches.Circle((-8.825,41.25), 0.025, edgecolor='red',alpha=0.4, facecolor='none')
circle6 = patches.Circle((-8.825,41.25), 0.02, linestyle='dashed',edgecolor='red',alpha=0.4, facecolor='none')

ax1.add_patch(circle1)
ax1.add_patch(circle2)
ax1.add_patch(circle3)
ax1.add_patch(circle4)
ax1.add_patch(circle5)
ax1.add_patch(circle6)
ax1.legend(fontsize=8)
ax2.legend(fontsize=10,loc='upper right')


def update(num):
    if num > 0:#从第二个预测轨迹点开始
        x_cnt.append(predicted_traj[num, 0])
        y_cnt.append(predicted_traj[num, 1])
        x_cnt_plus.append(predicted_traj_plus[num,0])
        y_cnt_plus.append(predicted_traj_plus[num,1])
        predict_point_ani.set_offsets(np.column_stack((x_cnt, y_cnt)))
        predict_point_ani_plus.set_offsets(np.column_stack((x_cnt_plus,y_cnt_plus)))
        x_text_ani_prediction.set_text('Predicted X1:%.3f' % predicted_traj[num, 0])
        y_text_ani_prediction.set_text('Predicted Y1:%.3f' % predicted_traj[num, 1])
        x_text_ani_prediction_plus.set_text('Predicted X2:%.3f' % predicted_traj_plus[num, 0])
        y_text_ani_prediction_plus.set_text('Predicted Y2:%.3f' % predicted_traj_plus[num, 1])

        arrow_annotation_plus.set_positions((predicted_traj_plus[num-1,0],predicted_traj_plus[num-1,1]),(predicted_traj_plus[num,0],predicted_traj_plus[num,1]))
        arrow_annotation.set_positions((predicted_traj[num-1, 0], predicted_traj[num-1, 1]), (predicted_traj[num, 0], predicted_traj[num, 1]))
        ax1.plot([predicted_traj_plus[num - 1, 0], predicted_traj_plus[num, 0]],
                 [predicted_traj_plus[num - 1, 1], predicted_traj_plus[num, 1]], linestyle='--', color='black',
                 alpha=0.2,label='Warning Trend')
        ax1.plot([predicted_traj[num-1, 0], predicted_traj[num, 0]], [predicted_traj[num-1, 1], predicted_traj[num, 1]], linestyle='--', color='green',alpha=0.2)
        """
        绘制ax2
        """
        system1_temp = Warning_system.warning_system(list(predicted_traj[num-1]),list(predicted_traj[num]),Confidential_Point_Set)
        system2_temp = Warning_system.warning_system(list(predicted_traj_plus[num-1]),list(predicted_traj_plus[num]),Confidential_Point_Set)
        res1_dic_temp = system1_temp.Calculate_Threat_level()
        res2_dic_temp = system2_temp.Calculate_Threat_level()
        res_temp = []
        for key in res1_dic_temp:
            res_temp.append(res1_dic_temp[key] + res2_dic_temp[key])
        res1_cnt_temp = res_temp[0]
        alarm1_cnt_sum.append(res1_cnt_temp)
        res2_cnt_temp = res_temp[1]
        alarm2_cnt_sum.append(res2_cnt_temp)
        ax2.plot(num, res1_cnt_temp, 'g', alpha=0.5, marker='o')
        ax2.plot(num, res2_cnt_temp, 'r', alpha=0.5, marker='o')
        if res1_cnt_temp>=1.14:
            ax2.text(num+0.1,alarm1_cnt_sum[num]+0.01, '!',fontsize=12,fontweight='bold',color='red')
            Thre_level_point1_ani_text.set_text('Threat Level of P1:%.3f' % res1_cnt_temp)
            Thre_level_point1_ani_text.set_color('red')
        else:
            Thre_level_point1_ani_text.set_text('Threat Level of P1:%.3f' % res1_cnt_temp)
            Thre_level_point1_ani_text.set_color('black')

        if res2_cnt_temp >= 1.14:
            ax2.text(num+0.1, alarm2_cnt_sum[num]+0.01,'!' ,fontsize=12, fontweight='bold', color='red')
            Thre_level_point2_ani_text.set_text('Threat Level of P2:%.3f' % res2_cnt_temp)
            Thre_level_point2_ani_text.set_color('red')
        else:
            Thre_level_point2_ani_text.set_text('Threat Level of P2:%.3f' % res2_cnt_temp)
            Thre_level_point2_ani_text.set_color('black')
        ax2.plot([num-1,num],[alarm1_cnt_sum[num-1],alarm1_cnt_sum[num]],linestyle='--',color='blue',alpha=0.2)
        ax2.plot([num - 1, num], [alarm2_cnt_sum[num - 1], alarm2_cnt_sum[num]], linestyle='--', color='blue',
                 alpha=0.2)
    return [predict_point_ani, x_text_ani_prediction, y_text_ani_prediction, arrow_annotation, predict_point_ani_plus, x_text_ani_prediction_plus, y_text_ani_prediction_plus, arrow_annotation_plus, Thre_level_point2_ani_text, Thre_level_point1_ani_text ]

ani = animation.FuncAnimation(fig=fig, func=update, frames=np.arange(0, 7), interval=1000, repeat=False)
ani.save('animation_v2.gif', writer='pillow',dpi=400)
plt.subplots_adjust()
plt.show()
















