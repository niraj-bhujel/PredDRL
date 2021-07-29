import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
# plt.rcParams['font.family'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


# compare-1

# continuous_TD3_1 = pd.read_csv('/home/ros_admin/Documents/compare-1/5.csv')
# continuous_TD3_3 = pd.read_csv('/home/ros_admin/Documents/compare-1/2.csv')
# continuous_TD3_2 = pd.read_csv('/home/ros_admin/Documents/compare-1/3.csv')
# continuous_TD3_4 = pd.read_csv('/home/ros_admin/Documents/compare-1/4.csv')

# continuous_data = [continuous_TD3_2,continuous_TD3_4]
# continuous_data_1 = [continuous_TD3_3, continuous_TD3_1]
# smoothed_data = []
# for y in continuous_data:
#     smoothed = []
#     weight = 0.9
#     last = y['Value'].values[0]
#     for point in y['Value'].values:
#         smoothed_val = last * weight + (1 - weight) * point
#         smoothed.append(smoothed_val)
#         last = smoothed_val
#     smoothed_data.append(smoothed)

# for y in continuous_data_1:
#     smoothed = []
#     weight = 0.95
#     last = y['Value'].values[0]
#     for point in y['Value'].values:
#         smoothed_val = last * weight + (1 - weight) * point
#         smoothed.append(smoothed_val)
#         last = smoothed_val
#     smoothed_data.append(smoothed)
# # print(type(x), type(y), len(x), len(y))
# plt.figure()

# plt.plot(continuous_TD3_2['Step'].values, smoothed_data[0], label = 'iTD3-CLN')
# plt.plot(continuous_TD3_4['Step'].values, smoothed_data[1], label = 'iTD3-SN')

# plt.plot(continuous_TD3_1['Step'].values, smoothed_data[3], label = 'TD3-CLN')
# plt.plot(continuous_TD3_3['Step'].values, smoothed_data[2], label = 'TD3-SN')

# plt.xlabel('Training Steps',weight='bold',size=15)
# plt.ylabel('Reward/Episode',weight='bold',size=15)
# plt.legend(bbox_to_anchor=(1,0.3),loc=7)
# plt.grid(linestyle='-.')
# plt.savefig('/home/ros_admin/Documents/compare-1/test.png')
# plt.show()

#

#compare-2

continuous_TD3_1 = pd.read_csv('/home/ros_admin/Documents/compare-4/run-compare1_td3per_nstep-tag-Common_training_return.csv')
continuous_TD3_2 = pd.read_csv('/home/ros_admin/Documents/compare-4/run-20201218T020505.722331_TD3_-tag-Common_training_return2.csv')
continuous_TD3_3 = pd.read_csv('/home/ros_admin/Documents/compare-4/run-20210726T175645.141199_TD3_-tag-Common_training_return1.csv')
continuous_TD3_4 = pd.read_csv('/home/ros_admin/Documents/compare-4/6.csv')
continuous_TD3_5 = pd.read_csv('/home/ros_admin/Documents/compare-4/run-compare1_td3only2-tag-Common_training_return.csv')

# continuous_TD3_6 = pd.read_csv('/home/ros_admin/Documents/compare-2/4.csv')


# for i in continuous_TD3_1['Step'].values:
#     # if i >= 280000:
#         # continuous_TD3_1['Value'].values -= 500
#     print (continuous_TD3_1['Value'].values)

# for i in range (0,len(continuous_TD3_1['Step'])): 
#     if continuous_TD3_1['Step'][i] >= 280000:          
#         continuous_TD3_1['Value'][i] -=300
    # print (i)
            
            
# print (continuous_TD3_1['Value'])

continuous_data = [continuous_TD3_1, continuous_TD3_2,continuous_TD3_3,continuous_TD3_4,continuous_TD3_5]
# continuous_data_1 = [continuous_TD3_3]
smoothed_data = []
for y in continuous_data:
    smoothed = []
    weight = 0.96
    last = y['Value'].values[0]
    for point in y['Value'].values:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    smoothed_data.append(smoothed)


# for y in continuous_data_1:
#     smoothed = []
#     weight = 0.95
#     last = y['Value'].values[0]
#     for point in y['Value'].values:
#         smoothed_val = last * weight + (1 - weight) * point
#         smoothed.append(smoothed_val)
#         last = smoothed_val
#     smoothed_data.append(smoothed)
# print(type(x), type(y), len(x), len(y))
plt.figure()
plt.plot(continuous_TD3_1['Step'].values, smoothed_data[0], label = 'iTD3-CLN')
plt.plot(continuous_TD3_3['Step'].values, smoothed_data[2], label = 'TD3-SN-Nstep')
plt.plot(continuous_TD3_4['Step'].values, smoothed_data[3], label = 'TD3-CLN')
plt.plot(continuous_TD3_5['Step'].values, smoothed_data[4], label = 'TD3-SN-PER')



plt.plot(continuous_TD3_2['Step'].values, smoothed_data[1], label = 'TD3-SN')




# # plt.plot(continuous_SAC['Step'].values, smoothed_data[1],  label='no_transfer')
# plt.plot(continuous_PPO['Step'].values, smoothed_data[2], 'g:')

# plt.plot(continuous_DDPG['Step'].values, smoothed_data[3], label='DDPG')
plt.xlabel('Training Steps',weight='bold',size=15)
plt.ylabel('Reward/Episode',weight='bold',size=15)
plt.legend(bbox_to_anchor=(1,0.3),loc=7)
plt.grid(linestyle='-.')
plt.savefig('/home/ros_admin/Documents/compare-2/test.png')
plt.show()




##compare-3

# continuous_TD3_2 = pd.read_csv('/home/ros_admin/Documents/compare-3/2.csv')
# continuous_TD3_3 = pd.read_csv('/home/ros_admin/Documents/compare-3/3.csv')
# continuous_TD3_4 = pd.read_csv('/home/ros_admin/Documents/compare-3/4.csv')

# continuous_data = [continuous_TD3_2,continuous_TD3_3,continuous_TD3_4]
# # continuous_data_1 = [continuous_TD3_3]
# smoothed_data = []
# for y in continuous_data:
#     smoothed = []
#     weight = 0.95
#     last = y['Value'].values[0]
#     for point in y['Value'].values:
#         smoothed_val = last * weight + (1 - weight) * point
#         smoothed.append(smoothed_val)
#         last = smoothed_val
#     smoothed_data.append(smoothed)

# # for y in continuous_data_1:
# #     smoothed = []
# #     weight = 0.95
# #     last = y['Value'].values[0]
# #     for point in y['Value'].values:
# #         smoothed_val = last * weight + (1 - weight) * point
# #         smoothed.append(smoothed_val)
# #         last = smoothed_val
# #     smoothed_data.append(smoothed)
# # print(type(x), type(y), len(x), len(y))
# plt.figure()


# plt.plot(continuous_TD3_4['Step'].values, smoothed_data[2], label = 'TD3')
# plt.plot(continuous_TD3_3['Step'].values, smoothed_data[1], label = 'DDPG')
# plt.plot(continuous_TD3_2['Step'].values, smoothed_data[0], label = 'PPO')


# # # plt.plot(continuous_SAC['Step'].values, smoothed_data[1],  label='no_transfer')
# # plt.plot(continuous_PPO['Step'].values, smoothed_data[2], 'g:')

# # plt.plot(continuous_DDPG['Step'].values, smoothed_data[3], label='DDPG')
# plt.xlabel('Training Steps',weight='bold',size=15)
# plt.ylabel('Reward/Episode',weight='bold',size=15)
# plt.legend(bbox_to_anchor=(1,0.3),loc=7)
# plt.grid(linestyle='-.')
# plt.savefig('/home/ros_admin/Documents/compare-2/test.png')
# plt.show()


#
#
# # discrete
# # continuous_PPO = pd.read_csv('/home/ywj/data/discrete/run-20191213T003216.106968_PPO_-tag-Common_training_return.csv')
# # continuous_SAC = pd.read_csv('/home/ywj/data/discrete/run-20191213T103406.918472_SAC_discrete_-tag-Common_training_return.csv')
# # continuous_Rainbow = pd.read_csv('/home/ywj/data/discrete/run-20191213T231752.919630_D6QN_-tag-Common_training_return.csv')
# #
# # continuous_data = [continuous_PPO, continuous_SAC, continuous_Rainbow]
# # smoothed_data = []
# # for y in continuous_data:
# #     smoothed = []
# #     weight = 0.99
# #     last = y['Value'].values[0]
# #     for point in y['Value'].values:
# #         smoothed_val = last * weight + (1 - weight) * point
# #         smoothed.append(smoothed_val)
# #         last = smoothed_val
# #     smoothed_data.append(smoothed)
# # # print(type(x), type(y), len(x), len(y))
# # plt.figure()
# # plt.plot(continuous_PPO['Step'].values, smoothed_data[0], label='PPO')
# # plt.plot(continuous_SAC['Step'].values, smoothed_data[1], label='SAC')
# # plt.plot(continuous_Rainbow['Step'].values, smoothed_data[2], label='Rainbow')
# # plt.xlabel('Steps')
# # plt.ylabel('Value')
# # plt.legend(loc='upper left')
# # plt.grid(linestyle='-.')
# # plt.savefig('/home/ywj/data/discrete/discrete.jpg')
# # plt.show()

# # turtlebot
# continuous_PPO = pd.read_csv('/home/ywj/data/test_500k_td3/run-20200311T005309.078819_TD3_-tag-Common_training_return.csv')
# smoothed_data = []
#
# smoothed = []
# weight = 0.9975
# last = continuous_PPO['Value'].values[0]
# for point in continuous_PPO['Value'].values:
#     smoothed_val = last * weight + (1 - weight) * point
#     smoothed.append(smoothed_val)
#     last = smoothed_val
# smoothed_data.append(smoothed)
# # print(type(x), type(y), len(x), len(y))
# plt.figure()
# plt.plot(continuous_PPO['Step'].values, smoothed_data[0], label='TD3')
#
# plt.xlabel('步数')
# plt.ylabel('奖励/回合')
# plt.legend(loc='upper left')
# plt.grid(linestyle='-.')
# plt.savefig('/home/ywj/data/picture/test_500k_td3/test_500k_td3.jpg')
# plt.show()
