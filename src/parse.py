import matplotlib.pyplot as plt

nums = []
for eachline in open('/Users/marcuslinture/Desktop/SuperMarioBrosRL/src/output.log', 'r'):
    if eachline.startswith('episode'):
        keep = eachline.index('reward:')
        keep += len('reward:')
        nums.append(float(eachline[keep:len(eachline)-1]))

episodes = [i for i in range(len(nums))]

avg = []
for i in range(int(len(nums)/10)):
    avg.append(sum(nums[i*10:(i+1)*10])/10)
setsten = [i*10 for i in range(len(avg))]

plt.plot(episodes, nums)
plt.plot(setsten, avg)
plt.show()

