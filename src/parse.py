import matplotlib.pyplot as plt

nums = []
for eachline in open('/Users/marcuslinture/Desktop/SuperMarioBrosRL/src/output.log', 'r'):
    if eachline.startswith('episode'):
        keep = eachline.index('reward:')
        keep += len('reward:')
        nums.append(float(eachline[keep:len(eachline)-1]))

dueling = []
for eachline in open('/Users/marcuslinture/Desktop/SuperMarioBrosRL/src/output-2.log', 'r'):
    if eachline.startswith('episode'):
        keep = eachline.index('reward:')
        keep += len('reward:')
        dueling.append(float(eachline[keep:len(eachline)-1]))

episodes = [i for i in range(len(nums))]
d_episodes = [i for i in range(len(dueling))]

avg = []
for i in range(int(len(nums)/10)):
    avg.append(sum(nums[i*10:(i+1)*10])/10)
setsten = [i*10 for i in range(len(avg))]

duelingavg = []
for i in range(int(len(dueling)/10)):
    duelingavg.append(sum(dueling[i*10:(i+1)*10])/10)
setsten_d = [i*10 for i in range(len(duelingavg))]

#plt.plot(episodes, nums)
#plt.plot(setsten, avg)
plt.plot(d_episodes, dueling)
plt.plot(setsten_d, duelingavg)

plt.show()

